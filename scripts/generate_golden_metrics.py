#!/usr/bin/env python3
"""
Generate GPU-specific baseline golden metrics for KernelBand.

This script benchmarks all (or selected) TritonBench kernels on the current GPU
and produces the {GPU_NAME}_golden_metrics/ folder required for speedup calculation.

Usage:
    python scripts/generate_golden_metrics.py                          # All kernels, auto-detect GPU
    python scripts/generate_golden_metrics.py --resume                 # Skip already-generated JSONs
    python scripts/generate_golden_metrics.py --kernels add_example    # Specific kernels only
    python scripts/generate_golden_metrics.py --multi-gpu              # Parallel across all GPUs
    python scripts/generate_golden_metrics.py --gpu-id 2               # Use specific GPU
    python scripts/generate_golden_metrics.py --dry-run                # Prepare scripts only
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Phase 0: Initialization helpers
# ---------------------------------------------------------------------------

def get_project_root():
    return Path(__file__).resolve().parent.parent


def detect_gpu_name(gpu_id=0):
    """Return GPU name with spaces replaced by underscores."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(gpu_id).replace(" ", "_")
    except Exception:
        return None


def detect_gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def detect_tritonbench_paths(project_root):
    """Detect TritonBench directory layout and return resolved paths.

    Returns:
        (perf_metrics_dir, golden_metrics_dir, kernels_dir)
    """
    # Try new (refactored) layout first
    new_kernels = project_root / "data" / "TritonBench" / "kernels"
    new_perf = project_root / "data" / "TritonBench" / "perf_metrics"
    new_golden = new_perf / "golden_metrics"

    if new_kernels.is_dir() and new_golden.is_dir():
        return new_perf, new_golden, new_kernels

    # Try old (original TritonBench repo) layout
    old_kernels = project_root / "data" / "TritonBench" / "data" / "TritonBench_G_v1"
    old_perf = project_root / "data" / "TritonBench" / "performance_metrics" / "perf_G"
    old_golden = old_perf / "golden_metrics"

    if old_kernels.is_dir() and old_golden.is_dir():
        return old_perf, old_golden, old_kernels

    # Fall back to new-format paths (will fail with a clear error later)
    return new_perf, new_golden, new_kernels


# ---------------------------------------------------------------------------
# Phase 1: Determine kernels
# ---------------------------------------------------------------------------

def discover_kernels(kernels_dir, golden_metrics_dir):
    """Return list of kernel names that have both a source file and a perf template."""
    kernel_files = sorted(
        f.stem for f in kernels_dir.iterdir()
        if f.suffix == ".py" and f.stem != "__init__" and not f.stem.startswith("__")
    )
    perf_templates = {f.stem.removesuffix("_perf") for f in golden_metrics_dir.iterdir() if f.suffix == ".py"}
    matched = [k for k in kernel_files if k in perf_templates]
    return matched


def apply_filters(kernels, selected_names, resume, output_dir):
    """Apply --kernels and --resume filters."""
    if selected_names:
        selected_set = set(selected_names)
        kernels = [k for k in kernels if k in selected_set]
        missing = selected_set - set(kernels)
        if missing:
            print(f"Warning: kernels not found: {', '.join(sorted(missing))}")

    if resume:
        kernels = [k for k in kernels if not (output_dir / f"{k}.json").exists()]

    return kernels


# ---------------------------------------------------------------------------
# Phase 2: Prepare perf scripts
# ---------------------------------------------------------------------------

def prepare_performance_utils(perf_metrics_dir, tmp_dir, output_dir):
    """Copy performance_utils.py to tmp_dir with folder_path patched."""
    src = perf_metrics_dir / "performance_utils.py"
    dst = tmp_dir / "performance_utils.py"

    with open(src, "r") as f:
        lines = f.readlines()

    patched_lines = []
    for line in lines:
        if "folder_path = " in line and "os.path" not in line:
            # Preserve indentation, replace the hardcoded path
            indent = line[: len(line) - len(line.lstrip())]
            line = f'{indent}folder_path = "{output_dir}"\n'
        patched_lines.append(line)

    with open(dst, "w") as f:
        f.writelines(patched_lines)


def prepare_perf_script(kernel_name, golden_metrics_dir, kernels_dir, tmp_dir,
                        output_dir, warmup, rep):
    """Apply transformations to a perf template and write to tmp_dir."""
    perf_file = golden_metrics_dir / f"{kernel_name}_perf.py"
    with open(perf_file, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        # T1: Replace sys.path.append to point at kernels dir and tmp dir
        if line.strip() == "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))":
            updated_lines.append(f"sys.path.append('{kernels_dir}')\n")
            updated_lines.append(f"sys.path.append('{tmp_dir}')\n")
            continue

        # T2: Direct kernel imports (remove TritonBench_v1 prefix)
        line = line.replace("from TritonBench_v1.", "from ")

        # T3: Inject warmup/rep into get_do_bench_config()
        line = line.replace(
            "op_perf.get_do_bench_config()",
            f"op_perf.get_do_bench_config(warmup={warmup}, rep={rep})"
        )

        # T4: Replace any remaining hardcoded folder_path strings
        if 'folder_path = ' in line and 'os.path' not in line and 'self.' not in line:
            indent = line[: len(line) - len(line.lstrip())]
            line = f'{indent}folder_path = "{output_dir}"\n'

        updated_lines.append(line)

    # T5: Wrap benchmark loop in try-except (for old-style templates that
    # have inline to_cuda -> results.append blocks instead of run_benchmark())
    content = "".join(updated_lines)
    content_lines = content.split("\n")
    tab = "    "
    index_1 = index_2 = None
    for i, cline in enumerate(content_lines):
        if "input_tensor = self.to_cuda(input_tensor_)" in cline:
            index_1 = i
        if "results.append(result)" in cline:
            index_2 = i + 1

    if index_1 is not None and index_2 is not None:
        for i in range(index_1, index_2):
            content_lines[i] = tab + content_lines[i]
        content_lines.insert(index_1, tab * 3 + "try:")
        content_lines.insert(index_2 + 1, tab * 3 + "except Exception as e:")
        content_lines.insert(index_2 + 2, tab * 4 + 'print(f"Failed to run benchmark for input tensor. Error: {e}")')
        content = "\n".join(content_lines)

    dst = tmp_dir / f"{kernel_name}_perf.py"
    with open(dst, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Phase 3: Execute benchmarks
# ---------------------------------------------------------------------------

def run_single_script(args):
    """Run a single perf script (used by both single-GPU and multi-GPU modes)."""
    script_path, gpu_id, timeout_sec, log_dir = args

    script_name = os.path.basename(script_path)
    log_file = log_dir / f"{script_name}.log"
    err_file = log_dir / f"{script_name}.err"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    timed_out = False
    return_code = None

    try:
        with open(log_file, "w") as log_f, open(err_file, "w") as err_f:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=log_f,
                stderr=err_f,
                env=env,
                timeout=timeout_sec,
            )
            return_code = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        with open(err_file, "a") as err_f:
            err_f.write(f"\nScript timed out after {timeout_sec} seconds\n")
    except Exception as e:
        with open(err_file, "a") as err_f:
            err_f.write(f"\nSubprocess failed: {e}\n")

    elapsed = time.time() - start
    success = return_code == 0 and not timed_out
    return script_name, success, elapsed, timed_out, return_code


def run_benchmarks_single(scripts, gpu_id, timeout, log_dir):
    """Run benchmarks sequentially on a single GPU."""
    results = []
    total = len(scripts)

    iterator = enumerate(scripts)
    if tqdm:
        pbar = tqdm(total=total, desc="Benchmarking")

    for idx, script in iterator:
        script_name, success, elapsed, timed_out, rc = run_single_script(
            (script, gpu_id, timeout, log_dir)
        )
        status = "TIMEOUT" if timed_out else ("OK" if success else f"FAIL(rc={rc})")
        msg = f"{status} {idx + 1}/{total}: {script_name} ({elapsed:.1f}s)"
        if tqdm:
            tqdm.write(msg)
            pbar.update(1)
        else:
            print(msg)

        results.append((script_name, success))

    if tqdm:
        pbar.close()

    return results


def run_benchmarks_multi(scripts, gpu_count, timeout, log_dir):
    """Run benchmarks in parallel across multiple GPUs."""
    args_list = [
        (scripts[i], i % gpu_count, timeout, log_dir)
        for i in range(len(scripts))
    ]
    results = []
    total = len(scripts)

    if tqdm:
        pbar = tqdm(total=total, desc=f"Benchmarking ({gpu_count} GPUs)")

    with Pool(processes=gpu_count) as pool:
        for script_name, success, elapsed, timed_out, rc in pool.imap_unordered(
            run_single_script, args_list
        ):
            status = "TIMEOUT" if timed_out else ("OK" if success else f"FAIL(rc={rc})")
            msg = f"{status}: {script_name} ({elapsed:.1f}s)"
            if tqdm:
                tqdm.write(msg)
                pbar.update(1)
            else:
                print(msg)
            results.append((script_name, success))

    if tqdm:
        pbar.close()

    return results


# ---------------------------------------------------------------------------
# Phase 4: Report
# ---------------------------------------------------------------------------

def report_results(output_dir, kernel_names, elapsed_total):
    """Print summary of generated golden metrics."""
    generated = []
    failed = []
    for name in kernel_names:
        json_path = output_dir / f"{name}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if len(data) > 0:
                    generated.append(name)
                    continue
            except (json.JSONDecodeError, Exception):
                pass
        failed.append(name)

    print(f"\n{'=' * 60}")
    print(f"  Golden Metrics Generation Report")
    print(f"{'=' * 60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Successful: {len(generated)}/{len(kernel_names)}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for name in failed:
            print(f"    - {name}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU-specific baseline golden metrics for KernelBand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_golden_metrics.py                          # All kernels
  python scripts/generate_golden_metrics.py --resume                 # Skip existing
  python scripts/generate_golden_metrics.py --kernels add_example    # Single kernel
  python scripts/generate_golden_metrics.py --multi-gpu              # All GPUs
  python scripts/generate_golden_metrics.py --dry-run                # Prep only
        """,
    )

    parser.add_argument("--kernels", nargs="+", default=None,
                        help="Kernel names to benchmark (without .py)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip kernels with existing JSON output")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU device ID for single-GPU mode (default: 0)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all available GPUs in parallel")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-script timeout in seconds (default: 600)")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Benchmark warmup iterations (default: 100)")
    parser.add_argument("--rep", type=int, default=1000,
                        help="Benchmark repetitions (default: 1000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override auto-detected output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare perf scripts without executing")
    parser.add_argument("--clean", action="store_true",
                        help="Remove _tmp/ and _logs/ directories after success")

    args = parser.parse_args()

    # ── Phase 0: Init ─────────────────────────────────────────────────────
    project_root = get_project_root()
    perf_metrics_dir, golden_metrics_dir, kernels_dir = detect_tritonbench_paths(project_root)

    if not kernels_dir.exists():
        print(f"Error: kernels directory not found: {kernels_dir}")
        return 1
    if not golden_metrics_dir.exists():
        print(f"Error: golden_metrics directory not found: {golden_metrics_dir}")
        return 1

    gpu_name = detect_gpu_name(args.gpu_id)
    if gpu_name is None:
        print("Error: No CUDA GPU detected. Make sure torch.cuda is available.")
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = perf_metrics_dir / f"{gpu_name}_golden_metrics"

    tmp_dir = output_dir / "_tmp"
    log_dir = output_dir / "_logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU detected: {gpu_name}")
    print(f"Output directory: {output_dir}")

    # ── Phase 1: Determine kernels ────────────────────────────────────────
    all_kernels = discover_kernels(kernels_dir, golden_metrics_dir)
    kernels = apply_filters(all_kernels, args.kernels, args.resume, output_dir)

    if not kernels:
        print("No kernels to benchmark (all filtered out or none matched).")
        return 0

    print(f"Kernels to benchmark: {len(kernels)} (of {len(all_kernels)} available)")
    if len(kernels) <= 10:
        for k in kernels:
            print(f"  - {k}")

    # ── Phase 2: Prepare perf scripts ─────────────────────────────────────
    print("\nPreparing perf scripts...")
    prepare_performance_utils(perf_metrics_dir, tmp_dir, output_dir)

    for kernel_name in kernels:
        prepare_perf_script(
            kernel_name, golden_metrics_dir, kernels_dir, tmp_dir,
            output_dir, args.warmup, args.rep,
        )

    print(f"Prepared {len(kernels)} scripts in {tmp_dir}")

    if args.dry_run:
        print("\n--dry-run: scripts prepared but not executed.")
        return 0

    # ── Phase 3: Execute benchmarks ───────────────────────────────────────
    scripts = sorted([tmp_dir / f"{k}_perf.py" for k in kernels])
    start_time = time.time()

    if args.multi_gpu:
        gpu_count = detect_gpu_count()
        if gpu_count < 2:
            print(f"Warning: only {gpu_count} GPU(s) detected, falling back to single-GPU mode.")
            results = run_benchmarks_single(scripts, args.gpu_id, args.timeout, log_dir)
        else:
            print(f"Running on {gpu_count} GPUs...")
            results = run_benchmarks_multi(scripts, gpu_count, args.timeout, log_dir)
    else:
        results = run_benchmarks_single(scripts, args.gpu_id, args.timeout, log_dir)

    elapsed_total = time.time() - start_time

    # ── Phase 4: Report ───────────────────────────────────────────────────
    report_results(output_dir, kernels, elapsed_total)

    # ── Phase 5: Cleanup ──────────────────────────────────────────────────
    if args.clean:
        all_ok = all(success for _, success in results)
        if all_ok:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(log_dir, ignore_errors=True)
            print("Cleaned up _tmp/ and _logs/ directories.")
        else:
            print("Skipping cleanup due to failures (logs preserved for debugging).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
