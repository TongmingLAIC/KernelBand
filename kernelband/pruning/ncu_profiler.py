"""
NCU Profiler for Strategy Pruning

This module profiles Triton kernels using NVIDIA Nsight Compute (NCU) to extract
accurate hardware performance metrics for profiler-based pruning.

Key features:
- Direct hardware measurement (no formula approximation)
- Multi-dimensional bottleneck analysis (DRAM/L1/L2/SM/Occupancy)
- Bank conflict detection
- Disk caching for performance
"""

import json
import os
import sys
import csv
import io
import re
import tempfile
import shutil
import subprocess
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import torch
from loguru import logger


class NCUProfilingResult:
    """Container for NCU profiling results"""

    def __init__(
        self,
        kernel_name: str,
        iteration: int,
        # Core hardware metrics
        dram_throughput_pct: float,
        l2_throughput_pct: float,
        l1_throughput_pct: float,
        sm_throughput_pct: float,
        occupancy_pct: float,
        # Auxiliary metrics
        duration_us: float,
        registers_per_thread: int,
        shared_memory_bank_conflicts: int,
        # Derived
        bottleneck_type: str,
        num_shapes: int,
        shape_results: Optional[List[Dict]] = None,
    ):
        self.kernel_name = kernel_name
        self.iteration = iteration

        # Core metrics
        self.dram_throughput_pct = dram_throughput_pct
        self.l2_throughput_pct = l2_throughput_pct
        self.l1_throughput_pct = l1_throughput_pct
        self.sm_throughput_pct = sm_throughput_pct
        self.occupancy_pct = occupancy_pct

        # Auxiliary metrics
        self.duration_us = duration_us
        self.registers_per_thread = registers_per_thread
        self.shared_memory_bank_conflicts = shared_memory_bank_conflicts

        # Derived
        self.bottleneck_type = bottleneck_type
        self.num_shapes = num_shapes
        self.shape_results = shape_results if shape_results else []

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "kernel_name": self.kernel_name,
            "iteration": self.iteration,
            "num_shapes": self.num_shapes,

            "core_metrics": {
                "dram_throughput_pct": self.dram_throughput_pct,
                "l2_throughput_pct": self.l2_throughput_pct,
                "l1_throughput_pct": self.l1_throughput_pct,
                "sm_throughput_pct": self.sm_throughput_pct,
                "occupancy_pct": self.occupancy_pct,
            },

            "auxiliary_metrics": {
                "duration_us": self.duration_us,
                "registers_per_thread": self.registers_per_thread,
                "shared_memory_bank_conflicts": self.shared_memory_bank_conflicts,
            },

            "derived": {
                "bottleneck_type": self.bottleneck_type,
            },

            "shape_results": self.shape_results,
        }


class NCUProfiler:
    """
    Profiler for Triton kernels using NVIDIA Nsight Compute (NCU).

    This profiler uses NCU command-line tool to extract accurate hardware
    performance metrics for profiler-based pruning decisions.

    Integrates with TritonBench dataset by extracting test_code from kernel files.
    """

    # NCU metrics to collect
    NCU_METRICS = [
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'gpu__time_duration.sum',
        'launch__registers_per_thread',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum',
    ]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        py_folder: Optional[str] = None,
    ):
        """
        Initialize NCU profiler.

        Args:
            cache_dir: Directory to cache profiling results. If None, no caching.
            py_folder: Path to TritonBench kernel folder (required for test_code extraction)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.py_folder = py_folder

        # Validate py_folder
        if py_folder:
            py_folder_path = Path(py_folder)
            if not py_folder_path.exists():
                logger.warning(f"py_folder not found: {py_folder}")
                self.py_folder = None

        # Validate GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. NCU profiler requires NVIDIA GPU.")

        self.gpu_name = torch.cuda.get_device_name(0)

        if "NVIDIA" not in self.gpu_name:
            raise RuntimeError(
                f"NCU profiler requires NVIDIA GPU, but found: {self.gpu_name}\n"
                f"NCU (NVIDIA Nsight Compute) only works with NVIDIA GPUs."
            )

        # Check NCU availability
        if not self._check_ncu_available():
            raise RuntimeError(
                "NCU (NVIDIA Nsight Compute) not found.\n"
                "Please install NCU from: https://developer.nvidia.com/nsight-compute"
            )

        logger.info(f"NCU Profiler initialized for GPU: {self.gpu_name}")
        logger.info(f"NCU version: {self._get_ncu_version()}")
        if self.py_folder:
            logger.info(f"TritonBench folder: {self.py_folder}")

        # Cache for profiling results
        self._cache = {}

        # Cache for test_code extraction
        self._test_code_cache = {}

        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory: {self.cache_dir}")

    def _compute_code_hash(self, code: str) -> str:
        """
        Compute hash of code for caching purposes.

        Normalizes code by removing comments and excessive whitespace,
        then computes MD5 hash.

        Args:
            code: Triton kernel code string

        Returns:
            16-character hash string
        """
        # Normalize code:
        # 1. Remove comments (lines starting with #)
        normalized = re.sub(r'#.*', '', code)
        # 2. Compress whitespace (multiple spaces/tabs/newlines -> single space)
        normalized = re.sub(r'\s+', ' ', normalized)
        # 3. Strip leading/trailing whitespace
        normalized = normalized.strip()

        # Compute MD5 hash (first 16 characters)
        hash_obj = hashlib.md5(normalized.encode('utf-8'))
        return hash_obj.hexdigest()[:16]

    def _check_ncu_available(self) -> bool:
        """Check if NCU is available in PATH"""
        try:
            result = subprocess.run(
                ['ncu', '--version'],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_ncu_version(self) -> str:
        """Get NCU version string"""
        try:
            result = subprocess.run(
                ['ncu', '--version'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return "unknown"

    def profile_best_code(
        self,
        kernel_name: str,
        best_code: str,
        iteration: int,
    ) -> Optional[NCUProfilingResult]:
        """
        Profile a best_code using NCU.

        Args:
            kernel_name: Name of the kernel (e.g., "add_example.py")
            best_code: Best code string from memory['best_code'][0]
            iteration: Current iteration number (for caching)

        Returns:
            NCUProfilingResult object or None if profiling failed
        """
        # Validate kernel_name to prevent path traversal attacks
        if not kernel_name or not isinstance(kernel_name, str):
            raise ValueError(f"Invalid kernel_name: {kernel_name}")

        if not kernel_name.endswith('.py'):
            raise ValueError(f"kernel_name must end with .py, got: {kernel_name}")

        # Prevent path traversal attacks
        if '..' in kernel_name or '/' in kernel_name or '\\' in kernel_name:
            raise ValueError(
                f"kernel_name contains invalid path characters: {kernel_name}. "
                f"Expected simple filename like 'add_example.py'"
            )

        # Validate iteration
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError(f"iteration must be non-negative int, got: {iteration}")

        # Compute code hash for caching
        code_hash = self._compute_code_hash(best_code)
        cache_key = f"{kernel_name}_{code_hash}"

        # Check memory cache
        if cache_key in self._cache:
            logger.info(f"    💾 Using memory cache: {cache_key} (iter={iteration})")
            return self._cache[cache_key]

        # Check disk cache
        if self.cache_dir:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                self._cache[cache_key] = cached_result
                logger.info(f"    💾 Using disk cache: {cache_key} (iter={iteration}, {cached_result.num_shapes} shapes)")
                return cached_result

        logger.info(f"    🔬 Running NCU profiler: {cache_key} (iter={iteration}, no cache found)")

        # Create temporary script
        temp_dir = None
        result = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="ncu_profile_")
            script_path = Path(temp_dir) / "kernel_script.py"

            # ============================================================
            # Step 1: Generate kernel script (preserve autotune)
            # ============================================================
            script_content = self._generate_kernel_script(best_code, kernel_name)
            with open(script_path, 'w') as f:
                f.write(script_content)

            logger.debug(f"Profiling {kernel_name} with warmup+profile strategy in {temp_dir}")

            # ============================================================
            # Step 2: Warmup phase - cache ALL autotune configs
            # ============================================================
            logger.info(f"       Step 1/2: Warmup (caching autotune configs)...")
            warmup_success = self._run_warmup(script_path)

            if not warmup_success:
                logger.warning(f"Warmup failed for {kernel_name}, fallback to no-autotune mode")
                # Fallback: Remove autotune and regenerate script
                best_code_clean = self._remove_autotune(best_code)
                script_content = self._generate_kernel_script(best_code_clean, kernel_name)
                with open(script_path, 'w') as f:
                    f.write(script_content)

            # ============================================================
            # Step 3: NCU profile phase - profile ALL launches
            # ============================================================
            logger.info(f"       Step 2/2: NCU profile (measuring hardware metrics)...")
            ncu_output = self._run_ncu(script_path, launch_count=-1)

            # ============================================================
            # Retry mechanism: If NCU failed and code has autotune, retry without it
            # ============================================================
            if not ncu_output and '@triton.autotune' in best_code:
                logger.warning(
                    f"⚠️  NCU profiling failed for {kernel_name} (likely due to buggy autotune configs)"
                )
                logger.warning(
                    f"    Retrying without @triton.autotune decorator..."
                )

                # Remove autotune and regenerate script
                best_code_clean = self._remove_autotune(best_code)
                script_content = self._generate_kernel_script(best_code_clean, kernel_name)
                with open(script_path, 'w') as f:
                    f.write(script_content)

                # Retry NCU profiling
                ncu_output = self._run_ncu(script_path, launch_count=-1)

                if ncu_output:
                    logger.info(
                        f"✅ NCU profiling succeeded after removing @triton.autotune"
                    )
                else:
                    logger.error(
                        f"❌ NCU profiling failed for {kernel_name} even without autotune"
                    )
                    logger.error(
                        f"    Likely cause: Kernel code has bugs (illegal memory access, etc.)"
                    )
                    return None

            elif not ncu_output:
                logger.error(f"NCU profiling failed for {kernel_name}")
                logger.error(f"    Code does not use @triton.autotune, cannot retry")
                return None

            # Parse NCU CSV output
            result = self._parse_ncu_output(
                ncu_output=ncu_output,
                kernel_name=kernel_name,
                iteration=iteration,
            )

            if result:
                # Cache result
                self._cache[cache_key] = result

                # Save to disk cache (with code_hash metadata)
                if self.cache_dir:
                    self._save_to_cache(cache_key, result, code_hash=code_hash)

                logger.debug(
                    f"NCU profiling complete for {kernel_name}: "
                    f"DRAM={result.dram_throughput_pct:.1f}%, "
                    f"L1={result.l1_throughput_pct:.1f}%, "
                    f"L2={result.l2_throughput_pct:.1f}%, "
                    f"SM={result.sm_throughput_pct:.1f}%, "
                    f"Occ={result.occupancy_pct:.1f}%, "
                    f"Bottleneck={result.bottleneck_type}"
                )

            return result

        except subprocess.TimeoutExpired:
            logger.error(f"NCU profiling timed out for {kernel_name}")
            return None
        except (ValueError, TypeError) as e:
            # Input validation errors (from validation checks in this method)
            logger.error(f"Invalid input parameters for {kernel_name}: {e}")
            raise  # Re-raise validation errors - these indicate programming errors
        except OSError as e:
            # File system errors (temp dir creation, file operations)
            logger.error(f"File system error during NCU profiling of {kernel_name}: {e}")
            return None
        except Exception as e:
            # Unexpected errors - keep detailed debugging information
            logger.error(f"Unexpected error during NCU profiling of {kernel_name}: {e}")
            logger.debug(f"Stack trace for {kernel_name}:", exc_info=True)
            return None

        finally:
            # Cleanup temp folder (keep on failure for debugging)
            if temp_dir and os.path.exists(temp_dir):
                if result is not None:  # Success - cleanup
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp folder {temp_dir}: {e}")
                else:  # Failure - keep for debugging
                    logger.warning(f"Keeping temp folder for debugging: {temp_dir}")

    def _extract_test_code(self, kernel_name: str) -> Optional[str]:
        """
        Extract test_code from TritonBench kernel file.

        Test code is separated by a line of 146 # characters.
        This follows the same approach as GEAK-eval's code_call_exec_success_allclose().

        Args:
            kernel_name: Kernel filename (e.g., "add_example.py")

        Returns:
            Test code string, or None if extraction failed
        """
        # Check cache first
        if kernel_name in self._test_code_cache:
            return self._test_code_cache[kernel_name]

        if not self.py_folder:
            logger.warning(f"py_folder not set, cannot extract test_code for {kernel_name}")
            return None

        kernel_path = Path(self.py_folder) / kernel_name
        if not kernel_path.exists():
            logger.warning(f"Kernel file not found: {kernel_path}")
            return None

        try:
            with open(kernel_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Find the separator line (146 # characters)
            hash_line = "#" * 146
            separator_idx = None
            for idx, line in enumerate(lines):
                if line.strip() == hash_line:
                    separator_idx = idx
                    break

            if separator_idx is None:
                logger.warning(f"Could not find separator line (146 #'s) in {kernel_name}")
                return None

            # Extract test code (everything after separator)
            test_code_lines = lines[separator_idx + 1:]
            test_code = ''.join(test_code_lines)

            # Verify test_code contains test function
            if 'def test_' not in test_code:
                logger.warning(f"No test function found in test_code for {kernel_name}")
                return None

            # Cache the result
            self._test_code_cache[kernel_name] = test_code

            logger.debug(f"Extracted test_code from {kernel_name} ({len(test_code_lines)} lines)")
            return test_code

        except Exception as e:
            logger.error(f"Failed to extract test_code from {kernel_name}: {e}")
            return None

    def _remove_autotune(self, code: str) -> str:
        """
        Remove @triton.autotune decorators from code to avoid profiling all configs.

        When best_code contains @triton.autotune, NCU will profile ALL config attempts
        (warmup + benchmarking), resulting in hundreds of kernel launches per shape.

        This method removes the autotune decorator but keeps a reasonable default config.

        Args:
            code: Triton kernel code string

        Returns:
            Code with autotune removed
        """
        if '@triton.autotune' not in code:
            return code

        logger.debug("Removing @triton.autotune decorator to avoid excessive NCU profiling")

        import re

        # Remove @triton.autotune(...) decorator
        # Pattern: @triton.autotune( ... ) spanning multiple lines
        pattern = r'@triton\.autotune\s*\([^)]*(?:\([^)]*\)[^)]*)*\)\s*\n'

        # Try simple removal first
        code_clean = re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)

        # If pattern didn't match (complex nested structure), try line-by-line
        if '@triton.autotune' in code_clean:
            lines = code.split('\n')
            cleaned_lines = []
            skip_until_close = 0

            for line in lines:
                if '@triton.autotune' in line:
                    # Start skipping
                    skip_until_close = line.count('(') - line.count(')')
                    continue
                elif skip_until_close > 0:
                    # Count parentheses to find the end
                    skip_until_close += line.count('(') - line.count(')')
                    if skip_until_close <= 0:
                        skip_until_close = 0
                    continue
                else:
                    cleaned_lines.append(line)

            code_clean = '\n'.join(cleaned_lines)

        if '@triton.autotune' not in code_clean:
            logger.debug("Successfully removed @triton.autotune decorator")
        else:
            logger.warning("Could not fully remove @triton.autotune, profiling may be slow")

        return code_clean

    def _find_perf_file(self, kernel_name: str) -> Optional[Path]:
        """
        Find the corresponding perf.py file for a kernel.

        Tries both the refactored layout and the original TritonBench layout:
          - New: {py_folder}/../perf_metrics/golden_metrics/{kernel}_perf.py
          - Old: {py_folder}/../../performance_metrics/perf_G/golden_metrics/{kernel}_perf.py

        Args:
            kernel_name: Kernel filename (e.g., "softmax_triton1.py")

        Returns:
            Path to perf.py file, or None if not found
        """
        if not self.py_folder:
            return None

        py_folder_path = Path(self.py_folder)

        # Convert kernel_name to perf filename
        # "softmax_triton1.py" -> "softmax_triton1_perf.py"
        base_name = kernel_name.replace('.py', '')
        perf_filename = f"{base_name}_perf.py"

        # Try both directory layouts
        candidates = [
            py_folder_path.parent / "perf_metrics" / "golden_metrics",
            py_folder_path.parent.parent / "performance_metrics" / "perf_G" / "golden_metrics",
        ]

        for perf_folder in candidates:
            perf_file = perf_folder / perf_filename
            if perf_file.exists():
                return perf_file

        logger.debug(f"Perf file not found for {kernel_name} in any known layout")
        return None

    def _extract_method_impl(self, perf_content: str, method_name: str) -> Optional[str]:
        """
        Extract a method implementation from perf.py file content.

        Args:
            perf_content: Content of the perf.py file
            method_name: Name of the method to extract (e.g., "get_input_tensors")

        Returns:
            Method implementation with proper indentation, or None if not found
        """
        lines = perf_content.split('\n')

        # Find the method definition
        method_start = None
        for idx, line in enumerate(lines):
            if f'def {method_name}(' in line:
                method_start = idx
                break

        if method_start is None:
            return None

        # Extract method body (until next method or class end)
        method_lines = [lines[method_start]]

        # Get base indentation (usually 4 or 8 spaces)
        base_indent = len(lines[method_start]) - len(lines[method_start].lstrip())

        # Collect method body
        for idx in range(method_start + 1, len(lines)):
            line = lines[idx]

            # Empty lines are part of the method
            if not line.strip():
                method_lines.append(line)
                continue

            # Check indentation
            line_indent = len(line) - len(line.lstrip())

            # If line has same or less indentation than method def, we've reached the end
            if line_indent <= base_indent:
                break

            method_lines.append(line)

        # Return with consistent indentation (4 spaces)
        result = []
        for line in method_lines:
            if line.strip():
                # Remove base indentation and add 4 spaces
                stripped = line[base_indent:] if len(line) > base_indent else line.lstrip()
                result.append('    ' + stripped)
            else:
                result.append('')

        return '\n'.join(result)

    def _extract_kernel_function_name(self, perf_content: str) -> Optional[str]:
        """
        Extract the kernel function name from call_op() method in perf.py.

        For example:
        - "return softmax(x)" -> "softmax"
        - "return custom_add(a, b)" -> "custom_add"

        Args:
            perf_content: Content of the perf.py file

        Returns:
            Function name, or None if not found
        """
        # Find the call_op method
        call_op_impl = self._extract_method_impl(perf_content, 'call_op')
        if not call_op_impl:
            return None

        # Look for return statement
        for line in call_op_impl.split('\n'):
            if 'return' in line:
                # Extract function name using regex
                import re
                # Match: return function_name(...)
                match = re.search(r'return\s+(\w+)\s*\(', line)
                if match:
                    return match.group(1)

        return None

    def _generate_script_from_perf(self, best_code: str, kernel_name: str, perf_file: Path) -> str:
        """
        Generate NCU profiling script using perf.py shape generation logic.

        This is the CORRECT approach: use the same shapes as GEAK-eval performance testing.

        Args:
            best_code: Kernel code string
            kernel_name: Kernel name (e.g., "softmax_triton1.py")
            perf_file: Path to the perf.py file

        Returns:
            Complete Python script content
        """
        with open(perf_file, 'r', encoding='utf-8') as f:
            perf_content = f.read()

        # Extract key methods from perf.py
        get_input_tensors_impl = self._extract_method_impl(perf_content, 'get_input_tensors')
        to_cuda_impl = self._extract_method_impl(perf_content, 'to_cuda')
        call_op_impl = self._extract_method_impl(perf_content, 'call_op')

        if not (get_input_tensors_impl and to_cuda_impl and call_op_impl):
            logger.warning(
                f"Could not extract all required methods from {perf_file.name}. "
                f"Missing: get_input_tensors={get_input_tensors_impl is None}, "
                f"to_cuda={to_cuda_impl is None}, call_op={call_op_impl is None}"
            )
            return None

        # Generate script
        script = f"""#!/usr/bin/env python3
\"\"\"
NCU Profiling Script for {kernel_name}
Generated from: {perf_file.name}
Supports warmup mode for autotune caching
\"\"\"
import sys
import argparse
import torch
import triton
import triton.language as tl

# ============================================================
# Best Code (from GEAK optimization)
# Preserves @triton.autotune decorator for warmup strategy
# ============================================================
{best_code}

# ============================================================
# Performance Testing Logic (from perf.py)
# This uses the SAME shapes as GEAK-eval performance evaluation
# ============================================================
class PerfRunner:
    def __init__(self):
        self.input_tensors = []

{get_input_tensors_impl}

{to_cuda_impl}

{call_op_impl}

# ============================================================
# Main execution logic
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true",
                       help="Warmup run for autotune config caching")
    args = parser.parse_args()

    runner = PerfRunner()
    runner.get_input_tensors()
    num_shapes = len(runner.input_tensors)

    if args.warmup:
        # ============================================================
        # Warmup Mode: Cache autotune configs for ALL shapes
        # ============================================================
        print(f"🔥 Warmup: caching autotune configs for {{num_shapes}} shapes...")
        for idx, input_tensor_ in enumerate(runner.input_tensors):
            try:
                input_tensor = runner.to_cuda(input_tensor_)
                _ = runner.call_op(input_tensor)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: Shape {{idx}} failed during warmup: {{e}}", file=sys.stderr)
                continue
        print("✅ Warmup complete: all configs cached")

    else:
        # ============================================================
        # Profile Mode: Run all shapes (NCU will profile all launches)
        # ============================================================
        print(f"🔬 Profile: executing on {{num_shapes}} shapes...")
        successful_runs = 0
        for idx, input_tensor_ in enumerate(runner.input_tensors):
            try:
                input_tensor = runner.to_cuda(input_tensor_)
                result = runner.call_op(input_tensor)
                torch.cuda.synchronize()
                successful_runs += 1
            except Exception as e:
                print(f"Warning: Shape {{idx}} failed: {{e}}", file=sys.stderr)
                continue
        print(f"✅ Profile complete: {{successful_runs}}/{{num_shapes}} shapes succeeded")
"""

        logger.debug(f"Generated script using perf.py for {kernel_name} ({perf_file.name})")
        return script

    def _generate_script_from_test_code(self, best_code: str, kernel_name: str) -> str:
        """
        Generate NCU profiling script using test_code (correctness testing).

        This is a FALLBACK when perf.py is not available.
        Note: test_code has fewer shapes (3-4) compared to perf.py (9-18).

        Args:
            best_code: Kernel code string
            kernel_name: Kernel name (e.g., "add_example.py")

        Returns:
            Complete Python script content
        """
        test_code = self._extract_test_code(kernel_name)

        if test_code:
            # Wrap with argparse support for warmup mode
            hash_line = "#" * 146
            script = f"""#!/usr/bin/env python3
\"\"\"
NCU Profiling Script for {kernel_name}
Generated from test_code (fallback)
Supports warmup mode for autotune caching
\"\"\"
import sys
import argparse

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument("--warmup", action="store_true",
                   help="Warmup run for autotune config caching")
args = parser.parse_args()

if args.warmup:
    print("🔥 Warmup: caching autotune configs (test_code mode)...")
else:
    print("🔬 Profile: executing test code...")

# ============================================================
# Best Code (from GEAK optimization)
# Preserves @triton.autotune decorator for warmup strategy
# ============================================================
{best_code}

{hash_line}

# ============================================================
# Test Code (from TritonBench)
# ============================================================
{test_code}

if args.warmup:
    print("✅ Warmup complete: all configs cached")
else:
    print("✅ Profile complete")
"""
            logger.debug(f"Generated script using test_code for {kernel_name} (fallback, fewer shapes)")
            return script

        # If test_code also not found, return None (will use final fallback)
        return None

    def _generate_kernel_script(self, best_code: str, kernel_name: str) -> str:
        """
        Generate temporary Python script for NCU profiling.

        Strategy (in priority order):
        1. Use perf.py shapes (PREFERRED - matches GEAK-eval performance testing)
        2. Use test_code shapes (FALLBACK - fewer shapes, correctness testing)
        3. Use minimal fallback (LAST RESORT - very simple kernels only)

        Args:
            best_code: Kernel code string
            kernel_name: Kernel name (e.g., "add_example.py")

        Returns:
            Complete Python script content
        """
        # Try method 1: Use perf.py (PREFERRED)
        perf_file = self._find_perf_file(kernel_name)
        if perf_file:
            script = self._generate_script_from_perf(best_code, kernel_name, perf_file)
            if script:
                return script
            else:
                logger.warning(f"Failed to generate script from perf.py for {kernel_name}")

        # Try method 2: Use test_code (FALLBACK)
        script = self._generate_script_from_test_code(best_code, kernel_name)
        if script:
            return script

        # Method 3: Minimal fallback (LAST RESORT)
        logger.warning(
            f"Could not find perf.py or test_code for {kernel_name}. "
            f"Using minimal fallback (may not work correctly)."
        )
        script = f"""
import torch
import triton
import triton.language as tl

# ============ Kernel Code ============
{best_code}

# ============ Minimal Test (Fallback) ============
# WARNING: This fallback only works for simple binary operations
# For reliable profiling, ensure perf.py files are available

try:
    # Try simple inputs
    size = 4096
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)

    # Attempt to call the last non-test, non-jit function
    # This is a best-effort attempt
    import sys
    import types

    # Execute the code to get functions in scope
    exec_globals = {{'torch': torch, 'triton': triton, 'tl': tl}}
    exec(compile('''
{best_code}
''', '<string>', 'exec'), exec_globals)

    # Find candidate wrapper functions
    candidates = []
    for name, obj in exec_globals.items():
        if isinstance(obj, types.FunctionType):
            if not name.startswith('_') and not name.startswith('test'):
                candidates.append((name, obj))

    if candidates:
        # Try the last candidate function
        func_name, func = candidates[-1]
        print(f"Attempting to profile function: {{func_name}}")

        # Try different call signatures
        try:
            result = func(x)  # Unary
        except:
            try:
                result = func(x, y)  # Binary
            except:
                print(f"Could not call {{func_name}} with simple inputs")

        torch.cuda.synchronize()
    else:
        print("No suitable function found for profiling")

except Exception as e:
    print(f"Fallback profiling failed: {{e}}")
    import traceback
    traceback.print_exc()
"""

        return script

    def _run_warmup(self, script_path: Path, timeout: int = 120) -> bool:
        """
        Run warmup to cache autotune configs for all shapes.

        This runs the script without NCU profiling to trigger autotune's
        config selection and caching. On the subsequent NCU run, autotune
        will read from cache instead of benchmarking all configs.

        Args:
            script_path: Path to kernel script
            timeout: Timeout in seconds (default: 120)

        Returns:
            True if warmup succeeded, False otherwise
        """
        try:
            # Run script with Python (no NCU)
            cmd = [
                sys.executable,
                str(script_path),
                "--warmup"  # Script must support this flag
            ]

            logger.debug(f"Running warmup: {' '.join(cmd)}")

            # Enable Triton autotune result caching (Triton 3.4.0+)
            env = os.environ.copy()
            env['TRITON_CACHE_AUTOTUNING'] = '1'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            if result.returncode == 0:
                logger.debug(f"Warmup completed successfully")
                return True
            else:
                logger.warning(f"Warmup failed with return code {result.returncode}")
                logger.warning(f"Warmup stderr: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"Warmup timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            return False

    def _run_ncu(
        self,
        script_path: Path,
        timeout: int = 300,
        launch_count: int = -1
    ) -> Optional[str]:
        """
        Run NCU profiling command.

        Args:
            script_path: Path to the kernel script
            timeout: Timeout in seconds (default: 300)
            launch_count: Number of kernel launches to profile
                -1: Profile ALL launches (recommended for warmup strategy)
                N: Profile first N launches only

        Returns:
            NCU CSV output as string, or None if failed
        """
        # Validate launch_count
        if not isinstance(launch_count, int):
            raise TypeError(f"launch_count must be int, got {type(launch_count).__name__}")

        if launch_count != -1 and not (1 <= launch_count <= 10000):
            raise ValueError(
                f"launch_count must be -1 (all) or 1-10000, got: {launch_count}"
            )

        # Validate timeout
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError(f"timeout must be positive int, got: {timeout}")

        # Validate script_path
        if not script_path.exists():
            raise FileNotFoundError(f"Script path does not exist: {script_path}")

        cmd = [
            'ncu',
            '--target-processes', 'all',
            '--kernel-name', 'regex:.*',  # Match all Triton kernels
            '--metrics', ','.join(self.NCU_METRICS),
            '--csv',
        ]

        # Add launch-count parameter if specified
        if launch_count != -1:
            cmd.extend(['--launch-count', str(launch_count)])
        # When launch_count=-1, omit the flag (NCU profiles all by default)

        cmd.extend(['python', str(script_path)])

        try:
            # Enable Triton autotune result caching (Triton 3.4.0+)
            # This ensures NCU profiling reads cached autotune results from warmup phase
            env = os.environ.copy()
            env['TRITON_CACHE_AUTOTUNING'] = '1'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            if result.returncode != 0:
                logger.error(f"NCU command failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                # NCU often outputs errors to stdout, not stderr
                logger.error(f"STDOUT (first 2000 chars): {result.stdout[:2000]}")
                logger.error(f"Command was: {' '.join(cmd)}")
                return None

            return result.stdout

        except subprocess.TimeoutExpired:
            logger.error(f"NCU profiling timed out ({timeout} seconds)")
            return None
        except Exception as e:
            logger.error(f"NCU execution failed: {e}")
            return None

    def _parse_ncu_output(
        self,
        ncu_output: str,
        kernel_name: str,
        iteration: int,
    ) -> Optional[NCUProfilingResult]:
        """
        Parse NCU CSV output and extract metrics.

        Args:
            ncu_output: NCU CSV output string
            kernel_name: Kernel name
            iteration: Iteration number

        Returns:
            NCUProfilingResult or None if parsing failed
        """
        try:
            # NCU output contains non-CSV lines (==PROF==, script stdout, etc.)
            # We need to find the CSV header line and start parsing from there
            lines = ncu_output.split('\n')
            csv_start_idx = None

            for idx, line in enumerate(lines):
                # CSV header starts with "ID","Process ID",...
                if line.startswith('"ID",'):
                    csv_start_idx = idx
                    break

            if csv_start_idx is None:
                logger.error("Could not find CSV header in NCU output")
                logger.debug(f"NCU output (first 500 chars):\n{ncu_output[:500]}")
                return None

            # Extract CSV portion only
            csv_lines = lines[csv_start_idx:]
            csv_content = '\n'.join(csv_lines)

            # Parse CSV
            reader = csv.DictReader(io.StringIO(csv_content))

            # NCU CSV format: One row per metric per kernel launch
            # We need to group by ID to collect all metrics for each launch
            kernel_launches = {}  # {ID: {kernel_name, metrics}}
            all_kernel_names = set()

            for row in reader:
                # Skip empty rows
                if not row or 'ID' not in row:
                    continue

                launch_id = row.get('ID', '')
                kernel_name_in_row = row.get('Kernel Name', '')
                metric_name = row.get('Metric Name', '')
                metric_value = row.get('Metric Value', '')

                all_kernel_names.add(kernel_name_in_row)

                # Filter out PyTorch/CUDA internal kernels
                if '::' in kernel_name_in_row or '<unnamed>' in kernel_name_in_row:
                    continue

                # Initialize launch entry if needed
                if launch_id not in kernel_launches:
                    kernel_launches[launch_id] = {
                        'kernel_name': kernel_name_in_row,
                        'metrics': {}
                    }

                # Store metric
                kernel_launches[launch_id]['metrics'][metric_name] = metric_value

            if not kernel_launches:
                logger.error(f"No valid Triton kernel launches found in NCU output")
                logger.debug(f"All kernels found: {list(all_kernel_names)[:5]}")
                return None

            # Extract metrics from grouped launches
            launches = []
            for launch_id, launch_info in kernel_launches.items():
                launch_data = self._extract_metrics_from_dict(launch_info['metrics'])
                if launch_data:
                    launch_data['kernel_name_from_ncu'] = launch_info['kernel_name']
                    launches.append(launch_data)

            logger.debug(f"Parsed {len(launches)} kernel launches from NCU output")

            # Aggregate metrics across all launches
            aggregated = self._aggregate_launch_metrics(launches)

            # Classify bottleneck
            bottleneck_type = self._classify_bottleneck(aggregated)

            # Create result
            result = NCUProfilingResult(
                kernel_name=kernel_name,
                iteration=iteration,
                dram_throughput_pct=aggregated['dram_throughput_pct'],
                l2_throughput_pct=aggregated['l2_throughput_pct'],
                l1_throughput_pct=aggregated['l1_throughput_pct'],
                sm_throughput_pct=aggregated['sm_throughput_pct'],
                occupancy_pct=aggregated['occupancy_pct'],
                duration_us=aggregated['duration_us'],
                registers_per_thread=aggregated['registers_per_thread'],
                shared_memory_bank_conflicts=aggregated['shared_memory_bank_conflicts'],
                bottleneck_type=bottleneck_type,
                num_shapes=len(launches),
                shape_results=launches,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to parse NCU output: {e}")
            logger.debug(f"NCU output:\n{ncu_output[:500]}...")
            return None

    def _extract_metrics_from_dict(self, metrics_dict: Dict) -> Optional[Dict]:
        """
        Extract metrics from a dictionary of metric_name -> metric_value.

        NCU CSV format has one row per metric, so we group by ID first,
        then extract all metrics for that kernel launch.

        Args:
            metrics_dict: Dictionary mapping metric names to values

        Returns:
            Dict with extracted metrics, or None if extraction failed
        """
        try:
            # Helper to safely extract float value
            def safe_float(key, default=0.0):
                val = metrics_dict.get(key, '')
                if val == '' or val == 'N/A':
                    return default
                try:
                    # Remove commas from number strings (NCU CSV format uses comma separators)
                    if isinstance(val, str):
                        val = val.replace(',', '')
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # Helper to safely extract int value
            def safe_int(key, default=0):
                val = metrics_dict.get(key, '')
                if val == '' or val == 'N/A':
                    return default
                try:
                    # Remove commas from number strings (NCU CSV format uses comma separators)
                    if isinstance(val, str):
                        val = val.replace(',', '')
                    return int(float(val))
                except (ValueError, TypeError):
                    return default

            # Extract metrics using NCU metric names
            # Duration is in nsecond, convert to microseconds
            duration_ns = safe_float('gpu__time_duration.sum', 0.0)
            duration_us = duration_ns / 1000.0 if duration_ns > 0 else 0.0

            metrics = {
                'dram_throughput_pct': safe_float('dram__throughput.avg.pct_of_peak_sustained_elapsed'),
                'l2_throughput_pct': safe_float('lts__throughput.avg.pct_of_peak_sustained_elapsed'),
                'l1_throughput_pct': safe_float('l1tex__throughput.avg.pct_of_peak_sustained_elapsed'),
                'sm_throughput_pct': safe_float('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
                'occupancy_pct': safe_float('sm__warps_active.avg.pct_of_peak_sustained_active'),
                'duration_us': duration_us,
                'registers_per_thread': safe_int('launch__registers_per_thread'),
                'shared_memory_bank_conflicts': safe_int('l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum'),
            }

            return metrics

        except Exception as e:
            logger.warning(f"Failed to extract metrics from dict: {e}")
            return None

    def _aggregate_launch_metrics(self, launches: List[Dict]) -> Dict:
        """
        Aggregate metrics across multiple kernel launches.

        Uses weighted average by execution time for throughput metrics.

        Args:
            launches: List of metric dicts from each launch

        Returns:
            Aggregated metrics dict
        """
        # Calculate total time for weighting
        total_time = sum(l['duration_us'] for l in launches)

        if total_time == 0:
            # Fallback to simple average if no timing data
            weights = [1.0 / len(launches)] * len(launches)
        else:
            weights = [l['duration_us'] / total_time for l in launches]

        # Weighted average for throughput and occupancy
        aggregated = {
            'dram_throughput_pct': sum(l['dram_throughput_pct'] * w for l, w in zip(launches, weights)),
            'l2_throughput_pct': sum(l['l2_throughput_pct'] * w for l, w in zip(launches, weights)),
            'l1_throughput_pct': sum(l['l1_throughput_pct'] * w for l, w in zip(launches, weights)),
            'sm_throughput_pct': sum(l['sm_throughput_pct'] * w for l, w in zip(launches, weights)),
            'occupancy_pct': sum(l['occupancy_pct'] * w for l, w in zip(launches, weights)),
            'duration_us': sum(l['duration_us'] for l in launches) / len(launches),  # Average
            'registers_per_thread': int(sum(l['registers_per_thread'] for l in launches) / len(launches)),  # Average
            'shared_memory_bank_conflicts': sum(l['shared_memory_bank_conflicts'] for l in launches),  # Sum
        }

        return aggregated

    def _classify_bottleneck(self, metrics: Dict) -> str:
        """
        Classify bottleneck type based on metrics.

        5-dimensional bottleneck classification:
        1. DRAM bound
        2. L1 Cache bound
        3. L2 Cache bound
        4. Compute bound
        5. Occupancy bound
        6. Balanced
        7. Well-optimized
        8. Underutilized
        9. Mixed

        Args:
            metrics: Aggregated metrics dict

        Returns:
            Bottleneck type string
        """
        dram = metrics['dram_throughput_pct']
        l1 = metrics['l1_throughput_pct']
        l2 = metrics['l2_throughput_pct']
        sm = metrics['sm_throughput_pct']
        occ = metrics['occupancy_pct']

        # 1. DRAM bound: DRAM high, SM low
        if dram > 75 and sm < 40:
            return "dram_bound"

        # 2. L1 Cache bound: L1 high, DRAM low
        if l1 > 70 and dram < 40:
            return "l1_bound"

        # 3. L2 Cache bound: L2 high, L1 low
        if l2 > 70 and l1 < 40:
            return "l2_bound"

        # 4. Compute bound: SM high, DRAM low
        if sm > 75 and dram < 40:
            return "compute_bound"

        # 5. Well-optimized: All metrics > 80%
        if all(x > 80 for x in [dram, l1, l2, sm, occ]):
            return "well_optimized"

        # 6. Balanced: All metrics 50-80%
        if all(50 <= x <= 80 for x in [dram, l1, l2, sm, occ]):
            return "balanced"

        # 7. Underutilized: All metrics < 40% (check before occupancy_bound)
        if all(x < 40 for x in [dram, l1, l2, sm, occ]):
            return "underutilized"

        # 8. Occupancy bound: Low occupancy but resources available
        if occ < 50 and (dram < 50 or sm < 50):
            return "occupancy_bound"

        return "mixed"

    def _save_to_cache(self, cache_key: str, result: NCUProfilingResult, code_hash: str = None):
        """
        Save profiling result to disk cache with metadata.

        Args:
            cache_key: Cache key (format: {kernel_name}_{code_hash})
            result: NCUProfilingResult to save
            code_hash: Optional code hash (extracted from cache_key if not provided)
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            # Prepare cache data with metadata
            cache_data = result.to_dict()

            # Extract code_hash from cache_key if not provided
            if code_hash is None:
                # cache_key format: "{kernel_name}_{code_hash}"
                parts = cache_key.rsplit('_', 1)
                if len(parts) == 2:
                    code_hash = parts[1]

            # Add metadata for debugging and cache management
            cache_data['_meta'] = {
                'kernel_name': result.kernel_name,
                'code_hash': code_hash,
                'iteration': result.iteration,  # For debugging/tracking
                'timestamp': time.time(),
                'cache_key': cache_key,
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved NCU profiling result to {cache_file} (iter={result.iteration})")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[NCUProfilingResult]:
        """Load profiling result from disk cache"""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Reconstruct NCUProfilingResult from cached data
            result = NCUProfilingResult(
                kernel_name=data["kernel_name"],
                iteration=data["iteration"],
                dram_throughput_pct=data["core_metrics"]["dram_throughput_pct"],
                l2_throughput_pct=data["core_metrics"]["l2_throughput_pct"],
                l1_throughput_pct=data["core_metrics"]["l1_throughput_pct"],
                sm_throughput_pct=data["core_metrics"]["sm_throughput_pct"],
                occupancy_pct=data["core_metrics"]["occupancy_pct"],
                duration_us=data["auxiliary_metrics"]["duration_us"],
                registers_per_thread=data["auxiliary_metrics"]["registers_per_thread"],
                shared_memory_bank_conflicts=data["auxiliary_metrics"]["shared_memory_bank_conflicts"],
                bottleneck_type=data["derived"]["bottleneck_type"],
                num_shapes=data["num_shapes"],
                shape_results=data.get("shape_results", []),
            )

            logger.debug(f"Loaded NCU profiling result from {cache_file}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None
