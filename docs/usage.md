# Usage Guide

Detailed documentation for configuring and running KernelBand.

## Table of Contents

- [Benchmark Data](#benchmark-data)
- [Configuration Reference](#configuration-reference)
- [MAB Configuration](#mab-configuration)
- [Running Optimization](#running-optimization)
- [Understanding Output](#understanding-output)
- [Kernel Selection Helper](#kernel-selection-helper)
- [Generating Golden Baselines](#generating-golden-baselines)
- [Troubleshooting](#troubleshooting)

## Benchmark Data

KernelBand evaluates on the **TritonBench-G** benchmark — a suite of 183 GPU kernel generation tasks. The benchmark data originates from [TritonBench](https://github.com/thunlp/TritonBench) (THUNLP), with corrections from [GEAK-eval](https://github.com/AMD-AIG-AIMA/GEAK) (AMD-AIG-AIMA) and additional KernelBand fixes for NVIDIA compatibility.

The data is included in this repository under `data/TritonBench/`. No additional download is required.

All config paths (`statis_path`, `py_folder`, etc.) reference files under `data/TritonBench/` by default.

For full attribution details, see [Acknowledgments](../README.md#acknowledgments) in the main README.

## Configuration Reference

Configuration files are in YAML format, located in `configs/examples/`. See [`full_config.yaml`](../configs/examples/full_config.yaml) for all available parameters.

```yaml
# LLM Configuration
api_key: ""                    # Your API key (or set OPENAI_API_KEY env var)
base_url: ""                   # Custom API endpoint (or MODEL_API_URL env var)
model_id: "gpt-4o-mini"        # LLM model identifier
temperature: 1.0               # LLM sampling temperature (paper default: 1.0)
max_tokens: 16384              # Max output tokens (paper default: 16384)

# Optimization Control (choose ONE mode)
max_iteration: 20              # MODE 1: Fixed iterations
# generation_budget_per_kernel: 200  # MODE 2: Budget mode (alternative)

# Execution
multi_thread: true             # Parallel code generation
thread_num: 3                  # Worker threads

# Kernel Selection
target_kernels:
  - "vector_addition_custom.py"
  - "add_value.py"

# Dataset Paths (relative to repo root)
statis_path: "data/TritonBench/statistics.json"
py_folder: "data/TritonBench/kernels"
instruction_path: "data/TritonBench/statistics.json"
corpus_path: "data/TritonBench/corpus/train_crawl.json"
golden_metrics: "data/TritonBench/perf_metrics/golden_metrics"
perf_G_path: "data/TritonBench/perf_metrics"
```

## MAB Configuration

The hierarchical multi-armed bandit framework balances exploration and exploitation:

```yaml
mab_config:
  ucb_c: 2.0                         # UCB exploration parameter (higher = more exploration)
  cluster_K: 3                        # Number of behavioral clusters
  cluster_tau: 10                     # Re-clustering interval (iterations)
  within_cluster_temperature: 1.0     # Softmax temperature for within-cluster sampling

  # Hardware saturation threshold (%)
  # Option 1 - Scalar (same threshold for all dimensions):
  theta_sat: 75.0

  # Option 2 - Per-dimension thresholds:
  # theta_sat:
  #   sm: 80.0     # SM (compute throughput) threshold
  #   dram: 70.0   # DRAM (memory bandwidth) threshold
  #   l2: 75.0     # L2 (cache locality) threshold

  # Strategy-to-hardware target mapping (optional, defaults shown):
  # strategy_target_map:
  #   tiling: "l2"
  #   vectorization: "dram"
  #   fusion: "dram"
  #   pipeline: "sm"
  #   reordering: "sm"
  #   access_layout: "dram"
```

## Running Optimization

```bash
# Run from the repo root directory
python -m kernelband --config configs/examples/test_config.yaml

# Or equivalently
python kernelband/main.py --config configs/examples/test_config.yaml
```

### What Happens During Optimization

Each iteration follows the **5-phase pipeline**:

1. **Phase 0: Clustering + Hardware Masking** - Cluster frontier kernels by behavioral features, profile cluster representatives via NCU, mask strategies targeting saturated resources
2. **Phase 1: UCB Selection + Code Generation** - Select (cluster, strategy) pair via Masked UCB, sample parent kernel, generate ONE optimized kernel via LLM
3. **Phase 2: Correctness Testing** - Validate generated kernels against ground truth
4. **Phase 3: Performance Testing** - Measure execution time on 16 input shapes (serial to avoid GPU conflicts)
5. **Phase 4: Reward + UCB Update + Frontier Update** - Compute reward, update UCB statistics, add valid kernels to frontier

## Understanding Output

### Memory File (`*_mem_{iter}.json`)

```json
{
  "vector_addition_custom.py": {
    "best_code": ["...", 1.0043, 9.95, 7.01, ["tiling"]],
    "branches": {
      "mab_tiling": ["...", 1.0043, 9.95, null],
      "mab_fusion": ["...", 1.22, 8.72, null]
    },
    "_generation_budget": 180,
    "_mab_state": {
      "K": 3, "S": 6,
      "total_t": 20,
      "frontier": [{"code": "...", "ms": 1.0043, "strategy": "tiling", "...": "..."}],
      "best_entry": {"code": "...", "ms": 1.0043, "strategy": "tiling", "...": "..."},
      "mu_hat": [[0.5, 0.3, "..."], "..."],
      "N": [[5, 3, "..."], "..."]
    }
  }
}
```

### Key Metrics

- **speedup**: Performance improvement vs golden baseline (higher is better)
- **ms**: Average execution time in milliseconds (lower is better)
- **efficiency**: Hardware utilization percentage (higher is better)

### Key Fields

- **best_code**: `[code, ms, efficiency, speedup, strategy_chain]` - Global best kernel
- **branches**: This iteration's result (`mab_{strategy}`: `[code, ms, eff, reflection]`)
- **_generation_budget**: Remaining budget (budget mode only)
- **_mab_state**: MAB statistics and frontier (mu_hat, N, masks, frontier, best_entry)

## Kernel Selection Helper

Use the kernel selection script to choose kernels for optimization:

```bash
# View kernel statistics by difficulty
python scripts/list_kernels.py --stats

# List easy kernels (difficulty 1-2)
python scripts/list_kernels.py --difficulty 1 2

# Update config file with selected kernels
python scripts/list_kernels.py --config configs/examples/test_config.yaml --difficulty 1 2 --range 0 5
```

See [list_kernels.md](list_kernels.md) for detailed documentation.

## Generating Golden Baselines

Speedup calculation requires GPU-specific golden metrics (baseline latency measurements). Pre-computed baselines may not exist for your GPU. Use the provided script to generate them:

```bash
# Generate baselines for all kernels on the current GPU (auto-detects GPU model)
python scripts/generate_golden_metrics.py

# Generate for specific kernels only
python scripts/generate_golden_metrics.py --kernels add_example vector_addition

# Resume interrupted generation (skip already-generated JSONs)
python scripts/generate_golden_metrics.py --resume

# Parallel generation across all available GPUs
python scripts/generate_golden_metrics.py --multi-gpu

# Prepare perf scripts without executing (useful for inspection)
python scripts/generate_golden_metrics.py --dry-run
```

The script outputs baseline JSON files to `data/TritonBench/perf_metrics/{GPU_NAME}_golden_metrics/`.

## Troubleshooting

### Speedup is null

Golden baselines for your GPU may not be pre-computed. Run `python scripts/generate_golden_metrics.py` to generate them (see [Generating Golden Baselines](#generating-golden-baselines)).

### NCU Profiling Fails

- Hardware masking requires NVIDIA GPU + Nsight Compute installation
- If NCU is unavailable, UCB selection still works using empirical rewards
