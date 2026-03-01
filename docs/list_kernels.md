# Kernel Selection Tool

Helper script to list and filter TritonBench kernels for batch evaluation in KernelBand experiments.

## Location

`scripts/list_kernels.py`

## Purpose

When running KernelBand optimization experiments, you often need to select specific kernels based on:
- Difficulty level (1-5)
- Specific index range
- Total count for testing

This tool helps you:
1. View available kernels and their difficulty ratings
2. Filter kernels by difficulty or index
3. Generate YAML configuration for direct use
4. Update config files with selected kernels

## Usage

### View Statistics

```bash
# Show kernel count by difficulty
python scripts/list_kernels.py --stats
```

Output:
```
======================================================================
Kernel Statistics
======================================================================

Total kernels: 163

Kernels by difficulty:
  Difficulty 1: 45 kernels (easiest)
  Difficulty 2: 58 kernels
  Difficulty 3: 35 kernels
  Difficulty 4: 15 kernels
  Difficulty 5: 10 kernels (hardest)
======================================================================
```

### List Kernels by Difficulty

```bash
# List all easy kernels (difficulty 1-2)
python scripts/list_kernels.py --difficulty 1 2

# List medium difficulty kernels
python scripts/list_kernels.py --difficulty 3

# List hard kernels
python scripts/list_kernels.py --difficulty 4 5
```

### List Kernels by Index Range

```bash
# List first 10 kernels
python scripts/list_kernels.py --range 0 10

# List kernels 20-30
python scripts/list_kernels.py --range 20 30
```

### Combine Filters

```bash
# First 5 easy kernels
python scripts/list_kernels.py --difficulty 1 2 --range 0 5
```

### Generate YAML Output

```bash
# Output in YAML format for config
python scripts/list_kernels.py --yaml --difficulty 1 2
```

Output:
```yaml
target_kernels:
  - "add_example.py"
  - "vector_addition.py"
  - "elementwise_mul.py"
  # ...
```

### Update Config File Directly

```bash
# Update target_kernels in existing config
python scripts/list_kernels.py --config configs/examples/my_config.yaml --difficulty 1 2

# Update with specific range
python scripts/list_kernels.py --config configs/examples/test_config.yaml --difficulty 1 --range 0 5
```

This will automatically replace the `target_kernels` field in your config file.

## Common Workflows

### 1. Quick Testing (1-2 Easy Kernels)

```bash
# Update test config with 2 easy kernels
python scripts/list_kernels.py --config configs/examples/test_config.yaml --difficulty 1 --range 0 2
```

### 2. Progressive Difficulty Experiments

```bash
# Create configs for different difficulty levels
python scripts/list_kernels.py --config configs/easy.yaml --difficulty 1 2
python scripts/list_kernels.py --config configs/moderate.yaml --difficulty 3
python scripts/list_kernels.py --config configs/hard.yaml --difficulty 4 5
```

### 3. Full Benchmark

```bash
# Update config with all kernels
python scripts/list_kernels.py --config configs/full_benchmark.yaml --all
```

## Options Reference

| Option | Description |
|--------|-------------|
| `--stats` | Show kernel statistics by difficulty |
| `--all` | List all kernels |
| `--difficulty N [N ...]` | Filter by difficulty levels (1-5) |
| `--range START END` | Filter by index range |
| `--yaml` | Output in YAML format |
| `--config PATH` | Update target_kernels in config file |

## Data Source

The tool reads kernel information from:
```
data/TritonBench/statistics.json
```

Each kernel entry contains:
- `file`: Kernel filename (e.g., "add_example.py")
- `difficulty`: Difficulty rating (1-5)
- `instruction`: Natural language description
- `label`: Reference implementation

## Related Documentation

- [README.md](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guide
