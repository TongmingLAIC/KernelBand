# TritonBench-G Dataset (KernelBand Version)

This repository contains the **TritonBench-G** dataset as modified and improved through multiple iterations for high-quality Triton kernel evaluation and development.

## Overview

TritonBench-G is a comprehensive benchmark dataset featuring **184 Triton GPU kernels** for evaluating kernel generation and optimization capabilities. This KernelBand version incorporates fixes and improvements from multiple research efforts:

- **184 executable kernel implementations** across various difficulty levels (1-5)
- **GPU-specific performance baselines** for accurate speedup calculation
- **Instruction datasets** in Alpaca format for LLM training
- **Training corpus** (8K samples) for retrieval-augmented generation
- **Comprehensive fixes** addressing evaluation accuracy, platform compatibility, and kernel correctness

## Modification History

This dataset has undergone several rounds of improvements:

### 1. Original TritonBench (Tsinghua University)
- **Source**: [thunlp/TritonBench](https://github.com/thunlp/TritonBench)
- **Paper**: [TRITONBENCH: Benchmarking Large Language Model Capabilities for Generating Triton Operators](https://arxiv.org/pdf/2502.14752)
- Initial benchmark dataset with 184 Triton kernels

### 2. GEAK-eval Improvements (AMD-AGI)
- **Repository**: [GEAK-eval](https://github.com/AMD-AGI/GEAK-eval)
- **Paper**: [Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks](https://arxiv.org/abs/2507.23194)
- **Key fixes** (see GEAK-eval README for details):
  - Replaced string matching with `torch.allclose` for accurate numerical comparison
  - Fixed ~150 ground truth files missing `print(result_gold)` statements
  - Fixed 7 kernels with memory access faults
  - Added missing test invocations (e.g., `result_gold = test_*()`)
  - Implemented consistent seed management for reproducibility
  - Integrated comprehensive performance measurement system

### 3. This Version: KernelBand NVIDIA GPU Compatibility Fixes
- **Paper**: [KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit](https://arxiv.org/abs/2511.18868)
- **Additional fixes**: **23 kernels** addressing NVIDIA GPU platform compatibility and performance test bugs

**Fix Categories:**
1. **Platform Compatibility** (12 kernels) - Replaced AMD HIP-specific functions with NVIDIA-compatible implementations
2. **Triton API Updates** (1 kernel) - Fixed deprecated API usage
3. **Operator Constraints** (1 kernel) - Enforced dimension requirements for `tl.dot`
4. **Kernel Logic Bugs** (2 kernels) - Fixed division-by-zero errors in attention kernels
5. **Performance Test Bugs** (7 kernels) - Corrected tensor shape/format/range issues:
   - `token_attn_llama2_perf.py`: Fixed tensor layout issues
   - `chunk_retention_perf.py`: Fixed `initial_state` shape from `(B,H,T,D)` to `(B,H,D,D)`
   - `fast_rope_embedding_perf.py`: Reduced test range to avoid OOM (`range(2,16)` → `range(2,15)`)
   - `attn_fwd_triton_perf.py`: Fixed HEAD_DIM (64→128), dtype (float16→bfloat16), and q_scale/k_scale shape (scalar→array)
   - `chunk_bwd_dqkg_perf.py`: Fixed h/dh tensor shape from `(B,H,V,K)` to `(B,H,NT*K,V)` to match kernel's block pointer access pattern
   - `chunk_gla_fwd_perf.py`: Fixed k tensor shape from `(B,H,K,T)` to `(B,H,T,K)` to match kernel stride expectations
   - `token_softmax_llama_perf.py`: Fixed Logics from 3D to 2D shape, and B_Start_Loc from simple range to cumulative positions

## Dataset Structure

```
TritonBench/
├── data/
│   ├── TritonBench_G_v1/              # 184 executable kernel implementations (.py files)
│   ├── TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json  # Complex instructions with difficulty ratings
│   ├── train_crawl.json                # 4024 GitHub samples (de-duplicated)
│   └── train_synth.json                # 4133 synthesized samples
├── performance_metrics/
│   └── perf_G/
│       └── NVIDIA_GeForce_RTX_4090_golden_metrics/  # Baseline performance metrics
├── NVIDIA_RTX4090_KERNELS_EVALUATION.md  # Detailed fix documentation
└── README.md                           # This file
```

## Data Files

### Kernel Implementations
- **Location**: `data/TritonBench_G_v1/*.py`
- **Count**: 184 kernels
- **Content**: Complete Triton kernel implementations with test functions
- **Difficulty Range**: 1-5 (annotated in instruction dataset)

### Instruction Dataset
- **File**: `TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json`
- **Format**: Alpaca-style instructions
- **Fields**:
  - `instruction`: Task description for kernel generation
  - `input`: Additional context or requirements
  - `output`: Expected kernel implementation
  - `difficulty`: Kernel complexity rating (1=easiest, 5=hardest)
  - `label`: Kernel identifier (matches filename)

### Training Corpus for RAG
- **train_crawl.json**: 4,024 real-world Triton kernels from GitHub (BERT-score de-duplicated)
- **train_synth.json**: 4,133 synthesized training examples
- **Combined 8K dataset** suitable for retrieval-augmented generation systems

### Performance Baselines
- **Included baselines**: NVIDIA RTX 4090 golden metrics (can generate for other GPUs via `geak-eval setup`)
- **Location**: `performance_metrics/perf_G/{GPU_NAME}_golden_metrics/*.json`
- **Content**: Per-kernel performance metrics across 16 input shapes
- **Metrics**:
  - Execution time (ms)
  - Bandwidth (GB/s)
  - Throughput (TFLOPS)
  - Hardware efficiency (%)
- **Usage**: Calculate speedup by comparing generated kernels against GPU-specific baselines

## Usage

This dataset is designed for use with the **GEAK-eval** evaluation framework.

For complete usage instructions, evaluation pipeline details, and development guidelines, please refer to the **GEAK-eval repository**:

**📦 [GEAK-eval Repository](https://github.com/AMD-AGI/GEAK-eval)**

### Quick Start

```bash
# Install GEAK-eval
pip install -e GEAK-eval

# Setup golden baselines for your GPU
geak-eval setup -ds tbg

# Evaluate a generated kernel
geak-eval -f your_kernel.py -o results.json -ds tbg -c
```

### Python Environment Requirements

```bash
triton >= 3.3.0
torch >= 2.5.1
```

For detailed instructions on:
- Installation and setup
- Evaluation pipeline (call accuracy, execution accuracy, performance testing)
- Understanding evaluation results
- Extending the framework
- Troubleshooting

Please see the [GEAK-eval README](https://github.com/AMD-AGI/GEAK-eval/blob/master/README.md).

### Direct Kernel Usage

Each kernel file in `data/TritonBench_G_v1/` is self-contained and can be imported directly:

```python
import sys
sys.path.append('data/TritonBench_G_v1/')

from add_example import test_add_kernel

# Run the kernel test
test_add_kernel()
```

## Kernel Difficulty Levels

Kernels are rated by implementation complexity (1=easiest, 5=hardest):

| Difficulty | Count | Description |
|------------|-------|-------------|
| 1 | 3 | Simple operations (element-wise, basic reductions) |
| 2 | 27 | Moderate complexity (layer norm, softmax, basic operations) |
| 3 | 65 | Advanced patterns (attention mechanisms, fused ops) |
| 4 | 84 | Complex algorithms (chunked attention, specialized matmul) |
| 5 | 5 | Highly optimized kernels (flash attention variants, quantization) |
| **Total** | **184** | |

Use the `scripts/list_kernels.py` tool in the main KernelBand repository to select kernels by difficulty for batch experiments.

## Performance Baseline Details

The golden metrics provide GPU-specific performance baselines for accurate speedup calculation:

- **Test Shapes**: 16 configurations ranging from 4K to 134M elements
- **Metrics Captured**: Execution time, memory bandwidth, compute throughput
- **Speedup Formula**: `avg(golden_ms) / avg(generated_ms)`
- **Efficiency Calculation**: `max(bandwidth_efficiency, compute_efficiency)` vs. theoretical peak

Users can generate baselines for other GPU models using `geak-eval setup -ds tbg`.

## Known Issues

Some kernels may fail on certain hardware due to:
- **Shared memory limits**: Kernels with large block sizes may exceed GPU shared memory capacity
- **Memory constraints**: Large input sizes may cause OOM errors on GPUs with limited VRAM
- **Kernel bugs**: A few kernels have illegal memory access issues at large input sizes

The number of affected kernels varies by GPU. Check the logs in `performance_metrics/perf_G/{GPU_NAME}_golden_metrics/logs/` for details.

## Detailed Logs

Kernel evaluation logs are available in:
```
performance_metrics/perf_G/{GPU_NAME}_golden_metrics/logs/
```

Each kernel has two log files:
- `{kernel_name}_perf.py.log` - Standard output
- `{kernel_name}_perf.py.err` - Error output

## Citation

If you use this dataset, please cite the relevant papers based on which components you use:

**For the dataset with all improvements:**
```bibtex
@misc{ran2025kernelband,
  title={KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit},
  author={Dezhi Ran and Shuxiao Xie and Mingfang Ji and Ziyue Hua and Mengzhou Wu and Yuan Cao and Yuzhe Guo and Yu Hao and Linyi Li and Yitao Hu and Tao Xie},
  year={2025},
  eprint={2511.18868},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2511.18868},
}
```

**For the GEAK-eval evaluation framework:**
```bibtex
@misc{wang2025geakintroducingtritonkernel,
  title={Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks},
  author={Jianghui Wang and Vinay Joshi and Saptarshi Majumder and Xu Chao and Bin Ding and Ziqiong Liu and Pratik Prabhanjan Brahma and Dong Li and Zicheng Liu and Emad Barsoum},
  year={2025},
  eprint={2507.23194},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2507.23194},
}
```

**For the original TritonBench dataset:**
```bibtex
@article{tritonbench2025,
  title={TRITONBENCH: Benchmarking Large Language Model Capabilities for Generating Triton Operators},
  year={2025},
  journal={arXiv preprint arXiv:2502.14752},
  url={https://arxiv.org/pdf/2502.14752}
}
```

## Related Resources

- **GEAK-eval Repository**: [https://github.com/AMD-AGI/GEAK-eval](https://github.com/AMD-AGI/GEAK-eval)
- **Original TritonBench**: [Hugging Face Collection](https://huggingface.co/collections/LiShangZ/tritonbench-67c0016bc8a8654cfd612a1a)
- **Original TritonBench GitHub**: [thunlp/TritonBench](https://github.com/thunlp/TritonBench)

## License

This dataset is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) file for details.

Key points:
- ✅ Free to use, modify, and distribute
- ✅ Commercial use permitted
- ✅ Must retain copyright and license notices
- ✅ Changes must be documented
- ⚠️ Provided "as-is" without warranties

## Contact

**For GEAK-eval and this dataset:**
- Open issues on [GEAK-eval GitHub](https://github.com/AMD-AGI/GEAK-eval/issues)
- See [GEAK-eval README](https://github.com/AMD-AGI/GEAK-eval/blob/master/README.md) for documentation

**For original TritonBench dataset:**
- **Email**: qshi9510@gmail.com
- **Paper**: [arXiv:2502.14752](https://arxiv.org/pdf/2502.14752)
