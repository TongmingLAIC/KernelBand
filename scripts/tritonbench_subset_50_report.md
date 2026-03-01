# TritonBench-G Subset Sampling Report

## Summary

- **Original dataset**: 184 kernels
- **Sampled subset**: 50 kernels (27.2%)
- **Random seed**: 42
- **Method**: Two-Phase Stratified Sampling

## Difficulty Distribution

| Level | Original | Sampled | Deviation |
|-------|----------|---------|-----------|
| L1 | 3 (1.6%) | 1 (2.0%) | 0.4% |
| L2 | 27 (14.7%) | 7 (14.0%) | 0.7% |
| L3 | 65 (35.3%) | 18 (36.0%) | 0.7% |
| L4 | 84 (45.7%) | 23 (46.0%) | 0.3% |
| L5 | 5 (2.7%) | 1 (2.0%) | 0.7% |

**Max deviation**: 0.7%

## Category Distribution

| Category | Original | Sampled | Covered |
|----------|----------|---------|---------|
| Attention | 29 (15.8%) | 7 (14.0%) | Yes |
| Element-wise Ops | 16 (8.7%) | 3 (6.0%) | Yes |
| Embedding/RoPE | 11 (6.0%) | 3 (6.0%) | Yes |
| Fused Ops/Activation | 10 (5.4%) | 4 (8.0%) | Yes |
| Linear Attention/SSM | 17 (9.2%) | 4 (8.0%) | Yes |
| Loss Functions | 7 (3.8%) | 3 (6.0%) | Yes |
| MatMul/GEMM | 26 (14.1%) | 7 (14.0%) | Yes |
| Memory/Index Ops | 13 (7.1%) | 3 (6.0%) | Yes |
| Normalization | 18 (9.8%) | 4 (8.0%) | Yes |
| Other | 12 (6.5%) | 3 (6.0%) | Yes |
| Quantization | 8 (4.3%) | 2 (4.0%) | Yes |
| Reduction | 6 (3.3%) | 3 (6.0%) | Yes |
| Softmax | 11 (6.0%) | 4 (8.0%) | Yes |

**Categories covered**: 13/13 (100.0%)

## Code Complexity (LOC)

| Metric | Original | Sampled |
|--------|----------|---------|
| Mean | 114.3 | 103.5 |
| Median | 96.5 | 94.0 |
| Std Dev | 81.5 | 64.7 |
| Min | 14 | 18 |
| Max | 451 | 277 |

## Autotuning Kernels

- Original: 46/184 (25.0%)
- Sampled: 10/50 (20.0%)

## Sampled Kernels

| # | Difficulty | Category | Filename |
|---|------------|----------|----------|
| 1 | L1 | Element-wise Ops | cosine_compute.py |
| 2 | L2 | Attention | flash_decode2_phi.py |
| 3 | L2 | MatMul/GEMM | matmul_kernel.py |
| 4 | L2 | Memory/Index Ops | matrix_transpose.py |
| 5 | L2 | Normalization | triton_mul2.py |
| 6 | L2 | Other | square_matrix.py |
| 7 | L2 | Reduction | triton_argmax.py |
| 8 | L2 | Softmax | softmax_triton1.py |
| 9 | L3 | Attention | flash_decode2_llama.py |
| 10 | L3 | Element-wise Ops | pow_scalar_tensor.py |
| 11 | L3 | Embedding/RoPE | embedding_triton_kernel.py |
| 12 | L3 | Fused Ops/Activation | relu_strided_buffer.py |
| 13 | L3 | Fused Ops/Activation | swiglu_backward.py |
| 14 | L3 | Fused Ops/Activation | swiglu_triton.py |
| 15 | L3 | Linear Attention/SSM | chunk_cumsum_vector.py |
| 16 | L3 | Linear Attention/SSM | reversed_cumsum_scalar.py |
| 17 | L3 | Loss Functions | kldiv_triton.py |
| 18 | L3 | MatMul/GEMM | triton_matmul.py |
| 19 | L3 | Memory/Index Ops | var_len_copy.py |
| 20 | L3 | Normalization | layer_norm_welfold.py |
| 21 | L3 | Normalization | rmsnorm_fused_llama.py |
| 22 | L3 | Other | uniform_sampling.py |
| 23 | L3 | Quantization | quantize_kv_copy.py |
| 24 | L3 | Reduction | matrix_reduction.py |
| 25 | L3 | Softmax | softmax_triton2.py |
| 26 | L3 | Softmax | softmax_triton3.py |
| 27 | L4 | Attention | attention_fwd_triton1.py |
| 28 | L4 | Attention | attention_fwd_triton2.py |
| 29 | L4 | Attention | attention_kernel.py |
| 30 | L4 | Attention | triton_attention.py |
| 31 | L4 | Element-wise Ops | matrix_vector_multip.py |
| 32 | L4 | Embedding/RoPE | fast_rope_embedding.py |
| 33 | L4 | Embedding/RoPE | rope_backward_transform.py |
| 34 | L4 | Fused Ops/Activation | relu_triton_kernel.py |
| 35 | L4 | Linear Attention/SSM | chunk_gate_recurrence.py |
| 36 | L4 | Linear Attention/SSM | fused_recurrent_retention.py |
| 37 | L4 | Loss Functions | cross_entropy_ops.py |
| 38 | L4 | Loss Functions | fast_ce_loss.py |
| 39 | L4 | MatMul/GEMM | int8_matmul_quantization.py |
| 40 | L4 | MatMul/GEMM | int_scaled_matmul.py |
| 41 | L4 | MatMul/GEMM | matmul_dequantize_int4.py |
| 42 | L4 | MatMul/GEMM | rms_matmul_rbe.py |
| 43 | L4 | MatMul/GEMM | streamk_matmul.py |
| 44 | L4 | Memory/Index Ops | kcache_copy_triton.py |
| 45 | L4 | Normalization | fused_layernorm_triton.py |
| 46 | L4 | Other | bgmv_expand_slice.py |
| 47 | L4 | Quantization | quantize_copy_kv.py |
| 48 | L4 | Reduction | logsumexp_fwd.py |
| 49 | L4 | Softmax | ksoftmax_triton.py |
| 50 | L5 | Attention | context_attn_bloom.py |