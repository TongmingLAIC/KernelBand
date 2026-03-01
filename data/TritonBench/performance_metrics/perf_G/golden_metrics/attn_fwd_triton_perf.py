import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.attn_fwd_triton import forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attn_fwd_triton', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # The kernel uses hardcoded offs_k = tl.arange(0, 128) with mask < 96
        # HEAD_DIM must be 128 to match internal kernel assumptions
        # q_scale/k_scale must be arrays of shape (batch, heads, seq_len)
        for i in range(2, 12):  # Example sizes, adjust as needed
            batch_size = 2 ** i
            num_heads = 4  # Match test_forward() in kernel
            seq_len = 128  # Must match BLOCK_M = 128
            head_dim = 128  # Must be 128 to match kernel's internal tl.arange(0, 128)
            q = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16)
            k = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16)
            v = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16)
            # q_scale and k_scale must be per-block arrays matching kernel expectations
            q_scale = torch.randn((batch_size, num_heads, seq_len), dtype=torch.float32)
            k_scale = torch.randn((batch_size, num_heads, seq_len), dtype=torch.float32)
            self.input_tensors.append((q, k, v, q_scale, k_scale))

    def to_cuda(self, input_tensor):
        q, k, v, q_scale, k_scale = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), q_scale.cuda(), k_scale.cuda())

    def call_op(self, input_tensor):
        q, k, v, q_scale, k_scale = input_tensor
        return forward(q, k, v, q_scale, k_scale)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, q_scale, k_scale = input_tensor
        # Input: q, k, v + q_scale, k_scale; Output: o (same size as q)
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
        total_bytes += (q_scale.numel() + k_scale.numel()) * q_scale.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        q, k, v, q_scale, k_scale = input_tensor
        batch_size, num_heads, seq_len, head_dim = q.shape
        FLOPS = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
