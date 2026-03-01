import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_bwd_dqkg import chunk_bwd_dqkg_fn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_bwd_dqkg', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # The kernel accesses h/dh with logical shape (V, NT * K) where NT = T / BT
        # So h/dh must be sized as (B, H, V, NT * K) to avoid illegal memory access
        # With correct h/dh sizes, the full original range works (4.3GB max for T=2^19)
        for i in range(2, 20):  # T from 4 to 524288 (original range)
            B = 2
            H = 2
            T = 2 ** i
            K = 64
            V = 64
            BT = 64  # Block size used in kernel
            NT = (T + BT - 1) // BT  # triton.cdiv(T, BT)
            scale = 1.0 / (K ** 0.5)
            q = torch.rand((B, H, T, K), dtype=torch.float32)
            k = torch.rand((B, H, T, K), dtype=torch.float32)
            v = torch.rand((B, H, T, V), dtype=torch.float32)
            g = torch.rand((B, H, T), dtype=torch.float32)
            # h and dh must have shape (B, H, NT * K, V) to match kernel's expected access pattern
            # The kernel uses: tl.make_block_ptr(h, (V, NT * K), (1, s_h_t), ...)
            # With strides (1, s_h_t) where s_h_t = h.stride(2) = V
            # This means V dimension should be innermost (stride 1), so shape is (NT*K, V) per slice
            h = torch.rand((B, H, NT * K, V), dtype=torch.float32)
            do = torch.rand((B, H, T, V), dtype=torch.float32)
            dh = torch.rand((B, H, NT * K, V), dtype=torch.float32)
            self.input_tensors.append((do, q, k, v, g, h, dh, scale))

    def to_cuda(self, input_tensor):
        return tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in input_tensor)

    def call_op(self, input_tensor):
        do, q, k, v, g, h, dh, scale = input_tensor
        return chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)

    def get_gbps(self, input_tensor, runtime):
        do, q, k, v, g, h, dh, scale = input_tensor
        total_bytes = (do.numel() + q.numel() + k.numel() + v.numel() + g.numel() + h.numel() + dh.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        do, q, k, v, g, h, dh, scale = input_tensor
        B, H, T, K = q.shape
        V = v.shape[-1]
        # Estimate FLOPS based on operations in the kernel
        FLOPS = 2 * B * H * T * K * V
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
