import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.token_softmax_llama import token_softmax_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('token_softmax_llama', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # Original issue was incorrect B_Start_Loc and Logics shape
        # With correct shapes, the full range should work
        for i in range(4, 14):  # Example sizes from 16 to 8192 (original range)
            batch_size = 2 ** i
            head_num = 8  # Example number of heads
            max_input_len = 512  # Example maximum input length

            # Logics must be 2D: (head_num, total_tokens) where total_tokens = batch_size * max_input_len
            total_tokens = batch_size * max_input_len
            Logics = torch.rand((head_num, total_tokens), dtype=torch.float32)

            # B_Start_Loc must be cumulative start positions for each batch element
            # NOT simple range [0, 1, 2, ...] but [0, max_input_len, 2*max_input_len, ...]
            B_Start_Loc = torch.arange(0, batch_size * max_input_len, max_input_len, dtype=torch.int32)

            B_Seqlen = torch.full((batch_size,), max_input_len, dtype=torch.int32)
            Prob_Out = torch.empty_like(Logics)

            self.input_tensors.append((Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len))

    def to_cuda(self, input_tensor):
        Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len = input_tensor
        return (Logics.cuda(), B_Start_Loc.cuda(), B_Seqlen.cuda(), Prob_Out.cuda(), max_input_len)

    def call_op(self, input_tensor):
        Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len = input_tensor
        token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len)
        return Prob_Out

    def get_gbps(self, input_tensor, runtime):
        Logics, _, _, _, _ = input_tensor
        total_bytes = 2 * Logics.numel() * Logics.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        Logics, _, _, _, _ = input_tensor
        FLOPS = 2 * Logics.numel()  # 2 operations per element (exp and division)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
