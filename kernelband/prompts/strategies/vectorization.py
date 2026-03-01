"""
Vectorization Strategy Prompt

Merged from: memory_access.py (vector load/store) + precision_dtype.py
Focus: Maximize bandwidth utilization via vectorized, coalesced memory access
and mixed-precision numerics.
"""

prompt_vectorization = """
[Strategy Focus: Vectorization & Precision]
Goal: Maximize memory bandwidth utilization by (1) ensuring contiguous program indices map to contiguous memory addresses for coalesced access, (2) vectorizing loads/stores along the contiguous axis, and (3) using mixed-precision (fp16/bf16/int8 I/O with fp32 accumulation) to reduce memory traffic while maintaining accuracy.

Follow this plan and reflect it in code:

1) Identify the contiguous (fastest-changing) dimension
- Detect which loop/program index corresponds to the "column direction" within each program.
- Ensure that tl.arange(...) in that axis maps to the contiguous memory axis of the hot tensor.
- If shapes/strides mismatch across operands, optimize for the tensor with largest memory footprint.

2) Enforce contiguous mapping and vectorize
- Map program/thread indices so that thread_id % BLOCK_SIZE corresponds directly to a contiguous memory slice.
- Use tl.arange(0, BLOCK_SIZE) to create vectors aligned to memory transactions.
- Issue vectorized loads/stores with tl.load(ptr + offsets, ...) where offsets form an arithmetic progression with step=1.
- Align VEC to cache-line multiples (16B/32B/64B).
- Assert with tl.multiple_of(offset, VEC) when provable.
- Use tl.max_contiguous to signal guaranteed contiguity to the compiler.
- Downgrade VEC (16->8->4->2->1) if misalignment occurs.

3) Mixed-precision I/O for bandwidth
- Declare numeric policy: inputs may be fp16/bf16/int8; accumulators default to fp32 for dot/reduction ops.
- Convert operands to accumulator dtype before contraction/reduction:
  - a_f = a.to(ACC_DTYPE); b_f = b.to(ACC_DTYPE)
  - acc = tl.zeros([...], dtype=ACC_DTYPE)
- Apply epilogue math in ACC_DTYPE; cast to OUT_DTYPE only at tl.store.
- For fp16/bf16 inputs: load as native dtype (saves bandwidth), promote to fp32 for math.
- For int8 inputs: dequantize on load with per-tensor or per-channel (scale, zp).

4) Vectorization under mixed precision
- Choose VEC so element_size * VEC aligns to 16/32/64B.
- When OUT_DTYPE is narrower than ACC_DTYPE, keep epilogue vectors in ACC_DTYPE and cast before store.

5) Numeric stability
- Reductions (softmax, norm, dot): accumulate in fp32 even if inputs are fp16/bf16.
- Apply stability tricks (max-subtraction in softmax) in ACC_DTYPE.
- Only downcast once final values are formed; avoid ping-pong casting.

6) Handle boundaries and partial tiles
- For non-divisible dimensions, use masks instead of divergent branching:
  - mask = offsets < N
  - val = tl.load(ptr + offsets, mask=mask, other=0)
- Avoid out-of-bounds loads/stores at program edges.

Expose and USE these knobs (top-of-file):
- BLOCK_SIZE in {64,128,256} (program tile width along contiguous axis)
- VEC in {1,2,4,8,16}
- ALIGN_BYTES in {16,32,64}
- IN_DTYPE in {"fp16","bf16","fp32","int8"}
- ACC_DTYPE in {"fp32"}
- OUT_DTYPE in {"fp16","bf16","fp32","int8"}
"""
