"""
Reordering Strategy Prompt

Merged from: scheduling_autotuning.py + compute_instruction.py + parallelism_occupancy.py
Focus: Instruction-level compute optimization, occupancy tuning, grid scheduling,
and systematic autotuning of meta-parameters.
"""

prompt_reordering = """
[Strategy Focus: Reordering, Scheduling & Compute]
Goal: Map compute to hardware-friendly instructions (tensor cores / FMAs), shape tiles/loops so the compiler emits efficient MMA/ldst vectors, balance outer (grid) and inner (threads per program) parallelism, and systematically search meta-parameters to maximize throughput.

Follow this plan and reflect it in code:

1) Choose compute primitive & tile shapes
- Use tl.dot for contraction-style compute so the compiler lowers to tensor-core MMAs when eligible.
- Make BLOCK_M/BLOCK_N/BLOCK_K multiples of the hardware MMA tile (e.g., 16/32).
- Keep accumulators in fp32 when using low-precision inputs; set allow_tf32 if acceptable.

2) Arrange indices for vectorized, coalesced IO
- Construct tl.arange so the inner-most dimension is memory-contiguous for the hot tensor.
- Vectorize loads/stores along that axis (VEC elements per thread).
- Prove alignment with tl.multiple_of(..., 16/32/64) and contiguity with tl.max_contiguous.
- Prefer a single tl.store after the epilogue.

3) Encourage efficient instruction selection
- Write contractions as acc += tl.dot(a, b) with a/b in the promoted dtype and acc in fp32 to enable MMA/FMA.
- Keep arithmetic patterns fusable (mul-add becomes FMA) and avoid unnecessary casts between ops.
- Use predicate math (tl.where) instead of divergent branches for per-element conditions.

4) Overdecompose the grid for occupancy
- Map tiles to programs with grid = (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N) [, batch*heads * SPLIT_K]).
- Aim for total_programs >> SM_count (e.g., >= 4-8x more programs than SMs).
- Optionally group/swap traversal (GROUP_M or swizzle) so adjacent programs reuse cache.

5) Set intra-program parallelism
- Use num_warps to control per-CTA thread count (compute-heavy ops often like 4-8 warps).
- Choose per-thread work granularity (VEC and micro-tiles) so each thread does contiguous chunks without inflating live values.

6) Control pipeline depth vs. registers
- Use num_stages>1 to overlap K-chunk prefetch with compute; increase only while registers/SMEM remain within limits.
- If occupancy drops due to register pressure, reduce BLOCK_* or num_stages.

7) Autotuning with @triton.autotune
- Create a small, high-quality set of triton.Configs covering:
  - BLOCK_M/BLOCK_N in {64,128,256}, BLOCK_K in {16,32,64}
  - num_warps in {2,4,8}, num_stages in {1,2,3,4}
- Choose good autotune keys: key=['M','N','K', ...] (dimensions that alter performance).
- Pre-filter invalid configs (alignment, resource limits).
- Provide a safe default config (small BLOCK_*, VEC=1) as fallback.

8) Boundary-safe masking without divergence
- Compute masks for partial tiles and thread-level slices once; reuse across loads, compute, and stores.
- Keep compute branch-free where possible.

Expose and USE these knobs (top-of-file):
- BLOCK_M, BLOCK_N in {64,128,256} (multiples of 16/32 for MMA)
- BLOCK_K in {16,32,64}
- VEC in {1,2,4,8,16}
- NUM_WARPS in {2,4,8}
- NUM_STAGES in {1,2,3,4}
- GROUP_M in {1,2,4,8}
- SPLIT_K in {1,2,4}
- ALLOW_TF32 in {0,1}
"""
