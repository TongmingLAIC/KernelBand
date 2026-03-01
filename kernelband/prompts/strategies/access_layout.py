"""
Access Layout Strategy Prompt

Merged from: memory_access.py (coalescing parts) + memory_layout.py
Focus: Data layout transformations and memory access pattern optimization
for coalescing and cache-friendly access.
"""

prompt_access_layout = """
[Strategy Focus: Access Layout & Data Arrangement]
Goal: Reduce strided/global-memory traffic by (1) choosing an access permutation that maps the innermost loop to the most-contiguous axis, (2) vectorizing loads/stores along that axis to maximize coalescing, and (3) rewriting indexing formulas to guarantee linear stride-1 access patterns.

Follow this plan and reflect it in code:

1) Detect contiguity & broadcasting
- Compute per-axis element strides S (elements, not bytes) from logical shapes/strides, using S_eff = abs(S).
- Treat broadcast axes (S==0 or extent==1) as low-priority for vectorization.
- Choose the innermost axis ax* := argmin_nonzero(S_eff); on ties prefer the larger extent.

2) Decide access permutation (logical reindex only)
- Define permutation to place ax* last in the indexing tuple (fastest-changing).
- If tensors disagree on stride orders, optimize for the hot tensor (most bandwidth-heavy).

3) Rewrite indexing for coalescing
- Map program/thread indices so each thread handles VEC contiguous elements on ax*.
- Use tl.arange(...) to form a [VEC] range on the innermost axis.
- Each thread block should cover a rectangular tile where the column axis is memory-contiguous.
- Avoid strided jumps in tl.load/tl.store: rewrite indexing so offsets along tl.arange are linear with stride==1.

4) Vectorized & aligned memory ops
- Use tl.load/tl.store with contiguous offsets along ax*.
- Only when statically provable:
  - tl.max_contiguous(index_expr, VEC)
  - tl.multiple_of(offset_bytes, ALIGN_BYTES)
- If alignment isn't met, progressively downgrade VEC (16->8->4->2->1).

5) Tail & boundary handling (low divergence)
- Handle non-multiple-of-VEC tails via masks; optionally pad shapes to PAD_TO (16/32/64) only in safe buffers.
- Prefer masked ops over per-thread branching.
- Use mask = offsets < N; val = tl.load(ptr + offsets, mask=mask, other=0).

6) Layout-aware optimizations
- For AoS (Array-of-Structs) data: consider SoA (Struct-of-Arrays) transformation when field accesses are independent.
- Pad inner dimensions to avoid bank conflicts in shared memory (pad to next power-of-2 + offset).
- For batched problems, keep per-batch tiles contiguous in memory.
- Prefer GROUP_M swizzle so consecutive programs touch nearby rows/cols, improving L2/L1 reuse.

Expose and USE these knobs (top-of-file):
- VEC in {1,2,4,8,16} (may auto-downgrade for alignment)
- BLOCK_SIZE in {64,128,256}
- PAD_TO in {16,32,64}
- ALIGN_BYTES in {16,32,64}
- CONTIG_AXIS: axis index designated as memory-contiguous (e.g., -1 for last dim)
"""
