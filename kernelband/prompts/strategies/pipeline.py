"""
Pipeline Strategy Prompt

Merged from: asynchrony_latency.py + controlflow_loop.py (software pipelining parts)
Focus: Overlap global memory loads with compute via software pipelining,
reshape loops for tight branch-free inner loops with good locality.
"""

prompt_pipeline = """
[Strategy Focus: Pipelining & Loop Optimization]
Goal: Overlap global memory loads with compute by structuring the K-loop as a software pipeline. Reshape loops and control flow so the compiler emits tight, branch-free inner loops with good locality. Unroll where profitable, split/fuse loops to expose parallelism.

Follow this plan and reflect it in code:

1) Chunk the reduction and set up staging
- Iterate K in BLOCK_K-sized chunks.
- Allocate 2-4 staging slots (register tiles or small SMEM tiles) for operands.
- Precompute base pointers and per-stage offsets so each next load uses simple pointer adds.

2) Pipeline pattern (double/triple buffering)
- Prologue: preload stage 0 (and stage 1 if NUM_STAGES>1).
- Main loop:
  - Issue loads for the next stage early (masked tl.load) using a different staging slot.
  - Compute on the current stage (e.g., acc += tl.dot(a[k], b[k]) plus fused epilogue).
  - Rotate staging indices: (stage_id = (stage_id + 1) % NUM_STAGES).
- Epilogue: finish compute for the last prefetched stage; keep the pattern uniform.

3) Keep loads ahead and independent
- Hoist address arithmetic for "next" loads before the compute block.
- Avoid creating RAW dependencies between future loads and current accumulators.
- Use vectorized, coalesced loads; prove alignment with tl.multiple_of / tl.max_contiguous where safe.

4) Pick the canonical loop nest
- Put the hottest, memory-contiguous axis in the innermost loop.
- Carry only the minimal loop-carried state (indices, accumulators); hoist loop-invariant math out.

5) Unroll the critical inner loop
- Unroll short, fixed-trip loops (e.g., BLOCK_K steps) to expose ILP and enable FMA/tensor-core scheduling.
- Use compile-time unrolling when trip count is known; otherwise partially unroll with tail handled by masks.

6) Split or fuse loops to match locality/parallelism
- Split (tile) long loops into chunks that fit registers/SMEM; compute-use within the chunk before advancing.
- Fuse adjacent producer->consumer loops so values stay in registers/SRAM instead of round-tripping to DRAM.
- When multiple small elementwise passes exist, merge them into a single pass over the data.

7) Prefer predication over divergent branches
- Replace if/else within hot loops by masked math (tl.where) and masked tl.load/tl.store.
- Compute masks once per iteration and reuse; avoid warp-level control divergence.

8) Balance pipeline depth vs. pressure
- Increase NUM_STAGES until memory latency is hidden; if registers/SMEM become tight, reduce NUM_STAGES or BLOCK_K.
- Prefer smaller BLOCK_K with more stages when latency-dominated; prefer larger BLOCK_K with fewer stages when bandwidth-limited.

9) Handle tails without control-flow knots
- Structure the main loop for full tiles; handle edges via masks or a single cleanup iteration.
- Keep memory indices monotonic and contiguous even on tails.

Expose and USE these knobs (top-of-file):
- BLOCK_K in {16,32,64}
- NUM_STAGES in {1,2,3,4} (>1 enables software pipelining)
- UNROLL_K in {1,2,4,8} (inner reduction/compute unroll factor)
- SPLIT_INNER in {1,2,4,8} (split factor for long inner loops)
- FUSE_PASSES in {0,1} (fuse consecutive elementwise passes into one loop)
"""
