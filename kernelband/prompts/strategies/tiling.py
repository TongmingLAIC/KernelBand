"""
Tiling Strategy Prompt

Merged from: tiling_blocking.py
Focus: Partition iteration space into cache-/SRAM-/register-friendly tiles.
"""

prompt_tiling = """
[Strategy Focus: Tiling & Blocking]
Goal: Partition the iteration space into cache-/SRAM-/register-friendly tiles so each program (CTA) computes a rectangular block while threads within the program iterate contiguous micro-tiles. Balance tile size to avoid register/SMEM spill while providing enough parallelism to saturate the device.

Follow this plan and reflect it in code:

1) Define the tiling scheme
- Choose rectangular tiles for the dominant 2D plane (e.g., M * N for mat-like ops) and an optional reduction tile (K).
- Each program_id(0/1/2) selects a CTA-level tile (BLOCK_M * BLOCK_N [* BLOCK_K]).
- Inside a program, use tl.arange(...) to create thread-level indices that cover micro-tiles along the fastest-changing axis.

2) Map programs to tiles (grid decomposition)
- grid_m = ceil_div(M, BLOCK_M); grid_n = ceil_div(N, BLOCK_N). Consider swizzled/grouped traversal to improve L2/L1 locality:
  - GROUP_M groups several M-tiles per N-step to reuse cached rows/cols.
- Derive pid_m, pid_n from program_id(0/1) (and pid_k if split-K), then compute m0 = pid_m * BLOCK_M, n0 = pid_n * BLOCK_N.

3) Cooperative loading and compute
- Build per-thread offsets within the tile:
  - offs_m = m0 + tl.arange(0, BLOCK_M_STEP) * STRIDE_M_STEP
  - offs_n = n0 + tl.arange(0, BLOCK_N_STEP) * STRIDE_N_STEP
- Ensure inner-most tl.arange spans the memory-contiguous axis of the hot tensor to keep loads/stores coalesced.
- If using shared/SRAM or software-pipelined staging, stage sub-tiles per K-chunk (BLOCK_K) and reuse across the BLOCK_M * BLOCK_N compute.

4) Balance tile size vs. resource usage
- Too large tiles -> register/SMEM pressure -> potential spilling / low occupancy.
- Too small tiles -> poor arithmetic intensity & parallelism.
- Heuristics:
  - Keep per-thread live values bounded (e.g., BLOCK_M_STEP * BLOCK_N_STEP * accumulator_dtype) to fit registers.
  - Keep SMEM bytes per CTA within hardware limits; if > limit, reduce BLOCK_*, or NUM_STAGES.
  - Target enough CTAs in flight per SM to hide latency (guided by NUM_WARPS & occupancy).

5) Reduction and split-K (if applicable)
- For K-reductions, iterate K in chunks of BLOCK_K with software pipelining (NUM_STAGES).
- If K is huge or the kernel is bandwidth-bound, consider SPLIT_K > 1:
  - Each program handles a slice of K; reduce partials with atomic add or epilogue reduction.

6) Boundary handling (no divergence)
- Compute masks for partial tiles at edges:
  - mask_m = m0 + tl.arange(...) < M
  - mask_n = n0 + tl.arange(...) < N
- Apply masks to tl.load/tl.store to avoid OOB without warp-divergent branches.

7) Vectorization and alignment inside tiles
- Vectorize along the inner contiguous axis (VEC elements per thread). Align VEC to 16/32/64B when possible.
- Use tl.max_contiguous(...) and tl.multiple_of(..., ALIGN_BYTES) only when provable; otherwise downgrade VEC.

8) Locality optimizations
- Prefer GROUP_M (or a custom swizzle) so consecutive programs touch nearby rows/cols, improving L2/L1 reuse.
- For batched problems, carry batch as an outer axis and keep per-batch tiles contiguous in memory.

9) Autotuning / fallback
- Expose BLOCK_*, GROUP_M, VEC, NUM_WARPS, NUM_STAGES, SPLIT_K as knobs.
- Provide a safe fallback (smaller BLOCK_* and VEC=1) when alignment cannot be proven or when resource usage is too high.

Expose and USE these knobs (top-of-file):
- BLOCK_M, BLOCK_N in {32,64,128,256}
- BLOCK_K in {16,32,64} (for reductions)
- VEC in {1,2,4,8,16}
- GROUP_M in {1,2,4,8}
- SPLIT_K in {1,2,4,8}
- NUM_WARPS in {2,4,8}
- NUM_STAGES in {1,2,3,4}
- ALIGN_BYTES in {16,32,64}
"""
