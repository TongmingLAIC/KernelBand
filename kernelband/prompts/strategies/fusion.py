"""
Fusion Strategy Prompt

Merged from: operation_fusion.py + reductions_scans_atomics.py
Focus: Fuse producer-consumer ops into a single kernel to cut DRAM traffic,
and build bandwidth-friendly hierarchical reductions.
"""

prompt_fusion = """
[Strategy Focus: Operation Fusion & Reductions]
Goal: Fuse producer->consumer ops into a single Triton kernel to cut DRAM traffic, eliminate intermediate tensors, and improve arithmetic intensity. For reduction-heavy kernels, build bandwidth-friendly hierarchical patterns that keep partials in registers/SRAM and touch global memory once.

Follow this plan and reflect it in code:

1) Identify fusible regions
- Prioritize straight-line epilogues after a heavy producer (e.g., tl.dot / contraction): scale, bias(add), activation, dropout, clamp, cast, quant/dequant, residual add.
- Fuse simple elementwise producers feeding another elementwise/reduction consumer when shapes/strides align.
- Avoid fusing if it causes excessive register pressure or control-divergent dataflow.

2) Producer->consumer chaining
- Compute the main tile (e.g., BLOCK_M * BLOCK_N) once, keep results in registers (acc).
- Apply epilogue stack in-register: acc = acc * scale + bias; acc = act(acc); acc = tl.where(mask, acc_do, acc), etc.
- For residuals or skip-connections, co-load the residual tile coalesced and combine in registers.
- Write final results once (single tl.store) after all fused ops.

3) Hierarchical reductions
- Tile the reduction axis into BLOCK_K-sized chunks.
- Per-thread: compute micro-partials from a contiguous slice (vectorized loads).
- Per-CTA: combine micro-partials with tl.sum / tl.max / tl.min over axes of the tile.
- Accumulate in high precision (ACC_DTYPE, e.g., fp32) before combining; only cast down at the very end.

4) Shard to avoid hotspots (for large reductions)
- If many CTAs write to the same output index, split the reduction across independent shards (SPLIT_R > 1).
- Each shard writes to a disjoint slice. Finalize with either:
  - a single atomic per output (tl.atomic_add / atomic_max), or
  - a deterministic post-pass that sums the shard buffers.
- Issue at most one atomic per output element per CTA.

5) Scans (prefix-sum) within a CTA
- For inclusive/exclusive scans over a short axis, perform staged upsweep/downsweep or Hillis-Steele style loop.
- Keep all scan steps register- or SMEM-resident.

6) Dataflow, lifetimes, and resource control
- Shorten live ranges: consume intermediates immediately and overwrite temporaries.
- Use micro-tiles so per-thread accumulator arrays remain register-resident.
- If fusion increases live values too much, reduce VEC or BLOCK_*.

7) Memory traffic & coalescing
- Load all operands (scale, bias, residual, mask) coalesced along the contiguous axis.
- Prefer broadcasting via strided views (no materialization).
- Reuse indices from the main tile.

8) Numeric precision & stability
- Keep accumulations in higher precision (fp32) even if inputs/outputs are fp16/bf16; cast at the end.
- For softmax/exp/log patterns, do max-subtraction and fused scaling within the same tile.
- For dropout/noise, generate RNG per-element deterministically from (seed, tile_coords, element_id).

9) Common fusion patterns
- GEMM/CONV + Scale + Bias + Activation (+ Residual) + Cast/Quant
- Norm (mean/var reduce over K) + Affine + Activation
- Softmax (max-reduce, exp, sum-reduce, scale) + Dropout
- Elementwise chains: add->mul->gelu->clamp->cast
- Attention block tails: qk^T scale->softmax->p * v

Expose and USE these knobs (top-of-file):
- BLOCK_M, BLOCK_N in {32,64,128,256}
- BLOCK_K in {16,32,64} (for fused reductions/epilogues over K)
- VEC in {1,2,4,8,16}
- GROUP_M in {1,2,4,8}
- NUM_WARPS in {2,4,8}
- NUM_STAGES in {1,2,3,4}
- SPLIT_R in {1,2,4,8} (reduction sharding)
- ACC_DTYPE in {"fp32"}
- FUSE_DROPOUT in {0,1}, FUSE_BIAS in {0,1}, FUSE_RESID in {0,1}
"""
