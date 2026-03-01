"""
Paper's 6 Optimization Strategies and Target Mapping

Maps each strategy to the hardware resource dimension it primarily targets:
  0 = SM (compute throughput)
  1 = DRAM (memory bandwidth)
  2 = L2 (cache locality)

Strategy consolidation from original 11 -> 6:
  tiling         <- tiling_blocking
  vectorization  <- memory_access (vector load/store) + precision_dtype
  fusion         <- operation_fusion + reductions_scans_atomics
  pipeline       <- asynchrony_latency + controlflow_loop
  reordering     <- scheduling_autotuning + compute_instruction + parallelism_occupancy
  access_layout  <- memory_access (coalescing) + memory_layout
"""

# Hardware dimension name -> index mapping
HW_DIM_NAMES = {"sm": 0, "dram": 1, "l2": 2}

# Number of hardware dimensions
HW_DIM_COUNT = 3

PAPER_STRATEGY_NAMES = [
    "tiling",
    "vectorization",
    "fusion",
    "pipeline",
    "reordering",
    "access_layout",
]

# Maps strategy -> hardware signature index: 0=SM, 1=DRAM, 2=L2
STRATEGY_TARGET_MAP = {
    "tiling": 2,           # L2 (cache locality)
    "vectorization": 1,    # DRAM (memory throughput via vector loads)
    "fusion": 1,           # DRAM (reduce memory traffic)
    "pipeline": 0,         # SM (latency hiding)
    "reordering": 0,       # SM (ILP, instruction scheduling)
    "access_layout": 1,    # DRAM (coalescing, data layout)
}

# Number of strategies
NUM_STRATEGIES = len(PAPER_STRATEGY_NAMES)

# Strategy name -> index mapping
STRATEGY_INDEX = {name: i for i, name in enumerate(PAPER_STRATEGY_NAMES)}
