"""
MAB (Multi-Armed Bandit) Module for KernelBand

Implements the paper's hierarchical MAB framework:
- Behavioral clustering (K-Means, K=3)
- Hardware-informed masking
- Masked UCB selection
- Within-cluster softmax sampling
"""

from kernelband.mab.state import MABState, KernelEntry
from kernelband.mab.ucb import UCBSelector
from kernelband.mab.clustering import BehavioralClusterer
from kernelband.mab.masking import HardwareMasker
from kernelband.mab.strategies import (
    PAPER_STRATEGY_NAMES, STRATEGY_TARGET_MAP, HW_DIM_NAMES, HW_DIM_COUNT,
)

__all__ = [
    "MABState",
    "KernelEntry",
    "UCBSelector",
    "BehavioralClusterer",
    "HardwareMasker",
    "PAPER_STRATEGY_NAMES",
    "STRATEGY_TARGET_MAP",
    "HW_DIM_NAMES",
    "HW_DIM_COUNT",
]
