"""
Strategy configuration for KernelBand MAB framework.

This module defines the 6 optimization strategies from the paper
and their corresponding prompts.
"""

from kernelband.prompts.strategies.tiling import prompt_tiling
from kernelband.prompts.strategies.vectorization import prompt_vectorization
from kernelband.prompts.strategies.fusion import prompt_fusion
from kernelband.prompts.strategies.pipeline import prompt_pipeline
from kernelband.prompts.strategies.reordering import prompt_reordering
from kernelband.prompts.strategies.access_layout import prompt_access_layout
from kernelband.mab.strategies import PAPER_STRATEGY_NAMES


# Single source of truth: strategy names from mab/strategies.py
STRATEGY_NAMES = PAPER_STRATEGY_NAMES

# Mapping from strategy names to their specific prompts
STRATEGY_PROMPTS = {
    "tiling": prompt_tiling,
    "vectorization": prompt_vectorization,
    "fusion": prompt_fusion,
    "pipeline": prompt_pipeline,
    "reordering": prompt_reordering,
    "access_layout": prompt_access_layout,
}

# Verify all strategies have prompts
assert len(STRATEGY_NAMES) == len(STRATEGY_PROMPTS) == 6, \
    f"Strategy count mismatch: {len(STRATEGY_NAMES)} names vs {len(STRATEGY_PROMPTS)} prompts"

assert all(name in STRATEGY_PROMPTS for name in STRATEGY_NAMES), \
    "Some strategies are missing prompts"
