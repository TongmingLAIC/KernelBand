"""
Hardware-Informed Strategy Masking

Paper formula: M_{i,s} = I[h(k_c^{(i)})[Target(s)] < theta_sat]

For each cluster i, check the hardware signature of its representative kernel.
If the targeted resource (SM/DRAM/L2) is already saturated (above theta_sat),
mask out that strategy since further optimization in that direction has limited headroom.

Safety: always keep at least 1 strategy unmasked per cluster.
"""

import numpy as np
from typing import List, Optional, Dict, Union
from loguru import logger

from kernelband.mab.state import KernelEntry, MABState
from kernelband.mab.strategies import PAPER_STRATEGY_NAMES, STRATEGY_TARGET_MAP, NUM_STRATEGIES, HW_DIM_COUNT


class HardwareMasker:
    """Compute hardware masks M_{i,s} based on NCU profiling."""

    def __init__(self, theta_sat: Union[float, np.ndarray] = 75.0,
                 strategy_target_map: Optional[Dict[str, int]] = None):
        """
        Args:
            theta_sat: Saturation threshold in percent. Strategies targeting
                       a resource above this threshold are masked out.
                       Can be a scalar (applied to all dimensions) or an
                       ndarray of shape (3,) for per-dimension thresholds
                       [SM, DRAM, L2].
            strategy_target_map: Maps strategy name -> hardware dimension index
                                 (0=SM, 1=DRAM, 2=L2). Defaults to
                                 STRATEGY_TARGET_MAP from strategies module.
        """
        if isinstance(theta_sat, np.ndarray):
            self.theta_sat = theta_sat.astype(np.float64)
        else:
            self.theta_sat = np.full(HW_DIM_COUNT, float(theta_sat), dtype=np.float64)
        self.strategy_target_map = strategy_target_map if strategy_target_map is not None else STRATEGY_TARGET_MAP

    def compute_masks(self, mab_state: MABState,
                      cluster_hw_sigs: Dict[int, Optional[np.ndarray]]) -> np.ndarray:
        """
        Compute hardware masks for all (cluster, strategy) pairs.

        Args:
            mab_state: Current MAB state (for K, S dimensions)
            cluster_hw_sigs: Dict mapping cluster_id -> hw_sig array [SM%, DRAM%, L2%]
                            or None if profiling unavailable for that cluster.

        Returns:
            np.ndarray of shape (K, S) with binary mask values (1=active, 0=masked)
        """
        K, S = mab_state.K, mab_state.S
        masks = np.ones((K, S), dtype=np.float64)

        for cluster_id in range(K):
            hw_sig = cluster_hw_sigs.get(cluster_id)
            if hw_sig is None:
                continue

            for s_idx, strategy_name in enumerate(PAPER_STRATEGY_NAMES):
                target_dim = self.strategy_target_map[strategy_name]
                resource_utilization = hw_sig[target_dim]

                if resource_utilization >= self.theta_sat[target_dim]:
                    masks[cluster_id, s_idx] = 0.0
                    logger.debug(
                        f"Mask: cluster={cluster_id}, strategy={strategy_name}, "
                        f"resource_dim={target_dim}, util={resource_utilization:.1f}% >= {self.theta_sat[target_dim]:.1f}%"
                    )

            # Safety: ensure at least 1 strategy is unmasked per cluster
            if masks[cluster_id].sum() == 0:
                # Unmask the strategy targeting the least saturated resource
                least_saturated_idx = self._find_least_saturated_strategy(hw_sig)
                masks[cluster_id, least_saturated_idx] = 1.0
                logger.warning(
                    f"All strategies masked for cluster {cluster_id}, "
                    f"unmasking {PAPER_STRATEGY_NAMES[least_saturated_idx]}"
                )

        return masks

    def _find_least_saturated_strategy(self, hw_sig: np.ndarray) -> int:
        """Find the strategy index targeting the least saturated resource."""
        best_idx = 0
        best_headroom = -1.0

        for s_idx, strategy_name in enumerate(PAPER_STRATEGY_NAMES):
            target_dim = self.strategy_target_map[strategy_name]
            headroom = self.theta_sat[target_dim] - hw_sig[target_dim]
            if headroom > best_headroom:
                best_headroom = headroom
                best_idx = s_idx

        return best_idx
