"""
UCB Selection and Within-Cluster Sampling

Masked UCB selection (Eq. 3 in paper):
    (I_t, S_t) = argmax_{i,s : M_{i,s}=1} [mu_hat_{i,s} + c * sqrt(ln(t) / N_{i,s})]

Within-cluster sampling:
    P(k in C_i) proportional to exp(V_hw(k, S_t))
    V_hw(k, S_t) = theta_sat - h(k)[Target(S_t)]

Reward computation:
    r_t = max(0, (T(k_t) - T(k'_t)) / T(k_t))

UCB update (incremental mean):
    N_{i,s} += 1
    mu_hat_{i,s} += (r_t - mu_hat_{i,s}) / N_{i,s}
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

from kernelband.mab.state import MABState, KernelEntry
from kernelband.mab.strategies import PAPER_STRATEGY_NAMES, STRATEGY_TARGET_MAP, HW_DIM_COUNT


class UCBSelector:
    """Masked UCB arm selection and update logic."""

    def __init__(self, c: float = 2.0, theta_sat: Union[float, np.ndarray] = 75.0,
                 temperature: float = 1.0,
                 strategy_target_map: Optional[Dict[str, int]] = None):
        """
        Args:
            c: UCB exploration parameter
            theta_sat: Saturation threshold for V_hw computation.
                       Can be a scalar (applied to all dimensions) or an
                       ndarray of shape (3,) for per-dimension thresholds
                       [SM, DRAM, L2].
            temperature: Softmax temperature for within-cluster sampling
            strategy_target_map: Maps strategy name -> hardware dimension index
                                 (0=SM, 1=DRAM, 2=L2). Defaults to
                                 STRATEGY_TARGET_MAP from strategies module.
        """
        self.c = c
        if isinstance(theta_sat, np.ndarray):
            self.theta_sat = theta_sat.astype(np.float64)
        else:
            self.theta_sat = np.full(HW_DIM_COUNT, float(theta_sat), dtype=np.float64)
        self.temperature = temperature
        self.strategy_target_map = strategy_target_map if strategy_target_map is not None else STRATEGY_TARGET_MAP

    def select(self, mab_state: MABState) -> Tuple[int, int, str]:
        """
        Select (cluster_id, strategy_idx, strategy_name) via Masked UCB.

        Args:
            mab_state: Current MAB state

        Returns:
            (cluster_id, strategy_idx, strategy_name)
        """
        t = max(mab_state.total_t, 1)
        K, S = mab_state.K, mab_state.S

        # Compute UCB scores for all (i, s) pairs
        exploration = self.c * np.sqrt(np.log(t) / mab_state.N)
        ucb_scores = mab_state.mu_hat + exploration

        # Apply masks: set masked arms to -infinity
        masked_scores = np.where(mab_state.masks > 0, ucb_scores, -np.inf)

        # Find argmax
        best_idx = np.unravel_index(np.argmax(masked_scores), (K, S))
        cluster_id = int(best_idx[0])
        strategy_idx = int(best_idx[1])
        strategy_name = PAPER_STRATEGY_NAMES[strategy_idx]

        ucb_val = masked_scores[cluster_id, strategy_idx]
        exploit_val = mab_state.mu_hat[cluster_id, strategy_idx]
        explore_val = exploration[cluster_id, strategy_idx]

        logger.debug(
            f"UCB select: cluster={cluster_id}, strategy={strategy_name}, "
            f"UCB={ucb_val:.4f} (exploit={exploit_val:.4f}, explore={explore_val:.4f}), "
            f"N={int(mab_state.N[cluster_id, strategy_idx])}"
        )

        return cluster_id, strategy_idx, strategy_name

    def sample_from_cluster(self, frontier: List[KernelEntry],
                            cluster_id: int, strategy_name: str) -> Optional[KernelEntry]:
        """
        Sample a kernel from cluster C_i via softmax over V_hw scores.

        V_hw(k, S_t) = theta_sat - h(k)[Target(S_t)]
        Kernels with more headroom for the selected strategy get higher probability.

        Falls back to best-latency kernel if no hw signatures available.

        Args:
            frontier: All frontier kernels
            cluster_id: Selected cluster
            strategy_name: Selected strategy name

        Returns:
            Selected KernelEntry, or None if cluster is empty
        """
        # Get members of this cluster
        members = [e for e in frontier if e.cluster_id == cluster_id]
        if not members:
            # Fallback: use all frontier kernels
            members = frontier
        if not members:
            return None

        # Check if we have hw signatures for softmax sampling
        target_dim = self.strategy_target_map[strategy_name]
        has_hw = [e for e in members if e.hw_sig is not None]

        if has_hw:
            # Softmax sampling based on V_hw
            scores = []
            for entry in members:
                if entry.hw_sig is not None:
                    v_hw = self.theta_sat[target_dim] - entry.hw_sig[target_dim]
                else:
                    v_hw = 0.0  # Neutral score for entries without hw data
                scores.append(v_hw)

            scores = np.array(scores, dtype=np.float64)
            # Softmax with temperature
            scores = scores / max(self.temperature, 1e-6)
            scores = scores - scores.max()  # Numerical stability
            probs = np.exp(scores)
            probs = probs / (probs.sum() + 1e-10)

            idx = np.random.choice(len(members), p=probs)
            selected = members[idx]

            logger.debug(
                f"Within-cluster sample: cluster={cluster_id}, "
                f"strategy={strategy_name}, selected ms={selected.ms}"
            )
            return selected
        else:
            # Fallback: select best-performing kernel in cluster
            perf_members = [e for e in members if e.pass_perf and e.ms is not None]
            if perf_members:
                perf_members.sort(key=lambda e: e.ms)
                return perf_members[0]

            exe_members = [e for e in members if e.pass_exe]
            if exe_members:
                return exe_members[0]

            return members[0]

    @staticmethod
    def compute_reward(parent_ms: Optional[float], child_ms: Optional[float]) -> float:
        """
        Compute reward: r_t = max(0, (T(k_t) - T(k'_t)) / T(k_t))

        Args:
            parent_ms: Latency of parent kernel
            child_ms: Latency of child kernel (new generation)

        Returns:
            Reward in [0, inf), typically [0, 1]
        """
        if parent_ms is None or child_ms is None:
            return 0.0
        if parent_ms <= 0:
            return 0.0
        reward = max(0.0, (parent_ms - child_ms) / parent_ms)
        return reward

    @staticmethod
    def update(mab_state: MABState, cluster_id: int, strategy_idx: int, reward: float):
        """
        Incremental UCB update.

        N_{i,s} += 1
        mu_hat_{i,s} += (r_t - mu_hat_{i,s}) / N_{i,s}
        """
        mab_state.N[cluster_id, strategy_idx] += 1
        n = mab_state.N[cluster_id, strategy_idx]
        mab_state.mu_hat[cluster_id, strategy_idx] += (
            reward - mab_state.mu_hat[cluster_id, strategy_idx]
        ) / n
        mab_state.total_t += 1
