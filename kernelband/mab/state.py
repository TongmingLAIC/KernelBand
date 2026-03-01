"""
MAB State Data Structures

KernelEntry: Represents a single kernel in the frontier.
MABState:    Tracks UCB statistics, cluster assignments, and masks per kernel problem.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from kernelband.mab.strategies import NUM_STRATEGIES


@dataclass
class KernelEntry:
    """A single kernel in the frontier P."""
    code: str
    ms: Optional[float] = None
    efficiency: Optional[float] = None
    speedup: Optional[float] = None
    strategy: str = "baseline"
    strategy_chain: List[str] = field(default_factory=list)
    pass_call: bool = False
    pass_exe: bool = False
    pass_perf: bool = False
    call_err_msg: Optional[str] = None
    exe_err_msg: Optional[str] = None
    # Behavioral features (5-dim) for clustering
    phi: Optional[np.ndarray] = None
    # Hardware signature (3-dim) [SM%, DRAM%, L2%]
    hw_sig: Optional[np.ndarray] = None
    # Cluster assignment
    cluster_id: Optional[int] = None

    def to_dict(self):
        """Serialize to dict (for JSON output)."""
        return {
            "code": self.code,
            "ms": self.ms,
            "efficiency": self.efficiency,
            "speedup": self.speedup,
            "strategy": self.strategy,
            "strategy_chain": self.strategy_chain,
            "pass_call": self.pass_call,
            "pass_exe": self.pass_exe,
            "pass_perf": self.pass_perf,
            "call_err_msg": self.call_err_msg,
            "exe_err_msg": self.exe_err_msg,
        }

    @staticmethod
    def from_dict(d):
        """Reconstruct a KernelEntry from a serialized dict (inverse of to_dict())."""
        from kernelband.mab.features import extract_behavioral_features
        entry = KernelEntry(
            code=d.get("code", ""),
            ms=d.get("ms"),
            efficiency=d.get("efficiency"),
            speedup=d.get("speedup"),
            strategy=d.get("strategy", "baseline"),
            strategy_chain=d.get("strategy_chain", [d.get("strategy", "baseline")]),
            pass_call=d.get("pass_call", True),
            pass_exe=d.get("pass_exe", True),
            pass_perf=d.get("pass_perf", False),
            call_err_msg=d.get("call_err_msg"),
            exe_err_msg=d.get("exe_err_msg"),
        )
        entry.phi = extract_behavioral_features(entry.code, entry.ms)
        entry.cluster_id = 0  # Default cluster; will be reassigned on next clustering
        return entry


@dataclass
class MABState:
    """
    Per-kernel MAB state tracking UCB statistics and clustering.

    Attributes:
        frontier: All valid kernels discovered so far (P in the paper).
        mu_hat:   (K, S) empirical mean rewards, initialized to 0.5.
        N:        (K, S) visit counts, initialized to 1.
        masks:    (K, S) binary hardware masks, initialized to 1 (all unmasked).
        cluster_centers: (K, 5) cluster centroids after K-Means.
        last_cluster_iter: Last iteration when clustering was performed.
        total_t:  Total number of MAB rounds played.
        best_entry: The best kernel found so far (lowest latency, correct).
    """
    K: int = 3  # Number of clusters
    S: int = NUM_STRATEGIES

    frontier: List[KernelEntry] = field(default_factory=list)
    mu_hat: np.ndarray = field(default=None)
    N: np.ndarray = field(default=None)
    masks: np.ndarray = field(default=None)
    cluster_centers: Optional[np.ndarray] = None
    last_cluster_iter: int = -1
    total_t: int = 0
    best_entry: Optional[KernelEntry] = None

    def __post_init__(self):
        if self.mu_hat is None:
            self.mu_hat = np.full((self.K, self.S), 0.5)
        if self.N is None:
            self.N = np.ones((self.K, self.S), dtype=np.float64)
        if self.masks is None:
            self.masks = np.ones((self.K, self.S), dtype=np.float64)

    def to_summary_dict(self):
        """Serialize MAB state summary for memory file."""
        d = {
            "K": self.K,
            "S": self.S,
            "total_t": self.total_t,
            "mu_hat": self.mu_hat.tolist(),
            "N": self.N.tolist(),
            "masks": self.masks.tolist(),
            "last_cluster_iter": self.last_cluster_iter,
            "frontier": [e.to_dict() for e in self.frontier],
        }
        if self.best_entry is not None:
            d["best_entry"] = self.best_entry.to_dict()
        return d

    @staticmethod
    def from_summary_dict(d):
        """Reconstruct MABState from serialized summary."""
        K = d.get("K", 3)
        S = d.get("S", NUM_STRATEGIES)
        state = MABState(K=K, S=S)
        state.mu_hat = np.array(d["mu_hat"])
        state.N = np.array(d["N"])
        state.masks = np.array(d["masks"])
        state.last_cluster_iter = d.get("last_cluster_iter", -1)
        state.total_t = d.get("total_t", 0)
        # Reconstruct frontier
        if "frontier" in d:
            state.frontier = [KernelEntry.from_dict(e) for e in d["frontier"]]
        # Reconstruct best_entry
        if d.get("best_entry") is not None:
            state.best_entry = KernelEntry.from_dict(d["best_entry"])
        return state
