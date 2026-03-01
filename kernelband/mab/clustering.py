"""
Behavioral Clustering via K-Means++

Groups kernels with similar performance signatures into K=3 clusters
using 5-dimensional behavioral features, reducing exploration overhead.

Re-clusters every tau iterations when |P| >= 2K.
Uses pure numpy K-Means++ (no sklearn dependency).
"""

import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

from kernelband.mab.state import KernelEntry


class BehavioralClusterer:
    """K-Means++ clustering on behavioral features."""

    def __init__(self, K: int = 3, max_iters: int = 50, seed: int = 42):
        """
        Args:
            K: Number of clusters
            max_iters: Max K-Means iterations
            seed: Random seed for reproducibility
        """
        self.K = K
        self.max_iters = max_iters
        self.rng = np.random.RandomState(seed)
        self._norm_mean = None
        self._norm_std = None

    def should_recluster(self, frontier_size: int, current_iter: int,
                         last_cluster_iter: int, tau: int) -> bool:
        """Check if re-clustering should happen this iteration."""
        if frontier_size < 2 * self.K:
            return False
        if last_cluster_iter < 0:
            # Never clustered before, do it now if enough data
            return True
        return (current_iter - last_cluster_iter) >= tau

    def cluster(self, frontier: List[KernelEntry]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Assign cluster IDs to all frontier entries.

        Args:
            frontier: List of KernelEntry with phi features extracted

        Returns:
            (assignments, centers) where:
              assignments: np.ndarray of shape (len(frontier),) with cluster IDs
              centers: np.ndarray of shape (K, feature_dim) or None if clustering failed
        """
        features = []
        valid_indices = []
        for i, entry in enumerate(frontier):
            if entry.phi is not None:
                features.append(entry.phi)
                valid_indices.append(i)

        n = len(features)
        if n == 0:
            logger.warning("No behavioral features available for clustering")
            assignments = np.zeros(len(frontier), dtype=np.int32)
            return assignments, None

        X = np.array(features, dtype=np.float64)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0  # Avoid division by zero
        X_norm = (X - mean) / std

        # Store normalization parameters for denormalization
        self._norm_mean = mean
        self._norm_std = std

        K_actual = min(self.K, n)

        if K_actual <= 1:
            assignments = np.zeros(len(frontier), dtype=np.int32)
            for i, idx in enumerate(valid_indices):
                frontier[idx].cluster_id = 0
            centers = X_norm.mean(axis=0, keepdims=True)
            # Denormalize centers back to original space
            centers_orig = centers * std + mean
            return assignments, centers_orig

        centers = self._kmeans_plus_plus_init(X_norm, K_actual)

        for _ in range(self.max_iters):
            dists = self._compute_distances(X_norm, centers)
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(K_actual):
                members = X_norm[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    new_centers[k] = X_norm[self.rng.randint(n)]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        assignments = np.zeros(len(frontier), dtype=np.int32)
        for i, idx in enumerate(valid_indices):
            assignments[idx] = labels[i]
            frontier[idx].cluster_id = int(labels[i])

        # Assign entries without features to nearest cluster (cluster 0)
        valid_set = set(valid_indices)
        for i in range(len(frontier)):
            if i not in valid_set:
                frontier[i].cluster_id = 0

        logger.info(
            f"Clustering: {n} kernels -> {K_actual} clusters, "
            f"sizes={[int(np.sum(labels == k)) for k in range(K_actual)]}"
        )

        # Denormalize centers back to original space for correct distance computation
        centers_orig = centers * std + mean

        return assignments, centers_orig

    def _kmeans_plus_plus_init(self, X: np.ndarray, K: int) -> np.ndarray:
        """K-Means++ initialization."""
        n, d = X.shape
        centers = np.zeros((K, d))

        idx = self.rng.randint(n)
        centers[0] = X[idx]

        for k in range(1, K):
            dists = self._compute_distances(X, centers[:k])
            min_dists = np.min(dists, axis=1)

            probs = min_dists / (min_dists.sum() + 1e-10)
            idx = self.rng.choice(n, p=probs)
            centers[k] = X[idx]

        return centers

    @staticmethod
    def _compute_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances from each point to each center."""
        # X: (n, d), centers: (k, d) -> result: (n, k)
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    def get_cluster_representative(self, frontier: List[KernelEntry],
                                   cluster_id: int,
                                   centers: Optional[np.ndarray] = None) -> Optional[KernelEntry]:
        """
        Get the representative kernel for a cluster (closest to centroid in phi-space).

        Falls back to the best-performing kernel if centroids are unavailable.
        """
        members = [e for e in frontier if e.cluster_id == cluster_id and e.phi is not None]
        if members and centers is not None and cluster_id < len(centers):
            centroid = centers[cluster_id]
            return min(members, key=lambda e: np.linalg.norm(e.phi - centroid))

        members = [e for e in frontier if e.cluster_id == cluster_id and e.pass_perf]
        if not members:
            members = [e for e in frontier if e.cluster_id == cluster_id and e.pass_exe]
        if not members:
            members = [e for e in frontier if e.cluster_id == cluster_id]
        if not members:
            return None

        members.sort(key=lambda e: e.ms if e.ms is not None else float('inf'))
        return members[0]
