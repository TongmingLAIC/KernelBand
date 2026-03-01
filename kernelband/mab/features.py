"""
Feature Extraction for Behavioral Clustering and Hardware Signatures

Behavioral features phi(k) -- 5-dim vector for clustering:
  0: T_norm    - log-normalized execution time
  1: n_reg     - registers per thread (estimated from code)
  2: n_smem    - shared memory per block (estimated from code)
  3: d_block   - block dimension product (parsed from BLOCK_SIZE/BLOCK_M/BLOCK_N)
  4: eta_occ   - theoretical occupancy estimate

Hardware signature h(k) -- 3-dim vector from NCU profiling:
  0: SM throughput %
  1: DRAM throughput %
  2: L2 throughput %
"""

import re
import math
import numpy as np
from typing import Optional
from loguru import logger


def extract_behavioral_features(code: str, ms: Optional[float] = None,
                                kernel_metadata: Optional[dict] = None) -> np.ndarray:
    """
    Extract 5-dim behavioral feature vector phi(k) from kernel code.

    Uses actual compilation metadata when available (n_regs, shared, num_warps
    from Triton's compiled kernel cache), falling back to regex heuristics.

    Args:
        code: Triton kernel source code
        ms: Execution time in milliseconds (if available)
        kernel_metadata: Dict with actual compilation metadata from test subprocess:
                        {'n_regs': int, 'shared': int (bytes), 'num_warps': int}

    Returns:
        np.ndarray of shape (5,)
    """
    if ms is not None and ms > 0:
        t_norm = math.log1p(ms)
    else:
        t_norm = 0.0

    has_metadata = (kernel_metadata is not None
                    and kernel_metadata.get('n_regs') is not None
                    and kernel_metadata.get('shared') is not None
                    and kernel_metadata.get('num_warps') is not None)

    if has_metadata:
        n_reg = float(kernel_metadata['n_regs'])
        n_smem = kernel_metadata['shared'] / 1024.0  # bytes -> KB
        warp_size = 32
        threads = kernel_metadata['num_warps'] * warp_size
        d_block = math.log2(max(threads, 1.0))
        eta_occ = _compute_occupancy_from_metadata(n_reg, n_smem, d_block)
        logger.debug(
            f"Using actual metadata: n_regs={kernel_metadata['n_regs']}, "
            f"shared={kernel_metadata['shared']}B, num_warps={kernel_metadata['num_warps']}"
        )
    else:
        n_reg = _estimate_registers(code)
        n_smem = _estimate_shared_memory(code)
        d_block = _estimate_block_dims(code)
        eta_occ = _estimate_occupancy(n_reg, n_smem, d_block)

    return np.array([t_norm, n_reg, n_smem, d_block, eta_occ], dtype=np.float64)


def _compute_occupancy_from_metadata(n_reg: float, n_smem_kb: float,
                                     d_block_log2: float) -> float:
    """
    Compute occupancy from actual metadata, optionally using torch.cuda device props.
    """
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            max_regs_per_sm = 65536  # Standard for modern NVIDIA GPUs
            max_smem_per_sm = props.max_shared_memory_size  # bytes
            max_threads_per_sm = props.max_threads_per_multi_processor

            threads_per_block = 2 ** d_block_log2 if d_block_log2 > 0 else 128.0
            regs_per_block = max(n_reg, 16.0) * threads_per_block
            smem_per_block = n_smem_kb * 1024

            blocks_by_regs = max_regs_per_sm / max(regs_per_block, 1.0)
            blocks_by_smem = max_smem_per_sm / max(smem_per_block, 1.0) if smem_per_block > 0 else 32.0
            max_blocks = min(blocks_by_regs, blocks_by_smem, 32.0)
            active_threads = max_blocks * threads_per_block
            occupancy = min(active_threads / max_threads_per_sm, 1.0)
            return occupancy * 100.0
    except Exception:
        pass
    # Fall back to simplified estimate
    return _estimate_occupancy(n_reg, n_smem_kb, d_block_log2)


def extract_hw_signature(profiling_result) -> Optional[np.ndarray]:
    """
    Extract 3-dim hardware signature h(k) = [SM%, DRAM%, L2%] from NCU profiling.

    Args:
        profiling_result: NCUProfilingResult object

    Returns:
        np.ndarray of shape (3,) or None if profiling_result is None
    """
    if profiling_result is None:
        return None

    return np.array([
        profiling_result.sm_throughput_pct,
        profiling_result.dram_throughput_pct,
        profiling_result.l2_throughput_pct,
    ], dtype=np.float64)


def _estimate_registers(code: str) -> float:
    """Estimate register usage from code complexity."""
    assignments = re.findall(r'(\w+)\s*=', code)
    unique_vars = len(set(assignments))
    return min(float(unique_vars), 64.0)


def _estimate_shared_memory(code: str) -> float:
    """Estimate shared memory usage from code patterns."""
    smem_bytes = 0.0

    alloc_patterns = re.findall(
        r'tl\.(?:zeros|full)\s*\(\s*\[([^\]]+)\]',
        code
    )
    for dims_str in alloc_patterns:
        try:
            dims = [int(d.strip()) for d in dims_str.split(',') if d.strip().isdigit()]
            if dims:
                product = 1
                for d in dims:
                    product *= d
                smem_bytes += product * 4  # Assume fp32
        except (ValueError, TypeError):
            pass

    return smem_bytes / 1024.0


def _estimate_block_dims(code: str) -> float:
    """Estimate block dimension product from BLOCK_SIZE / BLOCK_M / BLOCK_N constants."""
    block_product = 1.0
    found = False

    for pattern in [
        r'BLOCK_SIZE\s*[=:]\s*(\d+)',
        r'BLOCK_M\s*[=:]\s*(\d+)',
        r'BLOCK_N\s*[=:]\s*(\d+)',
    ]:
        match = re.search(pattern, code)
        if match:
            block_product *= float(match.group(1))
            found = True

    for pattern in [
        r'BLOCK_SIZE\s*:\s*tl\.constexpr.*?#.*?(\d+)',
        r"BLOCK_SIZE.*?(?:default|=)\s*(\d+)",
    ]:
        match = re.search(pattern, code)
        if match and not found:
            block_product = float(match.group(1))
            found = True

    if not found:
        block_product = 128.0

    return math.log2(max(block_product, 1.0))


def _estimate_occupancy(n_reg: float, n_smem_kb: float, d_block_log2: float) -> float:
    """
    Simplified occupancy estimate based on resource usage.

    Real occupancy depends on GPU architecture, but this gives a relative ranking.
    Higher register/smem usage -> lower occupancy.
    """
    regs_per_thread = max(n_reg, 16.0)
    threads_per_block = 2 ** d_block_log2 if d_block_log2 > 0 else 128.0

    regs_per_block = regs_per_thread * threads_per_block
    smem_per_block = n_smem_kb * 1024

    blocks_by_regs = 65536.0 / max(regs_per_block, 1.0)
    blocks_by_smem = (100.0 * 1024) / max(smem_per_block, 1.0) if smem_per_block > 0 else 16.0

    max_blocks = min(blocks_by_regs, blocks_by_smem, 16.0)  # Max 16 blocks per SM
    active_threads = max_blocks * threads_per_block
    occupancy = min(active_threads / 2048.0, 1.0)

    return occupancy * 100.0  # Return as percentage
