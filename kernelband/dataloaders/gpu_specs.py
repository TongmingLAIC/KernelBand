"""
GPU Hardware Specifications Database

This module provides peak performance specifications for various GPUs,
used for calculating hardware efficiency metrics.

All specifications are for FP32 (single-precision floating point) operations
unless otherwise noted.
"""

# Compute Capability to Architecture Name mapping (NVIDIA GPUs)
# Reference: https://developer.nvidia.com/cuda-gpus
CC_TO_ARCH = {
    (9, 0): "Hopper",           # H100, H200
    (8, 9): "Ada Lovelace",     # RTX 4090, RTX 4080, L40
    (8, 6): "Ampere",           # RTX 3090, RTX 3080, RTX 3070, RTX 3060
    (8, 0): "Ampere",           # A100, A800, A10
    (7, 5): "Turing",           # RTX 2080 Ti, RTX 2080, T4
    (7, 0): "Volta",            # V100, Titan V
    (6, 1): "Pascal",           # GTX 1080 Ti, GTX 1080, P40, P4
    (6, 0): "Pascal",           # P100
    (5, 2): "Maxwell",          # GTX Titan X, GTX 980
    (3, 7): "Kepler",           # K80
    (3, 5): "Kepler",           # K40, K20
}

def get_arch_name(compute_capability: tuple) -> str:
    """
    Get architecture name from NVIDIA compute capability.

    Args:
        compute_capability: Tuple of (major, minor) version

    Returns:
        Architecture name if known, "Unknown" otherwise

    Examples:
        >>> get_arch_name((8, 9))
        'Ada Lovelace'
        >>> get_arch_name((7, 0))
        'Volta'
        >>> get_arch_name((9, 9))
        'Unknown'
    """
    return CC_TO_ARCH.get(compute_capability, "Unknown")

# GPU specifications: {normalized_name: (bandwidth_GB/s, peak_TFLOPS)}
# Bandwidth: Theoretical peak memory bandwidth
# Peak TFLOPS: Theoretical peak FP32 compute performance
GPU_SPECS = {
    # NVIDIA RTX 4090 (Ada Lovelace)
    "nvidia_geforce_rtx_4090": (1008, 82.6),
    "rtx_4090": (1008, 82.6),

    # NVIDIA A800 PCIe (Ampere)
    "nvidia_a800_80gb_pcie": (1935, 19.5),
    "nvidia_a800-pcie-80gb": (1935, 19.5),
    "a800": (1935, 19.5),

    # NVIDIA A100 PCIe (Ampere)
    "nvidia_a100_80gb_pcie": (1935, 19.5),
    "nvidia_a100-pcie-80gb": (1935, 19.5),
    "a100": (1935, 19.5),

    # NVIDIA H20 (Hopper, China market)
    "nvidia_h20": (4000, 44.0),
    "h20": (4000, 44.0),

    # Add more GPUs as needed
}

def normalize_gpu_name(gpu_name: str) -> str:
    """
    Normalize GPU name for matching against database.

    Args:
        gpu_name: Raw GPU name from torch.cuda.get_device_name()
                  Example: "NVIDIA GeForce RTX 4090"

    Returns:
        Normalized name in lowercase with spaces replaced by underscores
    """
    return gpu_name.lower().replace(" ", "_").replace("-", "_")

def get_gpu_specs(gpu_name: str) -> tuple:
    """
    Get GPU specifications (bandwidth, peak TFLOPS) for a given GPU.

    Args:
        gpu_name: GPU name from torch.cuda.get_device_name()

    Returns:
        Tuple of (bandwidth_GB/s, peak_TFLOPS) if GPU is found, None otherwise

    Examples:
        >>> get_gpu_specs("NVIDIA GeForce RTX 4090")
        (1008, 82.6)
        >>> get_gpu_specs("Unknown GPU")
        None
    """
    if not gpu_name:
        return None

    normalized = normalize_gpu_name(gpu_name)

    # First try exact match
    if normalized in GPU_SPECS:
        return GPU_SPECS[normalized]

    # Try partial matching (check if any key is substring of normalized name)
    for key, specs in GPU_SPECS.items():
        if key in normalized or normalized in key:
            return specs

    # GPU not found in database
    return None

def get_hardware_limits(compute_capability: tuple) -> dict:
    """
    Get hardware limits (shared memory, threads) based on compute capability.

    PyTorch's get_device_properties() doesn't provide these values, so we use
    architecture-specific known limits.

    Args:
        compute_capability: Tuple of (major, minor) version

    Returns:
        Dict with 'max_shared_mem_per_block' (bytes) and 'max_threads_per_block'

    Note:
        These are conservative estimates. Actual limits may be higher with opt-in.
    """
    major, minor = compute_capability

    # Default limits (conservative)
    limits = {
        'max_shared_mem_per_block': 48 * 1024,  # 48 KB
        'max_threads_per_block': 1024
    }

    if major >= 9:
        # Hopper (H100)
        limits = {
            'max_shared_mem_per_block': 227 * 1024,  # 227 KB with opt-in
            'max_threads_per_block': 1024
        }
    elif major == 8:
        if minor == 0:
            # Ampere (A100)
            limits = {
                'max_shared_mem_per_block': 163 * 1024,  # 164 KB with opt-in
                'max_threads_per_block': 1024
            }
        else:
            # Ada Lovelace (RTX 4090), Ampere (RTX 3090)
            limits = {
                'max_shared_mem_per_block': 99 * 1024,  # 99 KB
                'max_threads_per_block': 1024
            }
    elif major == 7:
        # Volta, Turing
        limits = {
            'max_shared_mem_per_block': 96 * 1024,  # 96 KB
            'max_threads_per_block': 1024
        }
    elif major == 6:
        # Pascal
        limits = {
            'max_shared_mem_per_block': 48 * 1024,  # 48 KB
            'max_threads_per_block': 1024
        }

    return limits

def list_supported_gpus():
    """
    List all GPUs currently supported in the database.

    Returns:
        List of GPU names (keys) in the database
    """
    return list(GPU_SPECS.keys())

# Example usage and testing
if __name__ == "__main__":
    print("Supported GPUs:")
    for gpu in list_supported_gpus():
        specs = GPU_SPECS[gpu]
        print(f"  {gpu}: {specs[0]} GB/s, {specs[1]} TFLOPS")

    print("\nTest GPU name matching:")
    test_names = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA A800-PCIE-80GB",
        "Unknown GPU Model"
    ]
    for name in test_names:
        specs = get_gpu_specs(name)
        if specs:
            print(f"  {name}: {specs[0]} GB/s, {specs[1]} TFLOPS")
        else:
            print(f"  {name}: Not found in database")
