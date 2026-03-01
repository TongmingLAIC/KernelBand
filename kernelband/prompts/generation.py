
prompt = """
You are an expert Python programmer specializing in NVIDIA Triton kernels, specifically targeting **NVIDIA GPUs via Triton's CUDA backend**.
Your task is to generate a Python code snippet containing a Triton kernel based on the following request:

**Target Platform:** NVIDIA GPU (CUDA)
**Target GPU:** {gpu_name}
**Compute Capability:** {compute_capability}
**GPU Architecture:** {architecture}

**Hardware Specifications:**
- SM Count: {sm_count}
- Max Shared Memory per Block: {max_shared_mem_per_block_kb} KB ({max_shared_mem_per_block_bytes} bytes)
- Max Threads per Block: {max_threads_per_block}
- Warp Size: {warp_size}
- Total GPU Memory: {total_memory_gb} GB

**CRITICAL HARDWARE CONSTRAINTS - MUST OBEY:**
1. **Shared Memory Limit**: Your kernel MUST NOT allocate more than {max_shared_mem_per_block_kb} KB of shared memory per block. Exceeding this will cause compilation errors!
2. **Thread Limit**: Each block MUST NOT exceed {max_threads_per_block} threads.
3. **Grid Dimensions**: Consider the {sm_count} SMs available when choosing grid size for optimal occupancy.
4. **Warp Alignment**: Memory accesses should be aligned to {warp_size}-thread warps for coalescing.

**Request:**
{instruction}

**CRITICAL FUNCTION INFORMATION:**
Based on analysis, the implementation requires these EXACT function signatures:
{function_signatures}

**Output Requirements:**
1.  **NVIDIA Compatibility:** Generate code compatible with NVIDIA GPUs and CUDA.
2.  **Complete Code:** Generate a single, complete, and syntactically correct Python code block.
3.  **Triton Kernel:** The core logic must be implemented within a Triton kernel function decorated with `@triton.jit`.
4.  **Imports:** ALWAYS include necessary imports at the beginning:
    ```python
    import torch
    import triton
    import triton.language as tl
    # import math # Only if standard math functions are truly needed outside the kernel
    ```
    Include other imports *only if absolutely necessary*.
5.  **Function Signature (CRITICAL):**
    *   Define EACH function with EXACTLY the signature shown above.
    *   DO NOT change parameter names, counts, or order.
    *   Ensure all parameters in function calls match their function definitions.
    *   **Type Hints:** Use PyTorch tensor type hints (e.g., `x: torch.Tensor`) for tensor arguments. **DO NOT use `tl.pointer`**. Use standard Python types (e.g., `int`, `float`) or `tl.constexpr` for others.
    *   **`constexpr`:** Use `tl.constexpr` **ONLY** for arguments that *must* be known at compile time, typically block sizes (like `BLOCK_SIZE`, `BLOCK_M`) or flags that change the kernel's structure (like `IS_EVEN_K`). Simple numerical values like `eps` or `dropout_p` are usually *not* `constexpr`.
6.  **Data Types:** Be precise with data types inside the kernel (e.g., `tl.float16`, `tl.float32`, `tl.int32`). Ensure type compatibility. Assume input tensors might be `torch.float16` or `torch.float32` unless specified otherwise. Pay attention to potential type promotion/conversion needs (e.g., using `.to(tl.float32)` for accumulations).
7.  **Triton Operations:**
    *   Use Triton language functions correctly (`tl.load`, `tl.store`, `tl.dot`, `tl.arange`, `tl.program_id`, `tl.where`, `tl.atomic_cas`, etc.).
    *   **Pointers & Masks:** Be extremely careful when constructing pointers using offsets and strides. Ensure masks in `tl.load`/`tl.store` are correctly computed and match pointer dimensions. Avoid `ValueError: Mask argument cannot be block type...` or `ValueError: Unsupported ptr type...`.
    *   **`tl.dot`:** Ensure inputs are 2D blocks and have compatible types (e.g., float16, bfloat16). Int32 is generally not supported directly as input.
    *   **`tl.arange`:** Arguments `start` and `end` **must be `tl.constexpr`**.
    *   **Math:** Use functions from `tl.math` where available (e.g., `tl.math.exp`, `tl.math.sqrt`). Check function existence; avoid assuming functions like `tanh` or `log1p` exist if they don't in `tl.math`.
8.  **Triton Version:** Assume Triton version 3.1.0 or later.

**FINAL VERIFICATION:**
Before completing, verify:
1. ALL functions defined in the code have EXACT signatures matching the required function signatures above.
2. ALL function calls exactly match their definitions in terms of parameter counts and names.
3. No functions are called without being defined.
4. No parameters are missing from your implementations.

**Generated NVIDIA CUDA Compatible Triton Kernel Code:**
"""
