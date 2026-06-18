"""
XPU JIT kernels for EAGLE tree operations using SYCL.

This module provides SYCL-based JIT kernels for Intel XPU to accelerate EAGLE
speculative decoding. The kernels are compiled using icpx and loaded via ctypes
(not TVM FFI, as XPU is not yet supported by TVM FFI).

Reference: PR #27136
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import pathlib
import subprocess
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Global cache for compiled modules
_COMPILED_MODULES = {}


def _get_cache_dir() -> pathlib.Path:
    """Get the cache directory for compiled SYCL modules."""
    cache_dir = os.environ.get("SGLANG_XPU_JIT_CACHE", "~/.cache/sglang/jit_sycl")
    cache_dir = pathlib.Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _compute_source_hash(source_files: list[pathlib.Path]) -> str:
    """Compute hash of source files for cache key."""
    hasher = hashlib.sha256()
    for source_file in sorted(source_files):
        if source_file.exists():
            hasher.update(source_file.read_bytes())
    return hasher.hexdigest()[:16]


def _compile_sycl_module(
    module_name: str,
    sycl_files: list[str],
    extra_cflags: Optional[list[str]] = None,
) -> pathlib.Path:
    """
    Compile SYCL source files into a shared library.

    Args:
        module_name: Unique identifier for the module
        sycl_files: List of SYCL header files (relative to csrc_sycl/)
        extra_cflags: Additional compiler flags

    Returns:
        Path to compiled shared library (.so file)
    """
    # Locate source files
    jit_kernel_path = pathlib.Path(__file__).parent.resolve()
    csrc_sycl_path = jit_kernel_path / "csrc_sycl"

    if not csrc_sycl_path.exists():
        raise RuntimeError(f"csrc_sycl directory not found at {csrc_sycl_path}")

    source_paths = []
    for sycl_file in sycl_files:
        source_path = csrc_sycl_path / sycl_file
        if not source_path.exists():
            raise FileNotFoundError(f"SYCL source file not found: {source_path}")
        source_paths.append(source_path)

    # Compute source hash for cache key
    source_hash = _compute_source_hash(source_paths)
    cache_key = f"{module_name}_{source_hash}"

    # Setup build directory
    cache_dir = _get_cache_dir()
    build_dir = cache_dir / cache_key
    build_dir.mkdir(parents=True, exist_ok=True)

    output_so = build_dir / f"lib{cache_key}.so"

    # Check if already compiled
    if output_so.exists():
        logger.debug(f"Using cached SYCL module: {cache_key}")
        return output_so

    # Compile SYCL module
    logger.info(f"Compiling SYCL module {module_name} for XPU...")

    # Create wrapper source that includes all SYCL headers
    wrapper_source = build_dir / "wrapper.cpp"
    with open(wrapper_source, "w") as f:
        for source_path in source_paths:
            f.write(f'#include "{source_path}"\n')

    # Default compiler flags
    default_cflags = [
        "-fsycl",  # Enable SYCL
        "-shared",  # Build shared library
        "-fPIC",  # Position-independent code
        "-O3",  # Optimization level 3
        "-std=c++20",  # C++20 standard
    ]

    if extra_cflags:
        default_cflags.extend(extra_cflags)

    # Compile with icpx (Intel oneAPI DPC++/C++ compiler)
    compile_cmd = [
        "icpx",
        *default_cflags,
        str(wrapper_source),
        "-o",
        str(output_so),
    ]

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(build_dir),
        )
        logger.debug(f"SYCL compilation successful: {module_name}")
        if result.stdout:
            logger.debug(f"Compiler stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = f"SYCL compilation failed for {module_name}\n"
        error_msg += f"Command: {' '.join(compile_cmd)}\n"
        error_msg += f"Return code: {e.returncode}\n"
        if e.stdout:
            error_msg += f"Stdout: {e.stdout}\n"
        if e.stderr:
            error_msg += f"Stderr: {e.stderr}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError:
        raise RuntimeError(
            "icpx compiler not found. Please install Intel oneAPI DPC++/C++ Compiler "
            "and ensure it's in your PATH. You may need to run:\n"
            "  source /opt/intel/oneapi/setvars.sh"
        ) from None

    logger.info(f"Successfully compiled SYCL module: {cache_key}")
    return output_so


def _load_sycl_library(module_name: str, sycl_files: list[str]) -> ctypes.CDLL:
    """
    Load SYCL library using ctypes.

    Args:
        module_name: Unique identifier for the module
        sycl_files: List of SYCL header files

    Returns:
        ctypes.CDLL handle to the loaded library
    """
    # Check cache first
    cache_key = (module_name, tuple(sycl_files))
    if cache_key in _COMPILED_MODULES:
        return _COMPILED_MODULES[cache_key]

    # Compile module
    lib_path = _compile_sycl_module(module_name, sycl_files)

    # Load library with ctypes
    try:
        lib = ctypes.CDLL(str(lib_path))
        _COMPILED_MODULES[cache_key] = lib
        logger.debug(f"Loaded SYCL library: {lib_path.name}")
        return lib
    except OSError as e:
        raise RuntimeError(
            f"Failed to load compiled SYCL library {lib_path}: {e}\n"
            "This may indicate ABI incompatibility between PyTorch XPU and oneAPI."
        ) from e


def sgl_build_tree_kernel_efficient_xpu(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
    optimized: bool = False,
):
    """
    XPU JIT kernel for building EAGLE tree structure.

    This SYCL-based implementation replaces the PyTorch fallback for better
    performance on Intel XPU devices.

    Set ``optimized=True`` to dispatch the re-parallelized kernel (one
    work-group per batch element, ``draft_token_num`` work-items per group,
    with SLM staging and int32 division). It produces identical outputs to the
    default scalar kernel.

    Args:
        parent_list: Parent indices [batch_size, draft_token_num-1] or [flattened]
        selected_index: Selected token indices [batch_size, draft_token_num-1]
        verified_seq_len: Sequence lengths [batch_size]
        tree_mask: Tree attention mask (output, modified in-place)
        positions: Token positions (output, modified in-place)
        retrive_index: Retrieval indices (output, modified in-place)
        retrive_next_token: Next token links (output, modified in-place)
        retrive_next_sibling: Next sibling links (output, modified in-place)
        topk: Top-k value for tree branching
        depth: Maximum tree depth
        draft_token_num: Number of draft tokens
        tree_mask_mode: Tree mask mode (0=FULL_MASK, 1=QLEN_ONLY)

    Raises:
        ValueError: If tensors are not on XPU device
        RuntimeError: If compilation or execution fails
    """
    if not parent_list.is_xpu:
        raise ValueError("All tensors must be on XPU device")

    batch_size = verified_seq_len.shape[0]

    # Calculate strides
    if parent_list.dim() > 1:
        parent_list_stride = parent_list.stride(0)
    else:
        parent_list_stride = parent_list.shape[0]

    selected_index_stride = selected_index.stride(0)

    # Load SYCL library
    lib = _load_sycl_library("eagle_tree", ["eagle_tree.hpp"])

    # Get SYCL queue pointer from XPU tensor
    queue_ptr = torch.xpu.current_stream(parent_list.device).sycl_queue

    # Setup C function signature
    # void sgl_build_tree_kernel_efficient_xpu(
    #     void* queue_ptr,
    #     const void* parent_list, const void* selected_index, const void* verified_seq_len,
    #     void* tree_mask, void* positions,
    #     void* retrive_index, void* retrive_next_token, void* retrive_next_sibling,
    #     int topk, int depth, int draft_token_num, int tree_mask_mode,
    #     int batch_size, int parent_list_stride, int selected_index_stride
    # )
    func_name = (
        "sgl_build_tree_kernel_efficient_xpu_optimized"
        if optimized
        else "sgl_build_tree_kernel_efficient_xpu"
    )
    func = getattr(lib, func_name)
    func.argtypes = [
        ctypes.c_void_p,  # queue_ptr
        ctypes.c_void_p,  # parent_list
        ctypes.c_void_p,  # selected_index
        ctypes.c_void_p,  # verified_seq_len
        ctypes.c_void_p,  # tree_mask
        ctypes.c_void_p,  # positions
        ctypes.c_void_p,  # retrive_index
        ctypes.c_void_p,  # retrive_next_token
        ctypes.c_void_p,  # retrive_next_sibling
        ctypes.c_int,  # topk
        ctypes.c_int,  # depth
        ctypes.c_int,  # draft_token_num
        ctypes.c_int,  # tree_mask_mode
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # parent_list_stride
        ctypes.c_int,  # selected_index_stride
    ]
    func.restype = None

    # Call SYCL kernel
    func(
        queue_ptr,
        parent_list.data_ptr(),
        selected_index.data_ptr(),
        verified_seq_len.data_ptr(),
        tree_mask.data_ptr(),
        positions.data_ptr(),
        retrive_index.data_ptr(),
        retrive_next_token.data_ptr(),
        retrive_next_sibling.data_ptr(),
        ctypes.c_int(topk),
        ctypes.c_int(depth),
        ctypes.c_int(draft_token_num),
        ctypes.c_int(tree_mask_mode),
        ctypes.c_int(batch_size),
        ctypes.c_int(parent_list_stride),
        ctypes.c_int(selected_index_stride),
    )


def verify_tree_greedy_xpu(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    optimized: bool = False,
):
    """
    XPU JIT kernel for verifying EAGLE tree greedily.

    This SYCL-based implementation replaces the PyTorch fallback for better
    performance on Intel XPU devices.

    Set ``optimized=True`` to dispatch the variant that packs multiple batch
    elements per work-group (filling the SIMD width), drops the redundant
    emulated 64-bit divide/modulo, and submits asynchronously. It produces
    identical outputs to the default kernel.

    Args:
        predicts: Predicted tokens (output, modified in-place)
        accept_index: Accepted token indices (output, modified in-place)
        accept_token_num: Number of accepted tokens per batch (output, modified in-place)
        candidates: Candidate draft tokens [batch_size, num_draft_tokens]
        retrive_index: Retrieval indices [batch_size, num_draft_tokens]
        retrive_next_token: Next token links [batch_size, num_draft_tokens]
        retrive_next_sibling: Next sibling links [batch_size, num_draft_tokens]
        target_predict: Target predictions [batch_size, num_draft_tokens]

    Raises:
        ValueError: If tensors are not on XPU device
        RuntimeError: If compilation or execution fails
    """
    if not candidates.is_xpu:
        raise ValueError("All tensors must be on XPU device")

    batch_size = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_speculative_tokens = accept_index.shape[1]

    # Load SYCL library
    lib = _load_sycl_library("eagle_verify", ["eagle_verify.hpp"])

    # Get SYCL queue pointer from XPU tensor
    queue_ptr = torch.xpu.current_stream(candidates.device).sycl_queue

    # Setup C function signature
    # void verify_tree_greedy_xpu(
    #     void* queue_ptr,
    #     void* predicts, void* accept_index, void* accept_token_num,
    #     const void* candidates, const void* retrive_index,
    #     const void* retrive_next_token, const void* retrive_next_sibling,
    #     const void* target_predict,
    #     int batch_size, int num_speculative_tokens, int num_draft_tokens
    # )
    func_name = (
        "verify_tree_greedy_xpu_optimized" if optimized else "verify_tree_greedy_xpu"
    )
    func = getattr(lib, func_name)
    func.argtypes = [
        ctypes.c_void_p,  # queue_ptr
        ctypes.c_void_p,  # predicts
        ctypes.c_void_p,  # accept_index
        ctypes.c_void_p,  # accept_token_num
        ctypes.c_void_p,  # candidates
        ctypes.c_void_p,  # retrive_index
        ctypes.c_void_p,  # retrive_next_token
        ctypes.c_void_p,  # retrive_next_sibling
        ctypes.c_void_p,  # target_predict
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # num_speculative_tokens
        ctypes.c_int,  # num_draft_tokens
    ]
    func.restype = None

    # Call SYCL kernel
    func(
        queue_ptr,
        predicts.data_ptr(),
        accept_index.data_ptr(),
        accept_token_num.data_ptr(),
        candidates.data_ptr(),
        retrive_index.data_ptr(),
        retrive_next_token.data_ptr(),
        retrive_next_sibling.data_ptr(),
        target_predict.data_ptr(),
        ctypes.c_int(batch_size),
        ctypes.c_int(num_speculative_tokens),
        ctypes.c_int(num_draft_tokens),
    )
