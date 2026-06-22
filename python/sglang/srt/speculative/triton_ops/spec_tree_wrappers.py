"""
Wrapper functions for optimized Triton kernels with automatic backend selection.

This module provides drop-in replacements for the existing kernel wrappers that
automatically select between:
1. Original Triton kernel (baseline)
2. Optimized v1 kernel (moderate optimization)
3. Optimized v2 kernel (aggressive optimization)

Based on workload characteristics (batch_size, draft_token_num) and environment settings.
"""

import os
from typing import Optional

import torch
import triton

from sglang.srt.speculative.triton_ops.spec_tree import (
    sgl_build_tree_kernel_efficient_triton,
    verify_tree_greedy_kernel_triton,
)
from sglang.srt.speculative.triton_ops.spec_tree_optimized import (
    sgl_build_tree_kernel_optimized_v1,
    sgl_build_tree_kernel_optimized_v2,
    verify_tree_greedy_kernel_optimized_v1,
    verify_tree_greedy_kernel_optimized_v2,
)


def synchronize_device(device_type: str = 'xpu'):
    """Synchronize the specified device."""
    if device_type == 'xpu' and hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
    elif device_type == 'cuda':
        synchronize_device(tree_mask.device.type)
    # For CPU, no synchronization needed


# Environment variable controls
ENABLE_OPTIMIZED_KERNELS = os.environ.get("SGLANG_ENABLE_OPTIMIZED_SPEC_KERNELS", "1") == "1"
OPTIMIZATION_LEVEL = int(os.environ.get("SGLANG_SPEC_KERNEL_OPT_LEVEL", "1"))  # 0=baseline, 1=v1, 2=v2


def sgl_build_tree_kernel_triton(
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
):
    """
    Smart wrapper for build_tree kernel that selects optimal implementation.

    Auto-selects between baseline, v1, v2 based on:
    - Environment variables (SGLANG_SPEC_KERNEL_OPT_LEVEL)
    - Workload characteristics (draft_token_num size)

    Selection logic:
    - draft_token_num <= 85 (common): Use v2 (aggressive opt, fits in cache)
    - draft_token_num <= 128: Use v1 (moderate opt, safe)
    - draft_token_num > 128: Use baseline (avoid register spilling)
    """
    if not ENABLE_OPTIMIZED_KERNELS or OPTIMIZATION_LEVEL == 0:
        # Use baseline kernel
        kernel = sgl_build_tree_kernel_efficient_triton
    elif OPTIMIZATION_LEVEL == 2 and draft_token_num <= 85:
        # Use v2 (aggressive) for small-medium workloads
        kernel = sgl_build_tree_kernel_optimized_v2
    elif draft_token_num <= 128:
        # Use v1 (moderate) for medium-large workloads
        kernel = sgl_build_tree_kernel_optimized_v1
    else:
        # Fall back to baseline for very large workloads
        kernel = sgl_build_tree_kernel_efficient_triton

    # Prepare inputs for kernel launch
    batch_size = verified_seq_len.shape[0]
    parent_list_stride = parent_list.shape[1]
    selected_index_stride = selected_index.shape[1]

    # Compute prefix sums
    seq_len_prefix_sum = torch.zeros_like(verified_seq_len)
    if batch_size > 1:
        seq_len_prefix_sum[1:] = torch.cumsum(verified_seq_len[:-1], dim=0)

    # Launch kernel with grid size = batch_size (one thread block per batch item)
    grid = (batch_size,)

    kernel[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=tree_mask_mode,
        batch_size=batch_size,
        parent_list_stride=parent_list_stride,
        selected_index_stride=selected_index_stride,
    )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """
    Smart wrapper for verify_tree_greedy kernel that selects optimal implementation.

    Auto-selects between baseline, v1, v2 based on:
    - Environment variables (SGLANG_SPEC_KERNEL_OPT_LEVEL)
    - Workload characteristics

    Selection logic:
    - OPTIMIZATION_LEVEL == 2: Use v2 (aggressive with early exits)
    - OPTIMIZATION_LEVEL == 1: Use v1 (moderate with select ops)
    - Otherwise: Use baseline
    """
    batch_size = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_speculative_tokens = accept_index.shape[1]

    if not ENABLE_OPTIMIZED_KERNELS or OPTIMIZATION_LEVEL == 0:
        kernel = verify_tree_greedy_kernel_triton
    elif OPTIMIZATION_LEVEL == 2:
        kernel = verify_tree_greedy_kernel_optimized_v2
    else:
        kernel = verify_tree_greedy_kernel_optimized_v1

    grid = (batch_size,)

    kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def get_kernel_info() -> dict:
    """
    Return information about currently selected kernel implementations.

    Useful for debugging and performance analysis.
    """
    return {
        "optimized_kernels_enabled": ENABLE_OPTIMIZED_KERNELS,
        "optimization_level": OPTIMIZATION_LEVEL,
        "build_tree_kernel_options": [
            "baseline (triton)",
            "optimized_v1 (moderate)",
            "optimized_v2 (aggressive)",
        ],
        "verify_tree_kernel_options": [
            "baseline (triton)",
            "optimized_v1 (moderate)",
            "optimized_v2 (aggressive)",
        ],
        "selection_strategy": {
            "build_tree": {
                "opt_level_2_and_draft_tokens_<=_85": "optimized_v2",
                "draft_tokens_<=_128": "optimized_v1",
                "draft_tokens_>_128": "baseline",
            },
            "verify_tree": {
                "opt_level_2": "optimized_v2",
                "opt_level_1": "optimized_v1",
                "opt_level_0": "baseline",
            },
        },
    }


def benchmark_kernel_variants(
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
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """
    Benchmark all kernel variants and return performance metrics.

    Useful for selecting optimal kernel for specific workload.

    Returns:
        dict with keys: 'baseline', 'optimized_v1', 'optimized_v2'
        each containing mean/median/p95/p99 latency in milliseconds
    """
    import time

    batch_size = verified_seq_len.shape[0]
    parent_list_stride = parent_list.shape[1]
    selected_index_stride = selected_index.shape[1]

    seq_len_prefix_sum = torch.zeros_like(verified_seq_len)
    if batch_size > 1:
        seq_len_prefix_sum[1:] = torch.cumsum(verified_seq_len[:-1], dim=0)

    grid = (batch_size,)

    kernels = {
        "baseline": sgl_build_tree_kernel_efficient_triton,
        "optimized_v1": sgl_build_tree_kernel_optimized_v1,
        "optimized_v2": sgl_build_tree_kernel_optimized_v2,
    }

    results = {}

    for name, kernel in kernels.items():
        latencies = []

        # Warmup
        for _ in range(warmup):
            kernel[grid](
                parent_list,
                selected_index,
                verified_seq_len,
                seq_len_prefix_sum,
                tree_mask,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                topk=topk,
                depth=depth,
                draft_token_num=draft_token_num,
                tree_mask_mode=tree_mask_mode,
                batch_size=batch_size,
                parent_list_stride=parent_list_stride,
                selected_index_stride=selected_index_stride,
            )
        synchronize_device(tree_mask.device.type)

        # Benchmark
        for _ in range(iterations):
            # Reset outputs
            tree_mask.fill_(False)
            positions.fill_(0)
            retrive_index.fill_(0)
            retrive_next_token.fill_(0)
            retrive_next_sibling.fill_(0)

            start = time.perf_counter()
            kernel[grid](
                parent_list,
                selected_index,
                verified_seq_len,
                seq_len_prefix_sum,
                tree_mask,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                topk=topk,
                depth=depth,
                draft_token_num=draft_token_num,
                tree_mask_mode=tree_mask_mode,
                batch_size=batch_size,
                parent_list_stride=parent_list_stride,
                selected_index_stride=selected_index_stride,
            )
            synchronize_device(tree_mask.device.type)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        latencies_sorted = sorted(latencies)
        results[name] = {
            "mean_ms": sum(latencies) / len(latencies),
            "median_ms": latencies_sorted[len(latencies) // 2],
            "p95_ms": latencies_sorted[int(len(latencies) * 0.95)],
            "p99_ms": latencies_sorted[int(len(latencies) * 0.99)],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }

    # Compute speedups
    baseline_mean = results["baseline"]["mean_ms"]
    for name in ["optimized_v1", "optimized_v2"]:
        results[name]["speedup_vs_baseline"] = baseline_mean / results[name]["mean_ms"]

    return results
