"""
Performance benchmark: SYCL JIT vs Triton vs PyTorch for EAGLE tree kernels.

Compares three implementations across various batch sizes and configurations:
1. SYCL JIT kernel (native XPU with icpx)
2. Triton kernel (Triton-based)
3. PyTorch reference (pure PyTorch)

Run with:
    python test/registered/jit/benchmark/bench_eagle_tree_xpu.py

Generate plots:
    python test/registered/jit/benchmark/bench_eagle_tree_xpu.py --plot
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    sgl_build_tree_kernel_efficient_pytorch,
    verify_tree_greedy_pytorch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    suite="base-b-kernel-benchmark-1-gpu-large",
    disabled="Requires XPU hardware",
)


# Benchmark configurations
BATCH_SIZE_LIST = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128],
    ci_range=[2, 8, 32],
)

DRAFT_TOKEN_NUM_LIST = get_benchmark_range(
    full_range=[4, 8, 12, 16, 20, 24],
    ci_range=[4, 8, 16],
)

TOPK = 2
DEPTH = 3
TREE_MASK_MODE = TreeMaskMode.QLEN_ONLY


def _make_build_tree_inputs(batch_size: int, draft_token_num: int, device: str):
    """Generate inputs for build_tree kernel."""
    parent_list = torch.randint(
        0,
        draft_token_num,
        (batch_size, draft_token_num - 1),
        dtype=torch.int64,
        device=device,
    )
    selected_index = torch.randint(
        0,
        TOPK * (draft_token_num - 1),
        (batch_size, draft_token_num - 1),
        dtype=torch.int64,
        device=device,
    )
    verified_seq_len = torch.randint(
        5, 15, (batch_size,), dtype=torch.int32, device=device
    )

    # Allocate outputs
    seq_lens_sum = int(verified_seq_len.sum())
    if TREE_MASK_MODE == TreeMaskMode.FULL_MASK:
        tree_mask_size = (
            seq_lens_sum * draft_token_num
            + draft_token_num * draft_token_num * batch_size
        )
    else:
        tree_mask_size = draft_token_num * batch_size * draft_token_num

    tree_mask = torch.full((tree_mask_size,), True, dtype=torch.bool, device=device)
    positions = torch.zeros(
        (batch_size * draft_token_num,), dtype=torch.int64, device=device
    )
    retrive_index = torch.full(
        (batch_size, draft_token_num), -1, dtype=torch.int64, device=device
    )
    retrive_next_token = torch.full(
        (batch_size, draft_token_num), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, draft_token_num), -1, dtype=torch.int64, device=device
    )

    return (
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


def _make_verify_tree_inputs(batch_size: int, num_draft_tokens: int, device: str):
    """Generate inputs for verify_tree kernel."""
    vocab_size = 32000

    candidates = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    target_predict = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    # Make some matches for realistic acceptance rate
    target_predict[:, 0] = candidates[:, 1]

    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=device
    ).reshape(batch_size, num_draft_tokens)
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    # Build simple tree structure
    for b in range(batch_size):
        for i in range(min(num_draft_tokens - 1, 3)):
            retrive_next_token[b, i] = i + 1

    # Allocate outputs
    predicts = torch.zeros(
        (batch_size * num_draft_tokens,), dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    return (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )


# ============================================================================
# Build Tree Benchmarks
# ============================================================================


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZE_LIST,
        line_arg="provider",
        line_vals=["sycl_jit", "triton", "pytorch"],
        line_names=["SYCL JIT", "Triton", "PyTorch"],
        styles=[("green", "-"), ("blue", "--"), ("red", ":")],
        ylabel="Time (us)",
        plot_name="build-tree-batch-size",
        args={"draft_token_num": 8},
    )
)
def bench_build_tree_batch_size(batch_size: int, draft_token_num: int, provider: str):
    """Benchmark build_tree kernel across batch sizes."""
    device = "xpu" if torch.xpu.is_available() else "cuda"

    inputs = _make_build_tree_inputs(batch_size, draft_token_num, device)
    parent_list, selected_index, verified_seq_len = inputs[:3]
    outputs = inputs[3:]

    if provider == "sycl_jit":
        try:
            from sglang.jit_kernel.eagle_tree_xpu import (
                sgl_build_tree_kernel_efficient_xpu,
            )

            fn = lambda: sgl_build_tree_kernel_efficient_xpu(
                parent_list,
                selected_index,
                verified_seq_len,
                *outputs,
                TOPK,
                DEPTH,
                draft_token_num,
                TREE_MASK_MODE,
            )
        except Exception:
            return float("nan")

    elif provider == "triton":
        try:
            from sglang.srt.speculative.eagle_utils import (
                sgl_build_tree_kernel_triton,
            )

            fn = lambda: sgl_build_tree_kernel_triton(
                parent_list,
                selected_index,
                verified_seq_len,
                *outputs,
                TOPK,
                DEPTH,
                draft_token_num,
                TREE_MASK_MODE,
            )
        except Exception:
            return float("nan")

    else:  # pytorch
        fn = lambda: sgl_build_tree_kernel_efficient_pytorch(
            parent_list,
            selected_index,
            verified_seq_len,
            *outputs,
            TOPK,
            DEPTH,
            draft_token_num,
            TREE_MASK_MODE,
        )

    median_us, max_us, min_us = run_benchmark(fn)
    return median_us


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["draft_token_num"],
        x_vals=DRAFT_TOKEN_NUM_LIST,
        line_arg="provider",
        line_vals=["sycl_jit", "triton", "pytorch"],
        line_names=["SYCL JIT", "Triton", "PyTorch"],
        styles=[("green", "-"), ("blue", "--"), ("red", ":")],
        ylabel="Time (us)",
        plot_name="build-tree-draft-tokens",
        args={"batch_size": 8},
    )
)
def bench_build_tree_draft_tokens(
    batch_size: int, draft_token_num: int, provider: str
):
    """Benchmark build_tree kernel across draft token counts."""
    device = "xpu" if torch.xpu.is_available() else "cuda"

    inputs = _make_build_tree_inputs(batch_size, draft_token_num, device)
    parent_list, selected_index, verified_seq_len = inputs[:3]
    outputs = inputs[3:]

    if provider == "sycl_jit":
        try:
            from sglang.jit_kernel.eagle_tree_xpu import (
                sgl_build_tree_kernel_efficient_xpu,
            )

            fn = lambda: sgl_build_tree_kernel_efficient_xpu(
                parent_list,
                selected_index,
                verified_seq_len,
                *outputs,
                TOPK,
                DEPTH,
                draft_token_num,
                TREE_MASK_MODE,
            )
        except Exception:
            return float("nan")

    elif provider == "triton":
        try:
            from sglang.srt.speculative.eagle_utils import (
                sgl_build_tree_kernel_triton,
            )

            fn = lambda: sgl_build_tree_kernel_triton(
                parent_list,
                selected_index,
                verified_seq_len,
                *outputs,
                TOPK,
                DEPTH,
                draft_token_num,
                TREE_MASK_MODE,
            )
        except Exception:
            return float("nan")

    else:  # pytorch
        fn = lambda: sgl_build_tree_kernel_efficient_pytorch(
            parent_list,
            selected_index,
            verified_seq_len,
            *outputs,
            TOPK,
            DEPTH,
            draft_token_num,
            TREE_MASK_MODE,
        )

    median_us, max_us, min_us = run_benchmark(fn)
    return median_us


# ============================================================================
# Verify Tree Benchmarks
# ============================================================================


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZE_LIST,
        line_arg="provider",
        line_vals=["sycl_jit", "triton", "pytorch"],
        line_names=["SYCL JIT", "Triton", "PyTorch"],
        styles=[("green", "-"), ("blue", "--"), ("red", ":")],
        ylabel="Time (us)",
        plot_name="verify-tree-batch-size",
        args={"num_draft_tokens": 8},
    )
)
def bench_verify_tree_batch_size(
    batch_size: int, num_draft_tokens: int, provider: str
):
    """Benchmark verify_tree kernel across batch sizes."""
    device = "xpu" if torch.xpu.is_available() else "cuda"

    inputs = _make_verify_tree_inputs(batch_size, num_draft_tokens, device)
    outputs = inputs[:3]
    other_inputs = inputs[3:]

    if provider == "sycl_jit":
        try:
            from sglang.jit_kernel.eagle_tree_xpu import verify_tree_greedy_xpu

            fn = lambda: verify_tree_greedy_xpu(*outputs, *other_inputs)
        except Exception:
            return float("nan")

    elif provider == "triton":
        try:
            from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

            fn = lambda: verify_tree_greedy_triton(*outputs, *other_inputs)
        except Exception:
            return float("nan")

    else:  # pytorch
        fn = lambda: verify_tree_greedy_pytorch(*outputs, *other_inputs)

    median_us, max_us, min_us = run_benchmark(fn)
    return median_us


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_draft_tokens"],
        x_vals=DRAFT_TOKEN_NUM_LIST,
        line_arg="provider",
        line_vals=["sycl_jit", "triton", "pytorch"],
        line_names=["SYCL JIT", "Triton", "PyTorch"],
        styles=[("green", "-"), ("blue", "--"), ("red", ":")],
        ylabel="Time (us)",
        plot_name="verify-tree-draft-tokens",
        args={"batch_size": 8},
    )
)
def bench_verify_tree_draft_tokens(
    batch_size: int, num_draft_tokens: int, provider: str
):
    """Benchmark verify_tree kernel across draft token counts."""
    device = "xpu" if torch.xpu.is_available() else "cuda"

    inputs = _make_verify_tree_inputs(batch_size, num_draft_tokens, device)
    outputs = inputs[:3]
    other_inputs = inputs[3:]

    if provider == "sycl_jit":
        try:
            from sglang.jit_kernel.eagle_tree_xpu import verify_tree_greedy_xpu

            fn = lambda: verify_tree_greedy_xpu(*outputs, *other_inputs)
        except Exception:
            return float("nan")

    elif provider == "triton":
        try:
            from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

            fn = lambda: verify_tree_greedy_triton(*outputs, *other_inputs)
        except Exception:
            return float("nan")

    else:  # pytorch
        fn = lambda: verify_tree_greedy_pytorch(*outputs, *other_inputs)

    median_us, max_us, min_us = run_benchmark(fn)
    return median_us


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE tree kernels (SYCL vs Triton vs PyTorch)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save performance plots",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EAGLE TREE KERNEL PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    if args.plot:
        print("Running benchmarks and generating plots...")
        print()

    # Run all benchmarks
    print("1. Build Tree - Batch Size Scaling")
    print("-" * 80)
    bench_build_tree_batch_size.run(print_data=True, show_plots=args.plot)
    print()

    print("2. Build Tree - Draft Token Scaling")
    print("-" * 80)
    bench_build_tree_draft_tokens.run(print_data=True, show_plots=args.plot)
    print()

    print("3. Verify Tree - Batch Size Scaling")
    print("-" * 80)
    bench_verify_tree_batch_size.run(print_data=True, show_plots=args.plot)
    print()

    print("4. Verify Tree - Draft Token Scaling")
    print("-" * 80)
    bench_verify_tree_draft_tokens.run(print_data=True, show_plots=args.plot)
    print()

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    if args.plot:
        print()
        print("Plots saved to current directory:")
        print("  - build-tree-batch-size.png")
        print("  - build-tree-draft-tokens.png")
        print("  - verify-tree-batch-size.png")
        print("  - verify-tree-draft-tokens.png")
