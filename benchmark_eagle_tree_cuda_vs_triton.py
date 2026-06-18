#!/usr/bin/env python3
"""
Standalone Performance Benchmark: CUDA (sgl-kernel) vs Triton

Focused comparison of the two production EAGLE tree kernel implementations
on NVIDIA GPUs:
- CUDA   (sgl_kernel.build_tree_kernel_efficient / verify_tree_greedy)
- Triton (sglang.srt.speculative.eagle_utils.*_triton)

Both kernels are benchmarked for:
- build_tree   (sgl_build_tree_kernel_efficient vs sgl_build_tree_kernel_triton)
- verify_tree  (verify_tree_greedy vs verify_tree_greedy_triton)

This is a CUDA-only sibling of benchmark_eagle_tree_xpu.py: PyTorch and SYCL
JIT paths are intentionally dropped so the table shows just the CUDA/Triton
head-to-head.

Usage:
    python benchmark_eagle_tree_cuda_vs_triton.py [--quick] [--csv output.csv]

Options:
    --quick     Run quick benchmark with fewer configurations
    --csv FILE  Save results to CSV file
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    name: str
    implementation: str
    batch_size: int
    draft_tokens: int
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    std_us: float
    iterations: int


class BenchmarkRunner:
    """Manages benchmark execution and reporting."""

    def __init__(self, warmup_iters: int = 10, measure_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.results: List[BenchmarkResult] = []

    def benchmark_function(self, fn: Callable, name: str = "kernel") -> BenchmarkResult:
        """Benchmark a function with warmup and multiple iterations."""
        # Warmup
        for _ in range(self.warmup_iters):
            fn()
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(self.measure_iters):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1e6)  # Convert to microseconds

        times_tensor = torch.tensor(times)
        return BenchmarkResult(
            name=name,
            implementation="",
            batch_size=0,
            draft_tokens=0,
            mean_us=float(times_tensor.mean()),
            median_us=float(times_tensor.median()),
            min_us=float(times_tensor.min()),
            max_us=float(times_tensor.max()),
            std_us=float(times_tensor.std()),
            iterations=self.measure_iters,
        )

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def print_summary(self):
        """Print benchmark summary table (CUDA baseline, Triton speedup)."""
        print("\n" + "=" * 110)
        print("PERFORMANCE SUMMARY (CUDA vs Triton)")
        print("=" * 110)
        print(
            f"{'Kernel':<14} {'Impl':<10} {'Batch':<8} {'Tokens':<8} "
            f"{'Mean (us)':<12} {'Median (us)':<14} {'Speedup vs CUDA':<16}"
        )
        print("-" * 110)

        # Group by kernel and config
        kernels = {}
        for r in self.results:
            key = (r.name, r.batch_size, r.draft_tokens)
            kernels.setdefault(key, {})[r.implementation] = r

        for (name, batch, tokens), impls in sorted(kernels.items()):
            cuda_r = impls.get("cuda")
            baseline_us = cuda_r.median_us if cuda_r else None

            for impl_name in ["cuda", "triton"]:
                display_name = {"cuda": "CUDA", "triton": "Triton"}[impl_name]
                if impl_name not in impls:
                    continue
                r = impls[impl_name]
                speedup_str = ""
                if baseline_us:
                    speedup = baseline_us / r.median_us
                    speedup_str = f"{speedup:.2f}x"
                print(
                    f"{name:<14} {display_name:<10} {batch:<8} {tokens:<8} "
                    f"{r.mean_us:<12.2f} {r.median_us:<14.2f} {speedup_str:<16}"
                )

        print("=" * 110)

    def save_csv(self, filename: str):
        """Save results to CSV file."""
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "kernel",
                    "implementation",
                    "batch_size",
                    "draft_tokens",
                    "mean_us",
                    "median_us",
                    "min_us",
                    "max_us",
                    "std_us",
                    "iterations",
                ],
            )
            writer.writeheader()
            for r in self.results:
                writer.writerow(
                    {
                        "kernel": r.name,
                        "implementation": r.implementation,
                        "batch_size": r.batch_size,
                        "draft_tokens": r.draft_tokens,
                        "mean_us": r.mean_us,
                        "median_us": r.median_us,
                        "min_us": r.min_us,
                        "max_us": r.max_us,
                        "std_us": r.std_us,
                        "iterations": r.iterations,
                    }
                )
        print(f"\n✓ Results saved to {filename}")


def make_valid_tree_inputs(batch_size, draft_token_num, topk, device):
    """Build a VALID EAGLE tree (no orphan tokens) for benchmarking.

    Mirrors the parity-test "valid-chain" construction: every non-root token's
    parent resolves to an earlier selected token, so the build_tree kernels do
    not hit the "invalid eagle tree" warning path.

    Layout (topk-spaced chain):
      selected_index[k] = k * topk        -> parent_tb_idx = k
      parent_list[k]    = (k-1) * topk     (parent_list[0] = 0)
    """
    n = draft_token_num - 1
    base = torch.arange(n, dtype=torch.int64, device=device) * topk
    selected_index = base.unsqueeze(0).repeat(batch_size, 1).contiguous()

    parent_list = torch.zeros((batch_size, n), dtype=torch.int64, device=device)
    if n > 1:
        parent_list[:, 1:] = base[:-1].unsqueeze(0).repeat(batch_size, 1)

    verified_seq_len = torch.full((batch_size,), 8, dtype=torch.int32, device=device)
    return parent_list, selected_index, verified_seq_len


def check_implementations():
    """Check which implementations are available."""
    impls = {"cuda": False, "triton": False}

    try:
        from sgl_kernel import build_tree_kernel_efficient  # noqa: F401
        from sgl_kernel import verify_tree_greedy  # noqa: F401

        impls["cuda"] = True
    except Exception:
        pass

    try:
        from sglang.srt.speculative.eagle_utils import (  # noqa: F401
            sgl_build_tree_kernel_triton,
            verify_tree_greedy_triton,
        )

        impls["triton"] = True
    except Exception:
        pass

    return impls


def _build_tree_outputs(batch_size, draft_token_num, tree_mask_size, device):
    """Allocate the 5 output tensors for build_tree, in wrapper arg order:
    (tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling)
    """
    return (
        torch.full((tree_mask_size,), True, dtype=torch.bool, device=device),
        torch.zeros(
            (batch_size * draft_token_num,), dtype=torch.int64, device=device
        ),
        torch.full(
            (batch_size, draft_token_num), -1, dtype=torch.int64, device=device
        ),
        torch.full(
            (batch_size, draft_token_num), -1, dtype=torch.int64, device=device
        ),
        torch.full(
            (batch_size, draft_token_num), -1, dtype=torch.int64, device=device
        ),
    )


def run_build_tree_benchmark(
    runner: BenchmarkRunner,
    batch_sizes: List[int],
    draft_token_nums: List[int],
    implementations: dict,
):
    """Benchmark build_tree kernel: CUDA vs Triton."""
    from sglang.srt.speculative.eagle_utils import TreeMaskMode

    device = "cuda"
    topk, depth = 2, 3
    tree_mask_mode = TreeMaskMode.QLEN_ONLY

    print("\n" + "=" * 80)
    print("BENCHMARKING: build_tree kernel (CUDA vs Triton)")
    print("=" * 80)

    for batch_size in batch_sizes:
        for draft_token_num in draft_token_nums:
            print(
                f"\nConfig: batch_size={batch_size}, draft_token_num={draft_token_num}"
            )

            parent_list, selected_index, verified_seq_len = make_valid_tree_inputs(
                batch_size, draft_token_num, topk, device
            )
            tree_mask_size = draft_token_num * batch_size * draft_token_num

            # Benchmark CUDA
            if implementations["cuda"]:
                from sgl_kernel import build_tree_kernel_efficient

                outputs = _build_tree_outputs(
                    batch_size, draft_token_num, tree_mask_size, device
                )

                fn = lambda: build_tree_kernel_efficient(
                    parent_list,
                    selected_index,
                    verified_seq_len,
                    *outputs,
                    topk,
                    depth,
                    draft_token_num,
                    int(tree_mask_mode),
                )

                result = runner.benchmark_function(fn, "build_tree")
                result.implementation = "cuda"
                result.batch_size = batch_size
                result.draft_tokens = draft_token_num
                runner.add_result(result)
                print(f"  CUDA:    {result.median_us:.2f} us (median)")

            # Benchmark Triton
            if implementations["triton"]:
                from sglang.srt.speculative.eagle_utils import (
                    sgl_build_tree_kernel_triton,
                )

                outputs = _build_tree_outputs(
                    batch_size, draft_token_num, tree_mask_size, device
                )

                fn = lambda: sgl_build_tree_kernel_triton(
                    parent_list,
                    selected_index,
                    verified_seq_len,
                    *outputs,
                    topk,
                    depth,
                    draft_token_num,
                    tree_mask_mode,
                )

                result = runner.benchmark_function(fn, "build_tree")
                result.implementation = "triton"
                result.batch_size = batch_size
                result.draft_tokens = draft_token_num
                runner.add_result(result)
                print(f"  Triton:  {result.median_us:.2f} us (median)")


def _verify_tree_inputs(batch_size, num_draft_tokens, vocab_size, device):
    """Build the shared (non-output) inputs for verify_tree."""
    candidates = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    target_predict = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    target_predict[:, 0] = candidates[:, 1]  # Some acceptance

    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=device
    ).reshape(batch_size, num_draft_tokens)
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    for b in range(batch_size):
        for i in range(min(num_draft_tokens - 1, 3)):
            retrive_next_token[b, i] = i + 1

    return candidates, target_predict, retrive_index, retrive_next_token, retrive_next_sibling


def _verify_tree_outputs(batch_size, num_draft_tokens, device):
    """Allocate the 3 output tensors for verify_tree, in arg order:
    (predicts, accept_index, accept_token_num)
    """
    return (
        torch.zeros(
            (batch_size * num_draft_tokens,), dtype=torch.int32, device=device
        ),
        torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device
        ),
        torch.zeros((batch_size,), dtype=torch.int32, device=device),
    )


def run_verify_tree_benchmark(
    runner: BenchmarkRunner,
    batch_sizes: List[int],
    draft_token_nums: List[int],
    implementations: dict,
):
    """Benchmark verify_tree kernel: CUDA vs Triton."""
    device = "cuda"
    vocab_size = 32000

    print("\n" + "=" * 80)
    print("BENCHMARKING: verify_tree kernel (CUDA vs Triton)")
    print("=" * 80)

    for batch_size in batch_sizes:
        for num_draft_tokens in draft_token_nums:
            print(
                f"\nConfig: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}"
            )

            (
                candidates,
                target_predict,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
            ) = _verify_tree_inputs(batch_size, num_draft_tokens, vocab_size, device)

            # Benchmark CUDA
            if implementations["cuda"]:
                from sgl_kernel import verify_tree_greedy

                outputs = _verify_tree_outputs(batch_size, num_draft_tokens, device)

                fn = lambda: verify_tree_greedy(
                    *outputs,
                    candidates,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    target_predict,
                )

                result = runner.benchmark_function(fn, "verify_tree")
                result.implementation = "cuda"
                result.batch_size = batch_size
                result.draft_tokens = num_draft_tokens
                runner.add_result(result)
                print(f"  CUDA:    {result.median_us:.2f} us (median)")

            # Benchmark Triton
            if implementations["triton"]:
                from sglang.srt.speculative.eagle_utils import (
                    verify_tree_greedy_triton,
                )

                outputs = _verify_tree_outputs(batch_size, num_draft_tokens, device)

                fn = lambda: verify_tree_greedy_triton(
                    *outputs,
                    candidates,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    target_predict,
                )

                result = runner.benchmark_function(fn, "verify_tree")
                result.implementation = "triton"
                result.batch_size = batch_size
                result.draft_tokens = num_draft_tokens
                runner.add_result(result)
                print(f"  Triton:  {result.median_us:.2f} us (median)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE tree kernels (CUDA vs Triton)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with fewer configurations",
    )
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    args = parser.parse_args()

    print("=" * 80)
    print("EAGLE TREE KERNEL PERFORMANCE BENCHMARK")
    print("CUDA (sgl-kernel) vs Triton")
    print("=" * 80)

    # Check device
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. This benchmark requires an NVIDIA GPU.")
        return 1

    print(f"\n✓ Using device: cuda:{torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name()})")

    # Check implementations
    impls = check_implementations()
    print("\n📦 Available implementations:")
    print(f"  {'CUDA:':<12} {'✓' if impls['cuda'] else '✗ (sgl_kernel missing)'}")
    print(f"  {'Triton:':<12} {'✓' if impls['triton'] else '✗'}")

    if not impls["cuda"] and not impls["triton"]:
        print("\n❌ Neither CUDA nor Triton implementation is importable. Aborting.")
        return 1

    # Configure benchmark
    if args.quick:
        batch_sizes = [2, 8]
        draft_token_nums = [4, 8]
        warmup_iters = 5
        measure_iters = 50
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32]
        draft_token_nums = [4, 8, 12, 16]
        warmup_iters = 10
        measure_iters = 100

    print("\n⚙️ Benchmark configuration:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Draft tokens: {draft_token_nums}")
    print(f"  Warmup iterations: {warmup_iters}")
    print(f"  Measurement iterations: {measure_iters}")

    # Run benchmarks
    runner = BenchmarkRunner(warmup_iters=warmup_iters, measure_iters=measure_iters)

    run_build_tree_benchmark(runner, batch_sizes, draft_token_nums, impls)
    run_verify_tree_benchmark(runner, batch_sizes, draft_token_nums, impls)

    # Print summary
    runner.print_summary()

    # Save CSV
    if args.csv:
        runner.save_csv(args.csv)

    print("\n✅ Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
