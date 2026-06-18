#!/usr/bin/env python3
"""
Standalone Performance Benchmark: SYCL JIT vs SYCL JIT (optimized) vs Triton

Comprehensive performance comparison of EAGLE tree kernel implementations:
- SYCL JIT (native XPU with icpx compilation, scalar kernel)
- SYCL JIT opt (re-parallelized SYCL kernel, optimized=True)
- Triton (Triton-based kernels)

Usage:
    python benchmark_eagle_tree_xpu.py [--quick] [--csv output.csv] [--plot]

Options:
    --quick     Run quick benchmark with fewer iterations
    --csv FILE  Save results to CSV file
    --plot      Generate performance plots (requires matplotlib)
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

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

    def benchmark_function(
        self, fn: Callable, name: str = "kernel"
    ) -> BenchmarkResult:
        """Benchmark a function with warmup and multiple iterations."""
        # Warmup
        for _ in range(self.warmup_iters):
            fn()
        torch.xpu.synchronize() if torch.xpu.is_available() else torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(self.measure_iters):
            start = time.perf_counter()
            fn()
            torch.xpu.synchronize() if torch.xpu.is_available() else torch.cuda.synchronize()
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
        """Print benchmark summary table."""
        print("\n" + "=" * 120)
        print("PERFORMANCE SUMMARY")
        print("=" * 120)
        print(
            f"{'Kernel':<20} {'Impl':<12} {'Batch':<8} {'Tokens':<8} "
            f"{'Mean (us)':<12} {'Median (us)':<12} {'Speedup vs SYCL':<10}"
        )
        print("-" * 120)

        # Group by kernel and config
        kernels = {}
        for r in self.results:
            key = (r.name, r.batch_size, r.draft_tokens)
            if key not in kernels:
                kernels[key] = {}
            kernels[key][r.implementation] = r

        for (name, batch, tokens), impls in sorted(kernels.items()):
            # Baseline for speedup is the scalar SYCL JIT kernel.
            baseline_r = impls.get("sycl_jit")
            baseline_us = baseline_r.median_us if baseline_r else None

            for impl_name in ["sycl_jit", "sycl_jit_opt", "triton"]:
                display_name = {
                    "sycl_jit": "SYCL JIT",
                    "sycl_jit_opt": "SYCL JIT opt",
                    "triton": "Triton",
                }[impl_name]

                if impl_name in impls:
                    r = impls[impl_name]
                    speedup_str = ""
                    if baseline_us and impl_name != "sycl_jit":
                        speedup = baseline_us / r.median_us
                        speedup_str = f"{speedup:.2f}x"

                    print(
                        f"{name:<20} {display_name:<12} {batch:<8} {tokens:<8} "
                        f"{r.mean_us:<12.2f} {r.median_us:<12.2f} {speedup_str:<10}"
                    )

        print("=" * 120)

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
    not hit the "invalid eagle tree" warning path (which would both spam logs
    and make timings unrealistic).

    Layout (topk-spaced chain):
      selected_index[k] = k * topk        -> parent_tb_idx = k
      parent_list[k]    = (k-1) * topk     (parent_list[0] = 0)
    so parent_list[parent_tb_idx] always equals an existing selected_index value.
    """
    n = draft_token_num - 1
    base = torch.arange(n, dtype=torch.int64, device=device) * topk
    selected_index = base.unsqueeze(0).repeat(batch_size, 1).contiguous()

    parent_list = torch.zeros((batch_size, n), dtype=torch.int64, device=device)
    if n > 1:
        parent_list[:, 1:] = base[:-1].unsqueeze(0).repeat(batch_size, 1)

    # Fixed seq lens keep the benchmark deterministic across runs/impls.
    verified_seq_len = torch.full(
        (batch_size,), 8, dtype=torch.int32, device=device
    )
    return parent_list, selected_index, verified_seq_len


def check_implementations():
    """Check which implementations are available."""
    impls = {
        "sycl_jit": False,
        "sycl_jit_opt": False,
        "triton": False,
    }

    try:
        from sglang.jit_kernel.eagle_tree_xpu import (
            sgl_build_tree_kernel_efficient_xpu,
        )

        # The optimized kernel shares the same module/entry point; availability
        # tracks the scalar SYCL JIT path.
        impls["sycl_jit"] = True
        impls["sycl_jit_opt"] = True
    except Exception:
        pass

    try:
        from sglang.srt.speculative.eagle_utils import sgl_build_tree_kernel_triton

        impls["triton"] = True
    except Exception:
        pass

    return impls


def run_build_tree_benchmark(
    runner: BenchmarkRunner,
    batch_sizes: List[int],
    draft_token_nums: List[int],
    implementations: dict,
):
    """Benchmark build_tree kernel."""
    from sglang.srt.speculative.eagle_utils import TreeMaskMode

    device = "xpu" if torch.xpu.is_available() else "cuda"
    topk, depth = 2, 3
    tree_mask_mode = TreeMaskMode.QLEN_ONLY

    print("\n" + "=" * 80)
    print("BENCHMARKING: build_tree kernel")
    print("=" * 80)

    for batch_size in batch_sizes:
        for draft_token_num in draft_token_nums:
            print(
                f"\nConfig: batch_size={batch_size}, draft_token_num={draft_token_num}"
            )

            # Generate VALID tree inputs (no orphan tokens -> no warning spam),
            # matching the parity-test construction.
            parent_list, selected_index, verified_seq_len = make_valid_tree_inputs(
                batch_size, draft_token_num, topk, device
            )

            seq_lens_sum = int(verified_seq_len.sum())
            tree_mask_size = draft_token_num * batch_size * draft_token_num

            # Benchmark SYCL JIT
            if implementations["sycl_jit"]:
                from sglang.jit_kernel.eagle_tree_xpu import (
                    sgl_build_tree_kernel_efficient_xpu,
                )

                outputs = (
                    torch.full(
                        (tree_mask_size,), True, dtype=torch.bool, device=device
                    ),
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

                fn = lambda: sgl_build_tree_kernel_efficient_xpu(
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
                result.implementation = "sycl_jit"
                result.batch_size = batch_size
                result.draft_tokens = draft_token_num
                runner.add_result(result)
                print(f"  SYCL JIT:  {result.median_us:.2f} us (median)")

            # Benchmark SYCL JIT (optimized kernel)
            if implementations["sycl_jit_opt"]:
                from sglang.jit_kernel.eagle_tree_xpu import (
                    sgl_build_tree_kernel_efficient_xpu,
                )

                outputs = (
                    torch.full(
                        (tree_mask_size,), True, dtype=torch.bool, device=device
                    ),
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

                fn = lambda: sgl_build_tree_kernel_efficient_xpu(
                    parent_list,
                    selected_index,
                    verified_seq_len,
                    *outputs,
                    topk,
                    depth,
                    draft_token_num,
                    tree_mask_mode,
                    optimized=True,
                )

                result = runner.benchmark_function(fn, "build_tree")
                result.implementation = "sycl_jit_opt"
                result.batch_size = batch_size
                result.draft_tokens = draft_token_num
                runner.add_result(result)
                print(f"  SYCL JIT (opt): {result.median_us:.2f} us (median)")

            # Benchmark Triton
            if implementations["triton"]:
                from sglang.srt.speculative.eagle_utils import (
                    sgl_build_tree_kernel_triton,
                )

                outputs = (
                    torch.full(
                        (tree_mask_size,), True, dtype=torch.bool, device=device
                    ),
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
                print(f"  Triton:    {result.median_us:.2f} us (median)")


def run_verify_tree_benchmark(
    runner: BenchmarkRunner,
    batch_sizes: List[int],
    draft_token_nums: List[int],
    implementations: dict,
):
    """Benchmark verify_tree kernel."""
    device = "xpu" if torch.xpu.is_available() else "cuda"
    vocab_size = 32000

    print("\n" + "=" * 80)
    print("BENCHMARKING: verify_tree kernel")
    print("=" * 80)

    for batch_size in batch_sizes:
        for num_draft_tokens in draft_token_nums:
            print(
                f"\nConfig: batch_size={batch_size}, num_draft_tokens={num_draft_tokens}"
            )

            # Generate inputs
            candidates = torch.randint(
                0,
                vocab_size,
                (batch_size, num_draft_tokens),
                dtype=torch.int32,
                device=device,
            )
            target_predict = torch.randint(
                0,
                vocab_size,
                (batch_size, num_draft_tokens),
                dtype=torch.int32,
                device=device,
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

            # Benchmark SYCL JIT
            if implementations["sycl_jit"]:
                from sglang.jit_kernel.eagle_tree_xpu import verify_tree_greedy_xpu

                outputs = (
                    torch.zeros(
                        (batch_size * num_draft_tokens,),
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.full(
                        (batch_size, num_draft_tokens),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.zeros((batch_size,), dtype=torch.int32, device=device),
                )

                fn = lambda: verify_tree_greedy_xpu(
                    *outputs,
                    candidates,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    target_predict,
                )

                result = runner.benchmark_function(fn, "verify_tree")
                result.implementation = "sycl_jit"
                result.batch_size = batch_size
                result.draft_tokens = num_draft_tokens
                runner.add_result(result)
                print(f"  SYCL JIT:  {result.median_us:.2f} us (median)")

            # Benchmark SYCL JIT (optimized kernel)
            if implementations["sycl_jit_opt"]:
                from sglang.jit_kernel.eagle_tree_xpu import verify_tree_greedy_xpu

                outputs = (
                    torch.zeros(
                        (batch_size * num_draft_tokens,),
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.full(
                        (batch_size, num_draft_tokens),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.zeros((batch_size,), dtype=torch.int32, device=device),
                )

                fn = lambda: verify_tree_greedy_xpu(
                    *outputs,
                    candidates,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    target_predict,
                    optimized=True,
                )

                result = runner.benchmark_function(fn, "verify_tree")
                result.implementation = "sycl_jit_opt"
                result.batch_size = batch_size
                result.draft_tokens = num_draft_tokens
                runner.add_result(result)
                print(f"  SYCL JIT (opt): {result.median_us:.2f} us (median)")

            # Benchmark Triton
            if implementations["triton"]:
                from sglang.srt.speculative.eagle_utils import (
                    verify_tree_greedy_triton,
                )

                outputs = (
                    torch.zeros(
                        (batch_size * num_draft_tokens),
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.full(
                        (batch_size, num_draft_tokens),
                        -1,
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.zeros((batch_size,), dtype=torch.int32, device=device),
                )

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
                print(f"  Triton:    {result.median_us:.2f} us (median)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE tree kernels (SYCL vs SYCL opt vs Triton)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with fewer configurations",
    )
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--plot", action="store_true", help="Generate performance plots"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EAGLE TREE KERNEL PERFORMANCE BENCHMARK")
    print("SYCL JIT vs SYCL JIT (optimized) vs Triton")
    print("=" * 80)

    # Check device
    if not torch.xpu.is_available():
        print("\n❌ XPU not available. This benchmark requires Intel XPU hardware.")
        return 1

    print(f"\n✓ Using device: xpu:{torch.xpu.current_device()}")

    # Check implementations
    impls = check_implementations()
    print("\n📦 Available implementations:")
    print(f"  {'SYCL JIT:':<15} {'✓' if impls['sycl_jit'] else '✗'}")
    print(f"  {'SYCL JIT opt:':<15} {'✓' if impls['sycl_jit_opt'] else '✗'}")
    print(f"  {'Triton:':<15} {'✓' if impls['triton'] else '✗ (optional)'}")

    if not impls["sycl_jit"]:
        print(
            "\n⚠ SYCL JIT not available. Run: source /opt/intel/oneapi/setvars.sh"
        )

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

    print(f"\n⚙️ Benchmark configuration:")
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

    # Generate plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            # TODO: Implement plotting
            print("\n⚠ Plotting not yet implemented")
        except ImportError:
            print("\n⚠ matplotlib not available, skipping plots")

    print("\n✅ Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
