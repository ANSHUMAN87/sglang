#!/usr/bin/env python3
"""
Device-time micro-benchmark for EAGLE tree kernels: SYCL vs Triton (baseline/v1/v2).

The existing benchmark_eagle_tree_xpu.py measures wall-clock per call, which is
dominated by host dispatch (Python + ctypes + queue submit + synchronize). That
~450us floor hides the kernel, so scalar-vs-optimized looks like noise.

This harness isolates kernel cost two ways:
  1. Batched submit: enqueue N launches back-to-back, synchronize ONCE, divide.
     Amortizes the per-call host/sync overhead across N launches.
  2. torch.xpu.Event elapsed_time around the batched submit (device timeline).

Now includes:
  - SYCL JIT (scalar and optimized)
  - Triton baseline
  - Triton v1 (moderate optimization)
  - Triton v2 (aggressive optimization)

Run inside the jh_py_env_up conda env with oneAPI sourced (for first-time JIT
compile of the .hpp). Usage:
    python bench_eagle_devtime.py
    python bench_eagle_devtime.py --triton-only  # Skip SYCL benchmarks
"""
import argparse
import time

import torch

from sglang.jit_kernel.eagle_tree_xpu import (
    sgl_build_tree_kernel_efficient_xpu,
    verify_tree_greedy_xpu,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode

# Check for Triton baseline kernels
TRITON_BASELINE_AVAILABLE = False
try:
    from sglang.srt.speculative.triton_ops.spec_tree import (
        sgl_build_tree_kernel_efficient_triton as sgl_build_tree_kernel_triton_baseline,
        verify_tree_greedy_kernel_triton as verify_tree_greedy_kernel_triton_baseline,
    )
    TRITON_BASELINE_AVAILABLE = True
except ImportError:
    pass

# Check for Triton optimized kernels
TRITON_OPTIMIZED_AVAILABLE = False
try:
    from sglang.srt.speculative.triton_ops.spec_tree_optimized import (
        sgl_build_tree_kernel_optimized_v1,
        sgl_build_tree_kernel_optimized_v2,
        verify_tree_greedy_kernel_optimized_v1,
        verify_tree_greedy_kernel_optimized_v2,
    )
    TRITON_OPTIMIZED_AVAILABLE = True
except ImportError:
    pass

DEVICE = "xpu"
TOPK, DEPTH = 2, 3
N_INNER = 200  # launches per measured batch (amortizes host/sync overhead)
N_OUTER = 30  # measured repetitions
N_WARMUP = 20


def make_build_inputs(batch_size, draft_token_num):
    n = draft_token_num - 1
    base = torch.arange(n, dtype=torch.int64, device=DEVICE) * TOPK
    selected_index = base.unsqueeze(0).repeat(batch_size, 1).contiguous()
    parent_list = torch.zeros((batch_size, n), dtype=torch.int64, device=DEVICE)
    if n > 1:
        parent_list[:, 1:] = base[:-1].unsqueeze(0).repeat(batch_size, 1)
    verified_seq_len = torch.full(
        (batch_size,), 8, dtype=torch.int32, device=DEVICE
    )
    return parent_list, selected_index, verified_seq_len


def make_build_outputs(batch_size, draft_token_num):
    tree_mask_size = draft_token_num * batch_size * draft_token_num
    return (
        torch.full((tree_mask_size,), True, dtype=torch.bool, device=DEVICE),
        torch.zeros((batch_size * draft_token_num,), dtype=torch.int64, device=DEVICE),
        torch.full((batch_size, draft_token_num), -1, dtype=torch.int64, device=DEVICE),
        torch.full((batch_size, draft_token_num), -1, dtype=torch.int64, device=DEVICE),
        torch.full((batch_size, draft_token_num), -1, dtype=torch.int64, device=DEVICE),
    )


def make_verify_inputs(batch_size, num_draft_tokens, vocab=32000):
    candidates = torch.randint(
        0, vocab, (batch_size, num_draft_tokens), dtype=torch.int32, device=DEVICE
    )
    target_predict = torch.randint(
        0, vocab, (batch_size, num_draft_tokens), dtype=torch.int32, device=DEVICE
    )
    target_predict[:, 0] = candidates[:, 1]
    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=DEVICE
    ).reshape(batch_size, num_draft_tokens)
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=DEVICE
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=DEVICE
    )
    for b in range(batch_size):
        for i in range(min(num_draft_tokens - 1, 3)):
            retrive_next_token[b, i] = i + 1
    return candidates, target_predict, retrive_index, retrive_next_token, retrive_next_sibling


def make_verify_outputs(batch_size, num_draft_tokens):
    return (
        torch.zeros((batch_size * num_draft_tokens,), dtype=torch.int32, device=DEVICE),
        torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int32, device=DEVICE),
        torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE),
    )


def time_batched(launch_one):
    """Return per-launch us, measured by amortizing N_INNER launches over one sync.

    Uses torch.xpu.Event (device timeline) so we capture device-side elapsed
    rather than per-call host round trips.
    """
    for _ in range(N_WARMUP):
        launch_one()
    torch.xpu.synchronize()

    per_launch = []
    for _ in range(N_OUTER):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(N_INNER):
            launch_one()
        end.record()
        torch.xpu.synchronize()
        ms = start.elapsed_time(end)  # milliseconds for N_INNER launches
        per_launch.append(ms * 1e3 / N_INNER)  # -> us per launch
    per_launch.sort()
    return per_launch[len(per_launch) // 2]  # median


def bench_build(batch_size, draft, include_sycl=True, include_triton=True, include_triton_baseline=True):
    pl, si, vsl = make_build_inputs(batch_size, draft)
    mode = TreeMaskMode.QLEN_ONLY

    results = {}

    # SYCL benchmarks
    if include_sycl:
        out_s = make_build_outputs(batch_size, draft)
        out_o = make_build_outputs(batch_size, draft)

        results['sycl_scalar'] = time_batched(
            lambda: sgl_build_tree_kernel_efficient_xpu(
                pl, si, vsl, *out_s, TOPK, DEPTH, draft, mode, optimized=False
            )
        )
        results['sycl_opt'] = time_batched(
            lambda: sgl_build_tree_kernel_efficient_xpu(
                pl, si, vsl, *out_o, TOPK, DEPTH, draft, mode, optimized=True
            )
        )

    # Triton baseline benchmark
    if include_triton and include_triton_baseline and TRITON_BASELINE_AVAILABLE:
        # Compute prefix sums for Triton kernels
        seq_len_prefix_sum = torch.zeros_like(vsl)
        if batch_size > 1:
            seq_len_prefix_sum[1:] = torch.cumsum(vsl[:-1], dim=0)

        grid = (batch_size,)
        parent_list_stride = pl.shape[1]
        selected_index_stride = si.shape[1]

        # Triton baseline
        out_base = make_build_outputs(batch_size, draft)
        results['triton_base'] = time_batched(
            lambda: sgl_build_tree_kernel_triton_baseline[grid](
                pl, si, vsl, seq_len_prefix_sum, *out_base,
                topk=TOPK, depth=DEPTH, draft_token_num=draft,
                tree_mask_mode=mode.value, batch_size=batch_size,
                parent_list_stride=parent_list_stride,
                selected_index_stride=selected_index_stride,
            )
        )

    # Triton optimized benchmarks
    if include_triton and TRITON_OPTIMIZED_AVAILABLE:
        # Compute prefix sums for Triton kernels
        seq_len_prefix_sum = torch.zeros_like(vsl)
        if batch_size > 1:
            seq_len_prefix_sum[1:] = torch.cumsum(vsl[:-1], dim=0)

        grid = (batch_size,)
        parent_list_stride = pl.shape[1]
        selected_index_stride = si.shape[1]

        # Triton v1
        out_v1 = make_build_outputs(batch_size, draft)
        results['triton_v1'] = time_batched(
            lambda: sgl_build_tree_kernel_optimized_v1[grid](
                pl, si, vsl, seq_len_prefix_sum, *out_v1,
                topk=TOPK, depth=DEPTH, draft_token_num=draft,
                tree_mask_mode=mode.value, batch_size=batch_size,
                parent_list_stride=parent_list_stride,
                selected_index_stride=selected_index_stride,
            )
        )

        # Triton v2
        out_v2 = make_build_outputs(batch_size, draft)
        results['triton_v2'] = time_batched(
            lambda: sgl_build_tree_kernel_optimized_v2[grid](
                pl, si, vsl, seq_len_prefix_sum, *out_v2,
                topk=TOPK, depth=DEPTH, draft_token_num=draft,
                tree_mask_mode=mode.value, batch_size=batch_size,
                parent_list_stride=parent_list_stride,
                selected_index_stride=selected_index_stride,
            )
        )

    return results


def bench_verify(batch_size, draft, include_sycl=True, include_triton=True, include_triton_baseline=True):
    cand, tp, ri, rnt, rns = make_verify_inputs(batch_size, draft)

    results = {}

    # SYCL benchmarks
    if include_sycl:
        out_s = make_verify_outputs(batch_size, draft)
        out_o = make_verify_outputs(batch_size, draft)

        results['sycl_scalar'] = time_batched(
            lambda: verify_tree_greedy_xpu(
                *out_s, cand, ri, rnt, rns, tp, optimized=False
            )
        )
        results['sycl_opt'] = time_batched(
            lambda: verify_tree_greedy_xpu(
                *out_o, cand, ri, rnt, rns, tp, optimized=True
            )
        )

    # Triton baseline benchmark
    if include_triton and include_triton_baseline and TRITON_BASELINE_AVAILABLE:
        grid = (batch_size,)

        # Triton baseline
        out_base = make_verify_outputs(batch_size, draft)
        results['triton_base'] = time_batched(
            lambda: verify_tree_greedy_kernel_triton_baseline[grid](
                *out_base, cand, ri, rnt, rns, tp,
                batch_size=batch_size,
                num_speculative_tokens=draft,
                num_draft_tokens=draft,
            )
        )

    # Triton optimized benchmarks
    if include_triton and TRITON_OPTIMIZED_AVAILABLE:
        grid = (batch_size,)

        # Triton v1
        out_v1 = make_verify_outputs(batch_size, draft)
        results['triton_v1'] = time_batched(
            lambda: verify_tree_greedy_kernel_optimized_v1[grid](
                *out_v1, cand, ri, rnt, rns, tp,
                batch_size=batch_size,
                num_speculative_tokens=draft,
                num_draft_tokens=draft,
            )
        )

        # Triton v2
        out_v2 = make_verify_outputs(batch_size, draft)
        results['triton_v2'] = time_batched(
            lambda: verify_tree_greedy_kernel_optimized_v2[grid](
                *out_v2, cand, ri, rnt, rns, tp,
                batch_size=batch_size,
                num_speculative_tokens=draft,
                num_draft_tokens=draft,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark EAGLE tree kernels")
    parser.add_argument("--triton-only", action="store_true", help="Only benchmark Triton kernels")
    parser.add_argument("--sycl-only", action="store_true", help="Only benchmark SYCL kernels")
    args = parser.parse_args()

    assert torch.xpu.is_available(), "XPU required"
    print(f"Device: {torch.xpu.get_device_name()}")
    print(f"N_INNER={N_INNER} (launches/measure), N_OUTER={N_OUTER}, warmup={N_WARMUP}")
    print(f"Per-launch device time (median, amortized over {N_INNER} submits)\n")

    # Check what's available
    include_sycl = not args.triton_only
    include_triton = not args.sycl_only
    include_triton_baseline = TRITON_BASELINE_AVAILABLE
    include_triton_optimized = TRITON_OPTIMIZED_AVAILABLE

    if include_triton and not include_triton_baseline and not include_triton_optimized:
        print("⚠️  Triton kernels not available (import failed)")
        print("    Will only benchmark SYCL kernels\n")
        include_triton = False

    print("📦 Benchmarking:")
    if include_sycl:
        print("   ✓ SYCL JIT (scalar + optimized)")
    if include_triton and include_triton_baseline:
        print("   ✓ Triton baseline")
    if include_triton and include_triton_optimized:
        print("   ✓ Triton v1 (moderate optimization)")
        print("   ✓ Triton v2 (aggressive optimization)")
    print()

    configs = [(1, 4), (8, 8), (32, 16), (256, 16), (1024, 16)]

    for kernel, fn in [("build_tree", bench_build), ("verify_tree", bench_verify)]:
        print("=" * 120)
        print(f"{kernel}")

        # Determine columns based on what we're benchmarking
        headers = ["batch", "draft"]
        if include_sycl:
            headers.extend(["SYCL scalar", "SYCL opt", "SYCL spd"])
        if include_triton and include_triton_baseline:
            headers.extend(["Triton base", "base spd"])
        if include_triton and include_triton_optimized:
            headers.extend(["Triton v1", "v1 spd", "Triton v2", "v2 spd"])

        # Print header
        col_widths = [8, 8] + [12] * (len(headers) - 2)
        header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        print(header_str)
        print("-" * 120)

        for bs, dr in configs:
            try:
                results = fn(bs, dr, include_sycl=include_sycl, include_triton=include_triton,
                           include_triton_baseline=include_triton_baseline)

                row = [f"{bs:<8}", f"{dr:<8}"]

                # Determine baseline for speedup calculations
                baseline = results.get('sycl_scalar', 1.0)

                def spd_col(spd):
                    return f"{spd:.2f}x".ljust(12)

                if include_sycl:
                    s = results['sycl_scalar']
                    o = results['sycl_opt']
                    spd = s / o if o else 0.0
                    row.extend([f"{s:<12.3f}", f"{o:<12.3f}", spd_col(spd)])

                if include_triton and include_triton_baseline and 'triton_base' in results:
                    tb = results['triton_base']
                    tb_spd = baseline / tb if tb else 0.0
                    row.extend([f"{tb:<12.3f}", spd_col(tb_spd)])

                if include_triton and include_triton_optimized and 'triton_v1' in results:
                    v1 = results['triton_v1']
                    v1_spd = baseline / v1 if v1 else 0.0
                    row.extend([f"{v1:<12.3f}", spd_col(v1_spd)])

                    v2 = results['triton_v2']
                    v2_spd = baseline / v2 if v2 else 0.0
                    row.extend([f"{v2:<12.3f}", spd_col(v2_spd)])

                print("".join(row))

            except Exception as e:
                print(f"{bs:<8}{dr:<8}ERROR: {e}")
                import traceback
                traceback.print_exc()
        print()

    # Summary
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print("Speedup columns show improvement vs SYCL scalar baseline")
    print()
    print("Expected speedups (vs SYCL scalar):")
    if include_sycl:
        print("  - SYCL opt:      1.2-1.5x")
    if include_triton and include_triton_baseline:
        print("  - Triton base:   1.0-1.2x (baseline Triton implementation)")
    if include_triton and include_triton_optimized:
        print("  - Triton v1:     1.5-2.0x (moderate optimization)")
        print("  - Triton v2:     2.0-3.0x (aggressive, best for draft_tokens ≤ 85)")
    print()
    print("Progression: SYCL scalar → SYCL opt → Triton base → Triton v1 → Triton v2")
    print()


if __name__ == "__main__":
    main()
