#!/usr/bin/env python3
"""
Device-time micro-benchmark for the EAGLE tree SYCL kernels.

The existing benchmark_eagle_tree_xpu.py measures wall-clock per call, which is
dominated by host dispatch (Python + ctypes + queue submit + synchronize). That
~450us floor hides the kernel, so scalar-vs-optimized looks like noise.

This harness isolates kernel cost two ways:
  1. Batched submit: enqueue N launches back-to-back, synchronize ONCE, divide.
     Amortizes the per-call host/sync overhead across N launches.
  2. torch.xpu.Event elapsed_time around the batched submit (device timeline).

Run inside the jh_py_env_up conda env with oneAPI sourced (for first-time JIT
compile of the .hpp). Usage:
    python bench_eagle_devtime.py
"""
import time

import torch

from sglang.jit_kernel.eagle_tree_xpu import (
    sgl_build_tree_kernel_efficient_xpu,
    verify_tree_greedy_xpu,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode

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


def bench_build(batch_size, draft):
    pl, si, vsl = make_build_inputs(batch_size, draft)
    out_s = make_build_outputs(batch_size, draft)
    out_o = make_build_outputs(batch_size, draft)
    mode = TreeMaskMode.QLEN_ONLY

    scalar = time_batched(
        lambda: sgl_build_tree_kernel_efficient_xpu(
            pl, si, vsl, *out_s, TOPK, DEPTH, draft, mode, optimized=False
        )
    )
    opt = time_batched(
        lambda: sgl_build_tree_kernel_efficient_xpu(
            pl, si, vsl, *out_o, TOPK, DEPTH, draft, mode, optimized=True
        )
    )
    return scalar, opt


def bench_verify(batch_size, draft):
    cand, tp, ri, rnt, rns = make_verify_inputs(batch_size, draft)
    out_s = make_verify_outputs(batch_size, draft)
    out_o = make_verify_outputs(batch_size, draft)

    scalar = time_batched(
        lambda: verify_tree_greedy_xpu(
            *out_s, cand, ri, rnt, rns, tp, optimized=False
        )
    )
    opt = time_batched(
        lambda: verify_tree_greedy_xpu(
            *out_o, cand, ri, rnt, rns, tp, optimized=True
        )
    )
    return scalar, opt


def main():
    assert torch.xpu.is_available(), "XPU required"
    print(f"Device: {torch.xpu.get_device_name()}")
    print(f"N_INNER={N_INNER} (launches/measure), N_OUTER={N_OUTER}, warmup={N_WARMUP}")
    print(f"Per-launch device time (median, amortized over {N_INNER} submits)\n")

    configs = [(1, 4), (8, 8), (32, 16), (256, 16), (1024, 16)]

    for kernel, fn in [("build_tree", bench_build), ("verify_tree", bench_verify)]:
        print("=" * 72)
        print(f"{kernel}")
        print(f"{'batch':<8}{'draft':<8}{'scalar us':<14}{'opt us':<14}{'speedup':<10}")
        print("-" * 72)
        for bs, dr in configs:
            try:
                s, o = fn(bs, dr)
                spd = s / o if o else 0.0
                print(f"{bs:<8}{dr:<8}{s:<14.3f}{o:<14.3f}{spd:<10.2f}x")
            except Exception as e:
                print(f"{bs:<8}{dr:<8}ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
