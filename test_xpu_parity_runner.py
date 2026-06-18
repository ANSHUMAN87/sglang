#!/usr/bin/env python3
"""
XPU EAGLE Tree Parity Test Runner

Runs parity tests comparing SYCL JIT, Triton, and PyTorch implementations.
Provides clear reporting on which implementations match and performance metrics.

Usage:
    python test_xpu_parity_runner.py [--quick] [--verbose]

Options:
    --quick    Run only small test cases (faster)
    --verbose  Show detailed per-test output
"""

import argparse
import sys
import time

import torch


def check_xpu_available():
    """Check if XPU is available."""
    if not torch.xpu.is_available():
        print("❌ XPU not available")
        print("   Please ensure:")
        print("   1. Intel XPU hardware is present")
        print("   2. Intel Extension for PyTorch is installed")
        print("   3. oneAPI drivers are loaded")
        return False
    print(f"✓ XPU available: {torch.xpu.device_count()} device(s)")
    return True


def check_sycl_kernel_available():
    """Check if SYCL kernels can be compiled."""
    try:
        from sglang.jit_kernel.eagle_tree_xpu import (
            sgl_build_tree_kernel_efficient_xpu,
        )

        print("✓ SYCL JIT kernel module loaded")
        return True
    except ImportError as e:
        print(f"❌ SYCL JIT kernel not available: {e}")
        return False


def check_triton_available():
    """Check if Triton kernels are available."""
    try:
        from sglang.srt.speculative.triton_ops.spec_tree import (
            sgl_build_tree_kernel_efficient_triton,
        )

        print("✓ Triton kernel module loaded")
        return True
    except ImportError:
        print("⚠ Triton kernels not available (optional)")
        return False


def check_icpx_available():
    """Check if icpx compiler is available."""
    import subprocess

    try:
        result = subprocess.run(
            ["which", "icpx"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            print(f"✓ icpx compiler found: {result.stdout.strip()}")
            # Get version
            version_result = subprocess.run(
                ["icpx", "--version"], capture_output=True, text=True, check=False
            )
            if version_result.returncode == 0:
                version_line = version_result.stdout.split("\n")[0]
                print(f"  {version_line}")
            return True
        else:
            print("❌ icpx compiler not found")
            print("   Run: source /opt/intel/oneapi/setvars.sh")
            return False
    except Exception as e:
        print(f"❌ Error checking icpx: {e}")
        return False


def run_quick_parity_test():
    """Run a quick parity test with small inputs."""
    print("\n" + "=" * 60)
    print("QUICK PARITY TEST")
    print("=" * 60)

    from sglang.jit_kernel.eagle_tree_xpu import (
        sgl_build_tree_kernel_efficient_xpu,
        verify_tree_greedy_xpu,
    )
    from sglang.srt.speculative.eagle_utils import (
        TreeMaskMode,
        sgl_build_tree_kernel_efficient_pytorch,
        verify_tree_greedy_pytorch,
    )

    batch_size, draft_token_num, topk = 2, 4, 2
    device = "xpu"

    print(f"\nTest config: batch_size={batch_size}, draft_token_num={draft_token_num}, topk={topk}")

    # Test build_tree
    print("\n1. Testing build_tree kernel...")
    parent_list = torch.randint(
        0, draft_token_num, (batch_size, draft_token_num - 1), dtype=torch.int64, device=device
    )
    selected_index = torch.randint(
        0, topk * draft_token_num, (batch_size, draft_token_num - 1), dtype=torch.int64, device=device
    )
    verified_seq_len = torch.tensor([5, 6], dtype=torch.int32, device=device)

    tree_mask_mode = TreeMaskMode.QLEN_ONLY
    seq_lens_sum = int(verified_seq_len.sum())
    tree_mask_size = draft_token_num * batch_size * draft_token_num

    outputs_sycl = (
        torch.full((tree_mask_size,), True, dtype=torch.bool, device=device),
        torch.zeros((batch_size * draft_token_num,), dtype=torch.int64, device=device),
        torch.full((batch_size, draft_token_num), -1, dtype=torch.int64, device=device),
        torch.full((batch_size, draft_token_num), -1, dtype=torch.int64, device=device),
        torch.full((batch_size, draft_token_num), -1, dtype=torch.int64, device=device),
    )

    outputs_pytorch = tuple(t.clone() for t in outputs_sycl)

    # Time SYCL
    start = time.time()
    sgl_build_tree_kernel_efficient_xpu(
        parent_list, selected_index, verified_seq_len,
        *outputs_sycl, topk, 3, draft_token_num, tree_mask_mode
    )
    torch.xpu.synchronize()
    sycl_time = (time.time() - start) * 1000

    # Time PyTorch
    start = time.time()
    sgl_build_tree_kernel_efficient_pytorch(
        parent_list, selected_index, verified_seq_len,
        *outputs_pytorch, topk, 3, draft_token_num, tree_mask_mode
    )
    torch.xpu.synchronize()
    pytorch_time = (time.time() - start) * 1000

    # Check parity
    all_match = all(
        torch.equal(sycl, pytorch)
        for sycl, pytorch in zip(outputs_sycl, outputs_pytorch)
    )

    if all_match:
        print(f"   ✓ build_tree: SYCL matches PyTorch")
        print(f"   ⏱ SYCL: {sycl_time:.2f}ms, PyTorch: {pytorch_time:.2f}ms")
        print(f"   🚀 Speedup: {pytorch_time/sycl_time:.2f}x")
    else:
        print(f"   ❌ build_tree: MISMATCH between SYCL and PyTorch")
        return False

    # Test verify_tree
    print("\n2. Testing verify_tree kernel...")
    num_draft_tokens = 4
    candidates = torch.randint(0, 1000, (batch_size, num_draft_tokens), dtype=torch.int32, device=device)
    target_predict = torch.randint(0, 1000, (batch_size, num_draft_tokens), dtype=torch.int32, device=device)
    # Make first token match to ensure some acceptance
    target_predict[:, 0] = candidates[:, 1]

    retrive_index = torch.arange(batch_size * num_draft_tokens, dtype=torch.int64, device=device).reshape(batch_size, num_draft_tokens)
    retrive_next_token = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device)

    for b in range(batch_size):
        for i in range(num_draft_tokens - 1):
            retrive_next_token[b, i] = i + 1

    verify_outputs_sycl = (
        torch.zeros((batch_size * num_draft_tokens,), dtype=torch.int32, device=device),
        torch.full((batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device),
        torch.zeros((batch_size,), dtype=torch.int32, device=device),
    )
    verify_outputs_pytorch = tuple(t.clone() for t in verify_outputs_sycl)

    # Time SYCL
    start = time.time()
    verify_tree_greedy_xpu(
        *verify_outputs_sycl, candidates, retrive_index,
        retrive_next_token, retrive_next_sibling, target_predict
    )
    torch.xpu.synchronize()
    sycl_time = (time.time() - start) * 1000

    # Time PyTorch
    start = time.time()
    verify_tree_greedy_pytorch(
        *verify_outputs_pytorch, candidates, retrive_index,
        retrive_next_token, retrive_next_sibling, target_predict
    )
    torch.xpu.synchronize()
    pytorch_time = (time.time() - start) * 1000

    all_match = all(
        torch.equal(sycl, pytorch)
        for sycl, pytorch in zip(verify_outputs_sycl, verify_outputs_pytorch)
    )

    if all_match:
        print(f"   ✓ verify_tree: SYCL matches PyTorch")
        print(f"   ⏱ SYCL: {sycl_time:.2f}ms, PyTorch: {pytorch_time:.2f}ms")
        print(f"   🚀 Speedup: {pytorch_time/sycl_time:.2f}x")
    else:
        print(f"   ❌ verify_tree: MISMATCH between SYCL and PyTorch")
        return False

    return True


def run_full_test_suite(verbose=False):
    """Run the full pytest suite."""
    print("\n" + "=" * 60)
    print("FULL PARITY TEST SUITE")
    print("=" * 60)

    import subprocess

    cmd = [
        "pytest",
        "test/registered/jit/test_eagle_tree_xpu_parity.py",
        "-v" if verbose else "-q",
        "--tb=short",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd="/home/jhanshu/jh-repos/jh-upstream/sglang")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="XPU EAGLE Tree Parity Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("XPU EAGLE TREE PARITY TEST RUNNER")
    print("=" * 60)

    print("\n📋 Checking prerequisites...\n")

    # Check all prerequisites
    checks = [
        ("XPU", check_xpu_available()),
        ("SYCL JIT", check_sycl_kernel_available()),
        ("Triton", check_triton_available()),
        ("icpx", check_icpx_available()),
    ]

    required_checks = ["XPU", "SYCL JIT", "icpx"]
    all_required_pass = all(passed for name, passed in checks if name in required_checks)

    if not all_required_pass:
        print("\n❌ Required prerequisites not met")
        print("   Cannot run parity tests")
        return 1

    # Run quick test
    if args.quick:
        success = run_quick_parity_test()
        if success:
            print("\n" + "=" * 60)
            print("✅ QUICK PARITY TEST PASSED")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ QUICK PARITY TEST FAILED")
            print("=" * 60)
            return 1

    # Run full test suite
    success = run_full_test_suite(args.verbose)
    if success:
        print("\n" + "=" * 60)
        print("✅ ALL PARITY TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME PARITY TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
