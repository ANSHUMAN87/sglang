"""
Parity tests for EAGLE tree XPU JIT kernels vs Triton vs PyTorch.

Tests verify that the SYCL JIT kernel, Triton kernel, and PyTorch reference
implementations produce identical results across various input configurations.

Three implementations tested:
1. SYCL JIT kernel (eagle_tree_xpu.py) - native XPU performance
2. Triton kernel (spec_tree.py) - Triton-based implementation
3. PyTorch reference (eagle_utils.py) - pure PyTorch fallback

Test structure follows patterns from:
- test_ngram_embedding.py (JIT vs general parity)
- test_spec_eagle_parity.py (EAGLE correctness)
- test_mla_cp_fa3_parity.py (numerical parity across implementations)
"""

import sys

import pytest
import torch

from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    sgl_build_tree_kernel_efficient_pytorch,
    verify_tree_greedy_pytorch,
)
from sglang.test.ci.ci_register import register_xpu_ci

# Register for XPU CI - these tests require XPU hardware with oneAPI.
register_xpu_ci(est_time=120, stage="stage-b", runner_config="1-gpu-xpu")


def _make_test_data_build_tree(
    batch_size: int,
    draft_token_num: int,
    topk: int,
    device: str = "cuda",
):
    """Generate realistic test data for build_tree kernel."""
    # Parent list: each draft token references a previous position
    # Valid range: [0, draft_token_num)
    parent_list = torch.randint(
        0,
        draft_token_num,
        (batch_size, draft_token_num - 1),
        dtype=torch.int64,
        device=device,
    )

    # Selected index: indices into flattened parent_list * topk space
    selected_index = torch.randint(
        0,
        topk * (draft_token_num - 1),
        (batch_size, draft_token_num - 1),
        dtype=torch.int64,
        device=device,
    )

    # Varied sequence lengths
    verified_seq_len = torch.randint(
        3, 20, (batch_size,), dtype=torch.int32, device=device
    )

    return parent_list, selected_index, verified_seq_len


def _make_output_buffers_build_tree(
    batch_size: int,
    draft_token_num: int,
    verified_seq_len: torch.Tensor,
    tree_mask_mode: TreeMaskMode,
    device: str = "cuda",
):
    """Allocate output buffers matching build_tree requirements."""
    seq_lens_sum = int(verified_seq_len.sum())

    # Tree mask size calculation per TreeMaskMode
    if tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask_size = (
            seq_lens_sum * draft_token_num
            + draft_token_num * draft_token_num * batch_size
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask_size = draft_token_num * batch_size * draft_token_num
    else:
        raise NotImplementedError(f"Unsupported tree_mask_mode: {tree_mask_mode}")

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
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


def _make_test_data_verify_tree(
    batch_size: int,
    num_draft_tokens: int,
    vocab_size: int = 32000,
    device: str = "cuda",
):
    """Generate test data for verify_tree kernel with realistic tree structure."""
    # Draft candidate tokens
    candidates = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )

    # Build simple retrieval structure (simulate build_tree output)
    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=device
    ).reshape(batch_size, num_draft_tokens)

    # Create tree with linear chains and some branches
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    # Build linear chains: 0 -> 1 -> 2 -> ...
    for b in range(batch_size):
        for i in range(min(num_draft_tokens - 1, 3)):
            retrive_next_token[b, i] = i + 1
        # Add a branch at position 1 -> 4 (if space available)
        if num_draft_tokens > 4:
            retrive_next_sibling[b, 2] = 4
            retrive_next_token[b, 1] = 2

    # Target model predictions (some match candidates for acceptance)
    target_predict = torch.randint(
        0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    # Make first token always match to ensure some acceptance
    target_predict[:, 0] = candidates[:, 1]

    return (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )


def _make_output_buffers_verify_tree(
    batch_size: int,
    num_draft_tokens: int,
    num_speculative_tokens: int,
    device: str = "cuda",
):
    """Allocate output buffers for verify_tree kernel."""
    predicts = torch.zeros(
        (batch_size * num_draft_tokens,), dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, num_speculative_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    return predicts, accept_index, accept_token_num


# ============================================================================
# Build Tree Kernel Parity Tests
# ============================================================================


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("draft_token_num", [4, 8, 16])
@pytest.mark.parametrize("topk", [1, 2, 3])
@pytest.mark.parametrize(
    "tree_mask_mode", [TreeMaskMode.FULL_MASK, TreeMaskMode.QLEN_ONLY]
)
@pytest.mark.parametrize("optimized", [False, True])
def test_build_tree_sycl_matches_pytorch(
    batch_size: int,
    draft_token_num: int,
    topk: int,
    tree_mask_mode: TreeMaskMode,
    optimized: bool,
):
    """SYCL JIT kernel produces identical output to PyTorch reference."""
    from sglang.jit_kernel.eagle_tree_xpu import (
        sgl_build_tree_kernel_efficient_xpu,
    )

    depth = 3
    device = "xpu"

    # Generate input data
    parent_list, selected_index, verified_seq_len = _make_test_data_build_tree(
        batch_size, draft_token_num, topk, device
    )

    # Allocate output buffers (two independent sets)
    outputs_sycl = _make_output_buffers_build_tree(
        batch_size, draft_token_num, verified_seq_len, tree_mask_mode, device
    )
    outputs_pytorch = _make_output_buffers_build_tree(
        batch_size, draft_token_num, verified_seq_len, tree_mask_mode, device
    )

    # Run SYCL kernel
    sgl_build_tree_kernel_efficient_xpu(
        parent_list,
        selected_index,
        verified_seq_len,
        *outputs_sycl,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
        optimized=optimized,
    )

    # Run PyTorch reference
    sgl_build_tree_kernel_efficient_pytorch(
        parent_list,
        selected_index,
        verified_seq_len,
        *outputs_pytorch,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )

    # Assert exact match (integer outputs, no tolerance needed)
    (
        tree_mask_sycl,
        positions_sycl,
        retrive_index_sycl,
        retrive_next_token_sycl,
        retrive_next_sibling_sycl,
    ) = outputs_sycl
    (
        tree_mask_pytorch,
        positions_pytorch,
        retrive_index_pytorch,
        retrive_next_token_pytorch,
        retrive_next_sibling_pytorch,
    ) = outputs_pytorch

    torch.testing.assert_close(tree_mask_sycl, tree_mask_pytorch, atol=0, rtol=0)
    torch.testing.assert_close(positions_sycl, positions_pytorch, atol=0, rtol=0)
    torch.testing.assert_close(
        retrive_index_sycl, retrive_index_pytorch, atol=0, rtol=0
    )
    torch.testing.assert_close(
        retrive_next_token_sycl, retrive_next_token_pytorch, atol=0, rtol=0
    )
    torch.testing.assert_close(
        retrive_next_sibling_sycl, retrive_next_sibling_pytorch, atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("draft_token_num", [4, 8])
def test_build_tree_triton_matches_pytorch(batch_size: int, draft_token_num: int):
    """Triton kernel produces identical output to PyTorch reference."""
    try:
        from sglang.srt.speculative.eagle_utils import sgl_build_tree_kernel_triton
    except ImportError:
        pytest.skip("Triton implementation not available")

    topk = 2
    depth = 3
    tree_mask_mode = TreeMaskMode.FULL_MASK
    device = "xpu"

    parent_list, selected_index, verified_seq_len = _make_test_data_build_tree(
        batch_size, draft_token_num, topk, device
    )

    outputs_triton = _make_output_buffers_build_tree(
        batch_size, draft_token_num, verified_seq_len, tree_mask_mode, device
    )
    outputs_pytorch = _make_output_buffers_build_tree(
        batch_size, draft_token_num, verified_seq_len, tree_mask_mode, device
    )

    # Run Triton kernel
    sgl_build_tree_kernel_triton(
        parent_list,
        selected_index,
        verified_seq_len,
        *outputs_triton,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )

    # Run PyTorch reference
    sgl_build_tree_kernel_efficient_pytorch(
        parent_list,
        selected_index,
        verified_seq_len,
        *outputs_pytorch,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )

    # Compare outputs
    for triton_out, pytorch_out in zip(outputs_triton, outputs_pytorch):
        torch.testing.assert_close(triton_out, pytorch_out, atol=0, rtol=0)


# ============================================================================
# Verify Tree Kernel Parity Tests
# ============================================================================


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_draft_tokens", [4, 8, 16])
@pytest.mark.parametrize("optimized", [False, True])
def test_verify_tree_sycl_matches_pytorch(
    batch_size: int, num_draft_tokens: int, optimized: bool
):
    """SYCL JIT kernel produces identical output to PyTorch reference."""
    from sglang.jit_kernel.eagle_tree_xpu import verify_tree_greedy_xpu

    num_speculative_tokens = num_draft_tokens
    vocab_size = 32000
    device = "xpu"

    # Generate test data
    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    ) = _make_test_data_verify_tree(
        batch_size, num_draft_tokens, vocab_size, device
    )

    # Allocate outputs
    outputs_sycl = _make_output_buffers_verify_tree(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )
    outputs_pytorch = _make_output_buffers_verify_tree(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )

    # Run SYCL kernel
    verify_tree_greedy_xpu(
        *outputs_sycl,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        optimized=optimized,
    )

    # Run PyTorch reference
    verify_tree_greedy_pytorch(
        *outputs_pytorch,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )

    # Compare outputs
    predicts_sycl, accept_index_sycl, accept_token_num_sycl = outputs_sycl
    predicts_pytorch, accept_index_pytorch, accept_token_num_pytorch = outputs_pytorch

    torch.testing.assert_close(predicts_sycl, predicts_pytorch, atol=0, rtol=0)
    torch.testing.assert_close(accept_index_sycl, accept_index_pytorch, atol=0, rtol=0)
    torch.testing.assert_close(
        accept_token_num_sycl, accept_token_num_pytorch, atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_draft_tokens", [4, 8])
def test_verify_tree_triton_matches_pytorch(batch_size: int, num_draft_tokens: int):
    """Triton kernel produces identical output to PyTorch reference."""
    try:
        from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton
    except ImportError:
        pytest.skip("Triton implementation not available")

    num_speculative_tokens = num_draft_tokens
    vocab_size = 32000
    device = "xpu"

    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    ) = _make_test_data_verify_tree(
        batch_size, num_draft_tokens, vocab_size, device
    )

    outputs_triton = _make_output_buffers_verify_tree(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )
    outputs_pytorch = _make_output_buffers_verify_tree(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )

    # Run Triton kernel
    verify_tree_greedy_triton(
        *outputs_triton,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )

    # Run PyTorch reference
    verify_tree_greedy_pytorch(
        *outputs_pytorch,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )

    for triton_out, pytorch_out in zip(outputs_triton, outputs_pytorch):
        torch.testing.assert_close(triton_out, pytorch_out, atol=0, rtol=0)


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
def test_build_tree_minimal_config():
    """Minimal viable config: single batch, 2 draft tokens."""
    from sglang.jit_kernel.eagle_tree_xpu import (
        sgl_build_tree_kernel_efficient_xpu,
    )

    batch_size, draft_token_num, topk, depth = 1, 2, 1, 1
    tree_mask_mode = TreeMaskMode.QLEN_ONLY
    device = "xpu"

    parent_list, selected_index, verified_seq_len = _make_test_data_build_tree(
        batch_size, draft_token_num, topk, device
    )

    outputs_sycl = _make_output_buffers_build_tree(
        batch_size, draft_token_num, verified_seq_len, tree_mask_mode, device
    )
    outputs_pytorch = _make_output_buffers_build_tree(
        batch_size, draft_token_num, verified_seq_len, tree_mask_mode, device
    )

    sgl_build_tree_kernel_efficient_xpu(
        parent_list,
        selected_index,
        verified_seq_len,
        *outputs_sycl,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )
    sgl_build_tree_kernel_efficient_pytorch(
        parent_list,
        selected_index,
        verified_seq_len,
        *outputs_pytorch,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )

    for sycl_out, pytorch_out in zip(outputs_sycl, outputs_pytorch):
        torch.testing.assert_close(sycl_out, pytorch_out, atol=0, rtol=0)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
def test_verify_tree_no_acceptance():
    """No draft tokens accepted (candidates never match target)."""
    from sglang.jit_kernel.eagle_tree_xpu import verify_tree_greedy_xpu

    batch_size, num_draft_tokens = 2, 4
    num_speculative_tokens = num_draft_tokens
    device = "xpu"

    # Candidates and targets never overlap
    candidates = torch.full(
        (batch_size, num_draft_tokens), 100, dtype=torch.int32, device=device
    )
    target_predict = torch.full(
        (batch_size, num_draft_tokens), 200, dtype=torch.int32, device=device
    )

    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=device
    ).reshape(batch_size, num_draft_tokens)
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    # Build simple chain
    for b in range(batch_size):
        for i in range(num_draft_tokens - 1):
            retrive_next_token[b, i] = i + 1

    outputs_sycl = _make_output_buffers_verify_tree(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )
    outputs_pytorch = _make_output_buffers_verify_tree(
        batch_size, num_draft_tokens, num_speculative_tokens, device
    )

    verify_tree_greedy_xpu(
        *outputs_sycl,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )
    verify_tree_greedy_pytorch(
        *outputs_pytorch,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )

    for sycl_out, pytorch_out in zip(outputs_sycl, outputs_pytorch):
        torch.testing.assert_close(sycl_out, pytorch_out, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
