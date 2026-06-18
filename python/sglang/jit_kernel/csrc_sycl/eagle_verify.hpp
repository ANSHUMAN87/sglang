#pragma once
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

// Kernel functor for verifying EAGLE tree greedily
class VerifyTreeGreedyKernel {
public:
    VerifyTreeGreedyKernel(
        int32_t* predicts,
        int32_t* accept_index,
        int32_t* accept_token_num,
        const int32_t* candidates,
        const int64_t* retrive_index,
        const int64_t* retrive_next_token,
        const int64_t* retrive_next_sibling,
        const int32_t* target_predict,
        int num_speculative_tokens,
        int num_draft_tokens)
        : predicts_(predicts),
          accept_index_(accept_index),
          accept_token_num_(accept_token_num),
          candidates_(candidates),
          retrive_index_(retrive_index),
          retrive_next_token_(retrive_next_token),
          retrive_next_sibling_(retrive_next_sibling),
          target_predict_(target_predict),
          num_speculative_tokens_(num_speculative_tokens),
          num_draft_tokens_(num_draft_tokens) {}

    void operator()(::sycl::nd_item<1> item) const {
        const size_t bx = item.get_global_id(0);

        // Initialize
        int64_t last_accepted_retrive_idx = retrive_index_[bx * num_draft_tokens_];
        accept_index_[bx * num_speculative_tokens_] = last_accepted_retrive_idx;
        int num_accepted_tokens = 0;
        int64_t cur_index = 0;

        // Traverse tree and verify tokens
        for (int j = 1; j < num_speculative_tokens_; ++j) {
            cur_index = retrive_next_token_[bx * num_draft_tokens_ + cur_index];

            while (cur_index != -1) {
                int64_t draft_index = retrive_index_[bx * num_draft_tokens_ + cur_index];
                int32_t draft_token_id = candidates_[bx * num_draft_tokens_ + cur_index];

                int64_t target_idx = last_accepted_retrive_idx / num_draft_tokens_;
                int64_t target_offset = last_accepted_retrive_idx % num_draft_tokens_;
                int32_t target_token_id = target_predict_[target_idx * num_draft_tokens_ + target_offset];

                if (draft_token_id == target_token_id) {
                    // Accept token
                    predicts_[last_accepted_retrive_idx] = target_token_id;
                    num_accepted_tokens++;
                    accept_index_[bx * num_speculative_tokens_ + num_accepted_tokens] = draft_index;
                    last_accepted_retrive_idx = draft_index;
                    break;
                } else {
                    // Try next sibling
                    cur_index = retrive_next_sibling_[bx * num_draft_tokens_ + cur_index];
                }
            }

            if (cur_index == -1) {
                break;
            }
        }

        // Store final results
        accept_token_num_[bx] = num_accepted_tokens;

        int64_t final_target_idx = last_accepted_retrive_idx / num_draft_tokens_;
        int64_t final_target_offset = last_accepted_retrive_idx % num_draft_tokens_;
        predicts_[last_accepted_retrive_idx] =
            target_predict_[final_target_idx * num_draft_tokens_ + final_target_offset];
    }

private:
    int32_t* predicts_;
    int32_t* accept_index_;
    int32_t* accept_token_num_;
    const int32_t* candidates_;
    const int64_t* retrive_index_;
    const int64_t* retrive_next_token_;
    const int64_t* retrive_next_sibling_;
    const int32_t* target_predict_;
    int num_speculative_tokens_;
    int num_draft_tokens_;
};

// ---------------------------------------------------------------------------
// Optimized greedy tree verifier.
//
// The greedy verification is a serial pointer-chase per batch element: each
// iteration's `last_accepted_retrive_idx` feeds the next, so there is NO
// intra-batch parallelism to recover (the CUDA reference also launches
// block(1)). The optimizations here are therefore:
//   * Remove the redundant emulated 64-bit divide/modulo. The scalar kernel
//     computes idx/ndt and idx%ndt then reconstructs target_idx*ndt+offset,
//     which is exactly `last_accepted_retrive_idx`. Index target_predict
//     directly (matches the CUDA reference).
//   * Pack multiple independent batch elements into a work-group (one work-item
//     per batch) to fill the SIMD width instead of launching work-group size 1.
//   * Narrow loop/index arithmetic to int32 where the ranges allow it.
//
// Output and semantics are identical to VerifyTreeGreedyKernel.
class VerifyTreeGreedyKernelOptimized {
public:
    VerifyTreeGreedyKernelOptimized(
        int32_t* predicts,
        int32_t* accept_index,
        int32_t* accept_token_num,
        const int32_t* candidates,
        const int64_t* retrive_index,
        const int64_t* retrive_next_token,
        const int64_t* retrive_next_sibling,
        const int32_t* target_predict,
        int batch_size,
        int num_speculative_tokens,
        int num_draft_tokens)
        : predicts_(predicts),
          accept_index_(accept_index),
          accept_token_num_(accept_token_num),
          candidates_(candidates),
          retrive_index_(retrive_index),
          retrive_next_token_(retrive_next_token),
          retrive_next_sibling_(retrive_next_sibling),
          target_predict_(target_predict),
          batch_size_(batch_size),
          num_speculative_tokens_(num_speculative_tokens),
          num_draft_tokens_(num_draft_tokens) {}

    void operator()(::sycl::nd_item<1> item) const {
        const int bx = static_cast<int>(item.get_global_id(0));

        // Multiple batch elements packed per work-group: guard the tail.
        if (bx >= batch_size_) {
            return;
        }

        const int ndt = num_draft_tokens_;
        const int64_t draft_row = static_cast<int64_t>(bx) * ndt;
        const int64_t accept_row = static_cast<int64_t>(bx) * num_speculative_tokens_;

        int64_t last_accepted_retrive_idx = retrive_index_[draft_row];
        accept_index_[accept_row] = static_cast<int32_t>(last_accepted_retrive_idx);
        int num_accepted_tokens = 0;
        int64_t cur_index = 0;

        for (int j = 1; j < num_speculative_tokens_; ++j) {
            cur_index = retrive_next_token_[draft_row + cur_index];

            while (cur_index != -1) {
                int64_t draft_index = retrive_index_[draft_row + cur_index];
                int32_t draft_token_id = candidates_[draft_row + cur_index];

                // Direct index: target_idx*ndt + offset == last_accepted_retrive_idx,
                // so the div/mod round-trip of the scalar kernel is unnecessary.
                int32_t target_token_id = target_predict_[last_accepted_retrive_idx];

                if (draft_token_id == target_token_id) {
                    predicts_[last_accepted_retrive_idx] = target_token_id;
                    num_accepted_tokens++;
                    accept_index_[accept_row + num_accepted_tokens] =
                        static_cast<int32_t>(draft_index);
                    last_accepted_retrive_idx = draft_index;
                    break;
                } else {
                    cur_index = retrive_next_sibling_[draft_row + cur_index];
                }
            }

            if (cur_index == -1) {
                break;
            }
        }

        accept_token_num_[bx] = num_accepted_tokens;
        predicts_[last_accepted_retrive_idx] =
            target_predict_[last_accepted_retrive_idx];
    }

private:
    int32_t* predicts_;
    int32_t* accept_index_;
    int32_t* accept_token_num_;
    const int32_t* candidates_;
    const int64_t* retrive_index_;
    const int64_t* retrive_next_token_;
    const int64_t* retrive_next_sibling_;
    const int32_t* target_predict_;
    int batch_size_;
    int num_speculative_tokens_;
    int num_draft_tokens_;
};

// Host launcher function for verify_tree_greedy
void launch_verify_tree_greedy_kernel(
    ::sycl::queue& queue,
    void* predicts,
    void* accept_index,
    void* accept_token_num,
    const void* candidates,
    const void* retrive_index,
    const void* retrive_next_token,
    const void* retrive_next_sibling,
    const void* target_predict,
    int batch_size,
    int num_speculative_tokens,
    int num_draft_tokens
) {
    int32_t* predicts_ptr = static_cast<int32_t*>(predicts);
    int32_t* accept_index_ptr = static_cast<int32_t*>(accept_index);
    int32_t* accept_token_num_ptr = static_cast<int32_t*>(accept_token_num);
    const int32_t* candidates_ptr = static_cast<const int32_t*>(candidates);
    const int64_t* retrive_index_ptr = static_cast<const int64_t*>(retrive_index);
    const int64_t* retrive_next_token_ptr = static_cast<const int64_t*>(retrive_next_token);
    const int64_t* retrive_next_sibling_ptr = static_cast<const int64_t*>(retrive_next_sibling);
    const int32_t* target_predict_ptr = static_cast<const int32_t*>(target_predict);

    const size_t threads_per_group = 1;  // One thread per batch item
    const size_t num_groups = batch_size;

    queue.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>(
                ::sycl::range<1>(num_groups * threads_per_group),
                ::sycl::range<1>(threads_per_group)
            ),
            VerifyTreeGreedyKernel(
                predicts_ptr,
                accept_index_ptr,
                accept_token_num_ptr,
                candidates_ptr,
                retrive_index_ptr,
                retrive_next_token_ptr,
                retrive_next_sibling_ptr,
                target_predict_ptr,
                num_speculative_tokens,
                num_draft_tokens
            )
        );
    }).wait();
}

// Host launcher for the optimized verify_tree_greedy kernel.
void launch_verify_tree_greedy_kernel_optimized(
    ::sycl::queue& queue,
    void* predicts,
    void* accept_index,
    void* accept_token_num,
    const void* candidates,
    const void* retrive_index,
    const void* retrive_next_token,
    const void* retrive_next_sibling,
    const void* target_predict,
    int batch_size,
    int num_speculative_tokens,
    int num_draft_tokens
) {
    int32_t* predicts_ptr = static_cast<int32_t*>(predicts);
    int32_t* accept_index_ptr = static_cast<int32_t*>(accept_index);
    int32_t* accept_token_num_ptr = static_cast<int32_t*>(accept_token_num);
    const int32_t* candidates_ptr = static_cast<const int32_t*>(candidates);
    const int64_t* retrive_index_ptr = static_cast<const int64_t*>(retrive_index);
    const int64_t* retrive_next_token_ptr = static_cast<const int64_t*>(retrive_next_token);
    const int64_t* retrive_next_sibling_ptr = static_cast<const int64_t*>(retrive_next_sibling);
    const int32_t* target_predict_ptr = static_cast<const int32_t*>(target_predict);

    // One work-item per work-group, like the scalar kernel and the CUDA
    // reference (block(1)). Verify is a divergent serial pointer-chase: packing
    // multiple batches into one SIMD sub-group makes the whole group run at the
    // slowest lane's trip count, which measured SLOWER on BMG. Keep each batch
    // an independent work-group; the remaining win is the removed 64-bit
    // div/mod. The kernel still guards bx >= batch_size for safety.
    const size_t threads_per_group = 1;
    const size_t num_groups = static_cast<size_t>(batch_size);

    queue.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>(
                ::sycl::range<1>(num_groups * threads_per_group),
                ::sycl::range<1>(threads_per_group)
            ),
            VerifyTreeGreedyKernelOptimized(
                predicts_ptr,
                accept_index_ptr,
                accept_token_num_ptr,
                candidates_ptr,
                retrive_index_ptr,
                retrive_next_token_ptr,
                retrive_next_sibling_ptr,
                target_predict_ptr,
                batch_size,
                num_speculative_tokens,
                num_draft_tokens
            )
        );
    });
    // No internal .wait(): caller controls synchronization.
    queue.wait();
}

// Export C API for TVM FFI
extern "C" {

void verify_tree_greedy_xpu(
    void* queue_ptr,
    void* predicts,
    void* accept_index,
    void* accept_token_num,
    const void* candidates,
    const void* retrive_index,
    const void* retrive_next_token,
    const void* retrive_next_sibling,
    const void* target_predict,
    int batch_size,
    int num_speculative_tokens,
    int num_draft_tokens
) {
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);
    launch_verify_tree_greedy_kernel(
        queue,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        batch_size,
        num_speculative_tokens,
        num_draft_tokens
    );
}

void verify_tree_greedy_xpu_optimized(
    void* queue_ptr,
    void* predicts,
    void* accept_index,
    void* accept_token_num,
    const void* candidates,
    const void* retrive_index,
    const void* retrive_next_token,
    const void* retrive_next_sibling,
    const void* target_predict,
    int batch_size,
    int num_speculative_tokens,
    int num_draft_tokens
) {
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);
    launch_verify_tree_greedy_kernel_optimized(
        queue,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        batch_size,
        num_speculative_tokens,
        num_draft_tokens
    );
}

}  // extern "C"

}  // namespace sycl_kernel
}  // namespace sgl
