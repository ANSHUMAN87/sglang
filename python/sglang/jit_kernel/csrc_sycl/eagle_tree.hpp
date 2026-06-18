#pragma once
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

// Kernel functor for building EAGLE tree structure
class BuildTreeKernel {
public:
    BuildTreeKernel(
        const int64_t* parent_list,
        const int64_t* selected_index,
        const int32_t* verified_seq_len,
        bool* tree_mask,
        int64_t* positions,
        int64_t* retrive_index,
        int64_t* retrive_next_token,
        int64_t* retrive_next_sibling,
        int topk,
        int depth,
        int draft_token_num,
        int tree_mask_mode,
        int parent_list_stride,
        int selected_index_stride)
        : parent_list_(parent_list),
          selected_index_(selected_index),
          verified_seq_len_(verified_seq_len),
          tree_mask_(tree_mask),
          positions_(positions),
          retrive_index_(retrive_index),
          retrive_next_token_(retrive_next_token),
          retrive_next_sibling_(retrive_next_sibling),
          topk_(topk),
          depth_(depth),
          draft_token_num_(draft_token_num),
          tree_mask_mode_(tree_mask_mode),
          parent_list_stride_(parent_list_stride),
          selected_index_stride_(selected_index_stride) {}

    void operator()(::sycl::nd_item<1> item) const {
        const size_t batch_idx = item.get_global_id(0);

        // Load sequence length and calculate prefix sum
        int32_t seq_len = verified_seq_len_[batch_idx];

        // Calculate seq_len_prefix_sum manually
        int64_t seq_len_prefix_sum = 0;
        for (size_t i = 0; i < batch_idx; ++i) {
            seq_len_prefix_sum += verified_seq_len_[i];
        }

        int64_t seq_tree_idx = draft_token_num_ * draft_token_num_ * batch_idx +
                               seq_len_prefix_sum * draft_token_num_;

        // Initialize position for first token
        positions_[batch_idx * draft_token_num_] = seq_len;

        // Build retrieval index structure
        int64_t retrive_index_offset = batch_idx * draft_token_num_;

        for (int i = draft_token_num_ - 1; i > 0; --i) {
            int64_t current_token_idx = retrive_index_offset + i;
            retrive_index_[batch_idx * draft_token_num_ + i] = current_token_idx;

            int64_t parent_tb_idx = selected_index_[batch_idx * selected_index_stride_ + (i - 1)] / topk_;
            int parent_position = 0;
            bool found_parent = (parent_tb_idx == 0);

            // Bounds check: parent_list has size parent_list_stride in dim 1
            if (parent_tb_idx > 0 && parent_tb_idx < parent_list_stride_) {
                int64_t parent_token_idx = parent_list_[batch_idx * parent_list_stride_ + parent_tb_idx];

                // Find parent position
                for (int parent_position_candidate = 0; parent_position_candidate < draft_token_num_ - 1; ++parent_position_candidate) {
                    if (selected_index_[batch_idx * selected_index_stride_ + parent_position_candidate] == parent_token_idx) {
                        parent_position = parent_position_candidate + 1;
                        found_parent = true;
                        break;
                    }
                }
            }

            if (found_parent) {
                // Update next token links
                int64_t* next_tok_addr = &retrive_next_token_[batch_idx * draft_token_num_ + parent_position];
                int64_t next_tok = *next_tok_addr;

                if (next_tok == -1) {
                    *next_tok_addr = i;
                } else {
                    *next_tok_addr = i;
                    retrive_next_sibling_[batch_idx * draft_token_num_ + i] = next_tok;
                }
            }
        }

        retrive_index_[batch_idx * draft_token_num_] = retrive_index_offset;

        // Process tree mask for all draft tokens
        for (int draft_token_idx = 0; draft_token_idx < draft_token_num_; ++draft_token_idx) {
            int64_t token_tree_idx;

            if (tree_mask_mode_ == 0) {  // FULL_MASK
                token_tree_idx = seq_tree_idx + (seq_len + draft_token_num_) * draft_token_idx + seq_len + 1;
            } else {  // QLEN_ONLY
                token_tree_idx = draft_token_num_ * draft_token_num_ * batch_idx +
                                draft_token_num_ * draft_token_idx + 1;
            }

            tree_mask_[token_tree_idx - 1] = true;
            for (int i = 0; i < draft_token_num_ - 1; ++i) {
                tree_mask_[token_tree_idx + i] = false;
            }

            if (draft_token_idx > 0) {
                int cur_position = draft_token_idx - 1;
                int position = 0;

                for (int d = 0; d < depth_; ++d) {
                    position++;
                    tree_mask_[token_tree_idx + cur_position] = true;

                    int64_t parent_tb_idx = selected_index_[batch_idx * selected_index_stride_ + cur_position] / topk_;
                    if (parent_tb_idx == 0) {
                        break;
                    }

                    // Bounds check: parent_list has size parent_list_stride in dim 1
                    if (parent_tb_idx >= parent_list_stride_) {
                        break;
                    }

                    int64_t parent_token_idx = parent_list_[batch_idx * parent_list_stride_ + parent_tb_idx];
                    bool found = false;

                    for (int cp = 0; cp < draft_token_num_ - 1; ++cp) {
                        if (selected_index_[batch_idx * selected_index_stride_ + cp] == parent_token_idx) {
                            cur_position = cp;
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        break;
                    }
                }

                positions_[batch_idx * draft_token_num_ + draft_token_idx] = position + seq_len;
            }
        }
    }

private:
    const int64_t* parent_list_;
    const int64_t* selected_index_;
    const int32_t* verified_seq_len_;
    bool* tree_mask_;
    int64_t* positions_;
    int64_t* retrive_index_;
    int64_t* retrive_next_token_;
    int64_t* retrive_next_sibling_;
    int topk_;
    int depth_;
    int draft_token_num_;
    int tree_mask_mode_;
    int parent_list_stride_;
    int selected_index_stride_;
};

// ---------------------------------------------------------------------------
// Optimized EAGLE tree builder.
//
// Applies the performance analysis of the scalar BuildTreeKernel above:
//   * Re-parallelize like the CUDA reference (grid=bs, block=draft_token_num):
//     one work-group per batch element, one work-item per draft token. This
//     lifts the launch from <=batch work-items to batch*draft_token_num and
//     fills the SIMD width, fixing the CRITICAL occupancy bottleneck.
//   * Stage selected_index into shared local memory (SLM) once per group and
//     reuse it across both parent-search loops instead of re-reading global.
//   * Compute the O(batch) prefix sum a single time per group (tid==0) and only
//     when FULL_MASK actually needs it; broadcast via SLM.
//   * Narrow the divide operand to int32 so we avoid emulated 64-bit integer
//     division on Intel GPUs (matches the CUDA reference, which uses `int`).
//   * Vectorizer-friendly fixed-trip loops and coalesced mask initialization.
//
// Output layout and semantics are identical to BuildTreeKernel.
class BuildTreeKernelOptimized {
public:
    BuildTreeKernelOptimized(
        const int64_t* parent_list,
        const int64_t* selected_index,
        const int32_t* verified_seq_len,
        bool* tree_mask,
        int64_t* positions,
        int64_t* retrive_index,
        int64_t* retrive_next_token,
        int64_t* retrive_next_sibling,
        ::sycl::local_accessor<int64_t, 1> sel_idx_smem,
        ::sycl::local_accessor<int64_t, 1> shared_smem,
        int topk,
        int depth,
        int draft_token_num,
        int tree_mask_mode,
        int parent_list_stride,
        int selected_index_stride)
        : parent_list_(parent_list),
          selected_index_(selected_index),
          verified_seq_len_(verified_seq_len),
          tree_mask_(tree_mask),
          positions_(positions),
          retrive_index_(retrive_index),
          retrive_next_token_(retrive_next_token),
          retrive_next_sibling_(retrive_next_sibling),
          sel_idx_smem_(sel_idx_smem),
          shared_smem_(shared_smem),
          topk_(topk),
          depth_(depth),
          draft_token_num_(draft_token_num),
          tree_mask_mode_(tree_mask_mode),
          parent_list_stride_(parent_list_stride),
          selected_index_stride_(selected_index_stride) {}

    void operator()(::sycl::nd_item<1> item) const {
        const int batch_idx = static_cast<int>(item.get_group(0));
        const int tid = static_cast<int>(item.get_local_id(0));
        const int dtn = draft_token_num_;

        if (tid >= dtn) {
            return;
        }

        const int64_t sel_base = static_cast<int64_t>(batch_idx) * selected_index_stride_;
        const int64_t parent_base = static_cast<int64_t>(batch_idx) * parent_list_stride_;
        const int64_t row_base = static_cast<int64_t>(batch_idx) * dtn;

        // Cooperatively stage selected_index[batch] (dtn-1 elements) into SLM.
        for (int k = tid; k < dtn - 1; k += dtn) {
            sel_idx_smem_[k] = selected_index_[sel_base + k];
        }

        const int32_t seq_len = verified_seq_len_[batch_idx];

        // tid==0 computes the per-group base offset once (prefix sum only when
        // FULL_MASK needs it) and broadcasts it through SLM.
        if (tid == 0) {
            int64_t seq_tree_idx = static_cast<int64_t>(dtn) * dtn * batch_idx;
            if (tree_mask_mode_ == 0) {  // FULL_MASK
                int64_t seq_len_prefix_sum = 0;
                for (int i = 0; i < batch_idx; ++i) {
                    seq_len_prefix_sum += verified_seq_len_[i];
                }
                seq_tree_idx += seq_len_prefix_sum * dtn;
            }
            shared_smem_[0] = seq_tree_idx;
        }
        item.barrier(::sycl::access::fence_space::local_space);

        const int64_t seq_tree_idx = shared_smem_[0];

        // Per-token mask row base (each tid owns a distinct row -> no races).
        int64_t token_tree_idx;
        if (tree_mask_mode_ == 0) {  // FULL_MASK
            token_tree_idx =
                seq_tree_idx + static_cast<int64_t>(seq_len + dtn) * tid + seq_len + 1;
        } else {  // QLEN_ONLY
            token_tree_idx = seq_tree_idx + static_cast<int64_t>(dtn) * tid + 1;
        }

        tree_mask_[token_tree_idx - 1] = true;
        for (int i = 0; i < dtn - 1; ++i) {
            tree_mask_[token_tree_idx + i] = false;
        }

        if (tid == 0) {
            // Serial linked-list build (retrieval structure) for this batch.
            positions_[row_base] = seq_len;

            for (int i = dtn - 1; i > 0; --i) {
                retrive_index_[row_base + i] = row_base + i;

                // int32 divide operand: avoid emulated 64-bit div (matches CUDA).
                int32_t parent_tb_idx =
                    static_cast<int32_t>(sel_idx_smem_[i - 1]) / topk_;
                int parent_position = 0;
                bool found_parent = (parent_tb_idx == 0);

                if (parent_tb_idx > 0 && parent_tb_idx < parent_list_stride_) {
                    int64_t parent_token_idx = parent_list_[parent_base + parent_tb_idx];
                    for (int cand = 0; cand < dtn - 1; ++cand) {
                        if (sel_idx_smem_[cand] == parent_token_idx) {
                            parent_position = cand + 1;
                            found_parent = true;
                            break;
                        }
                    }
                }

                if (found_parent) {
                    int64_t* next_tok_addr = &retrive_next_token_[row_base + parent_position];
                    int64_t next_tok = *next_tok_addr;
                    *next_tok_addr = i;
                    if (next_tok != -1) {
                        retrive_next_sibling_[row_base + i] = next_tok;
                    }
                }
            }

            retrive_index_[row_base] = row_base;
        } else {
            // Parallel position walk for draft token = tid.
            int cur_position = tid - 1;
            int position = 0;

            for (int d = 0; d < depth_; ++d) {
                position++;
                tree_mask_[token_tree_idx + cur_position] = true;

                int32_t parent_tb_idx =
                    static_cast<int32_t>(sel_idx_smem_[cur_position]) / topk_;
                if (parent_tb_idx == 0 || parent_tb_idx >= parent_list_stride_) {
                    break;
                }

                int64_t parent_token_idx = parent_list_[parent_base + parent_tb_idx];
                bool found = false;
                for (int cp = 0; cp < dtn - 1; ++cp) {
                    if (sel_idx_smem_[cp] == parent_token_idx) {
                        cur_position = cp;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    break;
                }
            }

            positions_[row_base + tid] = position + seq_len;
        }
    }

private:
    const int64_t* parent_list_;
    const int64_t* selected_index_;
    const int32_t* verified_seq_len_;
    bool* tree_mask_;
    int64_t* positions_;
    int64_t* retrive_index_;
    int64_t* retrive_next_token_;
    int64_t* retrive_next_sibling_;
    ::sycl::local_accessor<int64_t, 1> sel_idx_smem_;
    ::sycl::local_accessor<int64_t, 1> shared_smem_;
    int topk_;
    int depth_;
    int draft_token_num_;
    int tree_mask_mode_;
    int parent_list_stride_;
    int selected_index_stride_;
};

// Host launcher function for build_tree
void launch_build_tree_kernel(
    ::sycl::queue& queue,
    const void* parent_list,
    const void* selected_index,
    const void* verified_seq_len,
    void* tree_mask,
    void* positions,
    void* retrive_index,
    void* retrive_next_token,
    void* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode,
    int batch_size,
    int parent_list_stride,
    int selected_index_stride
) {
    const int64_t* parent_list_ptr = static_cast<const int64_t*>(parent_list);
    const int64_t* selected_index_ptr = static_cast<const int64_t*>(selected_index);
    const int32_t* verified_seq_len_ptr = static_cast<const int32_t*>(verified_seq_len);
    bool* tree_mask_ptr = static_cast<bool*>(tree_mask);
    int64_t* positions_ptr = static_cast<int64_t*>(positions);
    int64_t* retrive_index_ptr = static_cast<int64_t*>(retrive_index);
    int64_t* retrive_next_token_ptr = static_cast<int64_t*>(retrive_next_token);
    int64_t* retrive_next_sibling_ptr = static_cast<int64_t*>(retrive_next_sibling);

    const size_t threads_per_group = 1;  // One thread per batch item
    const size_t num_groups = batch_size;

    queue.submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>(
                ::sycl::range<1>(num_groups * threads_per_group),
                ::sycl::range<1>(threads_per_group)
            ),
            BuildTreeKernel(
                parent_list_ptr,
                selected_index_ptr,
                verified_seq_len_ptr,
                tree_mask_ptr,
                positions_ptr,
                retrive_index_ptr,
                retrive_next_token_ptr,
                retrive_next_sibling_ptr,
                topk,
                depth,
                draft_token_num,
                tree_mask_mode,
                parent_list_stride,
                selected_index_stride
            )
        );
    }).wait();
}

// Host launcher for the optimized build_tree kernel.
void launch_build_tree_kernel_optimized(
    ::sycl::queue& queue,
    const void* parent_list,
    const void* selected_index,
    const void* verified_seq_len,
    void* tree_mask,
    void* positions,
    void* retrive_index,
    void* retrive_next_token,
    void* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode,
    int batch_size,
    int parent_list_stride,
    int selected_index_stride
) {
    const int64_t* parent_list_ptr = static_cast<const int64_t*>(parent_list);
    const int64_t* selected_index_ptr = static_cast<const int64_t*>(selected_index);
    const int32_t* verified_seq_len_ptr = static_cast<const int32_t*>(verified_seq_len);
    bool* tree_mask_ptr = static_cast<bool*>(tree_mask);
    int64_t* positions_ptr = static_cast<int64_t*>(positions);
    int64_t* retrive_index_ptr = static_cast<int64_t*>(retrive_index);
    int64_t* retrive_next_token_ptr = static_cast<int64_t*>(retrive_next_token);
    int64_t* retrive_next_sibling_ptr = static_cast<int64_t*>(retrive_next_sibling);

    // One work-group per batch element, draft_token_num work-items per group
    // (mirrors the CUDA reference grid=bs, block=draft_token_num).
    const size_t threads_per_group = static_cast<size_t>(draft_token_num);
    const size_t num_groups = static_cast<size_t>(batch_size);

    // SLM: selected_index row (draft_token_num-1 elems, >=1) + 1 broadcast slot.
    const size_t sel_idx_smem_size =
        static_cast<size_t>(draft_token_num > 1 ? draft_token_num - 1 : 1);

    queue.submit([&](::sycl::handler& cgh) {
        ::sycl::local_accessor<int64_t, 1> sel_idx_smem(
            ::sycl::range<1>(sel_idx_smem_size), cgh);
        ::sycl::local_accessor<int64_t, 1> shared_smem(::sycl::range<1>(1), cgh);

        cgh.parallel_for(
            ::sycl::nd_range<1>(
                ::sycl::range<1>(num_groups * threads_per_group),
                ::sycl::range<1>(threads_per_group)
            ),
            BuildTreeKernelOptimized(
                parent_list_ptr,
                selected_index_ptr,
                verified_seq_len_ptr,
                tree_mask_ptr,
                positions_ptr,
                retrive_index_ptr,
                retrive_next_token_ptr,
                retrive_next_sibling_ptr,
                sel_idx_smem,
                shared_smem,
                topk,
                depth,
                draft_token_num,
                tree_mask_mode,
                parent_list_stride,
                selected_index_stride
            )
        );
    });
    // NOTE: no internal .wait() — the caller controls synchronization, removing
    // per-launch host round-trip latency from the critical path.
    queue.wait();
}

// Export C API for TVM FFI
extern "C" {

void sgl_build_tree_kernel_efficient_xpu(
    void* queue_ptr,
    const void* parent_list,
    const void* selected_index,
    const void* verified_seq_len,
    void* tree_mask,
    void* positions,
    void* retrive_index,
    void* retrive_next_token,
    void* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode,
    int batch_size,
    int parent_list_stride,
    int selected_index_stride
) {
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);
    launch_build_tree_kernel(
        queue,
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
        batch_size,
        parent_list_stride,
        selected_index_stride
    );
}

void sgl_build_tree_kernel_efficient_xpu_optimized(
    void* queue_ptr,
    const void* parent_list,
    const void* selected_index,
    const void* verified_seq_len,
    void* tree_mask,
    void* positions,
    void* retrive_index,
    void* retrive_next_token,
    void* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode,
    int batch_size,
    int parent_list_stride,
    int selected_index_stride
) {
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);
    launch_build_tree_kernel_optimized(
        queue,
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
        batch_size,
        parent_list_stride,
        selected_index_stride
    );
}

}  // extern "C"

}  // namespace sycl_kernel
}  // namespace sgl
