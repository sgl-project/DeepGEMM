#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/smxx_fp8_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_clean_logits.hpp"
#endif

#include "layout.hpp"

namespace deep_gemm::attention {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void fp8_gemm_nt_skip_head_mid(const DGTensorView& a_data, const DGTensorView& a_sf,
                                      const DGTensorView& b_data, const DGTensorView& b_sf,
                                      const DGTensorView& d,
                                      const std::tuple<int, int, int>& head_splits,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    const auto& major_a = get_major_type_ab(a_data);
    const auto& major_b = get_major_type_ab(b_data);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    check_major_type_cd(d);

    const auto& [m , k ] = get_shape<2>(a_data);
    const auto& [n , k_] = get_shape<2>(b_data);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(dg_dtype_eq(a_data.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(b_data.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16) or dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));

    const auto& [left, mid, right] = head_splits;
    DG_HOST_ASSERT(n % (left + right) == 0 and n_ == n + n / (left + right) * mid);

    if (m == 0)
        return;

    const auto& [transformed_sfa_r, transformed_sfb_r, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a_sf, b_sf, m, n, k, recipe, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, disable_ue8m0_cast);
    const auto& sfa = transformed_sfa_r.view;
    const auto& sfb = transformed_sfb_r.view;
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    const auto& arch_major = device_runtime->get_arch_major();
    const auto& epilogue_type = fmt::format("EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
    if (arch_major == 9 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Float32) and std::get<1>(recipe.value()) != 1) {
        const auto& major_sfb = get_major_type_ab(sfb);
        sm90_fp8_gemm_1d2d(a_data, sfa, b_data, sfb, std::nullopt, d, m, n, k, major_a, major_b, major_sfb, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Int32)) {
        sm100_fp8_fp4_gemm_1d1d(a_data, sfa, b_data, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

// logits: pre-allocated buffer of shape (seq_len, visible_cols) with stride(0) = stride_logits
static void fp8_mqa_logits(const DGTensorView& q,
                           const DGTensorView& kv_data,
                           const DGTensorView& kv_sf,
                           const DGTensorView& weights,
                           const DGTensorView& cu_seq_len_k_start,
                           const DGTensorView& cu_seq_len_k_end,
                           DGTensorView logits,
                           const int& stride_logits,
                           const bool& clean_logits,
                           const int& max_seqlen_k) {
    const auto& [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_kv, head_dim_] = get_shape<2>(kv_data);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
    const auto& [seq_len_kv_] = get_shape<1>(kv_sf);

    DG_HOST_ASSERT(seq_len == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_ and head_dim == head_dim_);
    DG_HOST_ASSERT(seq_len_kv == seq_len_kv_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);

    DG_HOST_ASSERT(q.is_contiguous() and kv_data.is_contiguous());
    DG_HOST_ASSERT(kv_sf.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());

    DG_HOST_ASSERT(dg_dtype_eq(q.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(kv_data.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(kv_sf.scalar_type(), dg_dtype::Float32));
    DG_HOST_ASSERT(dg_dtype_eq(weights.scalar_type(), dg_dtype::Float32));
    DG_HOST_ASSERT(dg_dtype_eq(cu_seq_len_k_start.scalar_type(), dg_dtype::Int32));
    DG_HOST_ASSERT(dg_dtype_eq(cu_seq_len_k_end.scalar_type(), dg_dtype::Int32));
    DG_HOST_ASSERT(dg_dtype_eq(logits.scalar_type(), dg_dtype::Float32));

    constexpr int seq_len_alignment = 4;
    const auto seq_len_kv_visible = (max_seqlen_k == 0) ? seq_len_kv : max_seqlen_k;

    if (max_seqlen_k != 0)
        DG_HOST_ASSERT(not clean_logits);

    // Slice logits to the visible region
    auto logits_view = logits.slice(0, 0, seq_len);
    if (max_seqlen_k == 0)
        logits_view = logits_view.slice(1, 0, seq_len_kv);
    else
        logits_view = logits_view.slice(1, 0, max_seqlen_k);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_mqa_logits(q, kv_data, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits_view,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, seq_len_alignment);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    if (clean_logits)
        smxx_clean_logits(logits_view, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, stride_logits);
}

// schedule_metadata: pre-allocated output of shape (num_sms + 1, 2), dtype Int32
static void get_paged_mqa_logits_metadata(const DGTensorView& context_lens, int block_kv, int num_sms,
                                          DGTensorView schedule_metadata) {
    const bool is_context_lens_2d = context_lens.dim() == 2;
    int batch_size = 0, next_n = 0;
    if (is_context_lens_2d) {
        batch_size = context_lens.size(0);
        next_n = context_lens.size(1);
    } else {
        DG_HOST_ASSERT(context_lens.dim() == 1);
        batch_size = context_lens.size(0);
    }
    DG_HOST_ASSERT(dg_dtype_eq(context_lens.scalar_type(), dg_dtype::Int32));
    DG_HOST_ASSERT(context_lens.is_contiguous());

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n, block_kv, num_sms, is_context_lens_2d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

// logits: pre-allocated output of shape (batch_size * next_n, aligned_max_context_len), dtype Float32
static void fp8_paged_mqa_logits(const DGTensorView& q,
                                 const DGTensorView& fused_kv_cache,
                                 const DGTensorView& weights,
                                 const DGTensorView& context_lens,
                                 const DGTensorView& block_table,
                                 const DGTensorView& schedule_meta,
                                 const int& max_context_len,
                                 DGTensorView logits,
                                 const bool& clean_logits) {
    const auto& [batch_size, next_n, num_heads, head_dim] = get_shape<4>(q);
    const auto& [num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size_, max_block_len] = get_shape<2>(block_table);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);
    const auto& num_sms = device_runtime->get_num_sms();
    const auto kv_cache_stride_bytes = static_cast<int>(fused_kv_cache.stride(0));
    const auto block_table_stride = static_cast<int>(block_table.stride(0));

    const bool is_context_lens_2d = context_lens.dim() == 2;
    if (is_context_lens_2d) {
        const auto& [batch_size__, next_n_] = get_shape<2>(context_lens);
        DG_HOST_ASSERT(batch_size == batch_size__ and next_n == next_n_);
    } else {
        DG_HOST_ASSERT(context_lens.dim() == 1);
        const auto& [batch_size__] = get_shape<1>(context_lens);
        DG_HOST_ASSERT(batch_size == batch_size__);
    }

    DG_HOST_ASSERT(batch_size == batch_size_);
    DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    DG_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);
    DG_HOST_ASSERT(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
    DG_HOST_ASSERT(schedule_meta_size == num_sms + 1 and meta_info_size == 2);

    DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    DG_HOST_ASSERT(block_kv == 64);

    DG_HOST_ASSERT(q.is_contiguous());
    DG_HOST_ASSERT(kv_cache_stride_bytes % static_cast<int>(sizeof(float)) == 0);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());

    DG_HOST_ASSERT(dg_dtype_eq(q.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(fused_kv_cache.scalar_type(), dg_dtype::UInt8));
    DG_HOST_ASSERT(dg_dtype_eq(weights.scalar_type(), dg_dtype::Float32));
    DG_HOST_ASSERT(dg_dtype_eq(context_lens.scalar_type(), dg_dtype::Int32));
    DG_HOST_ASSERT(dg_dtype_eq(block_table.scalar_type(), dg_dtype::Int32));
    DG_HOST_ASSERT(dg_dtype_eq(schedule_meta.scalar_type(), dg_dtype::Int32));
    DG_HOST_ASSERT(dg_dtype_eq(logits.scalar_type(), dg_dtype::Float32));

    // Derive FP8 values and SF tensor from KV cache via pointer manipulation
    const auto kv_cache = DGTensorView::from_ptr_strided(
        fused_kv_cache.data_ptr(),
        dg_dtype::Float8E4M3,
        {static_cast<int64_t>(num_kv_blocks), static_cast<int64_t>(block_kv), static_cast<int64_t>(head_dim)},
        {static_cast<int64_t>(kv_cache_stride_bytes), static_cast<int64_t>(head_dim), 1LL},
        fused_kv_cache.device_id_val());

    const auto kv_cache_scales = DGTensorView::from_ptr_strided(
        static_cast<uint8_t*>(fused_kv_cache.data_ptr()) + block_kv * head_dim,
        dg_dtype::Float32,
        {static_cast<int64_t>(num_kv_blocks), static_cast<int64_t>(block_kv)},
        {static_cast<int64_t>(kv_cache_stride_bytes) / static_cast<int64_t>(sizeof(float)), 1LL},
        fused_kv_cache.device_id_val());

    // Slice logits to max_context_len
    constexpr int split_kv = 256;
    const auto aligned_max_context_len = static_cast<int>(logits.stride(0) > 0 ? logits.size(1) : align(max_context_len, split_kv));
    auto logits_view = logits.slice(-1, 0, max_context_len);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_paged_mqa_logits(q, kv_cache, kv_cache_scales, weights, context_lens, logits_view, block_table, schedule_meta,
                                  batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                  kv_cache_stride_bytes, aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    if (clean_logits) {
        DG_HOST_ASSERT(not is_context_lens_2d);
        smxx_clean_logits(logits_view, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    }
}

#endif

} // namespace deep_gemm::attention
