#pragma once

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"
#include "gemm.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm100_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"
#endif

namespace deep_gemm::einsum {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void bmk_bnk_mn(const DGTensorView& a, const DGTensorView& b, const DGTensorView& d,
                       const std::optional<DGTensorView>& c) {
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::Float32) &&
                   "bmk_bnk_mn only supports FP32 output; BF16 conversion should be handled externally");
    DG_HOST_ASSERT(c->data_ptr() == d.data_ptr() and c->same_shape(d) and c->same_strides(d));

    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    const auto& [s , m, k ] = get_shape<3>(a);
    const auto& [s_, n, k_] = get_shape<3>(b);
    DG_HOST_ASSERT(s == s_ and k == k_);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else if (arch_major == 10) {
        sm100_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhr_hdr_bhd(const DGTensorView& A, const DGTensorView& B, const DGTensorView& D, const bool& use_cublaslt) {
    const auto& [b , h  , r ] = get_shape<3>(A);
    const auto& [h_, d  , r_] = get_shape<3>(B);
    const auto& [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(dg_dtype_eq(A.scalar_type(), dg_dtype::BFloat16) and A.stride(2) == 1);
    DG_HOST_ASSERT(dg_dtype_eq(B.scalar_type(), dg_dtype::BFloat16) and B.stride(2) == 1);
    DG_HOST_ASSERT(dg_dtype_eq(D.scalar_type(), dg_dtype::BFloat16) and D.stride(2) == 1);

    const auto& arch_major = device_runtime->get_arch_major();
    if (use_cublaslt) {
        cublaslt_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhd_hdr_bhr(const DGTensorView& A, const DGTensorView& B, const DGTensorView& D, const bool& use_cublaslt) {
    const auto& [b , h  , d ] = get_shape<3>(A);
    const auto& [h_, d_ , r ] = get_shape<3>(B);
    const auto& [b_, h__, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(dg_dtype_eq(A.scalar_type(), dg_dtype::BFloat16) and A.stride(2) == 1);
    DG_HOST_ASSERT(dg_dtype_eq(B.scalar_type(), dg_dtype::BFloat16) and B.stride(2) == 1);
    DG_HOST_ASSERT(dg_dtype_eq(D.scalar_type(), dg_dtype::BFloat16) and D.stride(2) == 1);

    const auto& arch_major = device_runtime->get_arch_major();
    if (use_cublaslt) {
        cublaslt_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void einsum(const std::string& expr,
                   const DGTensorView& a,
                   const DGTensorView& b,
                   const DGTensorView& d,
                   const std::optional<DGTensorView>& c,
                   const bool& use_cublaslt) {
    DG_HOST_ASSERT(dg_dtype_eq(a.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(b.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16) or dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));
    if (c.has_value()) {
        DG_HOST_ASSERT(dg_dtype_eq(c->scalar_type(), dg_dtype::Float32));
        DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));
    }

    if (expr == "bmk,bnk->mn") {
        DG_HOST_ASSERT(not use_cublaslt);
        DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::Float32) &&
                       "BF16 output for bmk,bnk->mn requires external conversion; use FP32 output");
        bmk_bnk_mn(a, b, d, c);
    } else if (expr == "bhr,hdr->bhd") {
        DG_HOST_ASSERT(not c.has_value());
        bhr_hdr_bhd(a, b, d, use_cublaslt);
    } else if (expr == "bhd,hdr->bhr") {
        DG_HOST_ASSERT(not c.has_value());
        bhd_hdr_bhr(a, b, d, use_cublaslt);
    } else {
        DG_HOST_UNREACHABLE(fmt::format("Unsupported einsum expression: {}", expr));
    }
}

static void fp8_bmm(const DGTensorView& a, const DGTensorView& sfa,
                    const DGTensorView& b, const DGTensorView& sfb,
                    const DGTensorView& d,
                    const std::optional<DGTensorView>& c,
                    std::optional<std::tuple<int, int, int>> recipe,
                    const std::string& compiled_dims) {
    const auto& major_a = a.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    const auto& major_b = b.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    DG_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DG_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DG_HOST_ASSERT(d.stride(-1) == 1);

    const auto& [batch_size  , m , k ] = get_shape<3>(a);
    const auto& [batch_size_ , n , k_] = get_shape<3>(b);
    const auto& [batch_size__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(dg_dtype_eq(a.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(b.scalar_type(), dg_dtype::Float8E4M3));
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16) or dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));

    if (batch_size == 0 or gemm::early_return(m, n, k, d, c))
        return;

    const auto& sf_result = layout::transform_sf_pair_into_required_layout(
        sfa, sfb, m, n, k, recipe, std::nullopt, std::nullopt, batch_size, batch_size, false);
    const auto& transformed_sfa = sf_result.sfa.view;
    const auto& transformed_sfb = sf_result.sfb.view;

    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d, batch_size, m, n, k, major_a, major_b, compiled_dims);
    } else {
        const auto& major_sfb = get_major_type_ab(sfb);
        sm90_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d, batch_size, m, n, k, major_a, major_b, major_sfb, compiled_dims);
    }
}

static void fp8_einsum(const std::string& expr,
                       const DGTensorView& a_data, const DGTensorView& a_sf,
                       const DGTensorView& b_data, const DGTensorView& b_sf,
                       const DGTensorView& d,
                       const std::optional<DGTensorView>& c,
                       const std::tuple<int, int, int>& recipe) {
    const auto arch_major = device_runtime->get_arch_major();
    if (expr == "bhr,hdr->bhd") {
        const auto perm_a = a_data.permute({1, 0, 2});
        const auto perm_sfa = a_sf.permute({1, 0, 2});
        const auto perm_d = d.permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, b_data, b_sf, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,hdr->bhr" and arch_major == 10) {
        const auto perm_a = a_data.permute({1, 0, 2});
        const auto perm_sfa = a_sf.permute({1, 0, 2});
        const auto perm_b = b_data.permute({0, 2, 1});
        const auto perm_sfb = b_sf.permute({0, 2, 1});
        const auto perm_d = d.permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,bhr->hdr" and arch_major == 10) {
        const auto perm_a = a_data.permute({1, 2, 0});
        const auto perm_sfa = a_sf.permute({1, 2, 0});
        const auto perm_b = b_data.permute({1, 2, 0});
        const auto perm_sfb = b_sf.permute({1, 2, 0});
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, d, c, recipe, "mn");
    } else {
        DG_HOST_UNREACHABLE(fmt::format("Unsupported einsum expression: {}", expr));
    }
}
#endif

} // namespace deep_gemm::einsum
