#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#endif 

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const DGTensorView& d, const std::optional<DGTensorView>& c) {
    if (m == 0 or n == 0)
        return true;

    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->same_shape(d) and c->same_strides(d));
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));
        DG_HOST_ASSERT(dg_dtype_eq(c.value().scalar_type(), dg_dtype::Float32));
    }

    if (k == 0) {
        if (not is_cd_same) {
            if (c.has_value()) {
                DG_CUDA_RUNTIME_CHECK(cudaMemcpy(d.data_ptr(), c->data_ptr(), d.nbytes(), cudaMemcpyDeviceToDevice));
            } else {
                DG_CUDA_RUNTIME_CHECK(cudaMemset(d.data_ptr(), 0, d.nbytes()));
            }
        }
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        DG_CUDA_RUNTIME_CHECK(cudaMemcpy(d.data_ptr(), c->data_ptr(), d.nbytes(), cudaMemcpyDeviceToDevice));
    return false;
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

static void fp8_fp4_gemm_nt(const DGTensorView& a_data, const DGTensorView& a_sf,
                            const DGTensorView& b_data, const DGTensorView& b_sf,
                            const DGTensorView& d,
                            const std::optional<DGTensorView>& c,
                            std::optional<std::tuple<int, int, int>> recipe,
                            std::optional<std::tuple<int, int>> recipe_a,
                            std::optional<std::tuple<int, int>> recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    const auto& major_a = get_major_type_ab(a_data);
    const auto& major_b = get_major_type_ab(b_data);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    check_major_type_cd(d);

    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a_data, major_a, arch_major);
    const auto [n , k_] = check_ab_fp8_fp4(b_data, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16) or dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));

    if (early_return(m, n, k, d, c))
        return;

    auto sf_result = layout::transform_sf_pair_into_required_layout(
        a_sf, b_sf, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);
    const auto& sfa = sf_result.sfa.view;
    const auto& sfb = sf_result.sfb.view;
    const auto gran_k_a = sf_result.gran_k_a;
    const auto gran_k_b = sf_result.gran_k_b;

    if (arch_major == 9 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Float32)) {
        const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
        if (gran_n == 1) {
            sm90_fp8_gemm_1d1d(a_data, sfa, b_data, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
        } else {
            const auto& major_sfb = get_major_type_ab(sfb);
            sm90_fp8_gemm_1d2d(a_data, sfa, b_data, sfb, c, d, m, n, k, major_a, major_b, major_sfb, compiled_dims);
        }
    } else if (arch_major == 10 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Int32)) {
        sm100_fp8_fp4_gemm_1d1d(a_data, sfa, b_data, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void fp8_fp4_gemm_nn(const DGTensorView& a_data, const DGTensorView& a_sf,
                            const DGTensorView& b_data, const DGTensorView& b_sf,
                            const DGTensorView& d,
                            const std::optional<DGTensorView>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt(a_data, a_sf, b_data.transpose(0, 1), b_sf.transpose(0, 1),
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tn(const DGTensorView& a_data, const DGTensorView& a_sf,
                            const DGTensorView& b_data, const DGTensorView& b_sf,
                            const DGTensorView& d,
                            const std::optional<DGTensorView>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt(a_data.transpose(0, 1), a_sf.transpose(0, 1),
                    b_data.transpose(0, 1), b_sf.transpose(0, 1),
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tt(const DGTensorView& a_data, const DGTensorView& a_sf,
                            const DGTensorView& b_data, const DGTensorView& b_sf,
                            const DGTensorView& d,
                            const std::optional<DGTensorView>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt(a_data.transpose(0, 1), a_sf.transpose(0, 1), b_data, b_sf,
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(const DGTensorView& a_data, const DGTensorView& a_sf,
                                                 const DGTensorView& b_data, const DGTensorView& b_sf,
                                                 const DGTensorView& d,
                                                 const DGTensorView& grouped_layout,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const std::optional<int>& expected_m_for_psum_layout) {
    const auto& major_a = get_major_type_ab(a_data);
    const auto& major_b = get_major_type_ab(b_data);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a_data, major_a, arch_major);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b_data, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(grouped_layout.scalar_type(), dg_dtype::Int32));

    if (use_psum_layout) {
        const auto& [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto& [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    check_major_type_cd(d);

    if (m == 0)
        return;

    auto sf_result = layout::transform_sf_pair_into_required_layout(
        a_sf, b_sf, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, num_groups, disable_ue8m0_cast);
    const auto& sfa = sf_result.sfa.view;
    const auto& sfb = sf_result.sfb.view;
    const auto gran_k_a = sf_result.gran_k_a;
    const auto gran_k_b = sf_result.gran_k_b;

    if (arch_major == 9 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Float32)) {
        const auto& major_sfb = get_major_type_ab(sfb);
        DG_HOST_ASSERT(not use_psum_layout);
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a_data, sfa, b_data, sfb, d, grouped_layout,
                                                num_groups, m, n, k, major_a, major_b, major_sfb, compiled_dims);
    } else if (arch_major == 10 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Int32)) {
        sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a_data, sfa, b_data, sfb, d, grouped_layout,
                                                     num_groups, m, n, k, gran_k_a, gran_k_b, major_a, major_b,
                                                     compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void m_grouped_fp8_fp4_gemm_nn_contiguous(const DGTensorView& a_data, const DGTensorView& a_sf,
                                                 const DGTensorView& b_data, const DGTensorView& b_sf,
                                                 const DGTensorView& d,
                                                 const DGTensorView& grouped_layout,
                                                 const std::optional<std::tuple<int, int, int>>& recipe,
                                                 const std::optional<std::tuple<int, int>>& recipe_a,
                                                 const std::optional<std::tuple<int, int>>& recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout) {
    m_grouped_fp8_fp4_gemm_nt_contiguous(a_data, a_sf, b_data.transpose(1, 2), b_sf.transpose(1, 2),
                                         d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, std::nullopt);
}

static std::tuple<std::optional<int64_t>, std::optional<int64_t>> m_grouped_fp8_fp4_gemm_nt_masked(
                                             const DGTensorView& a_data, const DGTensorView& a_sf,
                                             const DGTensorView& b_data, const DGTensorView& b_sf,
                                             const DGTensorView& d,
                                             const DGTensorView& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast,
                                             const int& max_block_n,
                                             const bool& enable_overlap,
                                             const std::optional<DGTensorView>& signal) {
    const auto& major_a = get_major_type_ab(a_data);
    const auto& major_b = get_major_type_ab(b_data);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_groups  , m , k ] = check_grouped_ab_fp8_fp4(a_data, major_a, arch_major);
    const auto [num_groups_ , n , k_] = check_grouped_ab_fp8_fp4(b_data, major_b, arch_major);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(masked_m.scalar_type(), dg_dtype::Int32));

    check_major_type_cd(d);

    auto sf_result = layout::transform_sf_pair_into_required_layout(
        a_sf, b_sf, m, n, k, recipe, recipe_a, recipe_b, num_groups, num_groups, disable_ue8m0_cast);
    const auto& sfa = sf_result.sfa.view;
    const auto& sfb = sf_result.sfb.view;
    const auto gran_k_a = sf_result.gran_k_a;
    const auto gran_k_b = sf_result.gran_k_b;

    std::optional<std::pair<int, int>> overlap_result = std::nullopt;
    if (arch_major == 9 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Float32)) {
        const auto& major_sfb = get_major_type_ab(sfb);
        overlap_result = sm90_m_grouped_fp8_gemm_masked_1d2d(a_data, sfa, b_data, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, major_sfb, compiled_dims,
                                            max_block_n, enable_overlap, signal);
    } else if (arch_major == 10 and dg_dtype_eq(sfa.scalar_type(), dg_dtype::Int32)) {
        DG_HOST_ASSERT(not enable_overlap);
        sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(a_data, sfa, b_data, sfb, d, masked_m,
                                                 num_groups, m, n, k, expected_m, gran_k_a, gran_k_b,
                                                 major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }

    if (!overlap_result) {
        return std::make_tuple(std::nullopt, std::nullopt);
    }
    return std::make_tuple(
        std::optional<int64_t>(overlap_result->first),
        std::optional<int64_t>(overlap_result->second)
    );
}

static void k_grouped_fp8_gemm_tn_contiguous(const DGTensorView& a_data, const DGTensorView& a_sf,
                                             const DGTensorView& b_data, const DGTensorView& b_sf,
                                             const DGTensorView& d,
                                             const std::vector<int>& ks,
                                             const DGTensorView& ks_tensor,
                                             const std::optional<DGTensorView>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims) {
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));

    const auto& [num_groups, m, n] = get_shape<3>(d);
    const auto& [sum_k_ , m_] = get_shape<2>(a_data);
    const auto& [sum_k__, n_] = get_shape<2>(b_data);
    const int sum_k = std::accumulate(ks.begin(), ks.end(), 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    DG_HOST_ASSERT(a_data.is_contiguous());
    DG_HOST_ASSERT(b_data.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    if (early_return(m, n, std::accumulate(ks.begin(), ks.end(), 0), d, c))
        return;

    auto sfa_result = layout::transform_k_grouped_sf_into_required_layout(a_sf, ks, ks_tensor, recipe);
    auto sfb_result = layout::transform_k_grouped_sf_into_required_layout(b_sf, ks, ks_tensor, recipe);
    const auto& sfa = sfa_result.view;
    const auto& sfb = sfb_result.view;

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_k_grouped_fp8_gemm_1d1d(a_data, sfa, b_data, sfb, c, d, m, n, ks, ks_tensor,
                                      cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_fp8_gemm_nt_contiguous(const DGTensorView& a_data, const DGTensorView& a_sf,
                                             const DGTensorView& b_data, const DGTensorView& b_sf,
                                             const DGTensorView& d,
                                             const std::vector<int>& ks,
                                             const DGTensorView& ks_tensor,
                                             const std::optional<DGTensorView>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims) {
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));

    const auto& [num_groups, m, n] = get_shape<3>(d);
    const auto& sum_mk = a_data.numel();
    const auto& sum_nk = b_data.numel();
    const int sum_k = std::accumulate(ks.begin(), ks.end(), 0);
    DG_HOST_ASSERT(sum_mk == static_cast<int64_t>(sum_k) * m);
    DG_HOST_ASSERT(sum_nk == static_cast<int64_t>(sum_k) * n);

    DG_HOST_ASSERT(a_data.is_contiguous());
    DG_HOST_ASSERT(b_data.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    if (early_return(m, n, std::accumulate(ks.begin(), ks.end(), 0), d, c))
        return;

    auto sfa_result = layout::transform_k_grouped_sf_into_required_layout(a_sf, ks, ks_tensor, recipe);
    auto sfb_result = layout::transform_k_grouped_sf_into_required_layout(b_sf, ks, ks_tensor, recipe);
    const auto& sfa = sfa_result.view;
    const auto& sfb = sfb_result.view;

    // Allocate tensormap buffer (double buffering for both A and B: 2 * 2 = 4)
    const auto& num_sms = device_runtime->get_num_sms();
    const size_t tm_buf_size = static_cast<size_t>(num_sms) * 4 * sizeof(CUtensorMap);
    DGBuffer tm_buf(tm_buf_size);
    DGTensorView tensor_map_buffer = DGTensorView::from_ptr(
        tm_buf.ptr, dg_dtype::UInt8, {static_cast<int64_t>(tm_buf_size)}, a_data.device_id_val());

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_k_grouped_fp8_gemm_1d1d(a_data, sfa, b_data, sfb, c, d, m, n, ks, ks_tensor, tensor_map_buffer,
                                     cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

#if DG_TENSORMAP_COMPATIBLE
static void bf16_gemm_nt(const DGTensorView& a,
                         const DGTensorView& b,
                         const DGTensorView& d,
                         const std::optional<DGTensorView>& c,
                         const std::string& compiled_dims) {
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);

    check_major_type_cd(d);

    const auto& [m , k ] = get_shape<2>(a);
    const auto& [n , k_] = get_shape<2>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(dg_dtype_eq(a.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(b.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16) or dg_dtype_eq(d.scalar_type(), dg_dtype::Float32));

    if (early_return(m, n, k, d, c))
        return;

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nn(const DGTensorView& a,
                         const DGTensorView& b,
                         const DGTensorView& d,
                         const std::optional<DGTensorView>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tn(const DGTensorView& a,
                         const DGTensorView& b,
                         const DGTensorView& d,
                         const std::optional<DGTensorView>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tt(const DGTensorView& a,
                         const DGTensorView& b,
                         const DGTensorView& d,
                         const std::optional<DGTensorView>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const DGTensorView& a, const DGTensorView& b,
                                              const DGTensorView& d, const DGTensorView& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const std::optional<int>& expected_m_for_psum_layout) {
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    const auto& [m, k] = get_shape<2>(a);
    const auto& [num_groups, n, k_] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(dg_dtype_eq(a.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(b.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(grouped_layout.scalar_type(), dg_dtype::Int32));

    if (use_psum_layout) {
        const auto& [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto& [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    check_major_type_cd(d);

    if (m == 0)
        return;

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(not use_psum_layout);
        sm90_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void m_grouped_bf16_gemm_nn_contiguous(const DGTensorView& a, const DGTensorView& b,
                                              const DGTensorView& d, const DGTensorView& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout) {
    m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2),
                                      d, grouped_layout, compiled_dims, use_psum_layout, std::nullopt);
}

static void m_grouped_bf16_gemm_nt_masked(const DGTensorView& a, const DGTensorView& b,
                                          const DGTensorView& d, const DGTensorView& masked_m,
                                          const int& expected_m, const std::string& compiled_dims) {
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    const auto& [num_groups, m, k] = get_shape<3>(a);
    const auto& [num_groups_, n, k_] = get_shape<3>(b);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    const auto& num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(dg_dtype_eq(a.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(b.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(d.scalar_type(), dg_dtype::BFloat16));
    DG_HOST_ASSERT(dg_dtype_eq(masked_m.scalar_type(), dg_dtype::Int32));

    check_major_type_cd(d);

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_m_grouped_gemm_masked(a, b, d, masked_m,
                                        num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_masked(a, b, d, masked_m,
                                         num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_bf16_gemm_tn_contiguous(const DGTensorView& a,
                                              const DGTensorView& b,
                                              const DGTensorView& d,
                                              const std::vector<int>& ks,
                                              const DGTensorView& ks_tensor,
                                              const std::optional<DGTensorView>& c,
                                              const std::string& compiled_dims) {
    const auto& [num_groups, m, n] = get_shape<3>(d);
    const auto& [sum_k_ , m_] = get_shape<2>(a);
    const auto& [sum_k__, n_] = get_shape<2>(b);
    const int sum_k = std::accumulate(ks.begin(), ks.end(), 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    if (early_return(m, n, std::accumulate(ks.begin(), ks.end(), 0), d, c))
        return;

    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_k_grouped_gemm(a, b, c, d, m, n, ks, ks_tensor,
                                 cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_k_grouped_gemm(a, b, c, d, m, n, ks, ks_tensor,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

static void cublaslt_gemm_nt(const DGTensorView& a, const DGTensorView& b,
                             const DGTensorView& d, const std::optional<DGTensorView>& c) {
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);

    const auto& [m , k ] = get_shape<2>(a);
    const auto& [n , k_] = get_shape<2>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);

    if (early_return(m, n, k, d, c))
        return;

    cublaslt_gemm(a, b, c, d, m, n, k, major_a, major_b);
}

static void cublaslt_gemm_nn(const DGTensorView& a, const DGTensorView& b,
                             const DGTensorView& d, const std::optional<DGTensorView>& c) {
    cublaslt_gemm_nt(a, b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tn(const DGTensorView& a, const DGTensorView& b,
                             const DGTensorView& d, const std::optional<DGTensorView>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tt(const DGTensorView& a, const DGTensorView& b,
                             const DGTensorView& d, const std::optional<DGTensorView>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b, d, c);
}

} // namespace deep_gemm::gemm
