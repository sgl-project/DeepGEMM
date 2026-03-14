#pragma once

#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/smxx_layout.hpp"
#endif

namespace deep_gemm::layout {

struct TransformedSF {
    DGTensorView view;
    DGBuffer buf;
};

struct SFPairResult {
    TransformedSF sfa;
    TransformedSF sfb;
    int gran_k_a;
    int gran_k_b;
};

#if DG_TENSORMAP_COMPATIBLE
static TransformedSF transform_sf_into_required_layout(const DGTensorView& sf,
                                                       const int& mn, const int& k,
                                                       const std::optional<std::tuple<int, int, int>>& recipe,
                                                       const std::optional<std::tuple<int, int>>& recipe_ab,
                                                       const std::optional<int>& num_groups,
                                                       const bool& is_sfa,
                                                       const bool& disable_ue8m0_cast) {
    const auto& arch_major = device_runtime->get_arch_major();

    int gran_mn, gran_k;
    if (recipe.has_value()) {
        DG_HOST_ASSERT(not recipe_ab.has_value());
        gran_mn = is_sfa ? std::get<0>(recipe.value()) : std::get<1>(recipe.value());
        gran_k = std::get<2>(recipe.value());
    } else {
        DG_HOST_ASSERT(recipe_ab.has_value());
        std::tie(gran_mn, gran_k) = recipe_ab.value();
    }

    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Float32) and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast)) {
        const auto& [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf);
        const size_t out_size = static_cast<size_t>(ng) * sf_k_pp * tma_mn * sizeof(float);
        DGBuffer buf(out_size);
        DGTensorView out = dim == 2 ?
            DGTensorView::from_ptr_strided(buf.ptr, dg_dtype::Float32,
                {static_cast<int64_t>(mn_pp), static_cast<int64_t>(sf_k_pp)},
                {1LL, static_cast<int64_t>(tma_mn)}, sf.device_id_val()) :
            DGTensorView::from_ptr_strided(buf.ptr, dg_dtype::Float32,
                {static_cast<int64_t>(ng), static_cast<int64_t>(mn_pp), static_cast<int64_t>(sf_k_pp)},
                {static_cast<int64_t>(sf_k_pp) * tma_mn, 1LL, static_cast<int64_t>(tma_mn)}, sf.device_id_val());
        get_mn_major_tma_aligned_tensor(sf, out);
        return {out, std::move(buf)};
    }

    // (FP32, 128, 128) on SM90: no need to transform, check SFB requirements
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Float32) and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return {check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, dg_dtype::Float32), DGBuffer()};

    // (FP32, x, gran_k) on SM100: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Float32) and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        DG_HOST_ASSERT(gran_mn == 1 && "SM100 with gran_mn != 1 requires pre-broadcasting SF from Python");
        const auto& [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf);
        const auto packed_sf_k = ceil_div(sf_k_pp, 4);
        const auto tma_mn_int = get_tma_aligned_size(mn_pp, static_cast<int>(sizeof(int)));
        const size_t out_size = static_cast<size_t>(ng) * packed_sf_k * tma_mn_int * sizeof(int);
        DGBuffer buf(out_size);
        DGTensorView out = dim == 2 ?
            DGTensorView::from_ptr_strided(buf.ptr, dg_dtype::Int32,
                {static_cast<int64_t>(mn_pp), static_cast<int64_t>(packed_sf_k)},
                {1LL, static_cast<int64_t>(tma_mn_int)}, sf.device_id_val()) :
            DGTensorView::from_ptr_strided(buf.ptr, dg_dtype::Int32,
                {static_cast<int64_t>(ng), static_cast<int64_t>(mn_pp), static_cast<int64_t>(packed_sf_k)},
                {static_cast<int64_t>(packed_sf_k) * tma_mn_int, 1LL, static_cast<int64_t>(tma_mn_int)}, sf.device_id_val());
        get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, out);
        return {out, std::move(buf)};
    }

    // (INT, 1, gran_k) on SM100: transform to TMA-aligned and MN-major
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Int32) and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 10)
        return {check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, dg_dtype::Int32), DGBuffer()};

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static SFPairResult transform_sf_pair_into_required_layout(
        const DGTensorView& sfa, const DGTensorView& sfb,
        const int& m, const int& n, const int& k,
        std::optional<std::tuple<int, int, int>>& recipe,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b,
        const std::optional<int>& num_groups_a,
        const std::optional<int>& num_groups_b,
        const bool& disable_ue8m0_cast = false) {
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());
    auto transformed_sfa = transform_sf_into_required_layout(sfa, m, k, recipe, recipe_a, num_groups_a, true, disable_ue8m0_cast);
    auto transformed_sfb = transform_sf_into_required_layout(sfb, n, k, recipe, recipe_b, num_groups_b, false, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return {std::move(transformed_sfa), std::move(transformed_sfb), gran_k_a, gran_k_b};
}

static TransformedSF transform_k_grouped_sf_into_required_layout(const DGTensorView& sf,
                                                                  const std::vector<int>& ks,
                                                                  const DGTensorView& ks_tensor,
                                                                  const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    const auto& arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Float32) and arch_major == 9) {
        const auto& [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf);
        const size_t out_size = static_cast<size_t>(ng) * sf_k_pp * tma_mn * sizeof(float);
        DGBuffer buf(out_size);
        DGTensorView out = DGTensorView::from_ptr_strided(buf.ptr, dg_dtype::Float32,
            {static_cast<int64_t>(mn_pp), static_cast<int64_t>(sf_k_pp)},
            {1LL, static_cast<int64_t>(tma_mn)}, sf.device_id_val());
        get_mn_major_tma_aligned_tensor(sf, out);
        return {out, std::move(buf)};
    }

    // FP32 on SM100
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Float32) and arch_major == 10) {
        const auto& [sf_k, sf_mn] = get_shape<2>(sf);
        const auto& num_groups = static_cast<int>(ks.size());
        int packed_sf_k = 0;
        for (const auto& ki : ks)
            packed_sf_k += ceil_div(ki, 512);
        const auto tma_mn_int = get_tma_aligned_size(sf_mn, static_cast<int>(sizeof(int)));
        const size_t out_size = static_cast<size_t>(packed_sf_k) * tma_mn_int * sizeof(int);
        DGBuffer buf(out_size);
        DGTensorView out = DGTensorView::from_ptr_strided(buf.ptr, dg_dtype::Int32,
            {static_cast<int64_t>(sf_mn), static_cast<int64_t>(packed_sf_k)},
            {1LL, static_cast<int64_t>(tma_mn_int)}, sf.device_id_val());
        get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, out);
        return {out, std::move(buf)};
    }

    // INT on SM100
    if (dg_dtype_eq(sf.scalar_type(), dg_dtype::Int32) and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

#endif

} // namespace deep_gemm::layout
