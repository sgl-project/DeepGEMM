#pragma once

#include <cstdio>
#include <format>
#include <optional>
#include <torch/torch.h>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm100_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& d,
                                       const torch::Tensor& sqr_sum,
                                       const int& m, const int& n, const int& k,
                                       const int& num_splits,
                                       const std::optional<torch::Tensor>& b2 = std::nullopt,
                                       const std::optional<torch::Tensor>& d2 = std::nullopt) {
    DG_HOST_ASSERT(b2.has_value() == d2.has_value());
    const int num_b = b2.has_value() ? 2 : 1;
    const int block_m = deep_jit::get_env<int>("DG_HC_PRENORM_BLOCK_M", m <= 1024 ? 128 : 64);
    DG_HOST_ASSERT(block_m == 64 or block_m == 128);
    constexpr int block_k = 64;
    constexpr int num_mma_threads = 128;
    constexpr int num_cast_and_reduce_threads = 128;

    const int block_n = align(n, 16);
    DG_HOST_ASSERT(n <= block_n);
    DG_HOST_ASSERT(n <= 128 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    const auto swizzle_cd_mode = get_swizzle_mode(block_n, sizeof(float));
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              block_m, block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, a.element_size()), 0,
                                              true);
    const auto make_b_desc = [&](const torch::Tensor& t) {
        return make_tma_b_desc(cute::UMMA::Major::K, t, n, k,
                               block_n, block_k,
                               static_cast<int>(t.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                               get_swizzle_mode(block_k, t.element_size()), 0,
                               true);
    };
    const auto make_d_desc = [&](const torch::Tensor& t) {
        return num_splits == 1 ? make_tma_cd_desc(t, m, n,
                                                  block_m, block_n,
                                                  static_cast<int>(t.stride(-2)), 1,
                                                  swizzle_cd_mode)
                               : make_tma_3d_desc(t, n, m, num_splits,
                                                  block_n, block_m, 1,
                                                  static_cast<int>(t.stride(-2)),
                                                  static_cast<int>(t.stride(-3)),
                                                  swizzle_cd_mode);
    };
    const auto tensor_map_b = make_b_desc(b);
    const auto tensor_map_d = make_d_desc(d);
    const auto tensor_map_b2 = num_b == 2 ? make_b_desc(b2.value()) : tensor_map_b;
    const auto tensor_map_d2 = num_b == 2 ? make_d_desc(d2.value()) : tensor_map_d;

    // Calculate stages
    int num_stages = 12, smem_size = 0;
    while (num_stages > 0) {
        const int smem_a_per_stage = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
        const int smem_b_per_stage = num_b * block_n * block_k * static_cast<int>(sizeof(float));
        const int smem_cd = num_b * block_m * swizzle_cd_mode;
        const int smem_barriers = (num_stages * 4 + 1) * 8;
        const int smem_tmem_ptr = 4;
        smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages +
                    smem_cd + smem_barriers + smem_tmem_ptr;

        if (smem_size <= SM100ArchSpec::smem_capacity)
            break;
        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split K: %d"
               "stages: %d, shared memory: %d, swizzle CD: %d\n",
               m, n, k, block_m, block_n, block_k, num_splits,
               num_stages, smem_size, swizzle_cd_mode);
    }

    // Compile
    const auto kernel = jit->compile("sm100_tf32_hc_prenorm_gemm", std::format(R"(
#include <deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_tf32_hc_prenorm_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {},
        {}
    >);
}};
)",
        n, k,
        block_m, block_n, block_k,
        num_splits,
        swizzle_cd_mode,
        num_stages,
        num_mma_threads, num_cast_and_reduce_threads,
        num_b));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_splits * ceil_div(m, block_m), 1, 1),
            .block_dim = dim3(num_mma_threads + num_cast_and_reduce_threads, 1, 1),
        },
        m, tensor_map_a, tensor_map_b, tensor_map_d, sqr_sum.data_ptr<float>(),
        tensor_map_b2, tensor_map_d2
    );
}

} // namespace deep_gemm
