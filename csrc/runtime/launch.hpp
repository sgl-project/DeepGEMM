#pragma once

#include <algorithm>
#include "runtime.hpp"

namespace deep_gemm {

// Construct native DeepJIT options for the downstream multidimensional launchers.
inline deep_jit::cuda::LaunchOptions make_launch_options(
    const dim3& grid, const int num_threads, const int smem_size = 0,
    const int cluster_size = 1, const std::optional<bool> enable_pdl = std::nullopt) {
    return {
        .num_smem_bytes = smem_size,
        .grid_dim = grid,
        .block_dim = dim3(num_threads, 1, 1),
        .cluster_dim = dim3(std::max(1, cluster_size), 1, 1),
        .enable_pdl = enable_pdl,
    };
}

}  // namespace deep_gemm
