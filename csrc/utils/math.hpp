#pragma once

// #include <torch/python.h>
#include <cstdint>

#include "exception.hpp"

namespace deep_gemm {

template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int64_t get_tma_aligned_size(const int64_t& x, const int64_t& element_size) {
    constexpr int64_t kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

} // namespace deep_gemm
