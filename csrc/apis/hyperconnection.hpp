#pragma once

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp"

namespace deep_gemm::hyperconnection {

static void tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 const std::optional<int>& num_splits) {
    // A and B must be K-major, D must be N-major
    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    check_major_type_cd(d);

    // S must be contiguous
    DG_HOST_ASSERT(sqr_sum.is_contiguous());

    // Type and shape checks
    const auto [m, k ] = get_shape<2>(a);
    const auto [n, k_] = get_shape<2>(b);
    if (num_splits.has_value()) {
        const auto [num_splits_, m_, n_] = get_shape<3>(d);
        const auto [num_splits__, m__] = get_shape<2>(sqr_sum);
        DG_HOST_ASSERT(num_splits.value() == num_splits_ and num_splits.value() == num_splits__ and num_splits.value() >= 1);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    } else {
        const auto [m_, n_] = get_shape<2>(d);
        const auto [m__] = get_shape<1>(sqr_sum);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    }
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Dispatch into different implements
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 12) {
        sm120_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 9) {
        sm90_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 10) {
        sm100_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

// One pass over `a` for two projections: d = a @ b^T and d2 = a @ b2^T, each bitwise equal to its own
// `tf32_hc_prenorm_gemm` call; `sqr_sum` is written once
static void tf32_hc_prenorm_gemm_dual(const torch::Tensor& a,
                                      const torch::Tensor& b,
                                      const torch::Tensor& b2,
                                      const torch::Tensor& d,
                                      const torch::Tensor& d2,
                                      const torch::Tensor& sqr_sum,
                                      const std::optional<int>& num_splits) {
    DG_HOST_ASSERT(b.sizes() == b2.sizes() and b.strides() == b2.strides() and b2.scalar_type() == b.scalar_type());
    DG_HOST_ASSERT(d.sizes() == d2.sizes() and d.strides() == d2.strides() and d2.scalar_type() == d.scalar_type());
    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    check_major_type_cd(d);
    DG_HOST_ASSERT(sqr_sum.is_contiguous());

    const auto [m, k ] = get_shape<2>(a);
    const auto [n, k_] = get_shape<2>(b);
    if (num_splits.has_value()) {
        const auto [num_splits_, m_, n_] = get_shape<3>(d);
        const auto [num_splits__, m__] = get_shape<2>(sqr_sum);
        DG_HOST_ASSERT(num_splits.value() == num_splits_ and num_splits.value() == num_splits__ and num_splits.value() >= 1);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    } else {
        const auto [m_, n_] = get_shape<2>(d);
        const auto [m__] = get_shape<1>(sqr_sum);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    }
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat);

    if (m == 0)
        return;

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and "tf32_hc_prenorm_gemm_dual is SM100-only");
    sm100_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1, b2, d2);
}


#ifndef DG_USE_TORCH_LIBRARY

static void register_apis(pybind11::module_& m) {
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
    m.def("tf32_hc_prenorm_gemm_dual", &tf32_hc_prenorm_gemm_dual,
          py::arg("a"), py::arg("b"), py::arg("b2"), py::arg("d"), py::arg("d2"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
}

#endif

} // namespace deep_gemm::hyperconnection
