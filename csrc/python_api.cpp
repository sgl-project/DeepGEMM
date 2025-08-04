#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <string>
#include <optional>
#include <tuple>
#include <numeric> // For std::accumulate
#include <Python.h>

// Assuming these are your project headers and they do not include pybind11
#include "jit/compiler.hpp"
#include "jit/device_runtime.hpp"
#include "utils/layout.hpp"
#include "jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "jit_kernels/impls/sm100_fp8_gemm_1d2d.hpp"
#include "jit_kernels/impls/smxx_layout.hpp"

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

// ----------------------------------------------------------------------------
// Section 1: Core C++ Logic (Your original code)
// ----------------------------------------------------------------------------
// This part of the code remains unchanged as it is a pure C++ implementation.
namespace deep_gemm {
torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                const int& mn, const int& k,
                                                const std::tuple<int, int, int>& recipe,
                                                const std::optional<int>& num_groups,
                                                const bool& is_sfa,
                                                const bool& disable_ue8m0_cast) {
    const auto& gran_mn = is_sfa ? std::get<0>(recipe) : std::get<1>(recipe);
    const auto& gran_k = std::get<2>(recipe);
    const auto& arch_major = device_runtime->get_arch_major();

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(sf);
    }

    // (FP32, 128, 128) on SM90: no need to transform, check shape and contiguous
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto& broadcasted = sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(128));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
    }

    // (INT, 1, 128) on SM100: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and gran_k == 128 and arch_major == 10)
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                          const std::vector<int>& ks,
                                                          const torch::Tensor& ks_tensor,
                                                          const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    const auto& arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        DG_HOST_UNREACHABLE("Unimplemented");

    // FP32 on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks);

    // INT on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 std::optional<std::tuple<int, int, int>> recipe,
                 const std::string& compiled_dims,
                 const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check C as well
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    }

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = transform_sf_into_required_layout(b.second, n, k, recipe.value(), std::nullopt, false, disable_ue8m0_cast);

    // Dispatch into different implements
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

void fp8_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::string& compiled_dims,
                 const bool& disable_ue8m0_cast) {
    fp8_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

void fp8_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::string& compiled_dims,
                 const bool& disable_ue8m0_cast) {
    fp8_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)},
                {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

void fp8_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::string& compiled_dims,
                 const bool& disable_ue8m0_cast) {
    fp8_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

void m_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const torch::Tensor& m_indices,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [num_groups, n, k_] = get_shape<3>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = transform_sf_into_required_layout(b.second, n, k, recipe.value(),   num_groups, false, disable_ue8m0_cast);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, m_indices,
                                                num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, m_indices,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, m_indices,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

void m_grouped_fp8_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const torch::Tensor& m_indices,
                                      const std::optional<std::tuple<int, int, int>>& recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    m_grouped_fp8_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                     d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
}

void fp8_m_grouped_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                  const torch::Tensor& d,
                                  const torch::Tensor& masked_m,
                                  const int& expected_m,
                                  std::optional<std::tuple<int, int, int>> recipe,
                                  const std::string& compiled_dims,
                                  const bool& disable_ue8m0_cast) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto& [num_groups, m, k] = get_shape<3>(a.first);
    const auto& [num_groups_, n, k_] = get_shape<3>(b.first);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    const auto& num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Transform scaling factors
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups,  true, disable_ue8m0_cast);
    const auto& sfb = transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_fp8_m_grouped_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_m_grouped_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                             num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_fp8_m_grouped_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                             num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const std::vector<int>& ks,
                                      const torch::Tensor& ks_tensor,
                                      const std::optional<torch::Tensor>& c,
                                      const std::tuple<int, int, int>& recipe,
                                      const std::string& compiled_dims) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    if (c.has_value()) {
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().is_contiguous());
    }

    // Do nothing if empty
    if (std::accumulate(ks.begin(), ks.end(), 0) == 0)
        return;

    // Transform SF with padding
    const auto& [_, m] = get_shape<2>(a.first);
    const auto& [__, n] = get_shape<2>(b.first);
    const auto& sfa = transform_k_grouped_sf_into_required_layout(a.second, ks, ks_tensor, recipe);
    const auto& sfb = transform_k_grouped_sf_into_required_layout(b.second, ks, ks_tensor, recipe);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        fp8_k_grouped_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks, ks_tensor,
                                cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

} // namespace deep_gemm


// ----------------------------------------------------------------------------
// Section 2: Wrapper Functions (Adapter Layer)
// ----------------------------------------------------------------------------
// This namespace serves as an adapter between TORCH_LIBRARY and the core C++ logic.
// This part is well-designed. We just need to ensure all integers from TORCH_LIBRARY are handled as int64_t.
namespace deep_gemm_wrappers {

// --- Helper functions for recipe conversion ---
inline std::optional<std::tuple<int, int, int>> to_recipe_tuple(c10::optional<at::IntArrayRef> recipe_opt) {
    if (!recipe_opt.has_value()) {
        return std::nullopt;
    }
    auto recipe_ref = recipe_opt.value();
    TORCH_CHECK(recipe_ref.size() == 3, "Recipe must be a list/tuple of 3 integers.");
    return std::make_tuple(static_cast<int>(recipe_ref[0]), static_cast<int>(recipe_ref[1]), static_cast<int>(recipe_ref[2]));
}

inline std::tuple<int, int, int> to_recipe_tuple_default(at::IntArrayRef recipe_ref) {
    TORCH_CHECK(recipe_ref.size() == 3, "Recipe must be a list/tuple of 3 integers.");
    return std::make_tuple(static_cast<int>(recipe_ref[0]), static_cast<int>(recipe_ref[1]), static_cast<int>(recipe_ref[2]));
}

// --- Wrappers for GEMM functions ---
void fp8_gemm_nt_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const c10::optional<at::Tensor>& c,
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::fp8_gemm_nt(
        {a_val, a_scale}, {b_val, b_scale}, d, c,
        to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast
    );
}

void fp8_gemm_nn_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const c10::optional<at::Tensor>& c,
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::fp8_gemm_nn(
        {a_val, a_scale}, {b_val, b_scale}, d, c,
        to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast
    );
}

void fp8_gemm_tn_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const c10::optional<at::Tensor>& c,
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::fp8_gemm_tn(
        {a_val, a_scale}, {b_val, b_scale}, d, c,
        to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast
    );
}

void fp8_gemm_tt_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const c10::optional<at::Tensor>& c,
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::fp8_gemm_tt(
        {a_val, a_scale}, {b_val, b_scale}, d, c,
        to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast
    );
}

// --- Wrappers for M-Grouped GEMM functions ---
void m_grouped_fp8_gemm_nt_contiguous_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const at::Tensor& m_indices,
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::m_grouped_fp8_gemm_nt_contiguous(
        {a_val, a_scale}, {b_val, b_scale}, d, m_indices,
        to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast
    );
}

void m_grouped_fp8_gemm_nn_contiguous_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const at::Tensor& m_indices,
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::m_grouped_fp8_gemm_nn_contiguous(
        {a_val, a_scale}, {b_val, b_scale}, d, m_indices,
        to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast
    );
}

void fp8_m_grouped_gemm_nt_masked_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    const at::Tensor& masked_m,
    int64_t expected_m, // Accepts int64_t from PyTorch
    c10::optional<at::IntArrayRef> recipe,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast)
{
    deep_gemm::fp8_m_grouped_gemm_nt_masked(
        {a_val, a_scale}, {b_val, b_scale}, d, masked_m,
        static_cast<int>(expected_m), // Cast to int for the core C++ function
        to_recipe_tuple(recipe),
        compiled_dims, disable_ue8m0_cast
    );
}

// --- Wrapper for K-Grouped GEMM ---
void k_grouped_fp8_gemm_tn_contiguous_wrapper(
    const at::Tensor& a_val, const at::Tensor& a_scale,
    const at::Tensor& b_val, const at::Tensor& b_scale,
    const at::Tensor& d,
    at::IntArrayRef ks,
    const at::Tensor& ks_tensor,
    const c10::optional<at::Tensor>& c,
    at::IntArrayRef recipe,
    const std::string& compiled_dims)
{
    std::vector<int> ks_vec;
    ks_vec.reserve(ks.size());
    for(const auto& val : ks) {
        ks_vec.push_back(static_cast<int>(val));
    }

    deep_gemm::k_grouped_fp8_gemm_tn_contiguous(
        {a_val, a_scale}, {b_val, b_scale}, d,
        ks_vec, ks_tensor, c,
        to_recipe_tuple_default(recipe), compiled_dims
    );
}

// --- Wrapper for Layout Transformation ---
torch::Tensor transform_sf_into_required_layout_wrapper(
    const torch::Tensor& sf,
    int64_t mn, int64_t k, // Accepts int64_t
    at::IntArrayRef recipe,
    const c10::optional<int64_t>& num_groups, // Accepts optional<int64_t>
    bool is_sfa,
    bool disable_ue8m0_cast)
{
    c10::optional<int> num_groups_int;
    if (num_groups.has_value()) {
        num_groups_int = static_cast<int>(num_groups.value());
    }

    return deep_gemm::transform_sf_into_required_layout(
        sf, static_cast<int>(mn), static_cast<int>(k), // Cast to int
        to_recipe_tuple_default(recipe),
        num_groups_int, is_sfa, disable_ue8m0_cast
    );
}

} // namespace deep_gemm_wrappers


// ----------------------------------------------------------------------------
// Section 3: TORCH_LIBRARY Definition
// ----------------------------------------------------------------------------
// This is the new binding interface, which uses the wrapper functions above
// and fixes all integer type issues.
TORCH_LIBRARY(deep_gemm_cpp, m) {
    // --- Runtime Functions ---
    m.def("set_num_sms(int new_num_sms) -> ()");
    m.impl("set_num_sms", [](int64_t new_num_sms) {
        deep_gemm::device_runtime->set_num_sms(static_cast<int>(new_num_sms));
    });

    m.def("get_num_sms() -> int");
    m.impl("get_num_sms", []() -> int64_t {
        return static_cast<int64_t>(deep_gemm::device_runtime->get_num_sms());
    });

    m.def("set_tc_util(int new_tc_util) -> ()");
    m.impl("set_tc_util", [](int64_t new_tc_util) {
        deep_gemm::device_runtime->set_tc_util(static_cast<int>(new_tc_util));
    });

    m.def("get_tc_util() -> int");
    m.impl("get_tc_util", []() -> int64_t {
        return static_cast<int64_t>(deep_gemm::device_runtime->get_tc_util());
    });

    // --- JIT Function ---
    m.def("init(str library_root_path, str cuda_home_path_by_torch) -> ()");
    m.impl("init", [](const std::string& library_root_path, const std::string& cuda_home_path_by_torch) {
        deep_gemm::Compiler::prepare_init(library_root_path, cuda_home_path_by_torch);
        deep_gemm::KernelRuntime::prepare_init(cuda_home_path_by_torch);
    });

    // --- Stable Kernel APIs ---
    m.def("fp8_gemm_nt(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims=\"nk\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("fp8_gemm_nt", torch::kCUDA, &deep_gemm_wrappers::fp8_gemm_nt_wrapper);

    m.def("fp8_gemm_nn(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims=\"nk\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("fp8_gemm_nn", torch::kCUDA, &deep_gemm_wrappers::fp8_gemm_nn_wrapper);

    m.def("fp8_gemm_tn(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims=\"mn\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("fp8_gemm_tn", torch::kCUDA, &deep_gemm_wrappers::fp8_gemm_tn_wrapper);

    m.def("fp8_gemm_tt(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims=\"mn\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("fp8_gemm_tt", torch::kCUDA, &deep_gemm_wrappers::fp8_gemm_tt_wrapper);

    m.def("m_grouped_fp8_gemm_nt_contiguous(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor m_indices, int[]? recipe=None, str compiled_dims=\"nk\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("m_grouped_fp8_gemm_nt_contiguous", torch::kCUDA, &deep_gemm_wrappers::m_grouped_fp8_gemm_nt_contiguous_wrapper);

    m.def("m_grouped_fp8_gemm_nn_contiguous(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor m_indices, int[]? recipe=None, str compiled_dims=\"nk\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("m_grouped_fp8_gemm_nn_contiguous", torch::kCUDA, &deep_gemm_wrappers::m_grouped_fp8_gemm_nn_contiguous_wrapper);

    m.def("fp8_m_grouped_gemm_nt_masked(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, Tensor masked_m, int expected_m, int[]? recipe=None, str compiled_dims=\"nk\", bool disable_ue8m0_cast=False) -> ()");
    m.impl("fp8_m_grouped_gemm_nt_masked", torch::kCUDA, &deep_gemm_wrappers::fp8_m_grouped_gemm_nt_masked_wrapper);

    m.def("k_grouped_fp8_gemm_tn_contiguous(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor d, int[] ks, Tensor ks_tensor, Tensor? c=None, int[] recipe=[1, 1, 128], str compiled_dims=\"mn\") -> ()");
    m.impl("k_grouped_fp8_gemm_tn_contiguous", torch::kCUDA, &deep_gemm_wrappers::k_grouped_fp8_gemm_tn_contiguous_wrapper);

    // --- Layout Kernels ---
    m.def("transform_sf_into_required_layout(Tensor sf, int mn, int k, int[] recipe, int? num_groups=None, bool is_sfa=False, bool disable_ue8m0_cast=False) -> Tensor");
    m.impl("transform_sf_into_required_layout", torch::kCUDA, &deep_gemm_wrappers::transform_sf_into_required_layout_wrapper);

    // --- Raw Kernels or Utility Functions ---
    // [FIX 2] Add lambda wrappers for all functions using native int to ensure compatibility.
    m.def("get_tma_aligned_size(int size, int element_size) -> int");
    m.impl("get_tma_aligned_size", [](int64_t size, int64_t element_size) -> int64_t {
        return static_cast<int64_t>(
            deep_gemm::get_tma_aligned_size(static_cast<int>(size), static_cast<int>(element_size))
        );
    });

    m.def("get_mk_alignment_for_contiguous_layout() -> int");
    m.impl("get_mk_alignment_for_contiguous_layout", []() -> int64_t {
        return static_cast<int64_t>(
            deep_gemm::get_mk_alignment_for_contiguous_layout()
        );
    });

    m.def("get_mn_major_tma_aligned_tensor(Tensor a) -> Tensor");
    m.impl("get_mn_major_tma_aligned_tensor", torch::kCUDA, &deep_gemm::get_mn_major_tma_aligned_tensor);
    
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor a) -> Tensor");
    m.impl("get_mn_major_tma_aligned_packed_ue8m0_tensor", torch::kCUDA, &deep_gemm::get_mn_major_tma_aligned_packed_ue8m0_tensor);
    
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor a, Tensor ks_tensor, int[] ks) -> Tensor");
    m.impl("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", torch::kCUDA, [](const at::Tensor& a, const at::Tensor& ks_tensor, at::IntArrayRef ks_ref) -> at::Tensor {
        std::vector<int> ks_vec(ks_ref.begin(), ks_ref.end());
        return deep_gemm::get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(a, ks_tensor, ks_vec);
    });
}

REGISTER_EXTENSION(deep_gemm_cpp)
