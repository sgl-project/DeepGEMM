#include <torch/library.h>
#include <torch/types.h>
#include <vector>
#include <string>
#include <optional>
#include <tuple>
#include <numeric>
#include <Python.h>

#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/runtime.hpp"

#include "jit/compiler.hpp"
#include "jit/device_runtime.hpp"
#include "jit/kernel_runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_gemm
#endif


#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

namespace {

std::optional<std::tuple<int, int, int>> to_recipe_tuple(const c10::optional<c10::IntArrayRef>& recipe_opt) {
    if (!recipe_opt.has_value()) {
        return std::nullopt;
    }
    auto recipe_ref = recipe_opt.value();
    TORCH_CHECK(recipe_ref.size() == 3, "Recipe must be a list/tuple of 3 integers.");
    return std::make_tuple(static_cast<int>(recipe_ref[0]), static_cast<int>(recipe_ref[1]), static_cast<int>(recipe_ref[2]));
}

std::tuple<int, int, int> to_recipe_tuple_default(c10::IntArrayRef recipe_ref) {
    TORCH_CHECK(recipe_ref.size() == 3, "Recipe must be a list/tuple of 3 integers.");
    return std::make_tuple(static_cast<int>(recipe_ref[0]), static_cast<int>(recipe_ref[1]), static_cast<int>(recipe_ref[2]));
}

} // anonymous namespace

namespace deep_gemm_wrappers {

// Runtime wrappers
void set_num_sms_wrapper(int64_t new_num_sms) {
    deep_gemm::device_runtime->set_num_sms(new_num_sms);
}

int64_t get_num_sms_wrapper() {
    return deep_gemm::device_runtime->get_num_sms();
}

void set_tc_util_wrapper(int64_t new_tc_util) {
    deep_gemm::device_runtime->set_tc_util(new_tc_util);
}

int64_t get_tc_util_wrapper() {
    return deep_gemm::device_runtime->get_tc_util();
}

void init_wrapper(const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
    deep_gemm::Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
    deep_gemm::KernelRuntime::prepare_init(cuda_home_path_by_python);
}

// Scalar layout utility wrappers (int64_t signatures for PyTorch registration)
int64_t get_tma_aligned_size_wrapper(int64_t x, int64_t element_size);
int64_t get_mk_alignment_for_contiguous_layout_wrapper();

// Layout wrappers
torch::Tensor transform_sf_into_required_layout_wrapper(const torch::Tensor& sf, int64_t mn, int64_t k, c10::IntArrayRef recipe, const c10::optional<int64_t>& num_groups, bool is_sfa, bool disable_ue8m0_cast) {
    return deep_gemm::layout::transform_sf_into_required_layout(sf, mn, k, to_recipe_tuple_default(recipe), num_groups, is_sfa, disable_ue8m0_cast);
}

torch::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_wrapper(const torch::Tensor& sf, const torch::Tensor& ks_tensor, c10::List<int64_t> ks) {
    std::vector<int> ks_vec;
    ks_vec.reserve(ks.size());
    for (const auto& k_val : ks) {
        ks_vec.push_back(k_val);
    }
    return deep_gemm::get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks_vec);
}

// GEMM wrappers
void fp8_gemm_nt_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::fp8_gemm_nt({a_val, a_scale}, {b_val, b_scale}, d, c, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void fp8_gemm_nn_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::fp8_gemm_nn({a_val, a_scale}, {b_val, b_scale}, d, c, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void fp8_gemm_tn_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::fp8_gemm_tn({a_val, a_scale}, {b_val, b_scale}, d, c, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void fp8_gemm_tt_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::fp8_gemm_tt({a_val, a_scale}, {b_val, b_scale}, d, c, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void m_grouped_fp8_gemm_nt_contiguous_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const torch::Tensor& m_indices, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::m_grouped_fp8_gemm_nt_contiguous({a_val, a_scale}, {b_val, b_scale}, d, m_indices, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void m_grouped_fp8_gemm_nn_contiguous_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const torch::Tensor& m_indices, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::m_grouped_fp8_gemm_nn_contiguous({a_val, a_scale}, {b_val, b_scale}, d, m_indices, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void m_grouped_fp8_gemm_nt_masked_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const torch::Tensor& masked_m, int64_t expected_m, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::gemm::m_grouped_fp8_gemm_nt_masked({a_val, a_scale}, {b_val, b_scale}, d, masked_m, expected_m, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

void k_grouped_fp8_gemm_tn_contiguous_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, c10::List<int64_t> ks, const torch::Tensor& ks_tensor, const c10::optional<torch::Tensor>& c, c10::IntArrayRef recipe, const std::string& compiled_dims) {
    std::vector<int> ks_vec;
    ks_vec.reserve(ks.size());
    for(const auto i : ks) {
        ks_vec.push_back(i);
    }
    deep_gemm::gemm::k_grouped_fp8_gemm_tn_contiguous({a_val, a_scale}, {b_val, b_scale}, d, ks_vec, ks_tensor, c, to_recipe_tuple_default(recipe), compiled_dims);
}

void bf16_gemm_nt_wrapper(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_nt(a, b, d, c, compiled_dims);
}

void bf16_gemm_nn_wrapper(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_nn(a, b, d, c, compiled_dims);
}

void bf16_gemm_tn_wrapper(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_tn(a, b, d, c, compiled_dims);
}

void bf16_gemm_tt_wrapper(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_tt(a, b, d, c, compiled_dims);
}

void m_grouped_bf16_gemm_nt_contiguous_wrapper(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const torch::Tensor& m_indices, const std::string& compiled_dims) {
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, compiled_dims);
}

void m_grouped_bf16_gemm_nt_masked_wrapper(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const torch::Tensor& masked_m, int64_t expected_m, const std::string& compiled_dims) {
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims);
}

} // namespace deep_gemm_wrappers

TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
    // runtime APIs (explicit schema + impl for stable type behavior)
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

    m.def("init(str library_root_path, str cuda_home_path_by_torch) -> ()");
    m.impl("init", [](const std::string& library_root_path, const std::string& cuda_home_path_by_torch) {
        deep_gemm_wrappers::init_wrapper(library_root_path, cuda_home_path_by_torch);
    });

    // layout APIs
    m.def("transform_sf_into_required_layout(Tensor sf, int mn, int k, int[] recipe, int? num_groups=None, bool is_sfa=False, bool disable_ue8m0_cast=False) -> Tensor", deep_gemm_wrappers::transform_sf_into_required_layout_wrapper);

    m.def("get_tma_aligned_size(int size, int element_size) -> int");
    m.impl("get_tma_aligned_size", [](int64_t size, int64_t element_size) -> int64_t {
        return deep_gemm_wrappers::get_tma_aligned_size_wrapper(size, element_size);
    });

    m.def("get_mk_alignment_for_contiguous_layout() -> int");
    m.impl("get_mk_alignment_for_contiguous_layout", []() -> int64_t {
        return deep_gemm_wrappers::get_mk_alignment_for_contiguous_layout_wrapper();
    });
    m.def("get_mn_major_tma_aligned_tensor(Tensor a) -> Tensor");
    m.impl("get_mn_major_tma_aligned_tensor", [](const torch::Tensor& a) -> torch::Tensor {
        return deep_gemm::get_mn_major_tma_aligned_tensor(a);
    });

    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor a) -> Tensor");
    m.impl("get_mn_major_tma_aligned_packed_ue8m0_tensor", [](const torch::Tensor& a) -> torch::Tensor {
        return deep_gemm::get_mn_major_tma_aligned_packed_ue8m0_tensor(a);
    });

    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor a, Tensor ks_tensor, int[] ks) -> Tensor");
    m.impl("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", [](const torch::Tensor& a,
                                                                        const torch::Tensor& ks_tensor,
                                                                        at::IntArrayRef ks_ref) -> torch::Tensor {
        std::vector<int> ks_vec(ks_ref.begin(), ks_ref.end());
        return deep_gemm::get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(a, ks_tensor, ks_vec);
    });

    // gemm APIs (explicit schema + impl)
    m.def(R"(fp8_gemm_nt(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_nt", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                            const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        deep_gemm_wrappers::fp8_gemm_nt_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(fp8_gemm_nn(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_nn", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                            const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        deep_gemm_wrappers::fp8_gemm_nn_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(fp8_gemm_tn(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="mn", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_tn", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                            const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        deep_gemm_wrappers::fp8_gemm_tn_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(fp8_gemm_tt(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="mn", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_tt", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                            const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        deep_gemm_wrappers::fp8_gemm_tt_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(m_grouped_fp8_gemm_nt_contiguous(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor m_indices, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("m_grouped_fp8_gemm_nt_contiguous", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                                                 const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                                                 const torch::Tensor& d,
                                                                 const torch::Tensor& m_indices,
                                                                 const c10::optional<c10::IntArrayRef>& recipe,
                                                                 const std::string& compiled_dims,
                                                                 bool disable_ue8m0_cast) {
        deep_gemm_wrappers::m_grouped_fp8_gemm_nt_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(m_grouped_fp8_gemm_nn_contiguous(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor m_indices, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("m_grouped_fp8_gemm_nn_contiguous", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                                                 const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                                                 const torch::Tensor& d,
                                                                 const torch::Tensor& m_indices,
                                                                 const c10::optional<c10::IntArrayRef>& recipe,
                                                                 const std::string& compiled_dims,
                                                                 bool disable_ue8m0_cast) {
        deep_gemm_wrappers::m_grouped_fp8_gemm_nn_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(m_grouped_fp8_gemm_nt_masked(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, Tensor masked_m, int expected_m, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("m_grouped_fp8_gemm_nt_masked", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                                             const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                                             const torch::Tensor& d,
                                                             const torch::Tensor& masked_m,
                                                             int64_t expected_m,
                                                             const c10::optional<c10::IntArrayRef>& recipe,
                                                             const std::string& compiled_dims,
                                                             bool disable_ue8m0_cast) {
        deep_gemm_wrappers::m_grouped_fp8_gemm_nt_masked_wrapper(a_val, a_scale, b_val, b_scale, d, masked_m, expected_m, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(k_grouped_fp8_gemm_tn_contiguous(Tensor a_val, Tensor a_scale, Tensor b_val, Tensor b_scale, Tensor d, int[] ks, Tensor ks_tensor, Tensor? c, int[] recipe, str compiled_dims) -> ())");
    m.impl("k_grouped_fp8_gemm_tn_contiguous", torch::kCUDA, [](const torch::Tensor& a_val, const torch::Tensor& a_scale,
                                                                 const torch::Tensor& b_val, const torch::Tensor& b_scale,
                                                                 const torch::Tensor& d,
                                                                 at::IntArrayRef ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const c10::optional<torch::Tensor>& c,
                                                                 c10::IntArrayRef recipe,
                                                                 const std::string& compiled_dims) {
        std::vector<int64_t> ks64(ks.begin(), ks.end());
        c10::List<int64_t> ks_list(ks64);
        deep_gemm_wrappers::k_grouped_fp8_gemm_tn_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, ks_list, ks_tensor, c, recipe, compiled_dims);
    });

    m.def(R"(bf16_gemm_nt(Tensor a, Tensor b, Tensor d, Tensor? c=None, str compiled_dims="") -> ())");
    m.impl("bf16_gemm_nt", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c,
                                             const std::string& compiled_dims) {
        deep_gemm_wrappers::bf16_gemm_nt_wrapper(a, b, d, c, compiled_dims);
    });

    m.def(R"(bf16_gemm_nn(Tensor a, Tensor b, Tensor d, Tensor? c=None, str compiled_dims="") -> ())");
    m.impl("bf16_gemm_nn", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c,
                                             const std::string& compiled_dims) {
        deep_gemm_wrappers::bf16_gemm_nn_wrapper(a, b, d, c, compiled_dims);
    });

    m.def(R"(bf16_gemm_tn(Tensor a, Tensor b, Tensor d, Tensor? c=None, str compiled_dims="") -> ())");
    m.impl("bf16_gemm_tn", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c,
                                             const std::string& compiled_dims) {
        deep_gemm_wrappers::bf16_gemm_tn_wrapper(a, b, d, c, compiled_dims);
    });

    m.def(R"(bf16_gemm_tt(Tensor a, Tensor b, Tensor d, Tensor? c=None, str compiled_dims="") -> ())");
    m.impl("bf16_gemm_tt", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c,
                                             const std::string& compiled_dims) {
        deep_gemm_wrappers::bf16_gemm_tt_wrapper(a, b, d, c, compiled_dims);
    });

    m.def(R"(m_grouped_bf16_gemm_nt_contiguous(Tensor a, Tensor b, Tensor d, Tensor m_indices, str compiled_dims="") -> ())");
    m.impl("m_grouped_bf16_gemm_nt_contiguous", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b,
                                                                  const torch::Tensor& d, const torch::Tensor& m_indices,
                                                                  const std::string& compiled_dims) {
        deep_gemm_wrappers::m_grouped_bf16_gemm_nt_contiguous_wrapper(a, b, d, m_indices, compiled_dims);
    });

    m.def(R"(m_grouped_bf16_gemm_nt_masked(Tensor a, Tensor b, Tensor d, Tensor masked_m, int expected_m, str compiled_dims="") -> ())");
    m.impl("m_grouped_bf16_gemm_nt_masked", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                                              const torch::Tensor& masked_m, int64_t expected_m,
                                                              const std::string& compiled_dims) {
        deep_gemm_wrappers::m_grouped_bf16_gemm_nt_masked_wrapper(a, b, d, masked_m, expected_m, compiled_dims);
    });
}

// Provide single definitions for the declared wrappers
int64_t deep_gemm_wrappers::get_tma_aligned_size_wrapper(int64_t x, int64_t element_size) {
    return static_cast<int64_t>(deep_gemm::get_tma_aligned_size(static_cast<int>(x), static_cast<int>(element_size)));
}

int64_t deep_gemm_wrappers::get_mk_alignment_for_contiguous_layout_wrapper() {
    return static_cast<int64_t>(deep_gemm::get_mk_alignment_for_contiguous_layout());
}

REGISTER_EXTENSION(deep_gemm_cpp)
