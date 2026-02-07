#include <torch/library.h>
#include <torch/types.h>
#include <vector>
#include <string>
#include <optional>
#include <tuple>
#include <numeric>
#include <Python.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
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

// Accept Tensor, (Tensor, Tensor) tuple, or [Tensor, Tensor] list; return (tensor, scale)
std::pair<at::Tensor, at::Tensor> parse_tensor_or_tuple(const c10::IValue& input) {
    if (input.isTuple()) {
        auto tuple = input.toTuple();
        TORCH_CHECK(tuple->elements().size() >= 2, "Expected (Tensor, Tensor) tuple");
        return {tuple->elements()[0].toTensor(), tuple->elements()[1].toTensor()};
    } else if (input.isList()) {
        auto list = input.toList();
        TORCH_CHECK(list.size() >= 2, "Expected [Tensor, Tensor] list");
        return {list.get(0).toTensor(), list.get(1).toTensor()};
    } else if (input.isTensor()) {
        auto tensor = input.toTensor();
        auto scale = at::ones({1}, tensor.options().dtype(at::kFloat));
        return {tensor, scale};
    }
    TORCH_CHECK(false, "Expected Tensor, (Tensor, Tensor) tuple, or [Tensor, Tensor] list");
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

std::tuple<c10::optional<int64_t>, c10::optional<int64_t>> m_grouped_fp8_gemm_nt_masked_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, const torch::Tensor& masked_m, int64_t expected_m, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast, int64_t max_block_n, bool enable_overlap, const c10::optional<torch::Tensor>& signal) {
    auto result = deep_gemm::gemm::m_grouped_fp8_gemm_nt_masked({a_val, a_scale}, {b_val, b_scale}, d, masked_m, expected_m, to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast, max_block_n, enable_overlap, signal);

    if (!result) {
        return std::make_tuple(c10::nullopt, c10::nullopt);
    }
    return std::make_tuple(
        c10::optional<int64_t>(result->first),
        c10::optional<int64_t>(result->second)
    );
}

void k_grouped_fp8_gemm_nt_contiguous_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor& b_scale, const torch::Tensor& d, c10::List<int64_t> ks, const torch::Tensor& ks_tensor, const c10::optional<torch::Tensor>& c, c10::IntArrayRef recipe, const std::string& compiled_dims) {
    std::vector<int> ks_vec;
    ks_vec.reserve(ks.size());
    for(const auto i : ks) {
        ks_vec.push_back(i);
    }
    deep_gemm::gemm::k_grouped_fp8_gemm_nt_contiguous({a_val, a_scale}, {b_val, b_scale}, d, ks_vec, ks_tensor, c, to_recipe_tuple_default(recipe), compiled_dims);
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

// Attention wrappers
void fp8_gemm_nt_skip_head_mid_wrapper(const torch::Tensor& a_val, const torch::Tensor& a_scale, const torch::Tensor& b_val, const torch::Tensor b_scale, const torch::Tensor& d, const c10::IntArrayRef& head_splits, const c10::optional<c10::IntArrayRef>& recipe, const std::string& compiled_dims, bool disable_ue8m0_cast) {
    deep_gemm::attention::fp8_gemm_nt_skip_head_mid({a_val, a_scale}, {b_val, b_scale}, d, to_recipe_tuple_default(head_splits), to_recipe_tuple(recipe), compiled_dims, disable_ue8m0_cast);
}

torch::Tensor fp8_mqa_logits_wrapper(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& weight, const torch::Tensor& cu_seq_len_k_start, const torch::Tensor& cu_seq_len_k_end, bool clean_logits) {
    return deep_gemm::attention::fp8_mqa_logits(q, {k, v}, weight, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits);
}

torch::Tensor get_paged_mqa_logits_metadata_wrapper(const torch::Tensor& context_lens, int64_t block_kv, int64_t num_sms) {
    return deep_gemm::attention::get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms);
}

torch::Tensor fp8_paged_mqa_logits_wrapper(const torch::Tensor& q, const torch::Tensor& fused_kv_cache, const torch::Tensor& weight, const torch::Tensor& context_lens, const torch::Tensor& block_table, const torch::Tensor& schedule_meta, const int64_t max_context_len, bool clean_logits) {
    return deep_gemm::attention::fp8_paged_mqa_logits(q, fused_kv_cache, weight, context_lens, block_table, schedule_meta, max_context_len, clean_logits); 
}

} // namespace deep_gemm_wrappers

TORCH_LIBRARY(deep_gemm, m) {
    // runtime APIs (explicit schema + impl for stable type behavior)
    m.def("set_num_sms(int new_num_sms) -> ()");
    m.impl("set_num_sms", [](int64_t new_num_sms) {
        deep_gemm::device_runtime->set_num_sms(static_cast<int>(new_num_sms));
    });

    m.def("get_num_sms() -> int");
    m.impl("get_num_sms", []() -> int64_t {
        return static_cast<int64_t>(deep_gemm::device_runtime->get_num_sms());
    });

    m.def("set_compile_mode(int new_compile_mode) -> ()");
    m.impl("set_compile_mode", [](int64_t new_compile_mode) {
        deep_gemm::device_runtime->set_compile_mode(static_cast<int>(new_compile_mode));
    });

    m.def("get_compile_mode() -> int");
    m.impl("get_compile_mode", []() -> int64_t {
        return static_cast<int64_t>(deep_gemm::device_runtime->get_compile_mode());
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
    m.def(R"(fp8_gemm_nt(Any a, Any b, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_nt", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::fp8_gemm_nt_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(fp8_gemm_nn(Any a, Any b, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_nn", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::fp8_gemm_nn_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(fp8_gemm_tn(Any a, Any b, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="mn", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_tn", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::fp8_gemm_tn_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(fp8_gemm_tt(Any a, Any b, Tensor d, Tensor? c=None, int[]? recipe=None, str compiled_dims="mn", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_tt", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                            const torch::Tensor& d,
                                            const c10::optional<torch::Tensor>& c,
                                            const c10::optional<c10::IntArrayRef>& recipe,
                                            const std::string& compiled_dims,
                                            bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::fp8_gemm_tt_wrapper(a_val, a_scale, b_val, b_scale, d, c, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(m_grouped_fp8_gemm_nt_contiguous(Any a, Any b, Tensor d, Tensor m_indices, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("m_grouped_fp8_gemm_nt_contiguous", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                                                 const torch::Tensor& d,
                                                                 const torch::Tensor& m_indices,
                                                                 const c10::optional<c10::IntArrayRef>& recipe,
                                                                 const std::string& compiled_dims,
                                                                 bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::m_grouped_fp8_gemm_nt_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(m_grouped_fp8_gemm_nn_contiguous(Any a, Any b, Tensor d, Tensor m_indices, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("m_grouped_fp8_gemm_nn_contiguous", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                                                 const torch::Tensor& d,
                                                                 const torch::Tensor& m_indices,
                                                                 const c10::optional<c10::IntArrayRef>& recipe,
                                                                 const std::string& compiled_dims,
                                                                 bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::m_grouped_fp8_gemm_nn_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
    });

    m.def(R"(m_grouped_fp8_gemm_nt_masked(Any a, Any b, Tensor d, Tensor masked_m, int expected_m, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False, int max_block_n=256, bool enable_overlap=False, Tensor? signal=None) -> (int?, int?))");
    m.impl("m_grouped_fp8_gemm_nt_masked", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                                             const torch::Tensor& d,
                                                             const torch::Tensor& masked_m,
                                                             int64_t expected_m,
                                                             const c10::optional<c10::IntArrayRef>& recipe,
                                                             const std::string& compiled_dims,
                                                             bool disable_ue8m0_cast,
                                                             int64_t max_block_n,
                                                             bool enable_overlap,
                                                             const c10::optional<torch::Tensor>& signal) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        return deep_gemm_wrappers::m_grouped_fp8_gemm_nt_masked_wrapper(a_val, a_scale, b_val, b_scale, d, masked_m, expected_m, recipe, compiled_dims, disable_ue8m0_cast, max_block_n, enable_overlap, signal);
    });

    m.def(R"(k_grouped_fp8_gemm_nt_contiguous(Any a, Any b, Tensor d, int[] ks, Tensor ks_tensor, Tensor? c=None, int[] recipe=[1, 1, 128], str compiled_dims="mn") -> ())");
    m.impl("k_grouped_fp8_gemm_nt_contiguous", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                                                 const torch::Tensor& d,
                                                                 at::IntArrayRef ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const c10::optional<torch::Tensor>& c,
                                                                 c10::IntArrayRef recipe,
                                                                 const std::string& compiled_dims) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        std::vector<int64_t> ks64(ks.begin(), ks.end());
        c10::List<int64_t> ks_list(ks64);
        deep_gemm_wrappers::k_grouped_fp8_gemm_nt_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, ks_list, ks_tensor, c, recipe, compiled_dims);
    });

    m.def(R"(k_grouped_fp8_gemm_tn_contiguous(Any a, Any b, Tensor d, int[] ks, Tensor ks_tensor, Tensor? c=None, int[] recipe=[1, 1, 128], str compiled_dims="mn") -> ())");
    m.impl("k_grouped_fp8_gemm_tn_contiguous", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                                                 const torch::Tensor& d,
                                                                 at::IntArrayRef ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const c10::optional<torch::Tensor>& c,
                                                                 c10::IntArrayRef recipe,
                                                                 const std::string& compiled_dims) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        std::vector<int64_t> ks64(ks.begin(), ks.end());
        c10::List<int64_t> ks_list(ks64);
        deep_gemm_wrappers::k_grouped_fp8_gemm_tn_contiguous_wrapper(a_val, a_scale, b_val, b_scale, d, ks_list, ks_tensor, c, recipe, compiled_dims);
    });

    /*
     * BF16 GEMM
     */

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

    /*
     * cublas gemm
     */
    // cuBLASLt GEMMs
    m.def(R"(cublaslt_gemm_nt(Tensor a, Tensor b, Tensor d, Tensor? c) -> ())");
    m.impl("cublaslt_gemm_nt", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c) {
        deep_gemm::gemm::cublaslt_gemm_nt(a, b, d, c);
    });

    m.def(R"(cublaslt_gemm_nn(Tensor a, Tensor b, Tensor d, Tensor? c) -> ())");
    m.impl("cublaslt_gemm_nn", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c) {
        deep_gemm::gemm::cublaslt_gemm_nn(a, b, d, c);
    });

    m.def(R"(cublaslt_gemm_tn(Tensor a, Tensor b, Tensor d, Tensor? c) -> ())");
    m.impl("cublaslt_gemm_tn", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c) {
        deep_gemm::gemm::cublaslt_gemm_tn(a, b, d, c);
    });

    m.def(R"(cublaslt_gemm_tt(Tensor a, Tensor b, Tensor d, Tensor? c) -> ())");
    m.impl("cublaslt_gemm_tt", torch::kCUDA, [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                             const c10::optional<torch::Tensor>& c) {
        deep_gemm::gemm::cublaslt_gemm_tt(a, b, d, c);
    });

    /*
     * Attention
     */
    m.def(R"(fp8_gemm_nt_skip_head_mid(Any a, Any b, Tensor d, int[] head_splits, int[]? recipe=None, str compiled_dims="nk", bool disable_ue8m0_cast=False) -> ())");
    m.impl("fp8_gemm_nt_skip_head_mid", torch::kCUDA, [](const c10::IValue& a_input, const c10::IValue& b_input,
                                                        const torch::Tensor& d, 
                                                        const c10::IntArrayRef& head_splits, 
                                                        const c10::optional<c10::IntArrayRef>& recipe, 
                                                        const std::string& compiled_dims, 
                                                        bool disable_ue8m0_cast) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm_wrappers::fp8_gemm_nt_skip_head_mid_wrapper(a_val, a_scale, b_val, b_scale, d, head_splits, recipe, compiled_dims, disable_ue8m0_cast);
    });
    
    m.def(R"(fp8_mqa_logits(Tensor q, Any kv, Tensor weights, Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, bool clean_logits=True) -> Tensor)");
    m.impl("fp8_mqa_logits", torch::kCUDA, [](
        const torch::Tensor& q, 
        const c10::IValue& kv, 
        const torch::Tensor& weights, 
        const torch::Tensor& cu_seq_len_k_start, 
        const torch::Tensor& cu_seq_len_k_end, 
        bool clean_logits
    ) -> torch::Tensor {
        auto [k, v] = parse_tensor_or_tuple(kv);
        return deep_gemm_wrappers::fp8_mqa_logits_wrapper(q, k, v, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits);
    });

    m.def(R"(get_paged_mqa_logits_metadata(Tensor context_lens, int block_kv, int num_sms) -> Tensor)");
    m.impl("get_paged_mqa_logits_metadata", torch::kCUDA, [](
        const torch::Tensor& context_lens, 
        int64_t block_kv, 
        int64_t num_sms
    ) -> torch::Tensor {
        return deep_gemm_wrappers::get_paged_mqa_logits_metadata_wrapper(context_lens, block_kv, num_sms);
    });

    m.def(R"(fp8_paged_mqa_logits(Tensor q, Tensor fused_kv_cache, Tensor weights, Tensor context_lens, Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits) -> Tensor)");
    m.impl("fp8_paged_mqa_logits", torch::kCUDA, [](
        const torch::Tensor& q, 
        const torch::Tensor& fused_kv_cache, 
        const torch::Tensor& weights, 
        const torch::Tensor& context_lens, 
        const torch::Tensor& block_table, 
        const torch::Tensor& schedule_meta, 
        int64_t max_context_len, 
        bool clean_logits
    ) -> torch::Tensor {
        return deep_gemm_wrappers::fp8_paged_mqa_logits_wrapper(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits);
    });

    /*
     * einsum
     */
    m.def(R"(einsum(str expr, Tensor a, Tensor b, Tensor d, Tensor? c=None, bool use_cublaslt=False) -> ())");
    m.impl("einsum", torch::kCUDA, [](const std::string& expr, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const c10::optional<torch::Tensor>& c, bool use_cublaslt) {
        deep_gemm::einsum::einsum(expr, a, b, d, c, use_cublaslt);
    });

    m.def(R"(fp8_einsum(str expr, Any a, Any b, Tensor d, Tensor? c=None, int[] recipe=[1, 128, 128]) -> ())");
    m.impl("fp8_einsum", torch::kCUDA, [](const std::string& expr,
                                           const c10::IValue& a_input, const c10::IValue& b_input,
                                           const torch::Tensor& d,
                                           const c10::optional<torch::Tensor>& c,
                                           c10::IntArrayRef recipe) {
        auto [a_val, a_scale] = parse_tensor_or_tuple(a_input);
        auto [b_val, b_scale] = parse_tensor_or_tuple(b_input);
        deep_gemm::einsum::fp8_einsum(expr, {a_val, a_scale}, {b_val, b_scale}, d, c, to_recipe_tuple_default(recipe));
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