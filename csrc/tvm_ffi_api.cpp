#define DG_USE_TVM_FFI 1
#include <cstdint>
#include <optional>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/object.h>
#include <torch/torch.h>
#include <ATen/DLConvertor.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_gate.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/sm90_mega.hpp"
#include "utils/torch_compat.hpp"


using namespace deep_gemm;
using namespace tvm::ffi;

static std::optional<std::vector<int>> to_optional_int_vector(Optional<Array<int64_t>> values) {
    if (!values.has_value())
        return std::nullopt;
    std::vector<int> result;
    result.reserve(values.value().size());
    for (Array<int64_t>::iterator it = values.value().begin(); it != values.value().end(); ++it)
        result.push_back(static_cast<int>(*it));
    return result;
}


static std::optional<torch::Tensor> to_optional_tensor(Optional<TensorView> value) {
    return value.has_value() ? std::make_optional(convert_to_torch_tensor(value.value())) : std::nullopt;
}
static std::optional<float> to_optional_float(Optional<double> value) {
    return value.has_value() ? std::make_optional(static_cast<float>(value.value())) : std::nullopt;
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
void dg_init(std::string library_root_path, std::string cuda_home) {
    if (not cuda_home.empty())
        setenv("CUDA_HOME", cuda_home.c_str(), 0);
    init_jit(library_root_path);
}

int64_t dg_get_num_sms() { return runtime->get_num_sms(); }
void dg_set_num_sms(int64_t n) { runtime->set_num_sms(static_cast<int>(n)); }
int64_t dg_get_tc_util() { return runtime->get_tc_util(); }
void dg_set_tc_util(int64_t n) { runtime->set_tc_util(static_cast<int>(n)); }
bool dg_get_pdl() { return *jit->default_launch_options.enable_pdl; }
void dg_set_pdl(bool v) { jit->default_launch_options.enable_pdl = v; }
void dg_use_deterministic_algorithms(bool v) { heuristics_runtime->use_deterministic_algorithms(v); }
bool dg_get_deterministic_algorithms() { return heuristics_runtime->get_deterministic_algorithms(); }
void dg_set_ignore_compile_dims(bool v) { heuristics_runtime->set_ignore_compile_dims(v); }
void dg_set_block_size_multiple_of(int64_t m, int64_t n) {
    heuristics_runtime->set_block_size_multiple_of(static_cast<int>(m), static_cast<int>(n));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(use_deterministic_algorithms, dg_use_deterministic_algorithms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_deterministic_algorithms, dg_get_deterministic_algorithms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_ignore_compile_dims, dg_set_ignore_compile_dims);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_block_size_multiple_of, dg_set_block_size_multiple_of);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, dg_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_num_sms, dg_get_num_sms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_num_sms, dg_set_num_sms);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_compile_mode, dg_get_compile_mode);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_compile_mode, dg_set_compile_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_tc_util, dg_get_tc_util);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_tc_util, dg_set_tc_util);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_pdl, dg_get_pdl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_pdl, dg_set_pdl);

// ---------------------------------------------------------------------------
// Layout utilities
// ---------------------------------------------------------------------------
int64_t dg_get_tma_aligned_size(int64_t mn, int64_t element_size) {
    return get_tma_aligned_size(static_cast<int>(mn), static_cast<int>(element_size));
}

int64_t dg_get_mk_alignment_for_contiguous_layout() {
    return heuristics_runtime->get_mk_alignment_for_contiguous_layout();
}

void dg_set_mk_alignment_for_contiguous_layout(int64_t new_value) {
    heuristics_runtime->set_mk_alignment_for_contiguous_layout(static_cast<int>(new_value));
}

int64_t dg_get_theoretical_mk_alignment_for_contiguous_layout(Optional<int64_t> expected_m, Optional<int64_t> num_groups) {
    auto val = expected_m.has_value()? std::make_optional(static_cast<int>(expected_m.value())) : std::nullopt;
    return heuristics_runtime->get_theoretical_mk_alignment_for_contiguous_layout(
        val, num_groups.has_value() ? std::make_optional(static_cast<int>(num_groups.value())) : std::nullopt);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_tma_aligned_size, dg_get_tma_aligned_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mk_alignment_for_contiguous_layout, dg_get_mk_alignment_for_contiguous_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_mk_alignment_for_contiguous_layout, dg_set_mk_alignment_for_contiguous_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_theoretical_mk_alignment_for_contiguous_layout, dg_get_theoretical_mk_alignment_for_contiguous_layout);


// ---------------------------------------------------------------------------
// Layout kernels
// ---------------------------------------------------------------------------

tvm::ffi::Array<int64_t> dg_preprocess_sf(TensorView sf) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf_v);
    return {static_cast<int64_t>(dim),
            static_cast<int64_t>(ng),
            static_cast<int64_t>(mn_pp),
            static_cast<int64_t>(sf_k_pp),
            static_cast<int64_t>(tma_mn)};
}

Tensor dg_get_mn_major_tma_aligned_tensor(TensorView sf) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto result = get_mn_major_tma_aligned_tensor(sf_v);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_get_mn_major_tma_aligned_packed_ue8m0_tensor(TensorView sf, Optional<TensorView> psum_layout) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto result = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_v, to_optional_tensor(psum_layout));
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        TensorView sf, TensorView grouped_layout, Optional<Array<int64_t>> ks,
        int64_t gran_k, int64_t k_alignment, bool use_psum_layout, bool use_padded_sf_layout) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto grouped_layout_v = convert_to_torch_tensor(grouped_layout);
    auto ks_opt = to_optional_int_vector(ks);
    auto result = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf_v,
        grouped_layout_v,
        ks_opt,
        static_cast<int>(gran_k),
        static_cast<int>(k_alignment),
        use_psum_layout, use_padded_sf_layout
    );
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_transform_sf_into_required_layout(
        TensorView sf, int64_t mn, int64_t k,
        int64_t recipe_a, int64_t recipe_b, Optional<int64_t> recipe_c,
        Optional<int64_t> num_groups,
        Optional<bool> is_sfa,
        bool disable_ue8m0_cast, Optional<TensorView> psum_layout) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto is_sfa_val = is_sfa.has_value() ? std::make_optional(is_sfa.value()) : std::nullopt;
    auto ng = num_groups.has_value() ? std::make_optional(static_cast<int>(num_groups.value())) : std::nullopt;
    if(recipe_c.has_value()) {
        auto recipe = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c.value()));
        auto result = layout::transform_sf_into_required_layout(
            sf_v, static_cast<int>(mn), static_cast<int>(k),
            recipe, ng, is_sfa_val, disable_ue8m0_cast, to_optional_tensor(psum_layout));
        return Tensor::FromDLPack(at::toDLPack(result));
    } else {
        auto recipe = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b));
        auto result = layout::transform_sf_into_required_layout(
            sf_v, static_cast<int>(mn), static_cast<int>(k),
            recipe, ng, is_sfa_val, disable_ue8m0_cast, to_optional_tensor(psum_layout));
        return Tensor::FromDLPack(at::toDLPack(result));
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(preprocess_sf, dg_preprocess_sf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mn_major_tma_aligned_tensor, dg_get_mn_major_tma_aligned_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mn_major_tma_aligned_packed_ue8m0_tensor, dg_get_mn_major_tma_aligned_packed_ue8m0_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor, dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transform_sf_into_required_layout, dg_transform_sf_into_required_layout);


// ---------------------------------------------------------------------------
// cuBLASLt GEMMs (always available)
// ---------------------------------------------------------------------------
void dg_cublaslt_gemm_nt(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_nt(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}
void dg_cublaslt_gemm_nn(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_nn(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}
void dg_cublaslt_gemm_tn(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_tn(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}
void dg_cublaslt_gemm_tt(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_tt(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_nt, dg_cublaslt_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_nn, dg_cublaslt_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_tn, dg_cublaslt_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_tt, dg_cublaslt_gemm_tt);

// ---------------------------------------------------------------------------
// FP8/FP4 GEMMs and BF16 GEMMs
// ---------------------------------------------------------------------------

void dg_fp8_fp4_gemm_nt(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_nt(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast, to_optional_float(alpha));
}

void dg_fp8_fp4_gemm_nn(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_nn(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast, to_optional_float(alpha));
}

void dg_fp8_fp4_gemm_tn(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_tn(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast, to_optional_float(alpha));
}

void dg_fp8_fp4_gemm_tt(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_tt(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast, to_optional_float(alpha));
}

void dg_m_grouped_fp8_fp4_gemm_nt_contiguous(TensorView a, TensorView a_sf,
                                             TensorView b, TensorView b_sf,
                                             TensorView d,
                                             TensorView grouped_layout,
                                             Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                             Optional<Tuple<int64_t, int64_t>> recipe_a,
                                             Optional<Tuple<int64_t, int64_t>> recipe_b,
                                             std::string compiled_dims,
                                             bool disable_ue8m0_cast,
                                             bool use_psum_layout,
                                             bool ensure_zero_padding,
                                             Optional<int64_t> expected_m_for_psum_layout) {
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    auto expected_m_for_psum_layout_opts = expected_m_for_psum_layout.has_value()? std::make_optional((int) expected_m_for_psum_layout.value()) : std::nullopt;
    gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), convert_to_torch_tensor(grouped_layout),
        recipe_opt, recipe_a_opt, recipe_b_opt,
        compiled_dims, disable_ue8m0_cast,
        use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout_opts
    );
}

void dg_m_grouped_fp8_fp4_gemm_nn_contiguous(TensorView a, TensorView a_sf,
                                             TensorView b, TensorView b_sf,
                                             TensorView d,
                                             TensorView grouped_layout,
                                             Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                             Optional<Tuple<int64_t, int64_t>> recipe_a,
                                             Optional<Tuple<int64_t, int64_t>> recipe_b,
                                             std::string compiled_dims,
                                             bool disable_ue8m0_cast,
                                             bool use_psum_layout,
                                             bool ensure_zero_padding) {
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::m_grouped_fp8_fp4_gemm_nn_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), convert_to_torch_tensor(grouped_layout),
        recipe_opt, recipe_a_opt, recipe_b_opt,
        compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding
    );
}

void dg_m_grouped_fp8_fp4_gemm_nt_masked(TensorView a, TensorView a_sf,
                                 TensorView b, TensorView b_sf,
                                 TensorView d,
                                 TensorView masked_m,
                                 int64_t expected_m,
                                 Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                 Optional<Tuple<int64_t, int64_t>> recipe_a,
                                 Optional<Tuple<int64_t, int64_t>> recipe_b,
                                 std::string compiled_dims,
                                 bool disable_ue8m0_cast) {
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::m_grouped_fp8_fp4_gemm_nt_masked(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), convert_to_torch_tensor(masked_m),
        (int) expected_m, recipe_opt, recipe_a_opt, recipe_b_opt,
        compiled_dims, disable_ue8m0_cast
    );
}


void dg_bf16_gemm_nt(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_nt(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims, to_optional_float(alpha));
}

void dg_bf16_gemm_nn(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_nn(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims, to_optional_float(alpha));
}

void dg_bf16_gemm_tn(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_tn(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims, to_optional_float(alpha));
}

void dg_bf16_gemm_tt(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims, Optional<double> alpha) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_tt(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims, to_optional_float(alpha));
}

void dg_m_grouped_bf16_gemm_nt_contiguous(TensorView a, TensorView b, TensorView d,
                                          TensorView grouped_layout,
                                          std::string compiled_dims,
                                          bool use_psum_layout,
                                          bool ensure_zero_padding,
                                          Optional<int64_t> expected_m_for_psum_layout) {
    auto expected_m_opt = expected_m_for_psum_layout.has_value()? std::make_optional((int)expected_m_for_psum_layout.value()) : std::nullopt;
    gemm::m_grouped_bf16_gemm_nt_contiguous(
        convert_to_torch_tensor(a), convert_to_torch_tensor(b),
        convert_to_torch_tensor(d), convert_to_torch_tensor(grouped_layout),
        compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_opt
    );
}

void dg_m_grouped_bf16_gemm_nn_contiguous(TensorView a, TensorView b, TensorView d,
                                          TensorView grouped_layout,
                                          std::string compiled_dims,
                                          bool use_psum_layout,
                                          bool ensure_zero_padding) {
    gemm::m_grouped_bf16_gemm_nn_contiguous(
        convert_to_torch_tensor(a), convert_to_torch_tensor(b),
        convert_to_torch_tensor(d), convert_to_torch_tensor(grouped_layout),
        compiled_dims, use_psum_layout, ensure_zero_padding
    );
}

void dg_m_grouped_bf16_gemm_nt_masked(TensorView a, TensorView b, TensorView d,
                                      TensorView masked_m,
                                      int64_t expected_m,
                                      std::string compiled_dims) {
    gemm::m_grouped_bf16_gemm_nt_masked(
        convert_to_torch_tensor(a), convert_to_torch_tensor(b),
        convert_to_torch_tensor(d), convert_to_torch_tensor(masked_m),
        (int) expected_m, compiled_dims
    );
}

void dg_k_grouped_bf16_gemm_tn_contiguous(TensorView a, TensorView b, TensorView d,
                                          Optional<Array<int64_t>> ks, TensorView grouped_layout,
                                          Optional<TensorView> c,
                                          std::string compiled_dims,
                                          bool use_psum_layout) {
    auto ks_val = to_optional_int_vector(ks);
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::k_grouped_bf16_gemm_tn_contiguous(
        convert_to_torch_tensor(a), convert_to_torch_tensor(b),
        convert_to_torch_tensor(d), ks_val, convert_to_torch_tensor(grouped_layout),
        c_opt, compiled_dims, use_psum_layout
    );
}

void dg_k_grouped_fp8_gemm_tn_contiguous(TensorView a, TensorView a_sf,
                                         TensorView b, TensorView b_sf,
                                         TensorView d,
                                         Optional<Array<int64_t>> ks,
                                         TensorView grouped_layout,
                                         Optional<TensorView> c,
                                         Tuple<int64_t, int64_t, int64_t> recipe,
                                         std::string compiled_dims,
                                         bool use_psum_layout, bool use_padded_sf_layout) {
    auto ks_val = to_optional_int_vector(ks);
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe.get<0>()),
                                      static_cast<int>(recipe.get<1>()),
                                      static_cast<int>(recipe.get<2>()));
    gemm::k_grouped_fp8_gemm_tn_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), ks_val, convert_to_torch_tensor(grouped_layout),
        c_opt, recipe_val, compiled_dims, use_psum_layout, use_padded_sf_layout
    );
}

void dg_k_grouped_fp8_gemm_nt_contiguous(TensorView a, TensorView a_sf,
                                         TensorView b, TensorView b_sf,
                                         TensorView d,
                                         Optional<Array<int64_t>> ks,
                                         TensorView grouped_layout,
                                         Optional<TensorView> c,
                                         Tuple<int64_t, int64_t, int64_t> recipe,
                                         std::string compiled_dims,
                                         bool use_psum_layout, bool use_padded_sf_layout) {
    auto ks_val = to_optional_int_vector(ks);
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe.get<0>()),
                                      static_cast<int>(recipe.get<1>()),
                                      static_cast<int>(recipe.get<2>()));
    gemm::k_grouped_fp8_gemm_nt_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), ks_val, convert_to_torch_tensor(grouped_layout),
        c_opt, recipe_val, compiled_dims, use_psum_layout, use_padded_sf_layout
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_nt, dg_fp8_fp4_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_nn, dg_fp8_fp4_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_tn, dg_fp8_fp4_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_tt, dg_fp8_fp4_gemm_tt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_fp8_fp4_gemm_nt_contiguous, dg_m_grouped_fp8_fp4_gemm_nt_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_fp8_fp4_gemm_nn_contiguous, dg_m_grouped_fp8_fp4_gemm_nn_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_fp8_fp4_gemm_nt_masked, dg_m_grouped_fp8_fp4_gemm_nt_masked);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nt, dg_bf16_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nn, dg_bf16_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tn, dg_bf16_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tt, dg_bf16_gemm_tt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_bf16_gemm_nt_contiguous, dg_m_grouped_bf16_gemm_nt_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_bf16_gemm_nn_contiguous, dg_m_grouped_bf16_gemm_nn_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_bf16_gemm_nt_masked, dg_m_grouped_bf16_gemm_nt_masked);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_grouped_bf16_gemm_tn_contiguous, dg_k_grouped_bf16_gemm_tn_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_grouped_fp8_gemm_tn_contiguous, dg_k_grouped_fp8_gemm_tn_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_grouped_fp8_gemm_nt_contiguous, dg_k_grouped_fp8_gemm_nt_contiguous);

// Einsum
void dg_einsum(std::string expr, TensorView a, TensorView b, TensorView d,
               Optional<TensorView> c, bool use_cublaslt) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    einsum::einsum(expr, convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, use_cublaslt);
}

void dg_fp8_einsum(std::string expr,
                   TensorView a_data, TensorView a_sf,
                   TensorView b_data, TensorView b_sf,
                   TensorView d, Optional<TensorView> d_sf,
                   Optional<TensorView> c,
                   Tuple<int64_t, int64_t, int64_t> recipe) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = std::make_tuple((int)recipe.get<0>(), (int)recipe.get<1>(), (int)recipe.get<2>());
    std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>> output = convert_to_torch_tensor(d);
    if (d_sf.has_value())
        output = std::make_pair(convert_to_torch_tensor(d), convert_to_torch_tensor(d_sf.value()));
    einsum::fp8_einsum(expr, std::make_pair(convert_to_torch_tensor(a_data), convert_to_torch_tensor(a_sf)),
                       std::make_pair(convert_to_torch_tensor(b_data), convert_to_torch_tensor(b_sf)),
                       output, c_opt, recipe_opt);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(einsum, dg_einsum);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_einsum, dg_fp8_einsum);

// Hyperconnection
void dg_tf32_hc_prenorm_gemm(TensorView a, TensorView b, TensorView d,
                              TensorView sqr_sum, Optional<int64_t> num_splits) {
    auto ns = num_splits.has_value() ? std::make_optional(static_cast<int>(num_splits.value())) : std::nullopt;
    hyperconnection::tf32_hc_prenorm_gemm(convert_to_torch_tensor(a), convert_to_torch_tensor(b),
                                          convert_to_torch_tensor(d), convert_to_torch_tensor(sqr_sum), ns);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tf32_hc_prenorm_gemm, dg_tf32_hc_prenorm_gemm);

// Attention
void dg_fp8_gemm_nt_skip_head_mid(TensorView a_data, TensorView a_sf,
                                   TensorView b_data, TensorView b_sf,
                                   TensorView d,
                                   Tuple<int64_t, int64_t, int64_t> head_splits,
                                   Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                   std::string compiled_dims,
                                   bool disable_ue8m0_cast) {
    auto head_splits_opt = std::make_tuple((int)head_splits.get<0>(), (int)head_splits.get<1>(), (int)head_splits.get<2>());
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    attention::fp8_gemm_nt_skip_head_mid(
        std::make_pair(convert_to_torch_tensor(a_data), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b_data), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), head_splits_opt, recipe_opt, compiled_dims, disable_ue8m0_cast);
}

Tensor dg_fp8_mqa_logits(TensorView q, TensorView kv_data, TensorView kv_sf,
                        TensorView weights, TensorView cu_seq_len_k_start,
                        TensorView cu_seq_len_k_end,
                        bool clean_logits, int64_t max_seqlen_k) {
    auto result = attention::fp8_mqa_logits(
        convert_to_torch_tensor(q),
        std::make_pair(convert_to_torch_tensor(kv_data), convert_to_torch_tensor(kv_sf)),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(cu_seq_len_k_start),
        convert_to_torch_tensor(cu_seq_len_k_end), clean_logits, static_cast<int>(max_seqlen_k));
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_get_paged_mqa_logits_metadata(TensorView context_lens, int64_t block_kv,
                                       int64_t num_sms, Optional<TensorView> indices) {
    auto indices_val = indices.has_value()?
        std::optional<torch::Tensor>(convert_to_torch_tensor(indices.value()))
        : std::nullopt;
    auto result = attention::get_paged_mqa_logits_metadata(
        convert_to_torch_tensor(context_lens), static_cast<int>(block_kv),
        static_cast<int>(num_sms), indices_val);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_fp8_paged_mqa_logits(TensorView q, TensorView fused_kv_cache,
                              TensorView weights, TensorView context_lens,
                              TensorView block_table, TensorView schedule_meta,
                              int64_t max_context_len, bool clean_logits,
                              Optional<TensorView> indices) {
    auto indices_val = indices.has_value()? std::optional(convert_to_torch_tensor(indices.value())) : std::nullopt;
    auto result = attention::fp8_paged_mqa_logits(
        convert_to_torch_tensor(q), convert_to_torch_tensor(fused_kv_cache),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(context_lens),
        convert_to_torch_tensor(block_table), convert_to_torch_tensor(schedule_meta),
        static_cast<int>(max_context_len), clean_logits, indices_val);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_fp8_fp4_mqa_logits(TensorView q, Optional<TensorView> q_sf, TensorView kv_data, TensorView kv_sf,
                            TensorView weights, TensorView cu_seq_len_k_start,
                            TensorView cu_seq_len_k_end, bool clean_logits, int64_t max_seqlen_k,
                            std::string logits_dtype, Optional<TensorView> schedule_meta) {
    auto q_sf_val = q_sf.has_value()? std::make_optional(convert_to_torch_tensor(q_sf.value())) : std::nullopt;
    auto result = attention::fp8_fp4_mqa_logits(
        std::make_pair(convert_to_torch_tensor(q), q_sf_val),
        std::make_pair(convert_to_torch_tensor(kv_data), convert_to_torch_tensor(kv_sf)),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(cu_seq_len_k_start),
        convert_to_torch_tensor(cu_seq_len_k_end), clean_logits, static_cast<int>(max_seqlen_k),
        string_to_dtype(logits_dtype), to_optional_tensor(schedule_meta));
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_fp8_fp4_paged_mqa_logits(TensorView q, Optional<TensorView> q_sf, TensorView fused_kv_cache,
                              TensorView weights, TensorView context_lens,
                              TensorView block_table, TensorView schedule_meta,
                              int64_t max_context_len, bool clean_logits,
                              std::string logits_dtype, Optional<TensorView> indices) {
    auto q_sf_val = q_sf.has_value()? std::make_optional(convert_to_torch_tensor(q_sf.value())) : std::nullopt;
    auto indices_val = indices.has_value()? std::optional(convert_to_torch_tensor(indices.value())) : std::nullopt;
    auto result = attention::fp8_fp4_paged_mqa_logits(
        std::make_pair(convert_to_torch_tensor(q), q_sf_val),
        convert_to_torch_tensor(fused_kv_cache),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(context_lens),
        convert_to_torch_tensor(block_table), convert_to_torch_tensor(schedule_meta),
        static_cast<int>(max_context_len), clean_logits,
        string_to_dtype(logits_dtype), indices_val);
    return Tensor::FromDLPack(at::toDLPack(result));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm_nt_skip_head_mid, dg_fp8_gemm_nt_skip_head_mid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mqa_logits, dg_fp8_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_paged_mqa_logits_metadata, dg_get_paged_mqa_logits_metadata);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_paged_mqa_logits, dg_fp8_paged_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_mqa_logits, dg_fp8_fp4_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_paged_mqa_logits, dg_fp8_fp4_paged_mqa_logits);

// Mega MoE
int64_t dg_get_token_alignment_for_mega_moe() {
    return (int64_t)mega::get_token_alignment_for_mega_moe();
}

int64_t dg_get_block_m_for_mega_moe(int64_t num_ranks, int64_t num_experts,
                                    int64_t num_max_tokens_per_rank, int64_t num_tokens,
                                    int64_t num_topk, std::string mma_type) {
    return static_cast<int64_t>(mega::get_block_m_for_mega_moe(
        static_cast<int>(num_ranks),
        static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_tokens),
        static_cast<int>(num_topk),
        mma_type));
}

using MegaSliceResult = Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                              Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

int64_t dg_get_token_alignment_for_sm90_mega_moe() {
    return (int64_t)mega::get_token_alignment_for_sm90_mega_moe();
}

Tuple<int64_t, TypedFunction<MegaSliceResult(TensorView)>>
dg_get_symm_buffer_size_for_mega_moe(int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank, int64_t num_topk, int64_t hidden,
                                    int64_t intermediate_hidden, std::string mma_type, std::string activation,
                                    int64_t num_shared_experts) {
    auto [num_bytes, fn] = mega::get_symm_buffer_size_for_mega_moe(
        static_cast<int>(num_ranks),
        static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(hidden),
        static_cast<int>(intermediate_hidden),
        mma_type,
        activation,
        static_cast<int>(num_shared_experts)
    );

    auto slice_input_buffers = [=](TensorView buffer) {
        const auto buffer_torch = convert_to_torch_tensor(buffer);
        auto [x, x_sf, topk_idx, topk_weights,
              shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
              l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, x_scales] = fn(buffer_torch);
        // DLPack cannot carry FP8/FP4 dtypes, so activation views cross the
        // bridge as raw bytes; undefined views (BF16 SFs, absent shared
        // experts) cross as empty byte tensors.
        const auto as_bytes = [&](const torch::Tensor& t) {
            const auto t_val = t.defined()
                ? t
                : torch::empty({0}, torch::TensorOptions().dtype(torch::kChar).device(buffer_torch.device()));
            return Tensor::FromDLPack(at::toDLPack(
                t_val.scalar_type() == torch::kFloat8_e4m3fn or t_val.scalar_type() == torch::kUInt8
                    ? t_val.view(at::kChar) : t_val));
        };
        return MegaSliceResult(
            as_bytes(x), as_bytes(x_sf),
            as_bytes(topk_idx), as_bytes(topk_weights),
            as_bytes(shared_l1_acts), as_bytes(shared_l1_acts_sf),
            as_bytes(shared_l2_acts), as_bytes(shared_l2_acts_sf),
            as_bytes(l1_acts), as_bytes(l1_acts_sf),
            as_bytes(l2_acts), as_bytes(l2_acts_sf),
            as_bytes(x_scales)
        );
    };
    return Tuple<int64_t, TypedFunction<MegaSliceResult(TensorView)>>(
        num_bytes, slice_input_buffers);
}

Tuple<int64_t, TypedFunction<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(TensorView)>>
dg_get_symm_buffer_size_for_sm90_mega_moe(int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank, int64_t num_topk, int64_t hidden,
                                         int64_t intermediate_hidden, bool use_fp8_dispatch, std::string activation) {
    auto [num_bytes, fn] = mega::get_symm_buffer_size_for_sm90_mega_moe(
        static_cast<int>(num_ranks),
        static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(hidden),
        static_cast<int>(intermediate_hidden),
        use_fp8_dispatch,
        activation
    );

    auto slice_input_buffers = [=](TensorView buffer) {
        auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =  fn(convert_to_torch_tensor(buffer));
        return Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
            Tensor::FromDLPack(at::toDLPack(x.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(x_sf)),
            Tensor::FromDLPack(at::toDLPack(topk_idx)),
            Tensor::FromDLPack(at::toDLPack(topk_weights)),
            Tensor::FromDLPack(at::toDLPack(l1_acts.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(l1_acts_sf)),
            Tensor::FromDLPack(at::toDLPack(l2_acts.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(l2_acts_sf))
        );
    };
    return Tuple<int64_t, TypedFunction<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(TensorView)>>(
        num_bytes, slice_input_buffers);
}

void dg_fp8_fp4_mega_moe(TensorView y, TensorView l1_weights, TensorView l1_weights_sf, TensorView l2_weights, TensorView l2_weights_sf,
                        Optional<TensorView> shared_l1_weights, Optional<TensorView> shared_l1_weights_sf,
                        Optional<TensorView> shared_l2_weights, Optional<TensorView> shared_l2_weights_sf,
                        Optional<TensorView> cumulative_local_expert_recv_stats, TensorView sym_buffer, Array<int64_t> sym_buffer_ptrs,
                        int64_t rank_idx, int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                        Tuple<int64_t, int64_t, int64_t> recipe, std::string mma_type, std::string activation, Optional<double> activation_clamp_opt,
                        bool fast_math, bool use_x_scales, Optional<TensorView> l1_alphas,
                        Optional<TensorView> l2_alphas, Optional<TensorView> l2_act_scales) {
    auto c_val = cumulative_local_expert_recv_stats.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(cumulative_local_expert_recv_stats.value())) : std::nullopt;
    auto act_clamp_opt_val = activation_clamp_opt.has_value()? std::optional<float>(static_cast<float>(activation_clamp_opt.value())) : std::nullopt;
    std::vector<int64_t> sym_buffer_ptrs_val;
    sym_buffer_ptrs_val.reserve(sym_buffer_ptrs.size());

    for (Array<int64_t>::iterator it = sym_buffer_ptrs.begin(); it != sym_buffer_ptrs.end(); ++it) {
        sym_buffer_ptrs_val.push_back(*it);
    }
    auto [recipe_a, recipe_b, recipe_c] = recipe;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c));
    DG_HOST_ASSERT(shared_l1_weights.has_value() == shared_l1_weights_sf.has_value());
    DG_HOST_ASSERT(shared_l2_weights.has_value() == shared_l2_weights_sf.has_value());
    auto shared_l1_val = shared_l1_weights.has_value()
        ? std::optional<std::tuple<torch::Tensor, torch::Tensor>>(std::make_tuple(
              convert_to_torch_tensor(shared_l1_weights.value()), convert_to_torch_tensor(shared_l1_weights_sf.value())))
        : std::nullopt;
    auto shared_l2_val = shared_l2_weights.has_value()
        ? std::optional<std::tuple<torch::Tensor, torch::Tensor>>(std::make_tuple(
              convert_to_torch_tensor(shared_l2_weights.value()), convert_to_torch_tensor(shared_l2_weights_sf.value())))
        : std::nullopt;

    mega::fp8_fp4_mega_moe(
        convert_to_torch_tensor(y),
        std::make_tuple(convert_to_torch_tensor(l1_weights), convert_to_torch_tensor(l1_weights_sf)),
        std::make_tuple(convert_to_torch_tensor(l2_weights), convert_to_torch_tensor(l2_weights_sf)),
        shared_l1_val, shared_l2_val,
        c_val, convert_to_torch_tensor(sym_buffer), sym_buffer_ptrs_val, static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_experts),
        static_cast<int>(num_topk), recipe_val, mma_type, activation, act_clamp_opt_val, fast_math,
        use_x_scales,
        l1_alphas.has_value() ? std::optional<torch::Tensor>(convert_to_torch_tensor(l1_alphas.value())) : std::nullopt,
        l2_alphas.has_value() ? std::optional<torch::Tensor>(convert_to_torch_tensor(l2_alphas.value())) : std::nullopt,
        l2_act_scales.has_value() ? std::optional<torch::Tensor>(convert_to_torch_tensor(l2_act_scales.value())) : std::nullopt
    );
}

void dg_bf16_mega_moe(TensorView y, TensorView l1_weights, TensorView l2_weights,
                      Optional<TensorView> shared_l1_weights, Optional<TensorView> shared_l2_weights,
                      Optional<TensorView> cumulative_local_expert_recv_stats, TensorView sym_buffer,
                      Array<int64_t> sym_buffer_ptrs, int64_t rank_idx,
                      int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                      std::string activation, Optional<double> activation_clamp_opt, bool fast_math) {
    auto c_val = cumulative_local_expert_recv_stats.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(cumulative_local_expert_recv_stats.value())) : std::nullopt;
    auto act_clamp_opt_val = activation_clamp_opt.has_value()? std::optional<float>(static_cast<float>(activation_clamp_opt.value())) : std::nullopt;
    std::vector<int64_t> sym_buffer_ptrs_val;
    sym_buffer_ptrs_val.reserve(sym_buffer_ptrs.size());

    for (Array<int64_t>::iterator it = sym_buffer_ptrs.begin(); it != sym_buffer_ptrs.end(); ++it) {
        sym_buffer_ptrs_val.push_back(*it);
    }
    auto shared_l1_val = shared_l1_weights.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(shared_l1_weights.value())) : std::nullopt;
    auto shared_l2_val = shared_l2_weights.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(shared_l2_weights.value())) : std::nullopt;

    mega::bf16_mega_moe(
        convert_to_torch_tensor(y),
        convert_to_torch_tensor(l1_weights),
        convert_to_torch_tensor(l2_weights),
        shared_l1_val, shared_l2_val,
        c_val, convert_to_torch_tensor(sym_buffer), sym_buffer_ptrs_val, static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_experts),
        static_cast<int>(num_topk), activation, act_clamp_opt_val, fast_math
    );
}

void dg_fp8_fp4_mega_moe_sm90(TensorView y, TensorView l1_weights, TensorView l1_weights_sf, TensorView l2_weights, TensorView l2_weights_sf,
                              Optional<TensorView> cumulative_local_expert_recv_stats, TensorView sym_buffer, Array<int64_t> sym_buffer_ptrs,
                              int64_t rank_idx, int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                              Tuple<int64_t, int64_t, int64_t> recipe, std::string activation,
                              Optional<double> activation_clamp_opt,
                              Optional<double> activation_alpha_opt,
                              Optional<double> activation_linear_beta_opt,
                              bool fast_math) {
    auto c_val = cumulative_local_expert_recv_stats.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(cumulative_local_expert_recv_stats.value())) : std::nullopt;
    auto act_clamp_opt_val = activation_clamp_opt.has_value()? std::optional<float>(static_cast<float>(activation_clamp_opt.value())) : std::nullopt;
    auto act_alpha_opt_val = activation_alpha_opt.has_value()? std::optional<float>(static_cast<float>(activation_alpha_opt.value())) : std::nullopt;
    auto act_linear_beta_opt_val = activation_linear_beta_opt.has_value()? std::optional<float>(static_cast<float>(activation_linear_beta_opt.value())) : std::nullopt;
    std::vector<int64_t> sym_buffer_ptrs_val;
    sym_buffer_ptrs_val.reserve(sym_buffer_ptrs.size());

    for (Array<int64_t>::iterator it = sym_buffer_ptrs.begin(); it != sym_buffer_ptrs.end(); ++it) {
        sym_buffer_ptrs_val.push_back(*it);
    }
    auto [recipe_a, recipe_b, recipe_c] = recipe;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c));

    mega::fp8_fp4_mega_moe_sm90(
        convert_to_torch_tensor(y),
        std::make_pair(convert_to_torch_tensor(l1_weights), convert_to_torch_tensor(l1_weights_sf)),
        std::make_pair(convert_to_torch_tensor(l2_weights), convert_to_torch_tensor(l2_weights_sf)),
        c_val, convert_to_torch_tensor(sym_buffer), sym_buffer_ptrs_val, static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_experts),
        static_cast<int>(num_topk), recipe_val, activation, act_clamp_opt_val,
        act_alpha_opt_val, act_linear_beta_opt_val, fast_math
    );
}


void dg_fp8_mega_moe(TensorView y, TensorView l1_weights, TensorView l1_weights_sf, TensorView l2_weights, TensorView l2_weights_sf,
                    Optional<TensorView> cumulative_local_expert_recv_stats, TensorView sym_buffer, Array<int64_t> sym_buffer_ptrs,
                    int64_t rank_idx, int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                    Tuple<int64_t, int64_t, int64_t> recipe, std::string activation, Optional<double> activation_clamp_opt, bool fast_math) {
    auto c_val = cumulative_local_expert_recv_stats.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(cumulative_local_expert_recv_stats.value())) : std::nullopt;
    auto act_clamp_opt_val = activation_clamp_opt.has_value()? std::optional<float>(static_cast<float>(activation_clamp_opt.value())) : std::nullopt;
    std::vector<int64_t> sym_buffer_ptrs_val;
    sym_buffer_ptrs_val.reserve(sym_buffer_ptrs.size());

    for (Array<int64_t>::iterator it = sym_buffer_ptrs.begin(); it != sym_buffer_ptrs.end(); ++it) {
        sym_buffer_ptrs_val.push_back(*it);
    }
    auto [recipe_a, recipe_b, recipe_c] = recipe;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c));

    mega::fp8_mega_moe(
        convert_to_torch_tensor(y),
        std::make_pair(convert_to_torch_tensor(l1_weights), convert_to_torch_tensor(l1_weights_sf)),
        std::make_pair(convert_to_torch_tensor(l2_weights), convert_to_torch_tensor(l2_weights_sf)),
        c_val, convert_to_torch_tensor(sym_buffer), sym_buffer_ptrs_val, static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_experts),
        static_cast<int>(num_topk), recipe_val, activation, act_clamp_opt_val, fast_math
    );
}

void dg_mega_moe_pre_dispatch(
    TensorView x, TensorView topk_idx, TensorView topk_weights,
    TensorView buf_x, TensorView buf_x_sf,
    TensorView buf_topk_idx, TensorView buf_topk_weights,
    int64_t num_tokens, int64_t group_size, std::string mma_type,
    Optional<TensorView> buf_x_scales, Optional<TensorView> expert_scales) {
    mega_moe_pre_dispatch(
        convert_to_torch_tensor(x),
        convert_to_torch_tensor(topk_idx),
        convert_to_torch_tensor(topk_weights),
        convert_to_torch_tensor(buf_x),
        convert_to_torch_tensor(buf_x_sf),
        convert_to_torch_tensor(buf_topk_idx),
        convert_to_torch_tensor(buf_topk_weights),
        static_cast<int>(num_tokens),
        static_cast<int>(group_size),
        mma_type,
        buf_x_scales.has_value()
            ? std::optional<torch::Tensor>(convert_to_torch_tensor(buf_x_scales.value()))
            : std::nullopt,
        expert_scales.has_value()
            ? std::optional<torch::Tensor>(convert_to_torch_tensor(expert_scales.value()))
            : std::nullopt
    );
}

void dg_mega_moe_pre_dispatch_sm90(
    TensorView x, TensorView topk_idx, TensorView topk_weights,
    TensorView buf_x, TensorView buf_x_sf,
    TensorView buf_topk_idx, TensorView buf_topk_weights,
    int64_t num_tokens, int64_t group_size, double routed_scaling_factor) {
    mega::mega_moe_pre_dispatch_sm90(
        convert_to_torch_tensor(x),
        convert_to_torch_tensor(topk_idx),
        convert_to_torch_tensor(topk_weights),
        convert_to_torch_tensor(buf_x),
        convert_to_torch_tensor(buf_x_sf),
        convert_to_torch_tensor(buf_topk_idx),
        convert_to_torch_tensor(buf_topk_weights),
        static_cast<int>(num_tokens),
        static_cast<int>(group_size),
        static_cast<float>(routed_scaling_factor)
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_token_alignment_for_mega_moe, dg_get_token_alignment_for_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_block_m_for_mega_moe, dg_get_block_m_for_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_token_alignment_for_sm90_mega_moe, dg_get_token_alignment_for_sm90_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_symm_buffer_size_for_mega_moe, dg_get_symm_buffer_size_for_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_symm_buffer_size_for_sm90_mega_moe, dg_get_symm_buffer_size_for_sm90_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_mega_moe, dg_fp8_fp4_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_mega_moe, dg_bf16_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_mega_moe_sm90, dg_fp8_fp4_mega_moe_sm90);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mega_moe, dg_fp8_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_pre_dispatch, dg_mega_moe_pre_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_pre_dispatch_sm90, dg_mega_moe_pre_dispatch_sm90);


Tensor dg_get_mqa_logits_metadata(TensorView cu_seq_len_k_start,
    TensorView cu_seq_len_k_end,
    int64_t num_kv_tokens,
    int64_t num_heads) {
    auto result = attention::get_mqa_logits_metadata(
        convert_to_torch_tensor(cu_seq_len_k_start),
        convert_to_torch_tensor(cu_seq_len_k_end),
        static_cast<int>(num_kv_tokens),
        static_cast<int>(num_heads));
    return Tensor::FromDLPack(at::toDLPack(result));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mqa_logits_metadata, dg_get_mqa_logits_metadata);

Tensor dg_get_sparse_mqa_logits_metadata(TensorView cu_seq_len_k_start,
    TensorView cu_seq_len_k_end,
    int64_t num_kv_tokens,
    TensorView sparse_kv_block_indices,
    std::string qk_dtype,
    int64_t sparse_block_kv,
    bool use_unaligned_ks) {
    auto result = attention::get_sparse_mqa_logits_metadata(
        convert_to_torch_tensor(cu_seq_len_k_start),
        convert_to_torch_tensor(cu_seq_len_k_end),
        static_cast<int>(num_kv_tokens),
        convert_to_torch_tensor(sparse_kv_block_indices),
        string_to_dtype(qk_dtype),
        static_cast<int>(sparse_block_kv),
        use_unaligned_ks);
    return Tensor::FromDLPack(at::toDLPack(result));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_sparse_mqa_logits_metadata, dg_get_sparse_mqa_logits_metadata);

Tensor dg_get_paged_sparse_mqa_logits_metadata(TensorView context_lens,
    TensorView block_table,
    TensorView indices,
    int64_t page_kv,
    TensorView sparse_kv_block_indices,
    std::string qk_dtype,
    int64_t sparse_block_kv) {
    auto result = attention::get_paged_sparse_mqa_logits_metadata(
        convert_to_torch_tensor(context_lens),
        convert_to_torch_tensor(block_table),
        convert_to_torch_tensor(indices),
        static_cast<int>(page_kv),
        convert_to_torch_tensor(sparse_kv_block_indices),
        string_to_dtype(qk_dtype),
        static_cast<int>(sparse_block_kv));
    return Tensor::FromDLPack(at::toDLPack(result));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_paged_sparse_mqa_logits_metadata, dg_get_paged_sparse_mqa_logits_metadata);

Tensor dg_fp8_fp4_sparse_mqa_logits(TensorView q,
    Optional<TensorView> q_sf,
    TensorView kv,
    TensorView kv_sf,
    TensorView weights,
    TensorView metadata,
    int64_t num_max_sparse_blocks,
    int64_t sparse_block_kv,
    bool use_unaligned_ks) {
    auto result = attention::fp8_fp4_sparse_mqa_logits(
        std::make_tuple(convert_to_torch_tensor(q), to_optional_tensor(q_sf)),
        std::make_pair(convert_to_torch_tensor(kv), convert_to_torch_tensor(kv_sf)),
        convert_to_torch_tensor(weights),
        convert_to_torch_tensor(metadata),
        static_cast<int>(num_max_sparse_blocks),
        static_cast<int>(sparse_block_kv),
        use_unaligned_ks);
    return Tensor::FromDLPack(at::toDLPack(result));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_sparse_mqa_logits, dg_fp8_fp4_sparse_mqa_logits);

Tensor dg_fp8_fp4_paged_sparse_mqa_logits(TensorView q,
    Optional<TensorView> q_sf,
    TensorView fused_kv_cache,
    TensorView weights,
    TensorView metadata,
    int64_t num_max_sparse_blocks,
    int64_t sparse_block_kv) {
    auto result = attention::fp8_fp4_paged_sparse_mqa_logits(
        std::make_tuple(convert_to_torch_tensor(q), to_optional_tensor(q_sf)),
        convert_to_torch_tensor(fused_kv_cache),
        convert_to_torch_tensor(weights),
        convert_to_torch_tensor(metadata),
        static_cast<int>(num_max_sparse_blocks),
        static_cast<int>(sparse_block_kv));
    return Tensor::FromDLPack(at::toDLPack(result));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_paged_sparse_mqa_logits, dg_fp8_fp4_paged_sparse_mqa_logits);

void dg_mega_mhc(TensorView x,
    TensorView residual,
    Optional<TensorView> shifted_prev_mix,
    TensorView post_mix,
    TensorView comb_res_mix,
    TensorView fn,
    TensorView mix_scales,
    TensorView mix_bases,
    int64_t hc_mult,
    double hc_norm_eps,
    double hc_pre_eps,
    double hc_post_scale,
    double sinkhorn_eps,
    int64_t num_sinkhorn_iters,
    TensorView rmsnorm_weight,
    double rmsnorm_eps,
    double rmsnorm_scale,
    TensorView new_residual,
    Optional<TensorView> new_prev_mix,
    TensorView new_post_mix,
    TensorView new_comb_res_mix,
    Optional<TensorView> y_bf16,
    Optional<TensorView> y_fp8,
    Optional<TensorView> y_gemm_sf,
    Optional<TensorView> y_routed_sf,
    Optional<TensorView> y_shared_sf,
    int64_t shared_sf_block_m,
    Optional<TensorView> y_shared_sf_storage) {
    DG_HOST_ASSERT(y_shared_sf.has_value() == y_shared_sf_storage.has_value());
    auto shared_sf = to_optional_tensor(y_shared_sf);
    if (shared_sf.has_value()) {
        auto storage = convert_to_torch_tensor(y_shared_sf_storage.value());
        DG_HOST_ASSERT(storage.dim() == 1 and storage.is_contiguous());
        DG_HOST_ASSERT(storage.scalar_type() == shared_sf->scalar_type());
        DG_HOST_ASSERT(storage.device() == shared_sf->device());
        DG_HOST_ASSERT(storage.data_ptr() == shared_sf->data_ptr() or shared_sf->numel() == 0);
        // Preserve the actual allocation extent for the host API's padded-row
        // bounds check; a TensorView alone only carries the logical SF shape.
        shared_sf = storage.as_strided(shared_sf->sizes(), shared_sf->strides());
    }
    mega_mhc::mega_mhc(
        convert_to_torch_tensor(x),
        convert_to_torch_tensor(residual),
        to_optional_tensor(shifted_prev_mix),
        convert_to_torch_tensor(post_mix),
        convert_to_torch_tensor(comb_res_mix),
        convert_to_torch_tensor(fn),
        convert_to_torch_tensor(mix_scales),
        convert_to_torch_tensor(mix_bases),
        static_cast<int>(hc_mult),
        static_cast<float>(hc_norm_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_post_scale),
        static_cast<float>(sinkhorn_eps),
        static_cast<int>(num_sinkhorn_iters),
        convert_to_torch_tensor(rmsnorm_weight),
        static_cast<float>(rmsnorm_eps),
        static_cast<float>(rmsnorm_scale),
        convert_to_torch_tensor(new_residual),
        to_optional_tensor(new_prev_mix),
        convert_to_torch_tensor(new_post_mix),
        convert_to_torch_tensor(new_comb_res_mix),
        to_optional_tensor(y_bf16),
        to_optional_tensor(y_fp8),
        to_optional_tensor(y_gemm_sf),
        to_optional_tensor(y_routed_sf),
        shared_sf,
        static_cast<int>(shared_sf_block_m));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_mhc, dg_mega_mhc);

Tuple<Tensor, Tensor> dg_bf16_mega_gate(TensorView x,
    TensorView weight,
    int64_t num_topk,
    bool use_shared_as_routed,
    int64_t num_shared_experts,
    double routed_scaling_factor,
    int64_t ep_rank,
    std::string scoring_func,
    Optional<TensorView> mask,
    Optional<TensorView> bias,
    Optional<TensorView> image_bias,
    Optional<TensorView> image_token_mask,
    Optional<TensorView> fix_routing_mask,
    Optional<TensorView> to_physical_map,
    Optional<TensorView> logical_count,
    Optional<TensorView> unmapped_topk_idx,
    Optional<TensorView> force_random,
    Optional<TensorView> out_idx,
    Optional<TensorView> out_weights) {
    DG_HOST_ASSERT(out_idx.has_value() == out_weights.has_value());
    std::optional<std::tuple<torch::Tensor, torch::Tensor>> out_opt;
    if (out_idx.has_value())
        out_opt = std::make_tuple(convert_to_torch_tensor(out_idx.value()), convert_to_torch_tensor(out_weights.value()));
    auto result = mega_gate::bf16_mega_gate(
        convert_to_torch_tensor(x),
        convert_to_torch_tensor(weight),
        static_cast<int>(num_topk),
        use_shared_as_routed,
        static_cast<int>(num_shared_experts),
        static_cast<float>(routed_scaling_factor),
        static_cast<int>(ep_rank),
        scoring_func,
        to_optional_tensor(mask),
        to_optional_tensor(bias),
        to_optional_tensor(image_bias),
        to_optional_tensor(image_token_mask),
        to_optional_tensor(fix_routing_mask),
        to_optional_tensor(to_physical_map),
        to_optional_tensor(logical_count),
        to_optional_tensor(unmapped_topk_idx),
        to_optional_tensor(force_random),
        out_opt);
    return tvm::ffi::Tuple<Tensor, Tensor>(
        Tensor::FromDLPack(at::toDLPack(std::get<0>(result))),
        Tensor::FromDLPack(at::toDLPack(std::get<1>(result))));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_mega_gate, dg_bf16_mega_gate);

void dg_batched_syrk(TensorView a,
    TensorView d) {
    gemm::batched_syrk(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(d));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batched_syrk, dg_batched_syrk);

void dg_batched_symm(TensorView a,
    TensorView b,
    TensorView d) {
    gemm::batched_symm(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batched_symm, dg_batched_symm);

void dg_cublaslt_nvfp4_gemm_nt(TensorView a,
    TensorView a_sf,
    TensorView b,
    TensorView b_sf,
    TensorView d,
    Optional<TensorView> c) {
    gemm::cublaslt_nvfp4_gemm_nt(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d),
        to_optional_tensor(c));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_nvfp4_gemm_nt, dg_cublaslt_nvfp4_gemm_nt);

Map<String, int64_t> dg_get_bf16_mega_gate_config(int64_t num_tokens, int64_t hidden,
    int64_t num_routed_experts, int64_t num_topk) {
    const auto config = mega_gate::get_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts, num_topk);
    Map<String, int64_t> result;
    for (const auto& [key, value] : config)
        result.Set(key, value);
    return result;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_bf16_mega_gate_config, dg_get_bf16_mega_gate_config);

void dg_k_grouped_fp4_gemm_nt_contiguous(TensorView a, TensorView a_sf,
                                         TensorView b, TensorView b_sf,
                                         TensorView d,
                                         Optional<Array<int64_t>> ks,
                                         TensorView grouped_layout,
                                         Optional<TensorView> c,
                                         Tuple<int64_t, int64_t, int64_t> recipe,
                                         std::string compiled_dims,
                                         bool use_psum_layout) {
    auto ks_val = to_optional_int_vector(ks);
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe.get<0>()),
                                      static_cast<int>(recipe.get<1>()),
                                      static_cast<int>(recipe.get<2>()));
    gemm::k_grouped_fp4_gemm_nt_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), ks_val, convert_to_torch_tensor(grouped_layout),
        c_opt, recipe_val, compiled_dims, use_psum_layout
    );
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_grouped_fp4_gemm_nt_contiguous, dg_k_grouped_fp4_gemm_nt_contiguous);
