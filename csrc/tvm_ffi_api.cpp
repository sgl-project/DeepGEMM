#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/runtime.hpp"
#include "utils/torch_compat.hpp"

using namespace deep_gemm;

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
void dg_init(std::string library_root_path, std::string cuda_home) {
#if DG_TENSORMAP_COMPATIBLE
    Compiler::prepare_init(library_root_path, cuda_home);
    KernelRuntime::prepare_init(cuda_home);
#endif
}

int64_t dg_get_num_sms() { return device_runtime->get_num_sms(); }
void dg_set_num_sms(int64_t n) { device_runtime->set_num_sms(static_cast<int>(n)); }
int64_t dg_get_compile_mode() { return device_runtime->get_compile_mode(); }
void dg_set_compile_mode(int64_t n) { device_runtime->set_compile_mode(static_cast<int>(n)); }
int64_t dg_get_tc_util() { return device_runtime->get_tc_util(); }
void dg_set_tc_util(int64_t n) { device_runtime->set_tc_util(static_cast<int>(n)); }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, dg_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_num_sms, dg_get_num_sms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_num_sms, dg_set_num_sms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_compile_mode, dg_get_compile_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_compile_mode, dg_set_compile_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_tc_util, dg_get_tc_util);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_tc_util, dg_set_tc_util);

// ---------------------------------------------------------------------------
// Layout utilities
// ---------------------------------------------------------------------------
int64_t dg_get_tma_aligned_size(int64_t mn, int64_t element_size) {
    return get_tma_aligned_size(static_cast<int>(mn), static_cast<int>(element_size));
}

int64_t dg_get_mk_alignment_for_contiguous_layout() {
    return get_mk_alignment_for_contiguous_layout();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_tma_aligned_size, dg_get_tma_aligned_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mk_alignment_for_contiguous_layout, dg_get_mk_alignment_for_contiguous_layout);

// ---------------------------------------------------------------------------
// Layout kernels
// ---------------------------------------------------------------------------
#if DG_TENSORMAP_COMPATIBLE

tvm::ffi::Array<int64_t> dg_preprocess_sf(DLTensor* sf) {
    auto sf_v = dl_to_torch(sf);
    auto [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf_v);
    return {static_cast<int64_t>(dim),
            static_cast<int64_t>(ng),
            static_cast<int64_t>(mn_pp),
            static_cast<int64_t>(sf_k_pp),
            static_cast<int64_t>(tma_mn)};
}

void dg_get_mn_major_tma_aligned_tensor(DLTensor* sf, DLTensor* out) {
    auto sf_v = dl_to_torch(sf);
    auto out_v = dl_to_torch(out);
    auto result = get_mn_major_tma_aligned_tensor(sf_v);
    out_v.copy_(result);
}

void dg_get_mn_major_tma_aligned_packed_ue8m0_tensor(DLTensor* sf, DLTensor* out) {
    auto sf_v = dl_to_torch(sf);
    auto out_v = dl_to_torch(out);
    auto result = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_v);
    out_v.copy_(result);
}

void dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        DLTensor* sf, DLTensor* ks_tensor, DLTensor* ks_list,
        int64_t num_groups, DLTensor* out) {
    auto sf_v = dl_to_torch(sf);
    auto ks_tensor_v = dl_to_torch(ks_tensor);
    auto out_v = dl_to_torch(out);
    std::vector<int> ks;
    ks.reserve(num_groups);
    auto* ks_ptr = static_cast<int32_t*>(ks_list->data);
    for (int64_t i = 0; i < num_groups; ++i) {
        ks.push_back(static_cast<int>(ks_ptr[i]));
    }
    auto result = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf_v, ks_tensor_v, ks);
    out_v.copy_(result);
}

void dg_transform_sf_into_required_layout(
        DLTensor* sf, int64_t mn, int64_t k,
        int64_t recipe0, int64_t recipe1, int64_t recipe2,
        int64_t recipe_ab0, int64_t recipe_ab1,
        int64_t num_groups, bool is_sfa, bool disable_ue8m0_cast,
        DLTensor* out) {
    auto sf_v = dl_to_torch(sf);
    std::optional<std::tuple<int, int, int>> recipe =
        recipe0 >= 0 ? std::make_optional(std::make_tuple(
            static_cast<int>(recipe0), static_cast<int>(recipe1), static_cast<int>(recipe2)))
        : std::nullopt;
    std::optional<std::tuple<int, int>> recipe_ab =
        recipe_ab0 >= 0 ? std::make_optional(std::make_tuple(
            static_cast<int>(recipe_ab0), static_cast<int>(recipe_ab1)))
        : std::nullopt;
    std::optional<int> ng = num_groups >= 0 ? std::make_optional(static_cast<int>(num_groups)) : std::nullopt;
    auto result = layout::transform_sf_into_required_layout(
        sf_v, static_cast<int>(mn), static_cast<int>(k),
        recipe, recipe_ab, ng, is_sfa, disable_ue8m0_cast);
    if (result.defined() && result.numel() > 0) {
        auto out_v = dl_to_torch(out);
        out_v.copy_(result);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(preprocess_sf, dg_preprocess_sf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mn_major_tma_aligned_tensor, dg_get_mn_major_tma_aligned_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mn_major_tma_aligned_packed_ue8m0_tensor, dg_get_mn_major_tma_aligned_packed_ue8m0_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor, dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transform_sf_into_required_layout, dg_transform_sf_into_required_layout);

#endif  // DG_TENSORMAP_COMPATIBLE

// ---------------------------------------------------------------------------
// cuBLASLt GEMMs (always available)
// ---------------------------------------------------------------------------
void dg_cublaslt_gemm_nt(DLTensor* a, DLTensor* b, DLTensor* d) {
    gemm::cublaslt_gemm_nt(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), std::nullopt);
}
void dg_cublaslt_gemm_nn(DLTensor* a, DLTensor* b, DLTensor* d) {
    gemm::cublaslt_gemm_nn(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), std::nullopt);
}
void dg_cublaslt_gemm_tn(DLTensor* a, DLTensor* b, DLTensor* d) {
    gemm::cublaslt_gemm_tn(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), std::nullopt);
}
void dg_cublaslt_gemm_tt(DLTensor* a, DLTensor* b, DLTensor* d) {
    gemm::cublaslt_gemm_tt(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), std::nullopt);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_nt, dg_cublaslt_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_nn, dg_cublaslt_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_tn, dg_cublaslt_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_tt, dg_cublaslt_gemm_tt);

// ---------------------------------------------------------------------------
// FP8/FP4 GEMMs and BF16 GEMMs
// ---------------------------------------------------------------------------
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

void dg_fp8_fp4_gemm_nt(DLTensor* a_data, DLTensor* a_sf,
                         DLTensor* b_data, DLTensor* b_sf,
                         DLTensor* d,
                         int64_t has_c, DLTensor* c_or_dummy,
                         int64_t r0, int64_t r1, int64_t r2,
                         int64_t ra0, int64_t ra1,
                         int64_t rb0, int64_t rb1,
                         std::string compiled_dims,
                         bool disable_ue8m0_cast) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    auto recipe = r0 >= 0 ? std::make_optional(std::make_tuple((int)r0, (int)r1, (int)r2)) : std::nullopt;
    auto recipe_a = ra0 >= 0 ? std::make_optional(std::make_tuple((int)ra0, (int)ra1)) : std::nullopt;
    auto recipe_b = rb0 >= 0 ? std::make_optional(std::make_tuple((int)rb0, (int)rb1)) : std::nullopt;
    gemm::fp8_fp4_gemm_nt(std::make_pair(dl_to_torch(a_data), dl_to_torch(a_sf)),
                           std::make_pair(dl_to_torch(b_data), dl_to_torch(b_sf)),
                           dl_to_torch(d), c_opt,
                           recipe, recipe_a, recipe_b,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_fp4_gemm_nn(DLTensor* a_data, DLTensor* a_sf,
                         DLTensor* b_data, DLTensor* b_sf,
                         DLTensor* d,
                         int64_t has_c, DLTensor* c_or_dummy,
                         int64_t r0, int64_t r1, int64_t r2,
                         int64_t ra0, int64_t ra1,
                         int64_t rb0, int64_t rb1,
                         std::string compiled_dims,
                         bool disable_ue8m0_cast) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    auto recipe = r0 >= 0 ? std::make_optional(std::make_tuple((int)r0, (int)r1, (int)r2)) : std::nullopt;
    auto recipe_a = ra0 >= 0 ? std::make_optional(std::make_tuple((int)ra0, (int)ra1)) : std::nullopt;
    auto recipe_b = rb0 >= 0 ? std::make_optional(std::make_tuple((int)rb0, (int)rb1)) : std::nullopt;
    gemm::fp8_fp4_gemm_nn(std::make_pair(dl_to_torch(a_data), dl_to_torch(a_sf)),
                           std::make_pair(dl_to_torch(b_data), dl_to_torch(b_sf)),
                           dl_to_torch(d), c_opt,
                           recipe, recipe_a, recipe_b,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_fp4_gemm_tn(DLTensor* a_data, DLTensor* a_sf,
                         DLTensor* b_data, DLTensor* b_sf,
                         DLTensor* d,
                         int64_t has_c, DLTensor* c_or_dummy,
                         int64_t r0, int64_t r1, int64_t r2,
                         int64_t ra0, int64_t ra1,
                         int64_t rb0, int64_t rb1,
                         std::string compiled_dims,
                         bool disable_ue8m0_cast) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    auto recipe = r0 >= 0 ? std::make_optional(std::make_tuple((int)r0, (int)r1, (int)r2)) : std::nullopt;
    auto recipe_a = ra0 >= 0 ? std::make_optional(std::make_tuple((int)ra0, (int)ra1)) : std::nullopt;
    auto recipe_b = rb0 >= 0 ? std::make_optional(std::make_tuple((int)rb0, (int)rb1)) : std::nullopt;
    gemm::fp8_fp4_gemm_tn(std::make_pair(dl_to_torch(a_data), dl_to_torch(a_sf)),
                           std::make_pair(dl_to_torch(b_data), dl_to_torch(b_sf)),
                           dl_to_torch(d), c_opt,
                           recipe, recipe_a, recipe_b,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_fp4_gemm_tt(DLTensor* a_data, DLTensor* a_sf,
                         DLTensor* b_data, DLTensor* b_sf,
                         DLTensor* d,
                         int64_t has_c, DLTensor* c_or_dummy,
                         int64_t r0, int64_t r1, int64_t r2,
                         int64_t ra0, int64_t ra1,
                         int64_t rb0, int64_t rb1,
                         std::string compiled_dims,
                         bool disable_ue8m0_cast) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    auto recipe = r0 >= 0 ? std::make_optional(std::make_tuple((int)r0, (int)r1, (int)r2)) : std::nullopt;
    auto recipe_a = ra0 >= 0 ? std::make_optional(std::make_tuple((int)ra0, (int)ra1)) : std::nullopt;
    auto recipe_b = rb0 >= 0 ? std::make_optional(std::make_tuple((int)rb0, (int)rb1)) : std::nullopt;
    gemm::fp8_fp4_gemm_tt(std::make_pair(dl_to_torch(a_data), dl_to_torch(a_sf)),
                           std::make_pair(dl_to_torch(b_data), dl_to_torch(b_sf)),
                           dl_to_torch(d), c_opt,
                           recipe, recipe_a, recipe_b,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_bf16_gemm_nt(DLTensor* a, DLTensor* b, DLTensor* d,
                     int64_t has_c, DLTensor* c_or_dummy,
                     std::string compiled_dims) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    gemm::bf16_gemm_nt(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), c_opt, compiled_dims);
}
void dg_bf16_gemm_nn(DLTensor* a, DLTensor* b, DLTensor* d,
                     int64_t has_c, DLTensor* c_or_dummy,
                     std::string compiled_dims) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    gemm::bf16_gemm_nn(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), c_opt, compiled_dims);
}
void dg_bf16_gemm_tn(DLTensor* a, DLTensor* b, DLTensor* d,
                     int64_t has_c, DLTensor* c_or_dummy,
                     std::string compiled_dims) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    gemm::bf16_gemm_tn(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), c_opt, compiled_dims);
}
void dg_bf16_gemm_tt(DLTensor* a, DLTensor* b, DLTensor* d,
                     int64_t has_c, DLTensor* c_or_dummy,
                     std::string compiled_dims) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    gemm::bf16_gemm_tt(dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), c_opt, compiled_dims);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_nt, dg_fp8_fp4_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_nn, dg_fp8_fp4_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_tn, dg_fp8_fp4_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_tt, dg_fp8_fp4_gemm_tt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nt, dg_bf16_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nn, dg_bf16_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tn, dg_bf16_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tt, dg_bf16_gemm_tt);

// Einsum
void dg_einsum(std::string expr, DLTensor* a, DLTensor* b, DLTensor* d,
               int64_t has_c, DLTensor* c_or_dummy, bool use_cublaslt) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    einsum::einsum(expr, dl_to_torch(a), dl_to_torch(b), dl_to_torch(d), c_opt, use_cublaslt);
}

void dg_fp8_einsum(std::string expr,
                   DLTensor* a_data, DLTensor* a_sf,
                   DLTensor* b_data, DLTensor* b_sf,
                   DLTensor* d,
                   int64_t has_c, DLTensor* c_or_dummy,
                   int64_t r0, int64_t r1, int64_t r2) {
    auto c_opt = has_c ? std::make_optional(dl_to_torch(c_or_dummy)) : std::nullopt;
    auto recipe = std::make_tuple((int)r0, (int)r1, (int)r2);
    einsum::fp8_einsum(expr, std::make_pair(dl_to_torch(a_data), dl_to_torch(a_sf)),
                       std::make_pair(dl_to_torch(b_data), dl_to_torch(b_sf)),
                       dl_to_torch(d), c_opt, recipe);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(einsum, dg_einsum);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_einsum, dg_fp8_einsum);

// Hyperconnection
void dg_tf32_hc_prenorm_gemm(DLTensor* a, DLTensor* b, DLTensor* d,
                              DLTensor* sqr_sum, int64_t num_splits) {
    auto ns = num_splits > 0 ? std::make_optional(static_cast<int>(num_splits)) : std::nullopt;
    hyperconnection::tf32_hc_prenorm_gemm(dl_to_torch(a), dl_to_torch(b),
                                          dl_to_torch(d), dl_to_torch(sqr_sum), ns);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tf32_hc_prenorm_gemm, dg_tf32_hc_prenorm_gemm);

// Attention
void dg_fp8_gemm_nt_skip_head_mid(DLTensor* a_data, DLTensor* a_sf,
                                   DLTensor* b_data, DLTensor* b_sf,
                                   DLTensor* d,
                                   int64_t left, int64_t mid, int64_t right,
                                   int64_t r0, int64_t r1, int64_t r2,
                                   std::string compiled_dims,
                                   bool disable_ue8m0_cast) {
    auto head_splits = std::make_tuple((int)left, (int)mid, (int)right);
    auto recipe = r0 >= 0 ? std::make_optional(std::make_tuple((int)r0, (int)r1, (int)r2)) : std::nullopt;
    attention::fp8_gemm_nt_skip_head_mid(
        std::make_pair(dl_to_torch(a_data), dl_to_torch(a_sf)),
        std::make_pair(dl_to_torch(b_data), dl_to_torch(b_sf)),
        dl_to_torch(d), head_splits, recipe, compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_mqa_logits(DLTensor* q, DLTensor* kv_data, DLTensor* kv_sf,
                        DLTensor* weights, DLTensor* cu_seq_len_k_start,
                        DLTensor* cu_seq_len_k_end,
                        bool clean_logits, int64_t max_seqlen_k,
                        DLTensor* out) {
    auto result = attention::fp8_mqa_logits(
        dl_to_torch(q),
        std::make_pair(dl_to_torch(kv_data), dl_to_torch(kv_sf)),
        dl_to_torch(weights), dl_to_torch(cu_seq_len_k_start),
        dl_to_torch(cu_seq_len_k_end), clean_logits, static_cast<int>(max_seqlen_k));
    if (result.numel() > 0) {
        auto out_v = dl_to_torch(out);
        out_v.copy_(result);
    }
}

void dg_get_paged_mqa_logits_metadata(DLTensor* context_lens, int64_t block_kv,
                                       int64_t num_sms, DLTensor* out) {
    auto result = attention::get_paged_mqa_logits_metadata(
        dl_to_torch(context_lens), static_cast<int>(block_kv),
        static_cast<int>(num_sms));
    auto out_v = dl_to_torch(out);
    out_v.copy_(result);
}

void dg_fp8_paged_mqa_logits(DLTensor* q, DLTensor* fused_kv_cache,
                              DLTensor* weights, DLTensor* context_lens,
                              DLTensor* block_table, DLTensor* schedule_meta,
                              int64_t max_context_len, bool clean_logits,
                              DLTensor* out) {
    auto result = attention::fp8_paged_mqa_logits(
        dl_to_torch(q), dl_to_torch(fused_kv_cache),
        dl_to_torch(weights), dl_to_torch(context_lens),
        dl_to_torch(block_table), dl_to_torch(schedule_meta),
        static_cast<int>(max_context_len), clean_logits);
    if (result.numel() > 0) {
        auto out_v = dl_to_torch(out);
        out_v.copy_(result);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm_nt_skip_head_mid, dg_fp8_gemm_nt_skip_head_mid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mqa_logits, dg_fp8_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_paged_mqa_logits_metadata, dg_get_paged_mqa_logits_metadata);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_paged_mqa_logits, dg_fp8_paged_mqa_logits);

#endif  // DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
