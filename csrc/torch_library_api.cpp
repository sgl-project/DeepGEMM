// `TORCH_LIBRARY` bindings for the sgl-deep-gemm `_C.so`.
// The library has no CPython dependency: it is loaded with `torch.ops.load_library`, so a
// single build serves every Python version. Ops with tensor arguments are registered for the
// CUDA dispatch key; scalar-only runtime/config ops use catch-all kernels.
#define DG_USE_TORCH_LIBRARY 1
// The dispatcher already releases the GIL around `torch.ops` calls, so DeepJIT need not
// (and must not, without pybind11) do it again.
#define DJ_DISABLE_GIL 1

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <torch/library.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_gate.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/sm90_mega.hpp"

using namespace deep_gemm;
using torch::Tensor;
using OptTensor = std::optional<Tensor>;

namespace {

std::optional<int> to_optional_int(const std::optional<int64_t>& value) {
    return value.has_value() ? std::make_optional(static_cast<int>(value.value())) : std::nullopt;
}

std::optional<float> to_optional_float(const std::optional<double>& value) {
    return value.has_value() ? std::make_optional(static_cast<float>(value.value())) : std::nullopt;
}

std::optional<std::vector<int>> to_optional_int_vector(const at::OptionalIntArrayRef& values) {
    if (not values.has_value())
        return std::nullopt;
    return std::vector<int>(values->begin(), values->end());
}

std::tuple<int, int, int> to_tuple3(const at::IntArrayRef& values) {
    DG_HOST_ASSERT(values.size() == 3);
    return {static_cast<int>(values[0]), static_cast<int>(values[1]), static_cast<int>(values[2])};
}

std::optional<std::tuple<int, int, int>> to_optional_tuple3(const at::OptionalIntArrayRef& values) {
    return values.has_value() ? std::make_optional(to_tuple3(values.value())) : std::nullopt;
}

std::optional<std::tuple<int, int>> to_optional_tuple2(const at::OptionalIntArrayRef& values) {
    if (not values.has_value())
        return std::nullopt;
    DG_HOST_ASSERT(values->size() == 2);
    return std::make_tuple(static_cast<int>((*values)[0]), static_cast<int>((*values)[1]));
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
void dg_init(const std::string& library_root_path, const std::string& cuda_home) {
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

int64_t dg_get_theoretical_mk_alignment_for_contiguous_layout(std::optional<int64_t> expected_m, std::optional<int64_t> num_groups) {
    return heuristics_runtime->get_theoretical_mk_alignment_for_contiguous_layout(
        to_optional_int(expected_m), to_optional_int(num_groups));
}

// ---------------------------------------------------------------------------
// Layout kernels
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> dg_preprocess_sf(const Tensor& sf) {
    auto [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf);
    return {dim, ng, mn_pp, sf_k_pp, tma_mn};
}

Tensor dg_get_mn_major_tma_aligned_tensor(const Tensor& sf) {
    return get_mn_major_tma_aligned_tensor(sf);
}

Tensor dg_get_mn_major_tma_aligned_packed_ue8m0_tensor(const Tensor& sf, const OptTensor& psum_layout) {
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, psum_layout);
}

Tensor dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        const Tensor& sf, const Tensor& grouped_layout, at::OptionalIntArrayRef ks,
        int64_t gran_k, int64_t k_alignment, bool use_psum_layout, bool use_padded_sf_layout) {
    return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf, grouped_layout, to_optional_int_vector(ks),
        static_cast<int>(gran_k), static_cast<int>(k_alignment),
        use_psum_layout, use_padded_sf_layout);
}

Tensor dg_transform_sf_into_required_layout(
        const Tensor& sf, int64_t mn, int64_t k,
        int64_t recipe_a, int64_t recipe_b, std::optional<int64_t> recipe_c,
        std::optional<int64_t> num_groups, std::optional<bool> is_sfa,
        bool disable_ue8m0_cast, const OptTensor& psum_layout) {
    const auto ng = to_optional_int(num_groups);
    if (recipe_c.has_value()) {
        const auto recipe = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c.value()));
        return layout::transform_sf_into_required_layout(
            sf, static_cast<int>(mn), static_cast<int>(k), recipe, ng, is_sfa, disable_ue8m0_cast, psum_layout);
    }
    const auto recipe = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b));
    return layout::transform_sf_into_required_layout(
        sf, static_cast<int>(mn), static_cast<int>(k), recipe, ng, is_sfa, disable_ue8m0_cast, psum_layout);
}

// ---------------------------------------------------------------------------
// cuBLASLt GEMMs
// ---------------------------------------------------------------------------
void dg_cublaslt_gemm_nt(const Tensor& a, const Tensor& b, const Tensor& d, const OptTensor& c) {
    gemm::cublaslt_gemm_nt(a, b, d, c);
}
void dg_cublaslt_gemm_nn(const Tensor& a, const Tensor& b, const Tensor& d, const OptTensor& c) {
    gemm::cublaslt_gemm_nn(a, b, d, c);
}
void dg_cublaslt_gemm_tn(const Tensor& a, const Tensor& b, const Tensor& d, const OptTensor& c) {
    gemm::cublaslt_gemm_tn(a, b, d, c);
}
void dg_cublaslt_gemm_tt(const Tensor& a, const Tensor& b, const Tensor& d, const OptTensor& c) {
    gemm::cublaslt_gemm_tt(a, b, d, c);
}

void dg_cublaslt_nvfp4_gemm_nt(const Tensor& a, const Tensor& a_sf, const Tensor& b, const Tensor& b_sf,
                               const Tensor& d, const OptTensor& c) {
    gemm::cublaslt_nvfp4_gemm_nt({a, a_sf}, {b, b_sf}, d, c);
}

// ---------------------------------------------------------------------------
// FP8/FP4 GEMMs and BF16 GEMMs
// ---------------------------------------------------------------------------
#define DG_DEFINE_FP8_FP4_GEMM(name)                                                               \
void dg_##name(const Tensor& a, const Tensor& a_sf, const Tensor& b, const Tensor& b_sf,            \
               const Tensor& d, const OptTensor& c,                                                \
               at::OptionalIntArrayRef recipe, at::OptionalIntArrayRef recipe_a,                   \
               at::OptionalIntArrayRef recipe_b, const std::string& compiled_dims,                 \
               bool disable_ue8m0_cast, std::optional<double> alpha) {                             \
    gemm::name({a, a_sf}, {b, b_sf}, d, c,                                                         \
               to_optional_tuple3(recipe), to_optional_tuple2(recipe_a), to_optional_tuple2(recipe_b), \
               compiled_dims, disable_ue8m0_cast, to_optional_float(alpha));                       \
}
DG_DEFINE_FP8_FP4_GEMM(fp8_fp4_gemm_nt)
DG_DEFINE_FP8_FP4_GEMM(fp8_fp4_gemm_nn)
DG_DEFINE_FP8_FP4_GEMM(fp8_fp4_gemm_tn)
DG_DEFINE_FP8_FP4_GEMM(fp8_fp4_gemm_tt)
#undef DG_DEFINE_FP8_FP4_GEMM

void dg_m_grouped_fp8_fp4_gemm_nt_contiguous(const Tensor& a, const Tensor& a_sf,
                                             const Tensor& b, const Tensor& b_sf,
                                             const Tensor& d, const Tensor& grouped_layout,
                                             at::OptionalIntArrayRef recipe,
                                             at::OptionalIntArrayRef recipe_a,
                                             at::OptionalIntArrayRef recipe_b,
                                             const std::string& compiled_dims,
                                             bool disable_ue8m0_cast, bool use_psum_layout,
                                             bool ensure_zero_padding,
                                             std::optional<int64_t> expected_m_for_psum_layout) {
    gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
        {a, a_sf}, {b, b_sf}, d, grouped_layout,
        to_optional_tuple3(recipe), to_optional_tuple2(recipe_a), to_optional_tuple2(recipe_b),
        compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding,
        to_optional_int(expected_m_for_psum_layout));
}

void dg_m_grouped_fp8_fp4_gemm_nn_contiguous(const Tensor& a, const Tensor& a_sf,
                                             const Tensor& b, const Tensor& b_sf,
                                             const Tensor& d, const Tensor& grouped_layout,
                                             at::OptionalIntArrayRef recipe,
                                             at::OptionalIntArrayRef recipe_a,
                                             at::OptionalIntArrayRef recipe_b,
                                             const std::string& compiled_dims,
                                             bool disable_ue8m0_cast, bool use_psum_layout,
                                             bool ensure_zero_padding) {
    gemm::m_grouped_fp8_fp4_gemm_nn_contiguous(
        {a, a_sf}, {b, b_sf}, d, grouped_layout,
        to_optional_tuple3(recipe), to_optional_tuple2(recipe_a), to_optional_tuple2(recipe_b),
        compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding);
}

void dg_m_grouped_fp8_fp4_gemm_nt_masked(const Tensor& a, const Tensor& a_sf,
                                         const Tensor& b, const Tensor& b_sf,
                                         const Tensor& d, const Tensor& masked_m, int64_t expected_m,
                                         at::OptionalIntArrayRef recipe,
                                         at::OptionalIntArrayRef recipe_a,
                                         at::OptionalIntArrayRef recipe_b,
                                         const std::string& compiled_dims, bool disable_ue8m0_cast) {
    gemm::m_grouped_fp8_fp4_gemm_nt_masked(
        {a, a_sf}, {b, b_sf}, d, masked_m, static_cast<int>(expected_m),
        to_optional_tuple3(recipe), to_optional_tuple2(recipe_a), to_optional_tuple2(recipe_b),
        compiled_dims, disable_ue8m0_cast);
}

#define DG_DEFINE_BF16_GEMM(name)                                                                  \
void dg_##name(const Tensor& a, const Tensor& b, const Tensor& d, const OptTensor& c,              \
               const std::string& compiled_dims, std::optional<double> alpha) {                    \
    gemm::name(a, b, d, c, compiled_dims, to_optional_float(alpha));                               \
}
DG_DEFINE_BF16_GEMM(bf16_gemm_nt)
DG_DEFINE_BF16_GEMM(bf16_gemm_nn)
DG_DEFINE_BF16_GEMM(bf16_gemm_tn)
DG_DEFINE_BF16_GEMM(bf16_gemm_tt)
#undef DG_DEFINE_BF16_GEMM

void dg_m_grouped_bf16_gemm_nt_contiguous(const Tensor& a, const Tensor& b, const Tensor& d,
                                          const Tensor& grouped_layout, const std::string& compiled_dims,
                                          bool use_psum_layout, bool ensure_zero_padding,
                                          std::optional<int64_t> expected_m_for_psum_layout) {
    gemm::m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims,
                                            use_psum_layout, ensure_zero_padding,
                                            to_optional_int(expected_m_for_psum_layout));
}

void dg_m_grouped_bf16_gemm_nn_contiguous(const Tensor& a, const Tensor& b, const Tensor& d,
                                          const Tensor& grouped_layout, const std::string& compiled_dims,
                                          bool use_psum_layout, bool ensure_zero_padding) {
    gemm::m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims,
                                            use_psum_layout, ensure_zero_padding);
}

void dg_m_grouped_bf16_gemm_nt_masked(const Tensor& a, const Tensor& b, const Tensor& d,
                                      const Tensor& masked_m, int64_t expected_m,
                                      const std::string& compiled_dims) {
    gemm::m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, static_cast<int>(expected_m), compiled_dims);
}

void dg_k_grouped_bf16_gemm_tn_contiguous(const Tensor& a, const Tensor& b, const Tensor& d,
                                          at::OptionalIntArrayRef ks, const Tensor& grouped_layout,
                                          const OptTensor& c, const std::string& compiled_dims,
                                          bool use_psum_layout) {
    gemm::k_grouped_bf16_gemm_tn_contiguous(a, b, d, to_optional_int_vector(ks), grouped_layout,
                                            c, compiled_dims, use_psum_layout);
}

void dg_k_grouped_fp8_gemm_tn_contiguous(const Tensor& a, const Tensor& a_sf,
                                         const Tensor& b, const Tensor& b_sf,
                                         const Tensor& d, at::OptionalIntArrayRef ks,
                                         const Tensor& grouped_layout, const OptTensor& c,
                                         at::IntArrayRef recipe, const std::string& compiled_dims,
                                         bool use_psum_layout, bool use_padded_sf_layout) {
    gemm::k_grouped_fp8_gemm_tn_contiguous(
        {a, a_sf}, {b, b_sf}, d, to_optional_int_vector(ks), grouped_layout,
        c, to_tuple3(recipe), compiled_dims, use_psum_layout, use_padded_sf_layout);
}

void dg_k_grouped_fp8_gemm_nt_contiguous(const Tensor& a, const Tensor& a_sf,
                                         const Tensor& b, const Tensor& b_sf,
                                         const Tensor& d, at::OptionalIntArrayRef ks,
                                         const Tensor& grouped_layout, const OptTensor& c,
                                         at::IntArrayRef recipe, const std::string& compiled_dims,
                                         bool use_psum_layout, bool use_padded_sf_layout) {
    gemm::k_grouped_fp8_gemm_nt_contiguous(
        {a, a_sf}, {b, b_sf}, d, to_optional_int_vector(ks), grouped_layout,
        c, to_tuple3(recipe), compiled_dims, use_psum_layout, use_padded_sf_layout);
}

void dg_k_grouped_fp4_gemm_nt_contiguous(const Tensor& a, const Tensor& a_sf,
                                         const Tensor& b, const Tensor& b_sf,
                                         const Tensor& d, at::OptionalIntArrayRef ks,
                                         const Tensor& grouped_layout, const OptTensor& c,
                                         at::IntArrayRef recipe, const std::string& compiled_dims,
                                         bool use_psum_layout) {
    gemm::k_grouped_fp4_gemm_nt_contiguous(
        {a, a_sf}, {b, b_sf}, d, to_optional_int_vector(ks), grouped_layout,
        c, to_tuple3(recipe), compiled_dims, use_psum_layout);
}

void dg_batched_syrk(const Tensor& a, const Tensor& d) {
    gemm::batched_syrk(a, d);
}

void dg_batched_symm(const Tensor& a, const Tensor& b, const Tensor& d) {
    gemm::batched_symm(a, b, d);
}

// ---------------------------------------------------------------------------
// Einsum
// ---------------------------------------------------------------------------
void dg_einsum(const std::string& expr, const Tensor& a, const Tensor& b, const Tensor& d,
               const OptTensor& c, bool use_cublaslt) {
    einsum::einsum(expr, a, b, d, c, use_cublaslt);
}

void dg_fp8_einsum(const std::string& expr,
                   const Tensor& a_data, const Tensor& a_sf,
                   const Tensor& b_data, const Tensor& b_sf,
                   const Tensor& d, const OptTensor& d_sf,
                   const OptTensor& c, at::IntArrayRef recipe) {
    std::variant<Tensor, std::pair<Tensor, Tensor>> output = d;
    if (d_sf.has_value())
        output = std::make_pair(d, d_sf.value());
    einsum::fp8_einsum(expr, {a_data, a_sf}, {b_data, b_sf}, output, c, to_tuple3(recipe));
}

// ---------------------------------------------------------------------------
// Hyperconnection
// ---------------------------------------------------------------------------
void dg_tf32_hc_prenorm_gemm(const Tensor& a, const Tensor& b, const Tensor& d,
                             const Tensor& sqr_sum, std::optional<int64_t> num_splits) {
    hyperconnection::tf32_hc_prenorm_gemm(a, b, d, sqr_sum, to_optional_int(num_splits));
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------
void dg_fp8_gemm_nt_skip_head_mid(const Tensor& a_data, const Tensor& a_sf,
                                  const Tensor& b_data, const Tensor& b_sf,
                                  const Tensor& d, at::IntArrayRef head_splits,
                                  at::OptionalIntArrayRef recipe,
                                  const std::string& compiled_dims, bool disable_ue8m0_cast) {
    attention::fp8_gemm_nt_skip_head_mid(
        {a_data, a_sf}, {b_data, b_sf}, d, to_tuple3(head_splits),
        to_optional_tuple3(recipe), compiled_dims, disable_ue8m0_cast);
}

Tensor dg_fp8_mqa_logits(const Tensor& q, const Tensor& kv_data, const Tensor& kv_sf,
                         const Tensor& weights, const Tensor& cu_seq_len_k_start,
                         const Tensor& cu_seq_len_k_end, bool clean_logits, int64_t max_seqlen_k) {
    return attention::fp8_mqa_logits(q, {kv_data, kv_sf}, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                                     clean_logits, static_cast<int>(max_seqlen_k));
}

Tensor dg_get_paged_mqa_logits_metadata(const Tensor& context_lens, int64_t block_kv,
                                        int64_t num_sms, const OptTensor& indices) {
    return attention::get_paged_mqa_logits_metadata(context_lens, static_cast<int>(block_kv),
                                                    static_cast<int>(num_sms), indices);
}

Tensor dg_fp8_paged_mqa_logits(const Tensor& q, const Tensor& fused_kv_cache,
                               const Tensor& weights, const Tensor& context_lens,
                               const Tensor& block_table, const Tensor& schedule_meta,
                               int64_t max_context_len, bool clean_logits, const OptTensor& indices,
                               const OptTensor& histogram) {
    return attention::fp8_paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                           static_cast<int>(max_context_len), clean_logits, indices, histogram);
}

Tensor dg_fp8_fp4_mqa_logits(const Tensor& q, const OptTensor& q_sf, const Tensor& kv_data, const Tensor& kv_sf,
                             const Tensor& weights, const Tensor& cu_seq_len_k_start,
                             const Tensor& cu_seq_len_k_end, bool clean_logits, int64_t max_seqlen_k,
                             at::ScalarType logits_dtype, const OptTensor& schedule_meta) {
    return attention::fp8_fp4_mqa_logits(
        std::make_pair(q, q_sf), {kv_data, kv_sf}, weights, cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, static_cast<int>(max_seqlen_k), logits_dtype, schedule_meta);
}

Tensor dg_fp8_fp4_paged_mqa_logits(const Tensor& q, const OptTensor& q_sf, const Tensor& fused_kv_cache,
                                   const Tensor& weights, const Tensor& context_lens,
                                   const Tensor& block_table, const Tensor& schedule_meta,
                                   int64_t max_context_len, bool clean_logits,
                                   at::ScalarType logits_dtype, const OptTensor& indices,
                                   const OptTensor& histogram) {
    return attention::fp8_fp4_paged_mqa_logits(
        std::make_pair(q, q_sf), fused_kv_cache, weights, context_lens, block_table, schedule_meta,
        static_cast<int>(max_context_len), clean_logits, logits_dtype, indices, histogram);
}

Tensor dg_get_mqa_logits_metadata(const Tensor& cu_seq_len_k_start, const Tensor& cu_seq_len_k_end,
                                  int64_t num_kv_tokens, int64_t num_heads) {
    return attention::get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end,
                                              static_cast<int>(num_kv_tokens), static_cast<int>(num_heads));
}

Tensor dg_get_sparse_mqa_logits_metadata(const Tensor& cu_seq_len_k_start, const Tensor& cu_seq_len_k_end,
                                         int64_t num_kv_tokens, const Tensor& sparse_kv_block_indices,
                                         at::ScalarType qk_dtype, int64_t sparse_block_kv, bool use_unaligned_ks) {
    return attention::get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, static_cast<int>(num_kv_tokens), sparse_kv_block_indices,
        qk_dtype, static_cast<int>(sparse_block_kv), use_unaligned_ks);
}

Tensor dg_get_paged_sparse_mqa_logits_metadata(const Tensor& context_lens, const Tensor& block_table,
                                               const Tensor& indices, int64_t page_kv,
                                               const Tensor& sparse_kv_block_indices,
                                               at::ScalarType qk_dtype, int64_t sparse_block_kv) {
    return attention::get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, static_cast<int>(page_kv), sparse_kv_block_indices,
        qk_dtype, static_cast<int>(sparse_block_kv));
}

Tensor dg_fp8_fp4_sparse_mqa_logits(const Tensor& q, const OptTensor& q_sf, const Tensor& kv, const Tensor& kv_sf,
                                    const Tensor& weights, const Tensor& metadata,
                                    int64_t num_max_sparse_blocks, int64_t sparse_block_kv,
                                    bool use_unaligned_ks) {
    return attention::fp8_fp4_sparse_mqa_logits(
        std::make_tuple(q, q_sf), {kv, kv_sf}, weights, metadata,
        static_cast<int>(num_max_sparse_blocks), static_cast<int>(sparse_block_kv), use_unaligned_ks);
}

Tensor dg_fp8_fp4_paged_sparse_mqa_logits(const Tensor& q, const OptTensor& q_sf, const Tensor& fused_kv_cache,
                                          const Tensor& weights, const Tensor& metadata,
                                          int64_t num_max_sparse_blocks, int64_t sparse_block_kv) {
    return attention::fp8_fp4_paged_sparse_mqa_logits(
        std::make_tuple(q, q_sf), fused_kv_cache, weights, metadata,
        static_cast<int>(num_max_sparse_blocks), static_cast<int>(sparse_block_kv));
}

// ---------------------------------------------------------------------------
// Mega MoE
// ---------------------------------------------------------------------------
int64_t dg_get_token_alignment_for_mega_moe() {
    return mega::get_token_alignment_for_mega_moe();
}

int64_t dg_get_block_m_for_mega_moe(int64_t num_ranks, int64_t num_experts,
                                    int64_t num_max_tokens_per_rank, int64_t num_tokens,
                                    int64_t num_topk, const std::string& mma_type,
                                    std::optional<int64_t> global_num_tokens) {
    return mega::get_block_m_for_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_tokens),
        static_cast<int>(num_topk), mma_type, global_num_tokens);
}

// Schemas cannot return closures, so sizing and slicing are separate ops; the slicing op
// recomputes the (host-only) layout from the same arguments.
auto symm_buffer_layout_for_mega_moe(int64_t num_ranks, int64_t num_experts,
                                            int64_t num_max_tokens_per_rank, int64_t num_topk,
                                            int64_t hidden, int64_t intermediate_hidden,
                                            const std::string& mma_type, const std::string& activation,
                                            int64_t num_shared_experts) {
    return mega::get_symm_buffer_size_for_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        mma_type, activation, static_cast<int>(num_shared_experts));
}

int64_t dg_get_symm_buffer_size_for_mega_moe(int64_t num_ranks, int64_t num_experts,
                                             int64_t num_max_tokens_per_rank, int64_t num_topk,
                                             int64_t hidden, int64_t intermediate_hidden,
                                             const std::string& mma_type, const std::string& activation,
                                             int64_t num_shared_experts) {
    return std::get<0>(symm_buffer_layout_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_shared_experts));
}

std::vector<Tensor> dg_slice_symm_buffer_for_mega_moe(const Tensor& buffer,
                                                      int64_t num_ranks, int64_t num_experts,
                                                      int64_t num_max_tokens_per_rank, int64_t num_topk,
                                                      int64_t hidden, int64_t intermediate_hidden,
                                                      const std::string& mma_type, const std::string& activation,
                                                      int64_t num_shared_experts) {
    const auto [num_bytes, slice] = symm_buffer_layout_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_shared_experts);
    DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(num_bytes));
    const auto views = slice(buffer);
    // Absent views (BF16 SFs, no shared experts) are returned as empty byte tensors.
    const auto or_empty = [&](const Tensor& t) {
        return t.defined() ? t : torch::empty({0}, torch::TensorOptions().dtype(torch::kChar).device(buffer.device()));
    };
    return std::apply([&](const auto&... t) { return std::vector<Tensor>{or_empty(t)...}; }, views);
}

int64_t dg_get_symm_buffer_size_for_sm90_mega_moe(int64_t num_ranks, int64_t num_experts,
                                                  int64_t num_max_tokens_per_rank, int64_t num_topk,
                                                  int64_t hidden, int64_t intermediate_hidden,
                                                  bool use_fp8_dispatch, const std::string& activation) {
    return std::get<0>(mega::get_symm_buffer_size_for_sm90_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        use_fp8_dispatch, activation));
}

std::vector<Tensor> dg_slice_symm_buffer_for_sm90_mega_moe(const Tensor& buffer,
                                                           int64_t num_ranks, int64_t num_experts,
                                                           int64_t num_max_tokens_per_rank, int64_t num_topk,
                                                           int64_t hidden, int64_t intermediate_hidden,
                                                           bool use_fp8_dispatch, const std::string& activation) {
    const auto [num_bytes, slice] = mega::get_symm_buffer_size_for_sm90_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        use_fp8_dispatch, activation);
    DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(num_bytes));
    return std::apply([](const auto&... t) { return std::vector<Tensor>{t...}; }, slice(buffer));
}

void dg_fp8_fp4_mega_moe(const Tensor& y, const Tensor& l1_weights, const Tensor& l1_weights_sf,
                         const Tensor& l2_weights, const Tensor& l2_weights_sf,
                         const OptTensor& shared_l1_weights, const OptTensor& shared_l1_weights_sf,
                         const OptTensor& shared_l2_weights, const OptTensor& shared_l2_weights_sf,
                         const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
                         at::IntArrayRef sym_buffer_ptrs, int64_t rank_idx,
                         int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                         at::IntArrayRef recipe, const std::string& mma_type, const std::string& activation,
                         std::optional<double> activation_clamp, bool fast_math, bool use_x_scales,
                         const OptTensor& l1_alphas, const OptTensor& l2_alphas, const OptTensor& l2_act_scales,
                         bool post_expert_norm, const OptTensor& post_expert_norm_weight,
                         double post_expert_norm_eps, bool post_expert_norm_weight_first,
                         bool use_trtllm_weights, std::optional<int64_t> global_num_tokens) {
    DG_HOST_ASSERT(shared_l1_weights.has_value() == shared_l1_weights_sf.has_value());
    DG_HOST_ASSERT(shared_l2_weights.has_value() == shared_l2_weights_sf.has_value());
    const auto shared_l1 = shared_l1_weights.has_value()
        ? std::make_optional(std::make_tuple(shared_l1_weights.value(), shared_l1_weights_sf.value())) : std::nullopt;
    const auto shared_l2 = shared_l2_weights.has_value()
        ? std::make_optional(std::make_tuple(shared_l2_weights.value(), shared_l2_weights_sf.value())) : std::nullopt;
    mega::fp8_fp4_mega_moe(
        y, std::make_tuple(l1_weights, l1_weights_sf), std::make_tuple(l2_weights, l2_weights_sf),
        shared_l1, shared_l2, cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs.vec(),
        static_cast<int>(rank_idx), static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk), to_tuple3(recipe),
        mma_type, activation, to_optional_float(activation_clamp), fast_math, use_x_scales,
        l1_alphas, l2_alphas, l2_act_scales,
        post_expert_norm, post_expert_norm_weight, static_cast<float>(post_expert_norm_eps),
        post_expert_norm_weight_first, use_trtllm_weights, global_num_tokens);
}

void dg_bf16_mega_moe(const Tensor& y, const Tensor& l1_weights, const Tensor& l2_weights,
                      const OptTensor& shared_l1_weights, const OptTensor& shared_l2_weights,
                      const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
                      at::IntArrayRef sym_buffer_ptrs, int64_t rank_idx,
                      int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                      const std::string& activation, std::optional<double> activation_clamp, bool fast_math,
                      bool post_expert_norm, const OptTensor& post_expert_norm_weight,
                      double post_expert_norm_eps, bool post_expert_norm_weight_first,
                      bool use_trtllm_weights, std::optional<int64_t> global_num_tokens) {
    mega::bf16_mega_moe(
        y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs.vec(),
        static_cast<int>(rank_idx), static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        activation, to_optional_float(activation_clamp), fast_math,
        post_expert_norm, post_expert_norm_weight, static_cast<float>(post_expert_norm_eps),
        post_expert_norm_weight_first, use_trtllm_weights, global_num_tokens);
}

void dg_fp8_mega_moe(const Tensor& y, const Tensor& l1_weights, const Tensor& l1_weights_sf,
                     const Tensor& l2_weights, const Tensor& l2_weights_sf,
                     const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
                     at::IntArrayRef sym_buffer_ptrs, int64_t rank_idx,
                     int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                     at::IntArrayRef recipe, const std::string& activation,
                     std::optional<double> activation_clamp, bool fast_math) {
    mega::fp8_mega_moe(
        y, {l1_weights, l1_weights_sf}, {l2_weights, l2_weights_sf},
        cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs.vec(),
        static_cast<int>(rank_idx), static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk), to_tuple3(recipe),
        activation, to_optional_float(activation_clamp), fast_math);
}

void dg_mega_moe_pre_dispatch(const Tensor& x, const Tensor& topk_idx, const Tensor& topk_weights,
                              const Tensor& buf_x, const Tensor& buf_x_sf,
                              const Tensor& buf_topk_idx, const Tensor& buf_topk_weights,
                              int64_t num_tokens, int64_t group_size, const std::string& mma_type,
                              const OptTensor& buf_x_scales, const OptTensor& expert_scales) {
    mega_moe_pre_dispatch(x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
                          static_cast<int>(num_tokens), static_cast<int>(group_size), mma_type,
                          buf_x_scales, expert_scales);
}

void dg_mega_moe_pre_dispatch_sm90(const Tensor& x, const Tensor& topk_idx, const Tensor& topk_weights,
                                   const Tensor& buf_x, const Tensor& buf_x_sf,
                                   const Tensor& buf_topk_idx, const Tensor& buf_topk_weights,
                                   int64_t num_tokens, int64_t group_size, double routed_scaling_factor) {
    mega::mega_moe_pre_dispatch_sm90(x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
                                     static_cast<int>(num_tokens), static_cast<int>(group_size),
                                     static_cast<float>(routed_scaling_factor));
}

// ---------------------------------------------------------------------------
// Mega mHC / Mega Gate
// ---------------------------------------------------------------------------
void dg_mega_mhc(const Tensor& x, const Tensor& residual, const OptTensor& shifted_prev_mix,
                 const Tensor& post_mix, const Tensor& comb_res_mix, const Tensor& fn,
                 const Tensor& mix_scales, const Tensor& mix_bases, int64_t hc_mult,
                 double hc_norm_eps, double hc_pre_eps, double hc_post_scale,
                 double sinkhorn_eps, int64_t num_sinkhorn_iters,
                 const Tensor& rmsnorm_weight, double rmsnorm_eps, double rmsnorm_scale,
                 const Tensor& new_residual, const OptTensor& new_prev_mix,
                 const Tensor& new_post_mix, const Tensor& new_comb_res_mix,
                 const OptTensor& y_bf16, const OptTensor& y_fp8, const OptTensor& y_gemm_sf,
                 const OptTensor& y_routed_sf, const OptTensor& y_shared_sf, int64_t shared_sf_block_m) {
    mega_mhc::mega_mhc(
        x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases,
        static_cast<int>(hc_mult), static_cast<float>(hc_norm_eps), static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_post_scale), static_cast<float>(sinkhorn_eps), static_cast<int>(num_sinkhorn_iters),
        rmsnorm_weight, static_cast<float>(rmsnorm_eps), static_cast<float>(rmsnorm_scale),
        new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
        y_bf16, y_fp8, y_gemm_sf, y_routed_sf, y_shared_sf, static_cast<int>(shared_sf_block_m));
}

std::tuple<Tensor, Tensor> dg_bf16_mega_gate(const Tensor& x, const Tensor& weight, int64_t num_topk,
                                             bool use_shared_as_routed, int64_t num_shared_experts,
                                             double routed_scaling_factor, int64_t ep_rank,
                                             const std::string& scoring_func,
                                             const OptTensor& mask, const OptTensor& bias,
                                             const OptTensor& image_bias, const OptTensor& image_token_mask,
                                             const OptTensor& fix_routing_mask, const OptTensor& to_physical_map,
                                             const OptTensor& logical_count, const OptTensor& unmapped_topk_idx,
                                             const OptTensor& force_random,
                                             const OptTensor& out_idx, const OptTensor& out_weights) {
    DG_HOST_ASSERT(out_idx.has_value() == out_weights.has_value());
    std::optional<std::tuple<Tensor, Tensor>> out;
    if (out_idx.has_value())
        out = std::make_tuple(out_idx.value(), out_weights.value());
    return mega_gate::bf16_mega_gate(
        x, weight, static_cast<int>(num_topk), use_shared_as_routed, static_cast<int>(num_shared_experts),
        static_cast<float>(routed_scaling_factor), static_cast<int>(ep_rank), scoring_func,
        mask, bias, image_bias, image_token_mask, fix_routing_mask, to_physical_map,
        logical_count, unmapped_topk_idx, force_random, out);
}

c10::Dict<std::string, int64_t> dg_get_bf16_mega_gate_config(int64_t num_tokens, int64_t hidden,
                                                             int64_t num_routed_experts, int64_t num_topk) {
    c10::Dict<std::string, int64_t> result;
    for (const auto& [key, value]: mega_gate::get_bf16_mega_gate_config(
             static_cast<int>(num_tokens), static_cast<int>(hidden),
             static_cast<int>(num_routed_experts), static_cast<int>(num_topk)))
        result.insert(key, value);
    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
#define DG_FP8_FP4_GEMM_SCHEMA "(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, Tensor? c, int[]? recipe, int[]? recipe_a, int[]? recipe_b, str compiled_dims, bool disable_ue8m0_cast, float? alpha) -> ()"
#define DG_BF16_GEMM_SCHEMA "(Tensor a, Tensor b, Tensor(a!) d, Tensor? c, str compiled_dims, float? alpha) -> ()"
#define DG_CUBLASLT_GEMM_SCHEMA "(Tensor a, Tensor b, Tensor(a!) d, Tensor? c) -> ()"
#define DG_K_GROUPED_FP8_GEMM_SCHEMA "(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, int[]? ks, Tensor grouped_layout, Tensor? c, int[] recipe, str compiled_dims, bool use_psum_layout, bool use_padded_sf_layout) -> ()"
#define DG_SYMM_BUFFER_ARGS "int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, str mma_type, str activation, int num_shared_experts"
#define DG_SM90_SYMM_BUFFER_ARGS "int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, bool use_fp8_dispatch, str activation"

TORCH_LIBRARY(deep_gemm, m) {
    // Runtime and config (catch-all: no tensor arguments to dispatch on)
    m.def("init(str library_root_path, str cuda_home) -> ()", &dg_init);
    m.def("get_num_sms() -> int", &dg_get_num_sms);
    m.def("set_num_sms(int new_num_sms) -> ()", &dg_set_num_sms);
    m.def("get_tc_util() -> int", &dg_get_tc_util);
    m.def("set_tc_util(int new_tc_util) -> ()", &dg_set_tc_util);
    m.def("get_pdl() -> bool", &dg_get_pdl);
    m.def("set_pdl(bool enable) -> ()", &dg_set_pdl);
    m.def("use_deterministic_algorithms(bool enable) -> ()", &dg_use_deterministic_algorithms);
    m.def("get_deterministic_algorithms() -> bool", &dg_get_deterministic_algorithms);
    m.def("set_ignore_compile_dims(bool ignore) -> ()", &dg_set_ignore_compile_dims);
    m.def("set_block_size_multiple_of(int m, int n) -> ()", &dg_set_block_size_multiple_of);
    m.def("get_tma_aligned_size(int mn, int element_size) -> int", &dg_get_tma_aligned_size);
    m.def("get_mk_alignment_for_contiguous_layout() -> int", &dg_get_mk_alignment_for_contiguous_layout);
    m.def("set_mk_alignment_for_contiguous_layout(int new_value) -> ()", &dg_set_mk_alignment_for_contiguous_layout);
    m.def("get_theoretical_mk_alignment_for_contiguous_layout(int? expected_m, int? num_groups) -> int",
          &dg_get_theoretical_mk_alignment_for_contiguous_layout);
    m.def("get_token_alignment_for_mega_moe() -> int", &dg_get_token_alignment_for_mega_moe);
    m.def("get_block_m_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_tokens, "
          "int num_topk, str mma_type, int? global_num_tokens) -> int", &dg_get_block_m_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe(" DG_SYMM_BUFFER_ARGS ") -> int", &dg_get_symm_buffer_size_for_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe(" DG_SM90_SYMM_BUFFER_ARGS ") -> int",
          &dg_get_symm_buffer_size_for_sm90_mega_moe);
    m.def("get_bf16_mega_gate_config(int num_tokens, int hidden, int num_routed_experts, int num_topk) -> Dict(str, int)",
          &dg_get_bf16_mega_gate_config);

    // Layout
    m.def("preprocess_sf(Tensor sf) -> (int, int, int, int, int)");
    m.def("get_mn_major_tma_aligned_tensor(Tensor sf) -> Tensor");
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor sf, Tensor? psum_layout) -> Tensor");
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor sf, Tensor grouped_layout, int[]? ks, "
          "int gran_k, int k_alignment, bool use_psum_layout, bool use_padded_sf_layout) -> Tensor");
    m.def("transform_sf_into_required_layout(Tensor sf, int mn, int k, int recipe_a, int recipe_b, int? recipe_c, "
          "int? num_groups, bool? is_sfa, bool disable_ue8m0_cast, Tensor? psum_layout) -> Tensor");

    // GEMMs
    m.def("cublaslt_gemm_nt" DG_CUBLASLT_GEMM_SCHEMA);
    m.def("cublaslt_gemm_nn" DG_CUBLASLT_GEMM_SCHEMA);
    m.def("cublaslt_gemm_tn" DG_CUBLASLT_GEMM_SCHEMA);
    m.def("cublaslt_gemm_tt" DG_CUBLASLT_GEMM_SCHEMA);
    m.def("cublaslt_nvfp4_gemm_nt(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, Tensor? c) -> ()");
    m.def("fp8_fp4_gemm_nt" DG_FP8_FP4_GEMM_SCHEMA);
    m.def("fp8_fp4_gemm_nn" DG_FP8_FP4_GEMM_SCHEMA);
    m.def("fp8_fp4_gemm_tn" DG_FP8_FP4_GEMM_SCHEMA);
    m.def("fp8_fp4_gemm_tt" DG_FP8_FP4_GEMM_SCHEMA);
    m.def("m_grouped_fp8_fp4_gemm_nt_contiguous(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, "
          "Tensor grouped_layout, int[]? recipe, int[]? recipe_a, int[]? recipe_b, str compiled_dims, "
          "bool disable_ue8m0_cast, bool use_psum_layout, bool ensure_zero_padding, int? expected_m_for_psum_layout) -> ()");
    m.def("m_grouped_fp8_fp4_gemm_nn_contiguous(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, "
          "Tensor grouped_layout, int[]? recipe, int[]? recipe_a, int[]? recipe_b, str compiled_dims, "
          "bool disable_ue8m0_cast, bool use_psum_layout, bool ensure_zero_padding) -> ()");
    m.def("m_grouped_fp8_fp4_gemm_nt_masked(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, "
          "Tensor masked_m, int expected_m, int[]? recipe, int[]? recipe_a, int[]? recipe_b, str compiled_dims, "
          "bool disable_ue8m0_cast) -> ()");
    m.def("bf16_gemm_nt" DG_BF16_GEMM_SCHEMA);
    m.def("bf16_gemm_nn" DG_BF16_GEMM_SCHEMA);
    m.def("bf16_gemm_tn" DG_BF16_GEMM_SCHEMA);
    m.def("bf16_gemm_tt" DG_BF16_GEMM_SCHEMA);
    m.def("m_grouped_bf16_gemm_nt_contiguous(Tensor a, Tensor b, Tensor(a!) d, Tensor grouped_layout, "
          "str compiled_dims, bool use_psum_layout, bool ensure_zero_padding, int? expected_m_for_psum_layout) -> ()");
    m.def("m_grouped_bf16_gemm_nn_contiguous(Tensor a, Tensor b, Tensor(a!) d, Tensor grouped_layout, "
          "str compiled_dims, bool use_psum_layout, bool ensure_zero_padding) -> ()");
    m.def("m_grouped_bf16_gemm_nt_masked(Tensor a, Tensor b, Tensor(a!) d, Tensor masked_m, int expected_m, "
          "str compiled_dims) -> ()");
    m.def("k_grouped_bf16_gemm_tn_contiguous(Tensor a, Tensor b, Tensor(a!) d, int[]? ks, Tensor grouped_layout, "
          "Tensor? c, str compiled_dims, bool use_psum_layout) -> ()");
    m.def("k_grouped_fp8_gemm_tn_contiguous" DG_K_GROUPED_FP8_GEMM_SCHEMA);
    m.def("k_grouped_fp8_gemm_nt_contiguous" DG_K_GROUPED_FP8_GEMM_SCHEMA);
    m.def("k_grouped_fp4_gemm_nt_contiguous(Tensor a, Tensor a_sf, Tensor b, Tensor b_sf, Tensor(a!) d, int[]? ks, "
          "Tensor grouped_layout, Tensor? c, int[] recipe, str compiled_dims, bool use_psum_layout) -> ()");
    m.def("batched_syrk(Tensor a, Tensor(a!) d) -> ()");
    m.def("batched_symm(Tensor a, Tensor b, Tensor(a!) d) -> ()");

    // Einsum and hyperconnection
    m.def("einsum(str expr, Tensor a, Tensor b, Tensor(a!) d, Tensor? c, bool use_cublaslt) -> ()");
    m.def("fp8_einsum(str expr, Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, Tensor(a!) d, "
          "Tensor(b!)? d_sf, Tensor? c, int[] recipe) -> ()");
    m.def("tf32_hc_prenorm_gemm(Tensor a, Tensor b, Tensor(a!) d, Tensor(b!) sqr_sum, int? num_splits) -> ()");

    // Attention
    m.def("fp8_gemm_nt_skip_head_mid(Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, Tensor(a!) d, "
          "int[] head_splits, int[]? recipe, str compiled_dims, bool disable_ue8m0_cast) -> ()");
    m.def("fp8_mqa_logits(Tensor q, Tensor kv_data, Tensor kv_sf, Tensor weights, Tensor cu_seq_len_k_start, "
          "Tensor cu_seq_len_k_end, bool clean_logits, int max_seqlen_k) -> Tensor");
    m.def("get_paged_mqa_logits_metadata(Tensor context_lens, int block_kv, int num_sms, Tensor? indices) -> Tensor");
    m.def("fp8_paged_mqa_logits(Tensor q, Tensor fused_kv_cache, Tensor weights, Tensor context_lens, "
          "Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits, Tensor? indices, "
          "Tensor(a!)? histogram) -> Tensor");
    m.def("fp8_fp4_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv_data, Tensor kv_sf, Tensor weights, "
          "Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, bool clean_logits, int max_seqlen_k, "
          "ScalarType logits_dtype, Tensor? schedule_meta) -> Tensor");
    m.def("fp8_fp4_paged_mqa_logits(Tensor q, Tensor? q_sf, Tensor fused_kv_cache, Tensor weights, "
          "Tensor context_lens, Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits, "
          "ScalarType logits_dtype, Tensor? indices, Tensor(a!)? histogram) -> Tensor");
    m.def("get_mqa_logits_metadata(Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, int num_kv_tokens, "
          "int num_heads) -> Tensor");
    m.def("get_sparse_mqa_logits_metadata(Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, int num_kv_tokens, "
          "Tensor sparse_kv_block_indices, ScalarType qk_dtype, int sparse_block_kv, bool use_unaligned_ks) -> Tensor");
    m.def("get_paged_sparse_mqa_logits_metadata(Tensor context_lens, Tensor block_table, Tensor indices, int page_kv, "
          "Tensor sparse_kv_block_indices, ScalarType qk_dtype, int sparse_block_kv) -> Tensor");
    m.def("fp8_fp4_sparse_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv, Tensor kv_sf, Tensor weights, "
          "Tensor metadata, int num_max_sparse_blocks, int sparse_block_kv, bool use_unaligned_ks) -> Tensor");
    m.def("fp8_fp4_paged_sparse_mqa_logits(Tensor q, Tensor? q_sf, Tensor fused_kv_cache, Tensor weights, "
          "Tensor metadata, int num_max_sparse_blocks, int sparse_block_kv) -> Tensor");

    // Mega MoE
    m.def("slice_symm_buffer_for_mega_moe(Tensor(a) buffer, " DG_SYMM_BUFFER_ARGS ") -> Tensor(a)[]");
    m.def("slice_symm_buffer_for_sm90_mega_moe(Tensor(a) buffer, " DG_SM90_SYMM_BUFFER_ARGS ") -> Tensor(a)[]");
    m.def("fp8_fp4_mega_moe(Tensor(a!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, "
          "Tensor l2_weights_sf, Tensor? shared_l1_weights, Tensor? shared_l1_weights_sf, Tensor? shared_l2_weights, "
          "Tensor? shared_l2_weights_sf, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, "
          "int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, "
          "int[] recipe, str mma_type, str activation, float? activation_clamp, bool fast_math, bool use_x_scales, "
          "Tensor? l1_alphas, Tensor? l2_alphas, Tensor? l2_act_scales, "
          "bool post_expert_norm, Tensor? post_expert_norm_weight, float post_expert_norm_eps, "
          "bool post_expert_norm_weight_first, bool use_trtllm_weights, "
          "int? global_num_tokens) -> ()");
    m.def("bf16_mega_moe(Tensor(a!) y, Tensor l1_weights, Tensor l2_weights, Tensor? shared_l1_weights, "
          "Tensor? shared_l2_weights, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, "
          "int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, "
          "str activation, float? activation_clamp, bool fast_math, "
          "bool post_expert_norm, Tensor? post_expert_norm_weight, float post_expert_norm_eps, "
          "bool post_expert_norm_weight_first, bool use_trtllm_weights, "
          "int? global_num_tokens) -> ()");
    m.def("fp8_mega_moe(Tensor(a!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, "
          "Tensor l2_weights_sf, Tensor(b!)? cumulative_local_expert_recv_stats, Tensor(c!) sym_buffer, "
          "int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, "
          "int[] recipe, str activation, float? activation_clamp, bool fast_math) -> ()");
    m.def("mega_moe_pre_dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, Tensor(a!) buf_x, "
          "Tensor(b!) buf_x_sf, Tensor(c!) buf_topk_idx, Tensor(d!) buf_topk_weights, int num_tokens, int group_size, "
          "str mma_type, Tensor(e!)? buf_x_scales, Tensor? expert_scales) -> ()");
    m.def("mega_moe_pre_dispatch_sm90(Tensor x, Tensor topk_idx, Tensor topk_weights, Tensor(a!) buf_x, "
          "Tensor(b!) buf_x_sf, Tensor(c!) buf_topk_idx, Tensor(d!) buf_topk_weights, int num_tokens, int group_size, "
          "float routed_scaling_factor) -> ()");

    // Mega mHC / Mega Gate
    m.def("mega_mhc(Tensor x, Tensor residual, Tensor? shifted_prev_mix, Tensor post_mix, Tensor comb_res_mix, "
          "Tensor fn, Tensor mix_scales, Tensor mix_bases, int hc_mult, float hc_norm_eps, float hc_pre_eps, "
          "float hc_post_scale, float sinkhorn_eps, int num_sinkhorn_iters, Tensor rmsnorm_weight, float rmsnorm_eps, "
          "float rmsnorm_scale, Tensor(a!) new_residual, Tensor(b!)? new_prev_mix, Tensor(c!) new_post_mix, "
          "Tensor(d!) new_comb_res_mix, Tensor(e!)? y_bf16, Tensor(f!)? y_fp8, Tensor(g!)? y_gemm_sf, "
          "Tensor(h!)? y_routed_sf, Tensor(i!)? y_shared_sf, int shared_sf_block_m) -> ()");
    m.def("bf16_mega_gate(Tensor x, Tensor weight, int num_topk, bool use_shared_as_routed, int num_shared_experts, "
          "float routed_scaling_factor, int ep_rank, str scoring_func, Tensor? mask, Tensor? bias, Tensor? image_bias, "
          "Tensor? image_token_mask, Tensor? fix_routing_mask, Tensor? to_physical_map, Tensor? logical_count, "
          "Tensor(a!)? unmapped_topk_idx, Tensor? force_random, Tensor(b!)? out_idx, Tensor(c!)? out_weights) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("preprocess_sf", &dg_preprocess_sf);
    m.impl("get_mn_major_tma_aligned_tensor", &dg_get_mn_major_tma_aligned_tensor);
    m.impl("get_mn_major_tma_aligned_packed_ue8m0_tensor", &dg_get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.impl("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.impl("transform_sf_into_required_layout", &dg_transform_sf_into_required_layout);

    m.impl("cublaslt_gemm_nt", &dg_cublaslt_gemm_nt);
    m.impl("cublaslt_gemm_nn", &dg_cublaslt_gemm_nn);
    m.impl("cublaslt_gemm_tn", &dg_cublaslt_gemm_tn);
    m.impl("cublaslt_gemm_tt", &dg_cublaslt_gemm_tt);
    m.impl("cublaslt_nvfp4_gemm_nt", &dg_cublaslt_nvfp4_gemm_nt);
    m.impl("fp8_fp4_gemm_nt", &dg_fp8_fp4_gemm_nt);
    m.impl("fp8_fp4_gemm_nn", &dg_fp8_fp4_gemm_nn);
    m.impl("fp8_fp4_gemm_tn", &dg_fp8_fp4_gemm_tn);
    m.impl("fp8_fp4_gemm_tt", &dg_fp8_fp4_gemm_tt);
    m.impl("m_grouped_fp8_fp4_gemm_nt_contiguous", &dg_m_grouped_fp8_fp4_gemm_nt_contiguous);
    m.impl("m_grouped_fp8_fp4_gemm_nn_contiguous", &dg_m_grouped_fp8_fp4_gemm_nn_contiguous);
    m.impl("m_grouped_fp8_fp4_gemm_nt_masked", &dg_m_grouped_fp8_fp4_gemm_nt_masked);
    m.impl("bf16_gemm_nt", &dg_bf16_gemm_nt);
    m.impl("bf16_gemm_nn", &dg_bf16_gemm_nn);
    m.impl("bf16_gemm_tn", &dg_bf16_gemm_tn);
    m.impl("bf16_gemm_tt", &dg_bf16_gemm_tt);
    m.impl("m_grouped_bf16_gemm_nt_contiguous", &dg_m_grouped_bf16_gemm_nt_contiguous);
    m.impl("m_grouped_bf16_gemm_nn_contiguous", &dg_m_grouped_bf16_gemm_nn_contiguous);
    m.impl("m_grouped_bf16_gemm_nt_masked", &dg_m_grouped_bf16_gemm_nt_masked);
    m.impl("k_grouped_bf16_gemm_tn_contiguous", &dg_k_grouped_bf16_gemm_tn_contiguous);
    m.impl("k_grouped_fp8_gemm_tn_contiguous", &dg_k_grouped_fp8_gemm_tn_contiguous);
    m.impl("k_grouped_fp8_gemm_nt_contiguous", &dg_k_grouped_fp8_gemm_nt_contiguous);
    m.impl("k_grouped_fp4_gemm_nt_contiguous", &dg_k_grouped_fp4_gemm_nt_contiguous);
    m.impl("batched_syrk", &dg_batched_syrk);
    m.impl("batched_symm", &dg_batched_symm);

    m.impl("einsum", &dg_einsum);
    m.impl("fp8_einsum", &dg_fp8_einsum);
    m.impl("tf32_hc_prenorm_gemm", &dg_tf32_hc_prenorm_gemm);

    m.impl("fp8_gemm_nt_skip_head_mid", &dg_fp8_gemm_nt_skip_head_mid);
    m.impl("fp8_mqa_logits", &dg_fp8_mqa_logits);
    m.impl("get_paged_mqa_logits_metadata", &dg_get_paged_mqa_logits_metadata);
    m.impl("fp8_paged_mqa_logits", &dg_fp8_paged_mqa_logits);
    m.impl("fp8_fp4_mqa_logits", &dg_fp8_fp4_mqa_logits);
    m.impl("fp8_fp4_paged_mqa_logits", &dg_fp8_fp4_paged_mqa_logits);
    m.impl("get_mqa_logits_metadata", &dg_get_mqa_logits_metadata);
    m.impl("get_sparse_mqa_logits_metadata", &dg_get_sparse_mqa_logits_metadata);
    m.impl("get_paged_sparse_mqa_logits_metadata", &dg_get_paged_sparse_mqa_logits_metadata);
    m.impl("fp8_fp4_sparse_mqa_logits", &dg_fp8_fp4_sparse_mqa_logits);
    m.impl("fp8_fp4_paged_sparse_mqa_logits", &dg_fp8_fp4_paged_sparse_mqa_logits);

    m.impl("slice_symm_buffer_for_mega_moe", &dg_slice_symm_buffer_for_mega_moe);
    m.impl("slice_symm_buffer_for_sm90_mega_moe", &dg_slice_symm_buffer_for_sm90_mega_moe);
    m.impl("fp8_fp4_mega_moe", &dg_fp8_fp4_mega_moe);
    m.impl("bf16_mega_moe", &dg_bf16_mega_moe);
    m.impl("fp8_mega_moe", &dg_fp8_mega_moe);
    m.impl("mega_moe_pre_dispatch", &dg_mega_moe_pre_dispatch);
    m.impl("mega_moe_pre_dispatch_sm90", &dg_mega_moe_pre_dispatch_sm90);

    m.impl("mega_mhc", &dg_mega_mhc);
    m.impl("bf16_mega_gate", &dg_bf16_mega_gate);
}
