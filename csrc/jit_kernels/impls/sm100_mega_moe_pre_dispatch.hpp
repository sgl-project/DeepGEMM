#pragma once

#include <torch/python.h>

#include <optional>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/mega_moe.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

// JIT runtime for `sm100_mega_moe_pre_dispatch` (see
// `deep_gemm/include/deep_gemm/impls/sm100_mega_moe_pre_dispatch.cuh`).
// Templated on (kGroupSize, kMmaKind, kUsePDL); host fn picks the
// instantiation from explicit args.
class SM100MegaMoEPreDispatchRuntime final : public LaunchRuntime<SM100MegaMoEPreDispatchRuntime> {
public:
    struct Args {
        int group_size;
        MmaKind mma_kind;
        bool use_pdl;

        // Runtime args (passed to the kernel via the params struct).
        const void* x;
        const void* topk_idx;
        const void* topk_weights;
        const void* expert_scales;
        void*       buf_x;
        void*       buf_x_sf;
        void*       buf_x_scales;
        void*       buf_topk_idx;
        void*       buf_topk_weights;
        uint32_t    num_tokens;
        uint32_t    padded_max;
        uint32_t    hidden;
        uint32_t    num_groups;
        uint32_t    top_k;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mega_moe_pre_dispatch.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&mega_moe_pre_dispatch_kernel<
        {}, {}, {}
    >);
}};
)", args.group_size,
    to_mma_kind_name(args.mma_kind),
    args.use_pdl ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.x, args.topk_idx, args.topk_weights, args.expert_scales,
            args.buf_x, args.buf_x_sf, args.buf_x_scales,
            args.buf_topk_idx, args.buf_topk_weights,
            args.num_tokens, args.padded_max, args.hidden, args.num_groups, args.top_k));
    }
};

// Host entry point. Layout contract (matches DeepGEMM's mega symm buffer):
//   - x:            (M, H) bf16, contiguous.
//   - topk_idx:     (M, K) int32, contiguous.
//   - topk_weights: (M, K) float, contiguous.
//   - buf_x:        (P, H) fp8_e4m3 for `fp8xfp4`, else (P, H/2) int8 (packed FP4).
//   - buf_x_sf:     (P, G/4) int32, contiguous; G = H / group_size; each int32
//                   stores 4 UE8M0 bytes row-major.
//   - buf_topk_idx: (P, K) int64.
//   - buf_topk_weights: (P, K) float.
//
// Pad-fill: rows in [num_tokens, padded_max) of buf_topk_idx / buf_topk_weights
// are filled with (-1, 0). buf_x and buf_x_sf rows in that range are NOT
// touched (the kernel only writes valid-token rows; pad rows must have been
// pre-zeroed by the caller if they need defined values).
static void mega_moe_pre_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& buf_x,
    const torch::Tensor& buf_x_sf,
    const torch::Tensor& buf_topk_idx,
    const torch::Tensor& buf_topk_weights,
    const int& num_tokens,
    const int& group_size,
    const std::string& mma_type,
    const std::optional<torch::Tensor>& buf_x_scales,
    const std::optional<torch::Tensor>& expert_scales) {
    const auto mma_kind = parse_mma_kind(mma_type);
    DG_HOST_ASSERT(mma_kind != MmaKind::BF16);
    const bool is_packed_fp4 = mma_kind == MmaKind::MXFP4 or mma_kind == MmaKind::NVFP4;
    if (mma_kind == MmaKind::NVFP4) {
        // NVFP4 per-token acts: 16-elem UE4M3 block SFs plus one FP32 outer
        // scale per token (written to `buf_x_scales`).
        DG_HOST_ASSERT(group_size == 16);
        DG_HOST_ASSERT(buf_x_scales.has_value());
        DG_HOST_ASSERT(buf_x_scales->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(buf_x_scales->is_contiguous());
        DG_HOST_ASSERT(buf_x_scales->numel() >= num_tokens);
    } else {
        DG_HOST_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
        DG_HOST_ASSERT(!buf_x_scales.has_value());
    }
    if (expert_scales.has_value()) {
        DG_HOST_ASSERT(expert_scales->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(expert_scales->is_contiguous());
        DG_HOST_ASSERT(expert_scales->dim() == 1);
    }
    DG_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(x.is_contiguous());
    DG_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(topk_idx.is_contiguous() && topk_weights.is_contiguous());
    DG_HOST_ASSERT(x.dim() == 2 && topk_idx.dim() == 2 && topk_weights.dim() == 2);
    DG_HOST_ASSERT(buf_x.dim() == 2 && buf_x_sf.dim() == 2);
    DG_HOST_ASSERT(buf_topk_idx.dim() == 2 && buf_topk_weights.dim() == 2);
    DG_HOST_ASSERT(buf_topk_idx.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(buf_topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(buf_x_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(buf_x_sf.is_contiguous());

    const auto m = static_cast<int>(x.size(0));
    const auto hidden = static_cast<int>(x.size(1));
    const auto top_k = static_cast<int>(topk_idx.size(1));
    const auto padded_max = static_cast<int>(buf_x.size(0));

    DG_HOST_ASSERT(num_tokens == m);
    DG_HOST_ASSERT(num_tokens <= padded_max);
    DG_HOST_ASSERT(static_cast<int>(topk_idx.size(0)) == m);
    DG_HOST_ASSERT(static_cast<int>(topk_weights.size(0)) == m);
    DG_HOST_ASSERT(static_cast<int>(topk_weights.size(1)) == top_k);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_idx.size(0)) == padded_max);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_idx.size(1)) == top_k);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_weights.size(0)) == padded_max);
    DG_HOST_ASSERT(static_cast<int>(buf_topk_weights.size(1)) == top_k);

    DG_HOST_ASSERT(hidden % group_size == 0);
    const auto num_groups = hidden / group_size;
    DG_HOST_ASSERT(num_groups % 4 == 0);
    DG_HOST_ASSERT(static_cast<int>(buf_x_sf.size(0)) == padded_max);
    DG_HOST_ASSERT(static_cast<int>(buf_x_sf.size(1)) == num_groups / 4);

    if (is_packed_fp4) {
        // Packed FP4: (P, hidden/2) bytes. The symm-buffer slice views this
        // as kPackedFP4 (int8); accept either int8 / uint8 / float8_e4m3fn
        // re-views since callers may bind the slot differently.
        DG_HOST_ASSERT(static_cast<int>(buf_x.size(1)) == hidden / 2);
        DG_HOST_ASSERT(buf_x.element_size() == 1);
    } else {
        DG_HOST_ASSERT(buf_x.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(static_cast<int>(buf_x.size(1)) == hidden);
    }

    DG_HOST_ASSERT(hidden % 8 == 0);
    const auto num_threads = hidden / 8;
    DG_HOST_ASSERT(num_threads <= 1024);
    DG_HOST_ASSERT(num_threads >= top_k);

    const auto pad_slots = (padded_max - num_tokens) * top_k;
    const auto num_pad_blocks = pad_slots == 0 ? 0
        : math::ceil_div(pad_slots, num_threads);
    const auto num_total_blocks = num_tokens + num_pad_blocks;
    if (num_total_blocks == 0) return;

    const bool use_pdl = device_runtime->get_pdl();

    SM100MegaMoEPreDispatchRuntime::Args args = {
        .group_size = group_size,
        .mma_kind = mma_kind,
        .use_pdl = use_pdl,
        .x = x.const_data_ptr(),
        .topk_idx = topk_idx.const_data_ptr(),
        .topk_weights = topk_weights.const_data_ptr(),
        .expert_scales = expert_scales.has_value()
            ? expert_scales->const_data_ptr() : nullptr,
        .buf_x = buf_x.data_ptr(),
        .buf_x_sf = buf_x_sf.data_ptr(),
        .buf_x_scales = buf_x_scales.has_value()
            ? buf_x_scales->data_ptr() : nullptr,
        .buf_topk_idx = buf_topk_idx.data_ptr(),
        .buf_topk_weights = buf_topk_weights.data_ptr(),
        .num_tokens = static_cast<uint32_t>(num_tokens),
        .padded_max = static_cast<uint32_t>(padded_max),
        .hidden = static_cast<uint32_t>(hidden),
        .num_groups = static_cast<uint32_t>(num_groups),
        .top_k = static_cast<uint32_t>(top_k),
        .launch_args = LaunchArgs(num_total_blocks, num_threads, /*smem_size=*/0,
                                  /*cluster_dim=*/1, /*enable_pdl=*/use_pdl)
    };

    const auto code = SM100MegaMoEPreDispatchRuntime::generate(args);
    const auto runtime = compiler->build("sm100_mega_moe_pre_dispatch", code);
    SM100MegaMoEPreDispatchRuntime::launch(runtime, args);
}

} // namespace deep_gemm
