#pragma once

#include "../../runtime/runtime.hpp"

#include "../../utils/exception.hpp"
#include "../../runtime/launch.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SMXXCleanLogitsRuntime final {
public:
    struct Args {
        int next_n;
        int seq_len;
        int seq_len_kv;
        uint64_t stride_logits;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;
        at::ScalarType logits_dtype;

        int block_kv;
        int num_warps;

        deep_jit::cuda::LaunchOptions launch_args;
    };

    static std::string generate(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/smxx_clean_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_clean_logits<
        {}, {}, {}, {}
    >);
}};
)", args.next_n, args.block_kv, args.num_warps, to_string(args.logits_dtype));
    }

    static void launch(const std::shared_ptr<deep_jit::cuda::Kernel>& kernel, const Args& args) {
        jit->launch(kernel, args.launch_args,
            args.seq_len, args.seq_len_kv, static_cast<int64_t>(args.stride_logits),
            args.cu_seq_len_k_start, args.cu_seq_len_k_end, args.logits
        );
    }
};

static void smxx_clean_logits(const torch::Tensor& logits,
                              const std::optional<torch::Tensor>& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end,
                              const int& next_n,
                              const int& seq_len, const int& seq_len_kv,
                              const uint64_t &stride_logits) {
    const int block_kv = 8192;
    const int num_warps = 8;
    const int smem_size = block_kv * sizeof(float);

    // Launch
    const SMXXCleanLogitsRuntime::Args& args = {
        .next_n = next_n,
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_logits = stride_logits,
        .cu_seq_len_k_start = cu_seq_len_k_start.has_value() ? cu_seq_len_k_start.value().data_ptr<int>() : nullptr,
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .logits_dtype = logits.scalar_type(),
        .block_kv = block_kv,
        .num_warps = num_warps,
        .launch_args = make_launch_options(runtime->get_num_sms(),
                                  num_warps * 32, smem_size)
    };
    const auto code = SMXXCleanLogitsRuntime::generate(args);
    const auto kernel = jit->compile("smxx_clean_logits", code);
    SMXXCleanLogitsRuntime::launch(kernel, args);
}

} // namespace deep_gemm
