#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"
#include "../utils/compatibility.hpp"
#include "../utils/exception.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/heuristics/sm90.hpp"
#include "../jit_kernels/heuristics/sm100.hpp"
#include "../jit_kernels/heuristics/common.hpp"
#endif

namespace deep_gemm::runtime {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

static GemmType parse_gemm_type(const std::string& gemm_type) {
    if (gemm_type == "Normal")
        return GemmType::Normal;
    if (gemm_type == "MGroupedContiguous")
        return GemmType::MGroupedContiguous;
    if (gemm_type == "MGroupedMasked")
        return GemmType::MGroupedMasked;
    DG_HOST_UNREACHABLE("Unsupported gemm_type for get_best_gemm_config_key");
}

// Fields that determine the JIT kernel identity (see *Runtime::generate_impl).
static std::vector<int64_t> gemm_config_to_key_vec(const GemmDesc& desc, const GemmConfig& config) {
    return {
        static_cast<int64_t>(desc.gemm_type),
        static_cast<int64_t>(desc.kernel_type),
        static_cast<int64_t>(desc.get_mma_kind()),
        static_cast<int64_t>(desc.a_dtype),
        static_cast<int64_t>(desc.b_dtype),
        static_cast<int64_t>(desc.cd_dtype),
        static_cast<int64_t>(desc.major_a),
        static_cast<int64_t>(desc.major_b),
        static_cast<int64_t>(desc.with_accumulation),
        static_cast<int64_t>(config.layout.swap_ab),
        static_cast<int64_t>(config.layout.block_m),
        static_cast<int64_t>(config.layout.block_n),
        static_cast<int64_t>(config.layout.block_k),
        static_cast<int64_t>(config.layout.cluster_m),
        static_cast<int64_t>(config.layout.cluster_n),
        static_cast<int64_t>(config.pipeline_config.num_stages),
        static_cast<int64_t>(config.pipeline_config.smem_size),
        static_cast<int64_t>(config.launch_config.num_sms),
        static_cast<int64_t>(config.launch_config.num_sms_per_cluster),
        static_cast<int64_t>(config.launch_config.num_threads),
        static_cast<int64_t>(config.launch_config.num_tma_threads),
        static_cast<int64_t>(config.launch_config.num_math_threads),
        static_cast<int64_t>(config.launch_config.num_non_epilogue_threads),
        static_cast<int64_t>(config.launch_config.num_epilogue_threads),
        static_cast<int64_t>(config.storage_config.swizzle_a_mode),
        static_cast<int64_t>(config.storage_config.swizzle_b_mode),
        static_cast<int64_t>(config.storage_config.swizzle_cd_mode),
        static_cast<int64_t>(config.storage_config.load_block_m),
        static_cast<int64_t>(config.storage_config.load_block_n),
        static_cast<int64_t>(config.storage_config.store_block_m),
        static_cast<int64_t>(config.storage_config.store_block_n),
    };
}

template <typename ArchSpec>
static std::vector<int64_t> get_best_gemm_config_key_impl(const GemmDesc& desc) {
    const auto& config = get_best_config<ArchSpec>(desc);
    return gemm_config_to_key_vec(desc, config);
}

static std::vector<int64_t> get_best_gemm_config_key_vec(
    const std::string& gemm_type_str,
    const std::string& mma,
    const int& m, const int& n, const int& k, const int& num_groups) {
    const auto gemm_type = parse_gemm_type(gemm_type_str);
    const auto arch_major = device_runtime->get_arch_major();

    KernelType kernel_type;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    if (mma == "fp8") {
        a_dtype = torch::kFloat8_e4m3fn;
        b_dtype = torch::kFloat8_e4m3fn;
        cd_dtype = torch::kBFloat16;
        // SM90 FP8 NT uses 1D2D; SM100 FP8 uses 1D1D (matches gemm.hpp dispatch).
        kernel_type = (arch_major == 9) ? KernelType::Kernel1D2D : KernelType::Kernel1D1D;
    } else if (mma == "bf16") {
        a_dtype = torch::kBFloat16;
        b_dtype = torch::kBFloat16;
        cd_dtype = torch::kBFloat16;
        kernel_type = KernelType::KernelNoSF;
    } else if (mma == "bf16_fp32") {
        a_dtype = torch::kBFloat16;
        b_dtype = torch::kBFloat16;
        cd_dtype = torch::kFloat;
        kernel_type = KernelType::KernelNoSF;
    } else {
        DG_HOST_UNREACHABLE("Unsupported mma for get_best_gemm_config_key; expected fp8|bf16|bf16_fp32");
    }

    // For masked GEMM, callers pass expected_m as `m` (same as runtime).
    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = kernel_type,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a_dtype, .b_dtype = b_dtype, .cd_dtype = cd_dtype,
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = "",
    };

    if (arch_major == 9) {
        return get_best_gemm_config_key_impl<SM90ArchSpec>(desc);
    }
    if (arch_major == 10) {
        return get_best_gemm_config_key_impl<SM100ArchSpec>(desc);
    }
    DG_HOST_UNREACHABLE("Unsupported architecture for get_best_gemm_config_key");
}

#endif

#if 0

static void register_apis(pybind11::module_& m) {
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        device_runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return device_runtime->get_tc_util();
    });
    m.def("set_pdl", [&](const bool& new_enable_pdl) {
        device_runtime->set_pdl(new_enable_pdl);
    });
    m.def("get_pdl", [&]() {
        return device_runtime->get_pdl();
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("get_best_gemm_config_key",
          [](const std::string& gemm_type, const std::string& mma,
             const int& m, const int& n, const int& k, const int& num_groups) {
              const auto key = get_best_gemm_config_key_vec(gemm_type, mma, m, n, k, num_groups);
              pybind11::tuple result(key.size());
              for (size_t i = 0; i < key.size(); ++i)
                  result[i] = key[i];
              return result;
          },
          pybind11::arg("gemm_type"),
          pybind11::arg("mma"),
          pybind11::arg("m"),
          pybind11::arg("n"),
          pybind11::arg("k"),
          pybind11::arg("num_groups") = 1,
          "Return a hashable tuple identifying the best JIT GEMM config for (gemm_type, mma, m, n, k, num_groups).");
#endif
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
        IncludeParser::prepare_init(library_root_path);
#endif
    });
}

#endif

} // namespace deep_gemm::runtime
