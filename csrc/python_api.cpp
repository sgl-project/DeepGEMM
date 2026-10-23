#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

#include "utils/arch_traits.hpp"
#include "apis/config.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/epilogue_class.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/locality_domain.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // Reports the arch majors this binary's dispatcher accepts, before any CUDA
    // context exists -- so a stale `_C.so` under a newer Python side is caught.
    // NOTES: plain `m.def(name, fn)`; a type annotation in the name would become
    // part of the attribute name under pybind11.
    m.def("supported_arch_majors", []() {
        return std::vector<int64_t>(deep_gemm::arch::kSupportedArchMajors.begin(),
                                    deep_gemm::arch::kSupportedArchMajors.end());
    });

    // SM110 (Jetson Thor) policy table: which SM100-family features this arch has
    // actually been validated on. Reported as (name, sm100_state, sm110_state).
    m.def("arch_feature_status", []() {
        std::vector<std::tuple<std::string, int64_t, int64_t>> result;
        for (const auto& feature : deep_gemm::arch::kFeatureStates)
            result.emplace_back(feature.name,
                                static_cast<int64_t>(feature.state.sm100),
                                static_cast<int64_t>(feature.state.sm110));
        return result;
    });

    // Register JIT objects
    deep_jit::register_python_api(m, deep_gemm::jit);

    // Register config APIs
    deep_gemm::config::register_apis(m);

    // Register locality domain APIs
    deep_gemm::locality_domain::register_apis(m);

    // Register epilogue classes
    deep_gemm::epilogue_class::register_apis(m);

    // Register kernels
    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::mega_mhc::register_apis(m);
    deep_gemm::mega_gate::register_apis(m);
}
