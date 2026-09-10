#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION),
        {include_dir},
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        // Preserve the downstream family targets and explicit SM120 code generation.
        // DeepJIT requires CUDA 12.9+, where the family suffix is available.
        const auto major = runtime->device.get_arch_major();
        const auto minor = runtime->device.get_arch_minor();
        if (major == 10 and minor != 1)
            runtime->default_compiler_options.arch = "100f";
        if (major == 12) {
            runtime->default_compiler_options.arch = "120f";
            runtime->default_compiler_options.nvcc_flags->emplace_back(
                "-gencode=arch=compute_120f,code=sm_120f");
        }
        return runtime;
    });
}

}  // namespace deep_gemm
