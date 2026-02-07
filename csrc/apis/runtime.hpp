#pragma once

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"

namespace deep_gemm::runtime {

// The init and other functions are now exposed via TORCH_LIBRARY in python_api.cpp

} // namespace deep_gemm::runtime
