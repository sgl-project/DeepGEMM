#pragma once

#include <array>
#include <cstdint>

// ---------------------------------------------------------------------------
// The set of CUDA architecture majors this _C.so's host dispatcher accepts.
//
// Every dispatch predicate in csrc/ (gemm.hpp, attention.hpp, einsum.hpp,
// layout.hpp, utils/layout.hpp, mega_*.hpp, hyperconnection.hpp and the
// contiguous-K heuristic in jit_kernels/heuristics/runtime.hpp) gates kernels
// on `arch_major`. This list is the same information in a form a *program* can
// read, so that downstream frameworks (SGLang and friends) can decide whether
// to enable a kernel family without launching a kernel first -- the decision
// often has to be made at import time, before any CUDA context exists.
//
// It must stay in sync with those predicates. The reason it lives here, rather
// than being inferred from the device, is deliberate: the question is "does
// this binary know about SM110", not "is this machine an SM110". Those differ
// exactly when someone installs a wheel whose Python side is new but whose
// _C.so is stale, which is the failure mode this API exists to catch.
//
// `supported_arch_majors()` is exposed to Python as `deep_gemm._C
// .supported_arch_majors()` and re-exported as
// `deep_gemm.supported_arch_majors()`.
// ---------------------------------------------------------------------------
namespace deep_gemm::arch {

inline constexpr std::array<int64_t, 4> kSupportedArchMajors = {9, 10, 11, 12};

}  // namespace deep_gemm::arch
