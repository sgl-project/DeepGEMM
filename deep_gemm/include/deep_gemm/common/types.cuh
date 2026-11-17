#pragma once

#include <cute/arch/mma_sm100_desc.hpp>

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
    MXFP4       = 2,
    MXF4        = MXFP4,
    NVFP4       = 3,
};

// MegaMoE L1 epilogue activation
enum class MegaMoEActivation {
    SwiGLU      = 0,
    // SwiGLU with `sigmoid(1.702 * gate)`
    SwiGLUOAI   = 1,
    // Tanh-bounded gate and soft-clipped up
    SiTU        = 2,
};

// How MegaMoE expert weights and their SFs are stored, per expert. `g{j}` / `u{j}` are the gate / up rows
// of L1 output pair j. SF rows follow their weight rows; CUTLASS 128x4 atoms are
// `[N / 128][K / (4 * gran_k)][128 words]`, word 4 * (row % 32) + (row / 32) % 4.
enum class MegaMoEWeightLayout {
    // `transform_weights_for_mega_moe` output. L1 per 16 pairs: NVFP4 and MXF4 MMA
    // [g0 g2 .. g14, u0 u2 .. u14, g1 g3 .. g15, u1 u3 .. u15], other MMA [g0 .. g7, u0 .. u7, g8 .. g15, u8 .. u15].
    // L2 rows plain. SFs row-major, each 128-row block transposed for UTCCP (row 4i + j holds row 32j + i).
    GateUpBlock8        = 0,
    // TRT-LLM `shuffle_matrix_a` order. L1 per 16 pairs: [u0 u2 .. u14, g0 g2 .. g14, u1 u3 .. u15, g1 g3 .. g15].
    // L2: in each 32-row group, row 8 * i1 + i2 holds row 4 * i2 + i1. BF16 weights are also K-blocked,
    // `[E, K / 64, N, 64]`. SFs in CUTLASS 128x4 atoms.
    RowShuffled         = 1,
    // L1 [g0 u0 g1 u1 ..], L2 rows plain, SFs in CUTLASS 128x4 atoms.
    GateUpInterleaved   = 2,
    // FlashInfer CuTe DSL swap-AB order. L1 per 16 pairs: [u0 .. u15, g0 .. g15]. L2 rows plain,
    // SFs in CUTLASS 128x4 atoms.
    UpGateBlock16       = 3,
};

constexpr CUTLASS_HOST_DEVICE int get_element_size(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 2;
        case MmaKind::MXFP8FP4: return 1;
        case MmaKind::MXFP4:    return 1;
        case MmaKind::NVFP4:    return 1;
        default: return 0;
    }
}

constexpr CUTLASS_HOST_DEVICE int get_element_bits(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 16;
        case MmaKind::MXFP8FP4: return 8;
        case MmaKind::MXFP4:    return 4;
        case MmaKind::NVFP4:    return 4;
        default: return 0;
    }
}

constexpr CUTLASS_HOST_DEVICE int get_sf_gran_k(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::MXFP8FP4: return 32;
        case MmaKind::MXFP4:    return 32;
        case MmaKind::NVFP4:    return 16;
        default: return 0;
    }
}

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

constexpr CUTLASS_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

// Host-provided epilogue parameters. Operators keep their per-thread state separately;
// only the fields consumed by the selected operator are meaningful.
struct EpilogueOperatorArgs {
    uint32_t* sfd = nullptr;
    uint32_t sfd_stride = 0;
    // Non-zero selects FlashInfer's 128x4 swizzled SFD: words between 128-row tiles
    uint32_t sfd_mn_tile_stride = 0;
    uint32_t shape_m = 0, shape_n = 0;
    float alpha = 1.0f;
    float clamp_value = 0.0f;  // Zero disables clamping
    uint32_t* clamped_count = nullptr;
    uint32_t* clamp_count_ready = nullptr;  // Per-stream GPU launch tag for in-kernel initialization
};

static constexpr uint32_t kNumDeviceLocalityDomains = 2;

} // namespace deep_gemm
