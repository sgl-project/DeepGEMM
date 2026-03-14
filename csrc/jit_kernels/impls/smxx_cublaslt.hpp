#pragma once

#include <cublasLt.h>
#include <cute/arch/mma_sm100_umma.hpp>

#include "../../jit/device_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/compatibility.hpp"
#include "../../utils/tensor_view.hpp"

namespace deep_gemm {

static auto get_cublaslt_layout(const cudaDataType& type, const int& rows, const int& cols, const int& ld,
                                const std::optional<int>& batch_count = std::nullopt,
                                const std::optional<int>& batch_offset = std::nullopt) {
    cublasLtMatrixLayout_t layout;
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
    if (batch_count.has_value()) {
        DG_HOST_ASSERT(batch_offset.has_value());

        const int64_t batch_offset_int64 = batch_offset.value();
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count.value(), sizeof(batch_count.value())));
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset_int64, sizeof(batch_offset_int64)));
    }
    return layout;
}

static void call_cublaslt_api(const cublasOperation_t& trans_a,
                              const cublasOperation_t& trans_b,
                              const cublasLtMatrixLayout_t& layout_a,
                              const cublasLtMatrixLayout_t& layout_b,
                              const cublasLtMatrixLayout_t& layout_d,
                              const DGTensorView& a,
                              const DGTensorView& b,
                              const DGTensorView& d,
                              const bool& accumulate) {
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    cudaDataType_t scale_type = CUDA_R_32F;

    // Operation description
    cublasLtMatmulDesc_t desc;
    DG_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&desc, compute_type, scale_type));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

#if DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE
    const int& math_sms = device_runtime->get_num_sms();
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sms, sizeof(math_sms)));
#endif

#if DG_FP8_COMPATIBLE and DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE
    bool fp8_fast_accumulate = false;
    if (dg_dtype_eq(a.scalar_type(), dg_dtype::Float8E4M3))
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fp8_fast_accumulate, sizeof(fp8_fast_accumulate)));
#endif

    // Get cuBLASLt handle, workspace, and stream
    const auto& handle = device_runtime->get_cublaslt_handle();
    auto* workspace_ptr = device_runtime->get_cublaslt_workspace_ptr();
    const auto workspace_bytes = device_runtime->get_cublaslt_workspace_bytes();
    cudaStream_t stream = nullptr;
    DG_CUDA_RUNTIME_CHECK(cudaGetDefaultStream(&stream));

    // Algorithm selection
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_heuristic_results = 0;
    uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspace_bytes, sizeof(workspace_bytes)));
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                           &reduction_scheme_mask, sizeof(reduction_scheme_mask)));
    DG_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_d, layout_d,
                                                     pref, 1, &heuristic, &num_heuristic_results));
    DG_HOST_ASSERT(num_heuristic_results == 1 and "Unable to find any algorithm for the GEMM");

    // Call: D = alpha * (A @ B) + beta * C
    const float& alpha = 1.0, beta = accumulate ? 1.0 : 0.0;
    DG_CUBLASLT_CHECK(cublasLtMatmul(handle,                                // Light handle
                                     desc,                                  // Operation description
                                     &alpha,                                // Alpha
                                     b.data_ptr(), layout_a,                // A
                                     a.data_ptr(), layout_b,                // B
                                     &beta,                                 // Beta
                                     d.data_ptr(), layout_d,                // C
                                     d.data_ptr(), layout_d,                // D
                                     &heuristic.algo,                       // Algorithm
                                     workspace_ptr, workspace_bytes,        // Workspace
                                     stream));                              // Stream

    // Free memory
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_a));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_b));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_d));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescDestroy(desc));
}

static void cublaslt_gemm(const DGTensorView& lhs, const DGTensorView& rhs,
                          const std::optional<DGTensorView>& acc,
                          const DGTensorView& out,
                          const int& m, const int& n, const int& k,
                          const cute::UMMA::Major& a_major, const cute::UMMA::Major& b_major) {
    const auto& trans_a = b_major == cute::UMMA::Major::K ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto& trans_b = a_major == cute::UMMA::Major::K ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Duplicate the accumulator if necessary
    // TODO: remove this
    if (acc.has_value()) {
        if (acc->data_ptr() == out.data_ptr()) {
            bool layout_match = (acc->dim() == out.dim());
            if (layout_match) {
                for (int i = 0; i < out.dim(); ++i) {
                    if (acc->size(i) != out.size(i) || acc->stride(i) != out.stride(i)) {
                        layout_match = false;
                        break;
                    }
                }
            }
            DG_HOST_ASSERT(layout_match);
        } else {
            DG_CUDA_RUNTIME_CHECK(cudaMemcpy(out.data_ptr(), acc->data_ptr(), out.nbytes(), cudaMemcpyDeviceToDevice));
        }
    }

    // Matrix layouts
    const auto& cuda_type_a = dg_dtype_to_cublas(rhs.scalar_type());
    const auto& cuda_type_b = dg_dtype_to_cublas(lhs.scalar_type());
    const auto& cuda_type_d = dg_dtype_to_cublas(out.scalar_type());
    const auto& layout_a = b_major == cute::UMMA::Major::K ? get_cublaslt_layout(cuda_type_a, k, n, rhs.stride(0))
                                                           : get_cublaslt_layout(cuda_type_a, n, k, rhs.stride(1));
    const auto& layout_b = a_major == cute::UMMA::Major::K ? get_cublaslt_layout(cuda_type_b, k, m, lhs.stride(0))
                                                           : get_cublaslt_layout(cuda_type_b, m, k, lhs.stride(1));
    const auto& layout_d = get_cublaslt_layout(cuda_type_d, n, m, out.stride(0));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, acc.has_value());
}


static void cublaslt_bhr_hdr_bhd(const DGTensorView& lhs, const DGTensorView& rhs, const DGTensorView& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto& m = d, n = b, k = r;
    const auto& trans_a = CUBLAS_OP_T;
    const auto& trans_b = CUBLAS_OP_N;

    // Matrix layouts
    const auto& layout_a = get_cublaslt_layout(CUDA_R_16BF, k, m, rhs.stride(1), h, rhs.stride(0));
    const auto& layout_b = get_cublaslt_layout(CUDA_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto& layout_d = get_cublaslt_layout(CUDA_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}


static void cublaslt_bhd_hdr_bhr(const DGTensorView& lhs, const DGTensorView& rhs, const DGTensorView& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto& m = r, n = b, k = d;
    const auto& trans_a = CUBLAS_OP_N;
    const auto& trans_b = CUBLAS_OP_N;

    // Matrix layouts
    const auto& layout_a = get_cublaslt_layout(CUDA_R_16BF, m, k, rhs.stride(1), h, rhs.stride(0));
    const auto& layout_b = get_cublaslt_layout(CUDA_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto& layout_d = get_cublaslt_layout(CUDA_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}

} // namespace deep_gemm
