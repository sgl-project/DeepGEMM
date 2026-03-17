#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <dlpack/dlpack.h>

#include "exception.hpp"

namespace deep_gemm {

// ---------------------------------------------------------------------------
// DLDataType constants matching torch scalar types
// ---------------------------------------------------------------------------
namespace dg_dtype {
inline constexpr DLDataType Float32    = {kDLFloat, 32, 1};
inline constexpr DLDataType Float16    = {kDLFloat, 16, 1};
inline constexpr DLDataType BFloat16   = {kDLBfloat, 16, 1};
inline constexpr DLDataType Float8E4M3 = {kDLFloat8_e4m3fn, 8, 1};
inline constexpr DLDataType Int32      = {kDLInt, 32, 1};
inline constexpr DLDataType Int64      = {kDLInt, 64, 1};
inline constexpr DLDataType UInt8      = {kDLUInt, 8, 1};
inline constexpr DLDataType Byte       = {kDLUInt, 8, 1};
inline constexpr DLDataType PackedFP4  = {kDLUInt, 8, 1};
} // namespace dg_dtype

// Replaces the old torch-based kPackedFP4
inline constexpr DLDataType kPackedFP4 = dg_dtype::PackedFP4;

// ---------------------------------------------------------------------------
// DLDataType comparison operators
// ---------------------------------------------------------------------------
inline bool dg_dtype_eq(DLDataType a, DLDataType b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

inline bool dg_dtype_ne(DLDataType a, DLDataType b) {
    return !dg_dtype_eq(a, b);
}

// ---------------------------------------------------------------------------
// DLDataType utilities
// ---------------------------------------------------------------------------
inline int dg_element_size(DLDataType dtype) {
    return (dtype.bits * dtype.lanes + 7) / 8;
}

inline std::string dg_dtype_to_string(DLDataType dtype) {
    if (dg_dtype_eq(dtype, dg_dtype::Float32))    return "float32";
    if (dg_dtype_eq(dtype, dg_dtype::BFloat16))   return "bfloat16";
    if (dg_dtype_eq(dtype, dg_dtype::Float16))     return "float16";
    if (dg_dtype_eq(dtype, dg_dtype::Float8E4M3))  return "float8_e4m3fn";
    if (dg_dtype_eq(dtype, dg_dtype::Int32))       return "int32";
    if (dg_dtype_eq(dtype, dg_dtype::Int64))       return "int64";
    if (dg_dtype_eq(dtype, dg_dtype::UInt8))       return "uint8";
    return "unknown";
}

inline std::string dg_dtype_to_cuda_type_string(DLDataType dtype) {
    if (dg_dtype_eq(dtype, dg_dtype::Int32))       return "int";
    if (dg_dtype_eq(dtype, dg_dtype::Float32))     return "float";
    if (dg_dtype_eq(dtype, dg_dtype::BFloat16))    return "cutlass::bfloat16_t";
    if (dg_dtype_eq(dtype, dg_dtype::Float8E4M3))  return "cutlass::float_e4m3_t";
    if (dg_dtype_eq(dtype, dg_dtype::PackedFP4))   return "cutlass::detail::float_e2m1_unpacksmem_t";
    DG_HOST_UNREACHABLE("Unsupported dtype for CUDA type string");
}

inline cudaDataType_t dg_dtype_to_cublas(DLDataType dtype) {
    if (dg_dtype_eq(dtype, dg_dtype::Float32))     return CUDA_R_32F;
    if (dg_dtype_eq(dtype, dg_dtype::Float16))     return CUDA_R_16F;
    if (dg_dtype_eq(dtype, dg_dtype::BFloat16))    return CUDA_R_16BF;
    if (dg_dtype_eq(dtype, dg_dtype::Float8E4M3))  return CUDA_R_8F_E4M3;
    if (dg_dtype_eq(dtype, dg_dtype::Int32))       return CUDA_R_32I;
    DG_HOST_UNREACHABLE("Unsupported dtype for cuBLAS");
}

inline CUtensorMapDataType dg_dtype_to_tensormap(DLDataType dtype, bool allow_tf32 = false) {
    if (allow_tf32 && dg_dtype_eq(dtype, dg_dtype::Float32))
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
    if (dg_dtype_eq(dtype, dg_dtype::Int32))       return CU_TENSOR_MAP_DATA_TYPE_INT32;
    if (dg_dtype_eq(dtype, dg_dtype::Float32))     return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    if (dg_dtype_eq(dtype, dg_dtype::BFloat16))    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    if (dg_dtype_eq(dtype, dg_dtype::Float8E4M3))  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (dg_dtype_eq(dtype, dg_dtype::PackedFP4))   return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
    DG_HOST_UNREACHABLE("Unsupported dtype for TensorMap");
}

// ---------------------------------------------------------------------------
// DGTensorView: lightweight, non-owning tensor view wrapping DLTensor data.
// Replaces torch::Tensor for parameter passing and metadata queries.
// ---------------------------------------------------------------------------
class DGTensorView {
    static constexpr int kMaxDim = 8;

    void* data_ = nullptr;
    int ndim_ = 0;
    int64_t shape_[kMaxDim] = {};
    int64_t strides_[kMaxDim] = {};
    DLDataType dtype_ = {};
    int device_id_ = 0;

public:
    DGTensorView() = default;

    explicit DGTensorView(const DLTensor& t) {
        data_ = static_cast<char*>(t.data) + t.byte_offset;
        ndim_ = t.ndim;
        dtype_ = t.dtype;
        device_id_ = t.device.device_id;
        DG_HOST_ASSERT(ndim_ <= kMaxDim);
        for (int i = 0; i < ndim_; ++i)
            shape_[i] = t.shape[i];
        if (t.strides) {
            for (int i = 0; i < ndim_; ++i)
                strides_[i] = t.strides[i];
        } else {
            int64_t s = 1;
            for (int i = ndim_ - 1; i >= 0; --i) {
                strides_[i] = s;
                s *= shape_[i];
            }
        }
    }

    explicit DGTensorView(const DLTensor* t) : DGTensorView(*t) {}

    // -- Data access --------------------------------------------------------
    void* data_ptr() const { return data_; }
    template <typename T> T* data_ptr() const { return static_cast<T*>(data_); }

    // -- Shape / stride queries (torch::Tensor compatible names) ------------
    int dim() const { return ndim_; }
    int64_t size(int i) const { return shape_[i < 0 ? ndim_ + i : i]; }
    int64_t stride(int i) const { return strides_[i < 0 ? ndim_ + i : i]; }
    const int64_t* sizes() const { return shape_; }
    const int64_t* strides_ptr() const { return strides_; }

    DLDataType scalar_type() const { return dtype_; }
    int element_size() const { return dg_element_size(dtype_); }
    int device_id_val() const { return device_id_; }

    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < ndim_; ++i) n *= shape_[i];
        return n;
    }

    int64_t nbytes() const { return numel() * element_size(); }

    bool is_contiguous() const {
        int64_t expected = 1;
        for (int i = ndim_ - 1; i >= 0; --i) {
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

    // -- View operations (return new DGTensorView, zero-copy) ---------------
    DGTensorView transpose(int d0, int d1) const {
        DGTensorView r = *this;
        if (d0 < 0) d0 += ndim_;
        if (d1 < 0) d1 += ndim_;
        std::swap(r.shape_[d0], r.shape_[d1]);
        std::swap(r.strides_[d0], r.strides_[d1]);
        return r;
    }

    DGTensorView unsqueeze(int d) const {
        if (d < 0) d += ndim_ + 1;
        DG_HOST_ASSERT(ndim_ + 1 <= kMaxDim);
        DGTensorView r;
        r.data_ = data_;
        r.ndim_ = ndim_ + 1;
        r.dtype_ = dtype_;
        r.device_id_ = device_id_;
        for (int i = 0; i < d; ++i) {
            r.shape_[i] = shape_[i];
            r.strides_[i] = strides_[i];
        }
        r.shape_[d] = 1;
        r.strides_[d] = (d < ndim_) ? shape_[d] * strides_[d] : 1;
        for (int i = d; i < ndim_; ++i) {
            r.shape_[i + 1] = shape_[i];
            r.strides_[i + 1] = strides_[i];
        }
        return r;
    }

    DGTensorView squeeze(int d) const {
        if (d < 0) d += ndim_;
        DG_HOST_ASSERT(shape_[d] == 1);
        DGTensorView r;
        r.data_ = data_;
        r.ndim_ = ndim_ - 1;
        r.dtype_ = dtype_;
        r.device_id_ = device_id_;
        for (int i = 0; i < d; ++i) {
            r.shape_[i] = shape_[i];
            r.strides_[i] = strides_[i];
        }
        for (int i = d + 1; i < ndim_; ++i) {
            r.shape_[i - 1] = shape_[i];
            r.strides_[i - 1] = strides_[i];
        }
        return r;
    }

    DGTensorView slice(int dim_idx, int64_t start, int64_t end) const {
        if (dim_idx < 0) dim_idx += ndim_;
        DGTensorView r = *this;
        r.data_ = static_cast<char*>(data_) + start * strides_[dim_idx] * element_size();
        r.shape_[dim_idx] = end - start;
        return r;
    }

    DGTensorView permute(std::initializer_list<int> perm) const {
        DGTensorView r;
        r.data_ = data_;
        r.ndim_ = ndim_;
        r.dtype_ = dtype_;
        r.device_id_ = device_id_;
        DG_HOST_ASSERT(static_cast<int>(perm.size()) == ndim_);
        int i = 0;
        for (int p : perm) {
            r.shape_[i] = shape_[p];
            r.strides_[i] = strides_[p];
            i++;
        }
        return r;
    }

    bool same_shape(const DGTensorView& o) const {
        if (ndim_ != o.ndim_) return false;
        for (int i = 0; i < ndim_; ++i)
            if (shape_[i] != o.shape_[i]) return false;
        return true;
    }

    bool same_strides(const DGTensorView& o) const {
        if (ndim_ != o.ndim_) return false;
        for (int i = 0; i < ndim_; ++i)
            if (strides_[i] != o.strides_[i]) return false;
        return true;
    }

    // -- Static factories (create views from raw pointers) ------------------
    static DGTensorView from_ptr(void* data, DLDataType dtype,
                                 std::initializer_list<int64_t> shape_list,
                                 int device_id = 0) {
        DGTensorView r;
        r.data_ = data;
        r.dtype_ = dtype;
        r.device_id_ = device_id;
        r.ndim_ = static_cast<int>(shape_list.size());
        DG_HOST_ASSERT(r.ndim_ <= kMaxDim);
        int i = 0;
        for (auto s : shape_list) r.shape_[i++] = s;
        int64_t st = 1;
        for (int j = r.ndim_ - 1; j >= 0; --j) {
            r.strides_[j] = st;
            st *= r.shape_[j];
        }
        return r;
    }

    static DGTensorView from_ptr_strided(void* data, DLDataType dtype,
                                         std::initializer_list<int64_t> shape_list,
                                         std::initializer_list<int64_t> strides_list,
                                         int device_id = 0) {
        DGTensorView r;
        r.data_ = data;
        r.dtype_ = dtype;
        r.device_id_ = device_id;
        r.ndim_ = static_cast<int>(shape_list.size());
        DG_HOST_ASSERT(r.ndim_ <= kMaxDim);
        DG_HOST_ASSERT(static_cast<int>(strides_list.size()) == r.ndim_);
        int i = 0;
        for (auto s : shape_list) r.shape_[i++] = s;
        i = 0;
        for (auto s : strides_list) r.strides_[i++] = s;
        return r;
    }
};

// ---------------------------------------------------------------------------
// get_shape<N>: extract N-dim shape as a tuple (for structured bindings)
// Replaces the old torch-based version in utils/layout.hpp
// ---------------------------------------------------------------------------
template <int N>
static auto get_shape(const DGTensorView& t) {
    DG_HOST_ASSERT(t.dim() == N);
    return [&t]<size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.size(Is))...);
    }(std::make_index_sequence<N>());
}

// ---------------------------------------------------------------------------
// CUDA stream helper: get the current CUDA stream.
// With tvm-ffi this captures the framework's current stream via DLPack.
// Falls back to the default stream if unavailable.
// ---------------------------------------------------------------------------
inline cudaStream_t dg_get_current_cuda_stream() {
    // Use the default CUDA stream (stream 0).
    // With tvm-ffi, the framework manages stream context externally.
    return nullptr;
}

// ---------------------------------------------------------------------------
// CUDA memory helpers (replace torch::empty / torch::zeros)
// ---------------------------------------------------------------------------
struct DGBuffer {
    void* ptr = nullptr;
    size_t size = 0;

    DGBuffer() = default;
    explicit DGBuffer(size_t sz) : size(sz) {
        if (sz > 0)
            DG_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, sz));
    }

    ~DGBuffer() {
        if (ptr) cudaFree(ptr);
    }

    DGBuffer(const DGBuffer&) = delete;
    DGBuffer& operator=(const DGBuffer&) = delete;
    DGBuffer(DGBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; o.size = 0; }
    DGBuffer& operator=(DGBuffer&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; size = o.size; o.ptr = nullptr; o.size = 0; }
        return *this;
    }
};

} // namespace deep_gemm
