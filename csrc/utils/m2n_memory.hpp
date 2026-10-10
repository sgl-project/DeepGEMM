#pragma once

#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <tuple>

#include <torch/torch.h>

#include <deep_jit/backend/cuda/driver.hpp>
#include "exception.hpp"

namespace deep_gemm::m2n_memory {

enum class HandleKind : int64_t { Fabric = 1, Ipc = 2 };
using Metadata = std::array<int64_t, 10>;

#if CUDA_VERSION >= 12030
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuDeviceGetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemGetAllocationGranularity);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemCreate);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemExportToShareableHandle);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemImportFromShareableHandle);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemAddressReserve);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemMap);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemSetAccess);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemUnmap);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemAddressFree);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemRelease);
#endif

inline void report_cleanup_error(cudaError_t result, const char* operation) noexcept {
    if (result != cudaSuccess && result != cudaErrorCudartUnloading)
        std::fprintf(stderr, "DeepGEMM M2N cleanup: %s failed: %s\n",
                     operation, cudaGetErrorString(result));
}

inline void report_cleanup_error(CUresult result, const char* operation) noexcept {
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED)
        std::fprintf(stderr, "DeepGEMM M2N cleanup: %s failed: CUDA error %d\n",
                     operation, static_cast<int>(result));
}

struct Allocation {
    HandleKind kind;
    int device;
    size_t size = 0;
    void* pointer = nullptr;
    CUmemGenericAllocationHandle handle = 0;
    bool owns_handle = false;
    bool mapped = false;
    bool imported = false;

    Allocation(HandleKind kind, int device) : kind(kind), device(device) {}
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

    ~Allocation() noexcept {
        int previous_device = device;
        report_cleanup_error(cudaGetDevice(&previous_device), "cudaGetDevice");
        report_cleanup_error(cudaSetDevice(device), "cudaSetDevice");
        if (kind == HandleKind::Ipc) {
            if (pointer != nullptr) {
                if (imported)
                    report_cleanup_error(cudaIpcCloseMemHandle(pointer), "cudaIpcCloseMemHandle");
                else
                    report_cleanup_error(cudaFree(pointer), "cudaFree");
            }
        }
#if CUDA_VERSION >= 12030
        else {
            try {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                if (mapped) {
                    // VMM unmap does not wait for kernels using this mapping.
                    report_cleanup_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    report_cleanup_error(lazy_cuMemUnmap(address, size), "cuMemUnmap");
                }
                if (pointer != nullptr)
                    report_cleanup_error(lazy_cuMemAddressFree(address, size), "cuMemAddressFree");
                if (owns_handle)
                    report_cleanup_error(lazy_cuMemRelease(handle), "cuMemRelease");
            } catch (const DGException& error) {
                std::fprintf(stderr, "DeepGEMM M2N cleanup: %s\n", error.what());
            }
        }
#endif
        if (previous_device != device)
            report_cleanup_error(cudaSetDevice(previous_device), "cudaSetDevice");
    }
};

inline torch::Tensor make_tensor(std::shared_ptr<Allocation> allocation, int64_t num_bytes) {
    const auto options = torch::TensorOptions().dtype(torch::kInt8)
        .device(torch::kCUDA, allocation->device);
    void* pointer = allocation->pointer;
    return at::from_blob(pointer, {num_bytes},
                         [allocation = std::move(allocation)](void*) {},
                         options, options.device());
}

#if CUDA_VERSION >= 12030
inline bool fabric_unavailable(CUresult result) {
    if (result == CUDA_ERROR_NOT_PERMITTED)
        std::fprintf(stderr, "DeepGEMM M2N: CUDA FABRIC access is not permitted; "
                     "falling back to CUDA IPC, which requires same-host peers.\n");
    return result == CUDA_ERROR_NOT_SUPPORTED || result == CUDA_ERROR_NOT_PERMITTED;
}

inline void map_fabric(Allocation& allocation) {
    CUdeviceptr address = 0;
    DJ_CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&address, allocation.size, 0, 0, 0));
    allocation.pointer = reinterpret_cast<void*>(address);
    DJ_CUDA_DRIVER_CHECK(lazy_cuMemMap(address, allocation.size, 0, allocation.handle, 0));
    allocation.mapped = true;
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = allocation.device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DJ_CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(address, allocation.size, &access, 1));
}

inline std::shared_ptr<Allocation> allocate_fabric(int64_t num_bytes, int device, Metadata& metadata) {
    int supported = 0;
    DJ_CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    if (!supported)
        return nullptr;

    CUmemAllocationProp properties = {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t granularity = 0;
    auto result = lazy_cuMemGetAllocationGranularity(
        &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (fabric_unavailable(result))
        return nullptr;
    DJ_CUDA_DRIVER_CHECK(result);

    auto allocation = std::make_shared<Allocation>(HandleKind::Fabric, device);
    allocation->size = (static_cast<size_t>(num_bytes) + granularity - 1) / granularity * granularity;
    result = lazy_cuMemCreate(&allocation->handle, allocation->size, &properties, 0);
    if (fabric_unavailable(result))
        return nullptr;
    DJ_CUDA_DRIVER_CHECK(result);
    allocation->owns_handle = true;

    CUmemFabricHandle handle = {};
    static_assert(sizeof(handle) == 8 * sizeof(int64_t));
    result = lazy_cuMemExportToShareableHandle(&handle, allocation->handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (fabric_unavailable(result))
        return nullptr;
    DJ_CUDA_DRIVER_CHECK(result);
    map_fabric(*allocation);
    std::memcpy(metadata.data() + 2, &handle, sizeof(handle));
    return allocation;
}
#endif

inline std::tuple<torch::Tensor, torch::Tensor> allocate_buffer(int64_t num_bytes) {
    DG_HOST_ASSERT(num_bytes > 0);
    int device = 0;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device));
    Metadata metadata = {};
    std::shared_ptr<Allocation> allocation;
#if CUDA_VERSION >= 12030
    allocation = allocate_fabric(num_bytes, device, metadata);
#endif
    if (allocation == nullptr) {
        allocation = std::make_shared<Allocation>(HandleKind::Ipc, device);
        allocation->size = static_cast<size_t>(num_bytes);
        DG_CUDA_RUNTIME_CHECK(cudaMalloc(&allocation->pointer, allocation->size));
        cudaIpcMemHandle_t handle = {};
        static_assert(sizeof(handle) == 8 * sizeof(int64_t));
        DG_CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&handle, allocation->pointer));
        std::memcpy(metadata.data() + 2, &handle, sizeof(handle));
    }
    metadata[0] = static_cast<int64_t>(allocation->kind);
    metadata[1] = static_cast<int64_t>(allocation->size);
    auto metadata_tensor = torch::empty({static_cast<int64_t>(metadata.size())},
                                      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    std::memcpy(metadata_tensor.data_ptr(), metadata.data(), sizeof(metadata));
    return {make_tensor(std::move(allocation), num_bytes), metadata_tensor};
}

inline torch::Tensor import_buffer(const Metadata& metadata) {
    DG_HOST_ASSERT(metadata[1] > 0);
    const auto kind = static_cast<HandleKind>(metadata[0]);
    DG_HOST_ASSERT(kind == HandleKind::Fabric || kind == HandleKind::Ipc);
    int device = 0;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device));
    auto allocation = std::make_shared<Allocation>(kind, device);
    allocation->size = static_cast<size_t>(metadata[1]);
    allocation->imported = true;
    if (kind == HandleKind::Ipc) {
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, metadata.data() + 2, sizeof(handle));
        DG_CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(
            &allocation->pointer, handle, cudaIpcMemLazyEnablePeerAccess));
    } else {
#if CUDA_VERSION >= 12030
        CUmemFabricHandle handle;
        std::memcpy(&handle, metadata.data() + 2, sizeof(handle));
        DJ_CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(
            &allocation->handle, &handle, CU_MEM_HANDLE_TYPE_FABRIC));
        allocation->owns_handle = true;
        map_fabric(*allocation);
#else
        DG_HOST_UNREACHABLE("FABRIC memory import requires CUDA 12.3 or newer");
#endif
    }
    return make_tensor(std::move(allocation), metadata[1]);
}

} // namespace deep_gemm::m2n_memory
