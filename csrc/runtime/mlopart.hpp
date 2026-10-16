#pragma once

#include <unistd.h>

#include <cuda.h>
#include <deep_jit/backend/cuda/driver.hpp>
#ifndef DG_USE_TORCH_LIBRARY
#include <pybind11/pybind11.h>
#endif

#include "jit.hpp"

namespace deep_gemm::mlopart {

// We have to use IPC to allocate localized memory via MLOPart prior to CUDA 13.4
// For convenience this is done in Python
#ifdef DG_USE_TORCH_LIBRARY
// The library has no CPython dependency: `deep_gemm/locality_domain.py` installs `ctypes` callbacks
// into `deep_gemm.utils.mlopart` (ctypes takes the GIL itself when they are called)
struct Callbacks {
    bool (*is_available)(const uint8_t* device_uuid) = nullptr;
    int (*create_memory)(const uint8_t* device_uuid, size_t num_bytes, int domain_idx) = nullptr;
    void (*release)() = nullptr;
};

static Callbacks callbacks;

static void set_callbacks(const int64_t& is_available, const int64_t& create_memory, const int64_t& release) {
    callbacks.is_available = reinterpret_cast<decltype(Callbacks::is_available)>(is_available);
    callbacks.create_memory = reinterpret_cast<decltype(Callbacks::create_memory)>(create_memory);
    callbacks.release = reinterpret_cast<decltype(Callbacks::release)>(release);
}

static const uint8_t* get_device_uuid() {
    DG_HOST_ASSERT(callbacks.is_available != nullptr and "MLOPart callbacks are not installed");
    return reinterpret_cast<const uint8_t*>(jit->device.get_prop().uuid.bytes);
}

static bool is_available() {
    return callbacks.is_available(get_device_uuid());
}

static int create_memory_fd(const size_t& num_bytes, const int& domain_idx) {
    const auto fd = callbacks.create_memory(get_device_uuid(), num_bytes, domain_idx);
    DG_HOST_ASSERT(fd >= 0 and "MLOPart failed to create memory");
    return fd;
}

static void release() {
    get_device_uuid();
    callbacks.release();
}
#else
static pybind11::module_ get_module() {
    return pybind11::module_::import("deep_gemm.utils.mlopart");
}

static pybind11::bytes get_device_uuid() {
    const auto& uuid = jit->device.get_prop().uuid.bytes;
    return pybind11::bytes(uuid, sizeof(uuid));
}

static bool is_available() {
    return get_module().attr("is_available")(get_device_uuid()).cast<bool>();
}

static int create_memory_fd(const size_t& num_bytes, const int& domain_idx) {
    return get_module().attr("create_memory")(get_device_uuid(), num_bytes, domain_idx).cast<int>();
}

static void release() {
    get_module().attr("release")();
}
#endif

static CUmemGenericAllocationHandle create_memory(const size_t& num_bytes, const int& domain_idx) {
    const auto fd = create_memory_fd(num_bytes, domain_idx);
    CUmemGenericAllocationHandle handle = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    close(fd);
    return handle;
}

}  // namespace deep_gemm::mlopart
