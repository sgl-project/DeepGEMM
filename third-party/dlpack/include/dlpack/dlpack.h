/*
 * Minimal DLPack header for DeepGEMM.
 * Based on DLPack v0.8+ (https://github.com/dmlc/dlpack).
 * At build time, this may be overridden by the version from apache-tvm-ffi.
 *
 * Licensed under Apache 2.0.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DLPACK_VERSION
#define DLPACK_MAJOR_VERSION 0
#define DLPACK_MINOR_VERSION 8
#define DLPACK_VERSION (DLPACK_MAJOR_VERSION * 100 + DLPACK_MINOR_VERSION)
#endif

typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLCUDAManaged = 13,
} DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLFloat8_e4m3fn = 6U,
  kDLFloat8_e5m2 = 7U,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

typedef enum {
  kDLManagedTensorVersionNone = 0,
} DLManagedTensorVersion;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
}
#endif

#endif  /* DLPACK_DLPACK_H_ */
