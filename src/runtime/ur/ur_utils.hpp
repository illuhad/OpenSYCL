/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef UR_UTILS_HPP
#define UR_UTILS_HPP

#include "hipSYCL/runtime/error.hpp"

#include <string>
#include <ur_api.h>

namespace hipsycl::rt {
inline std::string ur_strerror(const ur_result_t res) {
  switch (res) {
  case UR_RESULT_SUCCESS:
    return "Success";
  case UR_RESULT_ERROR_INVALID_OPERATION:
    return "Invalid operation";
  case UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES:
    return "Invalid queue properties";
  case UR_RESULT_ERROR_INVALID_QUEUE:
    return "Invalid queue";
  case UR_RESULT_ERROR_INVALID_VALUE:
    return "Invalid value";
  case UR_RESULT_ERROR_INVALID_CONTEXT:
    return "Invalid context";
  case UR_RESULT_ERROR_INVALID_PLATFORM:
    return "Invalid platform";
  case UR_RESULT_ERROR_INVALID_BINARY:
    return "Invalid binary";
  case UR_RESULT_ERROR_INVALID_PROGRAM:
    return "Invalid program";
  case UR_RESULT_ERROR_INVALID_SAMPLER:
    return "Invalid sampler";
  case UR_RESULT_ERROR_INVALID_BUFFER_SIZE:
    return "Invalid buffer size";
  case UR_RESULT_ERROR_INVALID_MEM_OBJECT:
    return "Invalid memory object";
  case UR_RESULT_ERROR_INVALID_EVENT:
    return "Invalid event";
  case UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST:
    return "Invalid event wait list";
  case UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
    return "Misaligned sub-buffer offset";
  case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
    return "Invalid work group size";
  case UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE:
    return "Compiler not available";
  case UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE:
    return "Profiling info not available";
  case UR_RESULT_ERROR_DEVICE_NOT_FOUND:
    return "Device not found";
  case UR_RESULT_ERROR_INVALID_DEVICE:
    return "Invalid device";
  case UR_RESULT_ERROR_DEVICE_LOST:
    return "Device lost";
  case UR_RESULT_ERROR_DEVICE_REQUIRES_RESET:
    return "Device requires reset";
  case UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
    return "Device in low power state";
  case UR_RESULT_ERROR_DEVICE_PARTITION_FAILED:
    return "Device partition failed";
  case UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT:
    return "Invalid device partition count";
  case UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE:
    return "Invalid work item size";
  case UR_RESULT_ERROR_INVALID_WORK_DIMENSION:
    return "Invalid work dimension";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGS:
    return "Invalid kernel arguments";
  case UR_RESULT_ERROR_INVALID_KERNEL:
    return "Invalid kernel";
  case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "Invalid kernel name";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "Invalid kernel argument index";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "Invalid kernel argument size";
  case UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "Invalid kernel attribute value";
  case UR_RESULT_ERROR_INVALID_IMAGE_SIZE:
    return "Invalid image size";
  case UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "Invalid image format descriptor";
  case UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
    return "Memory object allocation failure";
  case UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE:
    return "Invalid program executable";
  case UR_RESULT_ERROR_UNINITIALIZED:
    return "Uninitialized error";
  case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "Out of host memory";
  case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "Out of device memory";
  case UR_RESULT_ERROR_OUT_OF_RESOURCES:
    return "Out of resources";
  case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
    return "Program build failure";
  case UR_RESULT_ERROR_PROGRAM_LINK_FAILURE:
    return "Program link failure";
  case UR_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "Unsupported version";
  case UR_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "Unsupported feature";
  case UR_RESULT_ERROR_INVALID_ARGUMENT:
    return "Invalid argument";
  case UR_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "Invalid null handle";
  case UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "Handle object in use";
  case UR_RESULT_ERROR_INVALID_NULL_POINTER:
    return "Invalid null pointer";
  case UR_RESULT_ERROR_INVALID_SIZE:
    return "Invalid size";
  case UR_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "Unsupported size";
  case UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "Unsupported alignment";
  case UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "Invalid synchronization object";
  case UR_RESULT_ERROR_INVALID_ENUMERATION:
    return "Invalid enumeration";
  case UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "Unsupported enumeration";
  case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "Unsupported image format";
  case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "Invalid native binary";
  case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "Invalid global name";
  case UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE:
    return "Function address not available";
  case UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "Invalid group size dimension";
  case UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "Invalid global width dimension";
  case UR_RESULT_ERROR_PROGRAM_UNLINKED:
    return "Program unlinked";
  case UR_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "Overlapping regions";
  case UR_RESULT_ERROR_INVALID_HOST_PTR:
    return "Invalid host pointer";
  case UR_RESULT_ERROR_INVALID_USM_SIZE:
    return "Invalid USM size";
  case UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE:
    return "Object allocation failure";
  case UR_RESULT_ERROR_ADAPTER_SPECIFIC:
    return "Adapter-specific error";
  case UR_RESULT_ERROR_LAYER_NOT_PRESENT:
    return "Layer not present";
  case UR_RESULT_ERROR_IN_EVENT_LIST_EXEC_STATUS:
    return "In-event list execution status error";
  case UR_RESULT_ERROR_DEVICE_NOT_AVAILABLE:
    return "Device not available";
  case UR_RESULT_ERROR_INVALID_SPEC_ID:
    return "Invalid specialization ID";
  case UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP:
    return "Invalid command buffer (experimental)";
  case UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP:
    return "Invalid command buffer sync point (experimental)";
  case UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP:
    return "Invalid command buffer sync point wait list (experimental)";
  case UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP:
    return "Invalid command buffer command handle (experimental)";
  case UR_RESULT_ERROR_UNKNOWN:
    return "Unknown error";
  case UR_RESULT_FORCE_UINT32:
    return "Force uint32 error";
  }

  return "Unknown error code";
}

inline error_info ur_error_info(const std::string &desc, const ur_result_t res) {
  return error_info{desc + ": " + ur_strerror(res), error_code{"ur", static_cast<int>(res)}};
}

inline error_info ur_error_info(const std::string &desc, const ur_result_t res,
                                const error_type etype) {
  return error_info{desc + ": " + ur_strerror(res), error_code{"ur", static_cast<int>(res)}, etype};
}

inline result make_ur_error(const source_location &origin, const std::string &desc) {
  const error_info info{desc};
  return make_error(origin, info);
}

} // namespace hipsycl::rt

#endif
