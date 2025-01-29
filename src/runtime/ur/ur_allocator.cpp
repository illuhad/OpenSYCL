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
#include "hipSYCL/runtime/ur/ur_allocator.hpp"

#include "hipSYCL/runtime/ur/ur_hardware_manager.hpp"
#include "ur_utils.hpp"

#include <ur_api.h>

using namespace hipsycl::rt;

namespace {} // namespace

ur_allocator::ur_allocator(ur_device_handle_t dev, ur_context_handle_t ctx,
                           const std::size_t device_index)
    : _dev(dev), _ctx(ctx),
      _dev_id{device_id{backend_descriptor{hardware_platform::unified_runtime,
                                           api_platform::unified_runtime},
                        static_cast<int>(device_index)}} {}

void *ur_allocator::raw_allocate(size_t, const size_t bytes) {
  void *out = nullptr;
  const ur_result_t res =
      urUSMDeviceAlloc(_ctx, _dev, nullptr, nullptr, bytes, &out);

  if (res != UR_RESULT_SUCCESS) {
    register_error(make_error(
        __acpp_here(), ur_error_info("urUSMDeviceAlloc() failed", res,
                                     error_type::memory_allocation_error)));
    return nullptr;
  }

  return out;
}

void *ur_allocator::raw_allocate_optimized_host(size_t, const size_t bytes) {
  void *out = nullptr;
  const ur_result_t res = urUSMHostAlloc(_ctx, nullptr, nullptr, bytes, &out);

  if (res != UR_RESULT_SUCCESS) {
    register_error(make_error(__acpp_here(),
                              ur_error_info("urUSMHostAlloc() failed", res)));
    return nullptr;
  }

  return out;
}
void *ur_allocator::raw_allocate_usm(const size_t bytes) {
  void *out = nullptr;

  const ur_result_t res =
      urUSMSharedAlloc(_ctx, _dev, nullptr, nullptr, bytes, &out);
  if (res != UR_RESULT_SUCCESS) {
    register_error(make_error(
        __acpp_here(), ur_error_info("urUSMSharedAlloc() failed", res,
                                     error_type::memory_allocation_error)));
    return nullptr;
  }

  return out;
}

void ur_allocator::raw_free(void *mem) {
  const ur_result_t res = urUSMFree(_ctx, mem);
  if (res != UR_RESULT_SUCCESS) {
    register_error(
        make_error(__acpp_here(), ur_error_info("urUSMFree() failed", res)));
  }
}

bool ur_allocator::is_usm_accessible_from(const backend_descriptor b) const {
  return b.hw_platform == hardware_platform::cpu ||
         b.hw_platform == hardware_platform::unified_runtime;
}

result ur_allocator::query_pointer(const void *ptr, pointer_info &out) const {

  ur_usm_type_t type;

  const ur_result_t res = urUSMGetMemAllocInfo(
      _ctx, ptr, UR_USM_ALLOC_INFO_TYPE, sizeof(type), &type, nullptr);

  if (res != UR_RESULT_SUCCESS) {
    return make_error(__acpp_here(),
                      ur_error_info("urUSMGetMemAllocInfo() failed", res));
  }

  if (type == UR_USM_TYPE_UNKNOWN) {
    return make_error(
        __acpp_here(),
        ur_error_info("urUSMGetMemAllocInfo() returned unknown type",
                      UR_RESULT_ERROR_INVALID_VALUE));
  }

  out.is_optimized_host = type == UR_USM_TYPE_HOST;
  out.is_usm = type == UR_USM_TYPE_SHARED;
  out.is_from_host_backend = false;
  out.dev = _dev_id;

  return make_success();
}
result ur_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                int advise) const {
  // TODO: Implement this
  // urEnqueueUSMAdvise()

  return make_success();
}

device_id ur_allocator::get_device() const { return _dev_id; }
