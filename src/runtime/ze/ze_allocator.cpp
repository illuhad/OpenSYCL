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
#include <level_zero/ze_api.h>

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/ze/ze_allocator.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

ze_allocator::ze_allocator(std::size_t device_index,
                           const ze_hardware_context *device,
                           const ze_hardware_manager *hw_manager)
    : _ctx{device->get_ze_context()}, _dev{device->get_ze_device()},
      _global_mem_ordinal{device->get_ze_global_memory_ordinal()},
      _hw_manager{hw_manager} {
  _dev_id = device_id{backend_descriptor{hardware_platform::level_zero,
                                         api_platform::level_zero},
                      static_cast<int>(device_index)};
}

void* ze_allocator::raw_allocate(size_t min_alignment, size_t size_bytes) {
  
  void* out = nullptr;

  ze_device_mem_alloc_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  desc.pNext = nullptr;
  desc.flags = 0;
  desc.ordinal = _global_mem_ordinal;

  ze_result_t err =
      zeMemAllocDevice(_ctx, &desc, size_bytes, min_alignment, _dev, &out);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ze_allocator: zeMemAllocDevice() failed",
                              error_code{"ze", static_cast<int>(err)},
                              error_type::memory_allocation_error});
    return nullptr; 
  }

  return out;
}

void* ze_allocator::raw_allocate_optimized_host(size_t min_alignment,
                                                size_t bytes) {
  void* out = nullptr;
  ze_host_mem_alloc_desc_t desc;
  
  desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  desc.pNext = nullptr;
  desc.flags = 0;

  ze_result_t err = zeMemAllocHost(_ctx, &desc, bytes, min_alignment, &out);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ze_allocator: zeMemAllocHost() failed",
                              error_code{"ze", static_cast<int>(err)},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return out;
}
  
void ze_allocator::raw_free(void *mem) {
  ze_result_t err = zeMemFree(_ctx, mem);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(), 
        error_info{"ze_allocator: zeMemFree() failed", 
            error_code{"ze",static_cast<int>(err)}});
  }
}

void* ze_allocator::raw_allocate_usm(size_t bytes) {

  void* out = nullptr;

  ze_device_mem_alloc_desc_t device_desc;

  device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  device_desc.pNext = nullptr;
  device_desc.flags = 0;
  device_desc.ordinal = _global_mem_ordinal;

  ze_host_mem_alloc_desc_t host_desc;

  host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  host_desc.pNext = nullptr;
  host_desc.flags = 0;

  ze_result_t err =
      zeMemAllocShared(_ctx, &device_desc, &host_desc, bytes, 0, _dev, &out);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ze_allocator: zeMemAllocShared() failed",
                              error_code{"ze", static_cast<int>(err)},
                              error_type::memory_allocation_error});
    return nullptr; 
  }

  return out;
}

bool ze_allocator::is_usm_accessible_from(backend_descriptor b) const {
  return b.hw_platform == hardware_platform::cpu ||
         b.hw_platform == hardware_platform::level_zero;
}

result ze_allocator::query_pointer(const void* ptr, pointer_info& out) const {

  ze_memory_allocation_properties_t props;
  props.pNext = nullptr;

  ze_device_handle_t dev;

  ze_result_t err = zeMemGetAllocProperties(_ctx, ptr, &props, &dev);

  if(err != ZE_RESULT_SUCCESS) {
    return make_error(__acpp_here(),
                   error_info{"ze_allocator: zeMemGetAllocProperties() failed",
                              error_code{"ze", static_cast<int>(err)}});
  }

  if(props.type == ZE_MEMORY_TYPE_UNKNOWN) {
    return make_error(
          __acpp_here(),
          error_info{"ze_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"ze", static_cast<int>(err)},
                     error_type::invalid_parameter_error});
  }

  out.is_optimized_host = props.type == ZE_MEMORY_TYPE_HOST;
  out.is_usm = props.type == ZE_MEMORY_TYPE_SHARED;

  out.is_from_host_backend = false;

  // Lastly, fill out.dev with ze_device_handle_t converted
  // to hipSYCL device_id. This might fail if the
  // ze_device_handle_t is unknown, so return the result of this function
  // for error handling.
  // However, if the allocation is shared or it is a host allocation,
  // no device might be associated, and the error is expected, so do
  // not fail in that case. Only if we have a device allocation do 
  // we really need to enforce that we get a valid value.
  auto dev_handle_err = _hw_manager->device_handle_to_device_id(dev, out.dev);
  if(props.type == ZE_MEMORY_TYPE_DEVICE && !dev_handle_err.is_success()){
    return dev_handle_err;
  }

  return make_success();
}

result ze_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                int advise) const {
  HIPSYCL_DEBUG_WARNING
      << "mem_advise is unsupported on Level Zero backend, ignoring"
      << std::endl;
  return make_success();
}

device_id ze_allocator::get_device() const {
  return _dev_id;
}

}
}
