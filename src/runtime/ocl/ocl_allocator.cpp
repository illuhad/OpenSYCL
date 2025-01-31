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
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"

#include "hipSYCL/runtime/ocl/ocl_allocator.hpp"
#include <cstddef>

namespace hipsycl {
namespace rt {

ocl_allocator::ocl_allocator(rt::device_id dev, ocl_usm* usm)
: _dev{dev}, _usm{usm} {}

void* ocl_allocator::raw_allocate(size_t min_alignment, size_t size_bytes,
                                  const allocation_hints &hints) {
  if(!_usm->is_available()) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return nullptr;
  }
  
  cl_int err;
  void* ptr = _usm->malloc_device(size_bytes, min_alignment, err);
  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: USM device allocation failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void *ocl_allocator::raw_allocate_optimized_host(size_t min_alignment,
                                                 size_t bytes,
                                                 const allocation_hints &hints) {
  if(!_usm->is_available()) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return nullptr;
  }
  cl_int err;
  void* ptr = _usm->malloc_host(bytes, min_alignment, err);
  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: USM host allocation failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void ocl_allocator::raw_free(void *mem) {
  if(!_usm->is_available()) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return;
  }
  cl_int err = _usm->free(mem);
  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: USM memory free() failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
  }
}

void *ocl_allocator::raw_allocate_usm(size_t bytes,
                                      const allocation_hints &hints) {
  if(!_usm->is_available()) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return nullptr;
  }
  cl_int err;
  void* ptr = _usm->malloc_shared(bytes, 0, err);
  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_allocator: USM shared allocation failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

bool ocl_allocator::is_usm_accessible_from(backend_descriptor b) const {
  // TODO: This is INCORRECT. We would need to compare the OpenCL platform, not
  // just the backend descriptor. The current API does not allow for that.
  // Probably, we should change this function to accept a device_id.
  // Or just remove this function entirely, as it does not seem to be used?
  return b.hw_platform == hardware_platform::ocl;
}

device_id ocl_allocator::get_device() const {
  return _dev;
}

result ocl_allocator::query_pointer(const void* ptr, pointer_info& out) const {
  if(!_usm->is_available()) {
    auto err = make_error(__acpp_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    register_error(err);
    return err;
  }
  cl_int err = _usm->get_alloc_info(ptr, out);
  if(err != CL_SUCCESS) {
    return make_error(
          __acpp_here(),
          error_info{"ocl_allocator: query_pointer(): query failed",
                     error_code{"CL", err}});
  }
  return make_success();
}

result ocl_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                          int advise) const {
  // TODO
  return make_success();
}


}
}

