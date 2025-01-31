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
#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/runtime/hip/hip_allocator.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"

namespace hipsycl {
namespace rt {

hip_allocator::hip_allocator(backend_descriptor desc, int hip_device)
    : _backend_descriptor{desc}, _dev{hip_device}
{}
      
void *hip_allocator::raw_allocate(size_t min_alignment, size_t size_bytes,
                                  const allocation_hints &hints)
{
  void *ptr;
  hip_device_manager::get().activate_device(_dev);
  hipError_t err = hipMalloc(&ptr, size_bytes);

  if (err != hipSuccess) {
    register_error(__acpp_here(),
                   error_info{"hip_allocator: hipMalloc() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

void *hip_allocator::raw_allocate_optimized_host(size_t min_alignment,
                                                 size_t bytes,
                                                 const allocation_hints &hints) {
  void *ptr;
  hip_device_manager::get().activate_device(_dev);

  hipError_t err = hipHostMalloc(&ptr, bytes, hipHostMallocDefault);

  if (err != hipSuccess) {
    register_error(__acpp_here(),
                   error_info{"hip_allocator: hipHostMalloc() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void hip_allocator::raw_free(void *mem) {

  pointer_info info;
  result query_result = query_pointer(mem, info);

  if (!query_result.is_success()) {
    register_error(query_result);
    return;
  }
  
  hipError_t err;
  if (info.is_optimized_host)
    err = hipHostFree(mem);
  else
    err = hipFree(mem);
  
  if (err != hipSuccess) {
    register_error(__acpp_here(),
                   error_info{"hip_allocator: hipFree() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
  }
}

void * hip_allocator::raw_allocate_usm(size_t bytes,
                                       const allocation_hints &hints)
{
  hip_device_manager::get().activate_device(_dev);

  void *ptr;
  auto err = hipMallocManaged(&ptr, bytes);
  if (err != hipSuccess) {
    register_error(__acpp_here(),
                   error_info{"hip_allocator: hipMallocManaged() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

bool hip_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  // TODO: Formulate this better - this assumes that either CUDA+CPU or
  // ROCm + CPU are active at the same time
  return true;
}

result hip_allocator::query_pointer(const void *ptr, pointer_info &out) const
{
  hipPointerAttribute_t attribs;

  auto err = hipPointerGetAttributes(&attribs, ptr);

  if (err != hipSuccess) {
    if (err == hipErrorInvalidValue)
      return make_error(
          __acpp_here(),
          error_info{"hip_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"HIP", err},
                     error_type::invalid_parameter_error});
    else
      return make_error(
          __acpp_here(),
          error_info{"hip_allocator: query_pointer(): query failed",
                     error_code{"HIP", err}});
  }
#if HIP_VERSION_MAJOR > 5
  const auto memoryType = attribs.type;
#else
  const auto memoryType = attribs.memoryType;
#endif

#if HIP_VERSION_MAJOR > 5
  // ROCm 6+ return hipMemoryTypeUnregistered and hipSuccess
  // for unknown host pointers
  if (attribs.type == hipMemoryTypeUnregistered) {
    return make_error(
          __acpp_here(),
          error_info{"hip_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"HIP", err},
                     error_type::invalid_parameter_error});
  }
#endif

  out.dev = rt::device_id{_backend_descriptor, attribs.device};
  out.is_from_host_backend = false;
  out.is_optimized_host = (memoryType == hipMemoryTypeHost);
#ifndef HIPSYCL_RT_NO_HIP_MANAGED_MEMORY
  out.is_usm = attribs.isManaged;
#else
  // query for hipMemoryTypeUnified as dummy; this is not actually
  // correct as HIP versions that do not support attribs.isManaged
  // have no way of querying whether something was malloc'd with
  // hipMallocManaged().
  out.is_usm = (memoryType == hipMemoryTypeUnified);
#endif
  
  return make_success();
}

result hip_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                int advise) const {
#ifndef HIPSYCL_RT_NO_HIP_MANAGED_MEMORY
  hipError_t err = hipMemAdvise(addr, num_bytes,
                                static_cast<hipMemoryAdvise>(advise), _dev);
  if(err != hipSuccess) {
    return make_error(
      __acpp_here(),
      error_info{"hip_allocator: hipMemAdvise() failed", error_code{"HIP", err}}
    );
  }
#else
  HIPSYCL_DEBUG_WARNING << "hip_allocator: Ignoring mem_advise() hint"
                        << std::endl;
#endif
  return make_success();
}

device_id hip_allocator::get_device() const {
  return device_id{_backend_descriptor, _dev};
}

}
}
