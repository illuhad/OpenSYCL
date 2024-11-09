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
#include <cuda_runtime_api.h>

#include "hipSYCL/runtime/cuda/cuda_allocator.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

cuda_allocator::cuda_allocator(backend_descriptor desc, int cuda_device)
    : _backend_descriptor{desc}, _dev{cuda_device}
{}
      
void *cuda_allocator::raw_allocate(size_t min_alignment, size_t size_bytes)
{
  void *ptr;
  cuda_device_manager::get().activate_device(_dev);
  cudaError_t err = cudaMalloc(&ptr, size_bytes);

  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_allocator: cudaMalloc() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

void *cuda_allocator::raw_allocate_optimized_host(size_t min_alignment,
                                                  size_t bytes) {
  void *ptr;
  cuda_device_manager::get().activate_device(_dev);

  cudaError_t err = cudaMallocHost(&ptr, bytes);

  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_allocator: cudaMallocHost() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void cuda_allocator::raw_free(void *mem) {

  pointer_info info;
  result query_result = query_pointer(mem, info);

  if (!query_result.is_success()) {
    register_error(query_result);
    return;
  }
  
  cudaError_t err;
  if (info.is_optimized_host)
    err = cudaFreeHost(mem);
  else
    err = cudaFree(mem);
  
  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_allocator: cudaFree() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
  }
}

void * cuda_allocator::raw_allocate_usm(size_t bytes)
{
  cuda_device_manager::get().activate_device(_dev);
  
  void *ptr;
  auto err = cudaMallocManaged(&ptr, bytes);
  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_allocator: cudaMallocManaged() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

bool cuda_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  // TODO: Formulate this better - this assumes that either CUDA+CPU or
  // ROCm + CPU are active at the same time
  return true;
}

result cuda_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  cudaPointerAttributes attribs;
  auto err = cudaPointerGetAttributes(&attribs, ptr);

  if (err != cudaSuccess) {
    if (err == cudaErrorInvalidValue)
      return make_error(
          __acpp_here(),
          error_info{"cuda_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"CUDA", err},
                     error_type::invalid_parameter_error});
    else
      return make_error(
          __acpp_here(),
          error_info{"cuda_allocator: query_pointer(): query failed",
                     error_code{"CUDA", err}});
  }

  // CUDA 11+ return cudaMemoryTypeUnregistered and cudaSuccess
  // for unknown host pointers
  if (attribs.type == cudaMemoryTypeUnregistered) {
    return make_error(
          __acpp_here(),
          error_info{"cuda_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"CUDA", err},
                     error_type::invalid_parameter_error});
  }

  out.dev = rt::device_id{_backend_descriptor, attribs.device};
  out.is_from_host_backend = false;
  out.is_optimized_host = attribs.type == cudaMemoryTypeHost;
  out.is_usm = attribs.type == cudaMemoryTypeManaged;
  
  return make_success();
}

result cuda_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const {
#ifndef _WIN32
  cudaError_t err = cudaMemAdvise(addr, num_bytes,
                                  static_cast<cudaMemoryAdvise>(advise), _dev);
  if(err != cudaSuccess) {
    return make_error(
      __acpp_here(),
      error_info{"cuda_allocator: cudaMemAdvise() failed", error_code{"CUDA", err}}
    );
  }
#else
  HIPSYCL_DEBUG_WARNING << "cuda_allocator: Ignoring mem_advise() hint"
                        << std::endl;
#endif // _WIN32
  return make_success();
}

device_id cuda_allocator::get_device() const {
  return device_id{_backend_descriptor, _dev};
}

}
}
