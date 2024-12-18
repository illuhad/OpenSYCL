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
#include <cstdlib>

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/omp/omp_allocator.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

omp_allocator::omp_allocator(const device_id &my_device)
    : _my_device{my_device} {}

void *omp_allocator::raw_allocate(size_t min_alignment, size_t size_bytes) {
  if(min_alignment < 32) {
    // Enforce alignment by default for performance reasons.
    // 32 is chosen since this is what is currently needed by the adaptivity
    // engine to consider an allocation strongly aligned.
    return raw_allocate(32, size_bytes);
  }

#if !defined(_WIN32)
  // posix requires alignment to be a multiple of sizeof(void*)
  if (min_alignment < sizeof(void*))
    return std::malloc(size_bytes);
#else
  /* The std::free function of the Microsoft C Runtime Library cannot handle
     aligned memory, therefore omp_allocator::free always calls _aligned_free.
     This, however, can only free memory allocated with _aligned_malloc, but
     _aligned_malloc returns NULL when min_alignment == 0.  */
  if (min_alignment == 0)
    min_alignment = 1;
#endif

  if(min_alignment > 0 && size_bytes % min_alignment != 0)
    return raw_allocate(min_alignment,
                        next_multiple_of(size_bytes, min_alignment));

    // ToDo: Mac OS CI has a problem with std::aligned_alloc
    // but it's unclear if it's a Mac, or libc++, or toolchain issue
#ifdef __APPLE__
  return aligned_alloc(min_alignment, size_bytes);
#elif !defined(_WIN32)
  return std::aligned_alloc(min_alignment, size_bytes);
#else
  min_alignment = power_of_2_ceil(min_alignment);
  return _aligned_malloc(size_bytes, min_alignment);
#endif
}

void *omp_allocator::raw_allocate_optimized_host(size_t min_alignment,
                                             size_t bytes) {
  return this->raw_allocate(min_alignment, bytes);
};

void omp_allocator::raw_free(void *mem) {
#if !defined(_WIN32)
  std::free(mem);
#else
  _aligned_free(mem);
#endif
}

void* omp_allocator::raw_allocate_usm(size_t bytes) {
  return this->raw_allocate(0, bytes);
}

bool omp_allocator::is_usm_accessible_from(backend_descriptor b) const {
  if(b.hw_platform == hardware_platform::cpu) {
    return true;
  }
  return false;
}

device_id omp_allocator::get_device() const {
  return _my_device;
}

result omp_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  
  // For a host device, USM is the same as host memory?
  out.is_optimized_host = true;
  out.is_usm = true;
  out.is_from_host_backend = true;
  out.dev = _my_device;

  return make_success();
}

result omp_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                 int advise) const {
  HIPSYCL_DEBUG_WARNING << "omp_allocator: Ignoring mem_advise() hint"
                        << std::endl;
  return make_success();
}

}
}
