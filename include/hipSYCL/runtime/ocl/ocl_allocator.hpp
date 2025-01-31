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
#ifndef HIPSYCL_OCL_ALLOCATOR_HPP
#define HIPSYCL_OCL_ALLOCATOR_HPP

#include <CL/opencl.hpp>

#include "../allocator.hpp"
#include "../hints.hpp"
#include "ocl_usm.hpp"

namespace hipsycl {
namespace rt {

class ocl_allocator : public backend_allocator 
{
public:
  ocl_allocator() = default;
  ocl_allocator(rt::device_id dev, ocl_usm* usm_provier);

  virtual void* raw_allocate(size_t min_alignment, size_t size_bytes,
                             const allocation_hints &hints = {}) override;

  virtual void *
  raw_allocate_optimized_host(size_t min_alignment, size_t bytes,
                              const allocation_hints &hints = {}) override;
  
  virtual void raw_free(void *mem) override;

  virtual void *raw_allocate_usm(size_t bytes,
                                 const allocation_hints &hints = {}) override;
  virtual bool is_usm_accessible_from(backend_descriptor b) const override;

  virtual result query_pointer(const void* ptr, pointer_info& out) const override;

  virtual result mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const override;

  virtual device_id get_device() const override;
private:
  ocl_usm* _usm;
  rt::device_id _dev;
};

}
}

#endif
