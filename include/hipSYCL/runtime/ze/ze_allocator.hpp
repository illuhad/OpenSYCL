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
#ifndef HIPSYCL_ZE_ALLOCATOR_HPP
#define HIPSYCL_ZE_ALLOCATOR_HPP

#include "../allocator.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "ze_hardware_manager.hpp"

#include <level_zero/ze_api.h>

namespace hipsycl {
namespace rt {

class ze_allocator : public backend_allocator 
{
public:
  ze_allocator(std::size_t device_index, const ze_hardware_context *dev,
               const ze_hardware_manager *hw_manager);

  virtual void* raw_allocate(size_t min_alignment, size_t size_bytes) override;

  virtual void *raw_allocate_optimized_host(size_t min_alignment,
                                            size_t bytes) override;
  
  virtual void raw_free(void *mem) override;

  virtual void *raw_allocate_usm(size_t bytes) override;
  virtual bool is_usm_accessible_from(backend_descriptor b) const override;

  virtual result query_pointer(const void* ptr, pointer_info& out) const override;

  virtual result mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const override;

  virtual device_id get_device() const override;
private:
  ze_context_handle_t _ctx;
  ze_device_handle_t _dev;
  uint32_t _global_mem_ordinal;
  device_id _dev_id;

  const ze_hardware_manager* _hw_manager;
};

}
}

#endif