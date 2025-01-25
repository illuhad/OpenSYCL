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
#ifndef HIPSYCL_UR_ALLOCATOR_HPP
#define HIPSYCL_UR_ALLOCATOR_HPP

#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/device_id.hpp"

#include <ur_api.h>

namespace hipsycl::rt {

class ur_hardware_context;
class ur_hardware_manager;

class ur_allocator final : public backend_allocator {
public:
  ur_allocator(ur_device_handle_t dev, ur_context_handle_t ctx,
               std::size_t device_index);

  [[nodiscard]] void *raw_allocate(size_t min_alignment, size_t bytes) override;
  [[nodiscard]] void *raw_allocate_optimized_host(size_t min_alignment,
                                                  size_t bytes) override;
  [[nodiscard]] void *raw_allocate_usm(size_t bytes) override;
  void raw_free(void *mem) override;

  [[nodiscard]] bool
  is_usm_accessible_from(backend_descriptor b) const override;

  result query_pointer(const void *ptr, pointer_info &out) const override;

  result mem_advise(const void *addr, std::size_t num_bytes,
                    int advise) const override;

  [[nodiscard]] device_id get_device() const override;

private:
  ur_device_handle_t _dev;
  ur_context_handle_t _ctx;

  device_id _dev_id;
};

} // namespace hipsycl::rt

#endif