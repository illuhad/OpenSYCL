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

#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/allocation_tracker.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/runtime_event_handlers.hpp"

namespace hipsycl {
namespace rt {

void *allocate_device(backend_allocator *alloc, size_t min_alignment,
                      size_t size_bytes) {
  auto *ptr = alloc->raw_allocate(min_alignment, size_bytes);
  if(ptr) {
    application::event_handler_layer().on_new_allocation(
        ptr, size_bytes,
        allocation_info{alloc->get_device(),
                        allocation_info::allocation_type::device});
  }
  return ptr;
}

void *allocate_host(backend_allocator *alloc, size_t min_alignment,
                              size_t bytes) {
  auto* ptr = alloc->raw_allocate_optimized_host(min_alignment, bytes);
  if(ptr) {
    application::event_handler_layer().on_new_allocation(
        ptr, bytes,
        allocation_info{alloc->get_device(),
                        allocation_info::allocation_type::host});
  }
  return ptr;
}

void *allocate_shared(backend_allocator *alloc, size_t bytes) {
  auto* ptr = alloc->raw_allocate_usm(bytes);
  if(ptr) {
    application::event_handler_layer().on_new_allocation(
        ptr, bytes,
        allocation_info{alloc->get_device(),
                        allocation_info::allocation_type::host});
  }
  return ptr;
}

void deallocate(backend_allocator* alloc, void *mem) {
  alloc->raw_free(mem);
  application::event_handler_layer().on_deallocation(mem);
}

}
}
