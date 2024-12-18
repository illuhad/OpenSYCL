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


#include "hipSYCL/runtime/runtime_event_handlers.hpp"
#include "hipSYCL/runtime/allocation_tracker.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/settings.hpp"

namespace hipsycl {
namespace rt {

runtime_event_handlers::runtime_event_handlers() {
  _needs_allocation_tracking = application::get_settings().get<
    setting::enable_allocation_tracking>();
}

void runtime_event_handlers::on_new_allocation(const void *ptr,
                                               std::size_t size,
                                               const allocation_info &info) {
  if (_needs_allocation_tracking) {
    allocation_tracker::register_allocation(ptr, size, info);
  }
}


void runtime_event_handlers::on_deallocation(const void* ptr) {
  if (_needs_allocation_tracking) {
    allocation_tracker::unregister_allocation(ptr);
  }
}

}
}
