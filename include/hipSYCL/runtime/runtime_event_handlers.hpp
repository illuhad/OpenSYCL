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
#ifndef ACPP_RT_EVENT_HANDLERS_HPP
#define ACPP_RT_EVENT_HANDLERS_HPP

#include <memory>

#include "backend.hpp"
#include "device_id.hpp"
#include "settings.hpp"

namespace hipsycl {
namespace rt {

struct allocation_info {
  enum class allocation_type {
    device,
    shared,
    host
  };

  rt::device_id dev;
  allocation_type alloc_type;
};

class runtime_event_handlers {
public:
  runtime_event_handlers();
  void on_new_allocation(const void*, std::size_t, const allocation_info& info);
  void on_deallocation(const void* ptr);
private:
  bool _needs_allocation_tracking;
};


}
}


#endif
