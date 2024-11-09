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
#ifndef ACPP_ALLOCATION_TRACKER_HPP
#define ACPP_ALLOCATION_TRACKER_HPP

#include <cstdint>
#include "runtime_event_handlers.hpp"
#include "hipSYCL/common/allocation_map.hpp"

namespace hipsycl {
namespace rt {

class allocation_tracker {
public:
  static bool query_allocation(const void *ptr, allocation_info &out,
                               uint64_t &root_address);
  static bool register_allocation(const void *ptr, std::size_t size,
                                  const allocation_info &info);
  static bool unregister_allocation(const void* ptr);
};

}
}

#endif