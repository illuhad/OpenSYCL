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
#ifndef HIPSYCL_UR_EVENT_HPP
#define HIPSYCL_UR_EVENT_HPP

#include "hipSYCL/runtime/inorder_queue_event.hpp"

#include <ur_api.h>

namespace hipsycl::rt {

class ur_node_event final : public inorder_queue_event<ur_event_handle_t> {
public:
  /// Takes ownership of supplied ze_event_handle_t
  explicit ur_node_event(ur_event_handle_t evt);
  ~ur_node_event() override;

  void wait() override;

  [[nodiscard]] bool is_complete() const override;
  [[nodiscard]] ur_event_handle_t request_backend_event() override;

  [[nodiscard]] ur_event_handle_t get_event_handle() const;

private:
  ur_event_handle_t _evt;
};

} // namespace hipsycl::rt

#endif
