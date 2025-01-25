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
#include "hipSYCL/runtime/ur/ur_event.hpp"

#include "ur_utils.hpp"

using namespace hipsycl::rt;

ur_node_event::ur_node_event(const ur_event_handle_t evt) : _evt(evt) {}
ur_node_event::~ur_node_event() = default;

ur_event_handle_t ur_node_event::get_event_handle() const { return _evt; }
ur_event_handle_t ur_node_event::request_backend_event() {
  return get_event_handle();
}

bool ur_node_event::is_complete() const {
  ur_event_status_t status;
  const ur_result_t err =
      urEventGetInfo(_evt, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                     sizeof(status), &status, nullptr);
  if (err != UR_RESULT_SUCCESS) {
    register_error(make_error(
        __acpp_here(),
        ur_error_info("ur_node_event: urEventGetInfo() failed", err)));

    return false;
  }

  return status == UR_EVENT_STATUS_COMPLETE || status == UR_EVENT_STATUS_ERROR;
}

void ur_node_event::wait() {
  const auto events = std::vector{_evt};
  const auto err = urEventWait(events.size(), events.data());
  if (err != UR_RESULT_SUCCESS) {
    register_error(
        make_error(__acpp_here(),
                   ur_error_info("ur_node_event: urEventWait() failed", err)));
  }
}