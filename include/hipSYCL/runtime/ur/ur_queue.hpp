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
#ifndef HIPSYCL_UR_QUEUE_HPP
#define HIPSYCL_UR_QUEUE_HPP

#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/reflection_map.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/ur/ur_code_object.hpp"

#include <future>

namespace hipsycl::rt {

class ur_hardware_manager;
class ur_queue;

class ur_queue final : public inorder_queue {
public:
  ur_queue(ur_hardware_manager *hw_manager, int device_index);
  ~ur_queue() override;

  /// Inserts an event into the stream
  std::shared_ptr<dag_node_event> insert_event() override;
  std::shared_ptr<dag_node_event> create_queue_completion_event() override;

  result submit_kernel(kernel_operation &, const dag_node_ptr &) override;
  result submit_memcpy(memcpy_operation &, const dag_node_ptr &) override;
  result submit_prefetch(prefetch_operation &, const dag_node_ptr &) override;
  result submit_memset(memset_operation &, const dag_node_ptr &) override;

  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  result submit_queue_wait_for(const dag_node_ptr &evt) override;
  result submit_external_wait_for(const dag_node_ptr &node) override;

  result wait() override;

  device_id get_device() const override;
  /// Return native type if supported, nullptr otherwise
  void *get_native_type() const override;

  result query_status(inorder_queue_status &status) override;

  [[nodiscard]] ur_queue_handle_t get_ur_queue() const;
  [[nodiscard]] ur_hardware_manager *get_hardware_manager() const;

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      std::string_view kernel_name, const hcf_kernel_info *kernel_info,
      const range<3> &num_groups, const range<3> &group_size,
      unsigned local_mem_size, void **args, std::size_t *arg_sizes,
      std::size_t num_args, const kernel_configuration &config);

private:
  void register_submitted_op(ur_event_handle_t evt);

  // Non-thread safe state should go here
  struct protected_state {
    auto get_most_recent_event() const {
      std::lock_guard lock{_mutex};
      return _most_recent_event;
    }

    template <class T> void set_most_recent_event(const T &x) {
      std::lock_guard lock{_mutex};
      _most_recent_event = x;
    }

  private:
    std::shared_ptr<dag_node_event> _most_recent_event = nullptr;
    mutable std::mutex _mutex;
  };

  protected_state _state;

  const int _device_index;
  ur_queue_handle_t _queue{};
  ur_hardware_manager *_hw_manager;
  ur_sscp_code_object_invoker _sscp_code_object_invoker;
  std::shared_ptr<dag_node_event> _last_submitted_op_event;
  std::vector<std::shared_ptr<dag_node_event>> _enqueued_synchronization_ops;
  std::vector<std::future<void>> _external_waits;
  std::shared_ptr<kernel_cache> _kernel_cache;

  // SSCP submission data
  glue::jit::cxx_argument_mapper _arg_mapper;
  kernel_configuration _config;
  glue::jit::reflection_map _reflection_map;

  common::spin_lock _sscp_submission_spin_lock;
};

} // namespace hipsycl::rt

#endif
