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
#ifndef HIPSYCL_ZE_QUEUE_HPP
#define HIPSYCL_ZE_QUEUE_HPP

#include <future>
#include <mutex>
#include <level_zero/ze_api.h>

#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/reflection_map.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "ze_code_object.hpp"


namespace hipsycl {
namespace rt {

class ze_hardware_manager;
class ze_queue;

class ze_queue : public inorder_queue
{
public:
  ze_queue(ze_hardware_manager* hw_manager, std::size_t device_index);

  virtual ~ze_queue();

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() override;
  virtual std::shared_ptr<dag_node_event> create_queue_completion_event() override;

  virtual result submit_memcpy(memcpy_operation&, const dag_node_ptr&) override;
  virtual result submit_kernel(kernel_operation&, const dag_node_ptr&) override;
  virtual result submit_prefetch(prefetch_operation &, const dag_node_ptr&) override;
  virtual result submit_memset(memset_operation&, const dag_node_ptr&) override;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(const dag_node_ptr& evt) override;
  virtual result submit_external_wait_for(const dag_node_ptr& node) override;

  virtual result wait() override;

  virtual device_id get_device() const override;
  /// Return native type if supported, nullptr otherwise
  virtual void* get_native_type() const override;

  virtual result query_status(inorder_queue_status& status) override;

  ze_command_list_handle_t get_ze_command_list() const {
    return _command_list;
  }

  ze_hardware_manager* get_hardware_manager() const {
    return _hw_manager;
  }

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      std::string_view kernel_name, const rt::hcf_kernel_info *kernel_info,
      const rt::range<3> &num_groups, const rt::range<3> &group_size,
      unsigned local_mem_size, void **args, std::size_t *arg_sizes,
      std::size_t num_args, const kernel_configuration &config);

private:
  const std::vector<std::shared_ptr<dag_node_event>>&
  get_enqueued_synchronization_ops() const;
  
  std::vector<ze_event_handle_t>
  get_enqueued_event_handles() const;

  void register_submitted_op(std::shared_ptr<dag_node_event> evt);

  std::shared_ptr<dag_node_event> create_event();

  ze_command_list_handle_t _command_list;
  ze_hardware_manager* _hw_manager;
  const std::size_t _device_index;

  ze_sscp_code_object_invoker _sscp_code_object_invoker;

  std::shared_ptr<dag_node_event> _last_submitted_op_event;
  std::vector<std::shared_ptr<dag_node_event>> _enqueued_synchronization_ops;

  std::vector<std::future<void>> _external_waits;

  std::shared_ptr<kernel_cache> _kernel_cache;
  
  // Most L0 API functions that add to a command list are not thread-safe.
  // Since most of the public API functions of this class do exactly that,
  // arguably the best strategy to achieve thread-safety is to just have a mutex
  // and lock in every public function.
  std::mutex _mutex;

  // SSCP submission data
  glue::jit::cxx_argument_mapper _arg_mapper;
  kernel_configuration _config;  
  glue::jit::reflection_map _reflection_map;
};

}
}

#endif
