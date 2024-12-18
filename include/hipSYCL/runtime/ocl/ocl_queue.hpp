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
#ifndef HIPSYCL_OCL_QUEUE_HPP
#define HIPSYCL_OCL_QUEUE_HPP

#include <CL/opencl.hpp>
#include <mutex>

#include "../executor.hpp"
#include "../inorder_queue.hpp"

#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/reflection_map.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/generic/async_worker.hpp"
#include "hipSYCL/runtime/ocl/ocl_code_object.hpp"

namespace hipsycl {
namespace rt {

class ocl_hardware_manager;

class ocl_queue : public inorder_queue
{
public:
  ocl_queue(ocl_hardware_manager* hw_manager, std::size_t device_index);
  
  ocl_queue(const ocl_queue&) = delete;
  ocl_queue& operator=(const ocl_queue&) = delete;

  virtual ~ocl_queue();

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

  ocl_hardware_manager* get_hardware_manager() const;

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string_view kernel_name,
      const rt::hcf_kernel_info *kernel_info, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const kernel_configuration &config);

private:
  void register_submitted_op(cl::Event);

  // These member variables have to be thread-safe.
  ocl_hardware_manager* _hw_manager;
  const std::size_t _device_index;

  cl::CommandQueue _queue;
  ocl_sscp_code_object_invoker _sscp_invoker;
  worker_thread _host_worker;

  std::shared_ptr<kernel_cache> _kernel_cache;

  // Non-thread safe state should go here
  struct protected_state {
  public:
    auto get_most_recent_event() const {
      std::lock_guard<std::mutex> lock {_mutex};
      return _most_recent_event;
    }

    template<class T>
    void set_most_recent_event(const T& x) {
      std::lock_guard<std::mutex> lock {_mutex};
      _most_recent_event = x;
    }
  private:
    std::shared_ptr<dag_node_event> _most_recent_event = nullptr;
    mutable std::mutex _mutex;
  };

  protected_state _state;

  // SSCP submission data
  common::spin_lock _sscp_submission_spin_lock;
  glue::jit::cxx_argument_mapper _arg_mapper;
  kernel_configuration _config;
  glue::jit::reflection_map _reflection_map;
};

}
}

#endif
