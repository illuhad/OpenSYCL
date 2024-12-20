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
#ifndef HIPSYCL_OMP_QUEUE_HPP
#define HIPSYCL_OMP_QUEUE_HPP

#include "../generic/async_worker.hpp"
#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "../device_id.hpp"
#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/reflection_map.hpp"

namespace hipsycl {
namespace rt {

class omp_queue;
class omp_backend;

class omp_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  omp_sscp_code_object_invoker(omp_queue* q)
  : _queue{q} {}

  virtual ~omp_sscp_code_object_invoker(){}

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               std::string_view kernel_name,
                               const rt::hcf_kernel_info* kernel_info,
                               const kernel_configuration& config) override;
  
  virtual rt::range<3> select_group_size(const rt::range<3> &num_groups,
                                         const rt::range<3> &group_size) const override;

private:
  omp_queue* _queue;
};

class omp_queue : public inorder_queue
{
public:
  omp_queue(omp_backend* be, int dev);
  virtual ~omp_queue();

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
  virtual void *get_native_type() const override;

  virtual result query_status(inorder_queue_status& status) override;

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string_view kernel_name,
      const rt::hcf_kernel_info *kernel_info, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const kernel_configuration &config);

  worker_thread& get_worker();
private:
  const backend_id _backend_id;
  worker_thread _worker;

  omp_sscp_code_object_invoker _sscp_code_object_invoker;
  std::shared_ptr<kernel_cache> _kernel_cache;

  // SSCP submission data
  common::spin_lock _sscp_submission_spin_lock;
  glue::jit::cxx_argument_mapper _arg_mapper;
  kernel_configuration _config;
  glue::jit::reflection_map _reflection_map;
};

}
}

#endif
