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
#include "hipSYCL/runtime/omp/omp_queue.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/generic/async_worker.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/instrumentation.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/omp/omp_event.hpp"
#include "hipSYCL/runtime/omp/omp_backend.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"
#include "hipSYCL/runtime/signal_channel.hpp"
#include "hipSYCL/runtime/util.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER
#include "hipSYCL/compiler/llvm-to-backend/host/LLVMToHostFactory.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/runtime/omp/omp_code_object.hpp"

#ifndef WIN32
#include <unistd.h>
#else
#include <Windows.h>
#endif
#endif

#include <omp.h>

#include <memory>

namespace hipsycl {
namespace rt {

namespace {

bool is_contigous(id<3> offset, range<3> r, range<3> allocation_shape) {
  if (r.size() == 0)
    return true;

  int dim = 3;
  if (r.get(0) == 1)
    dim = 2;
  if (r.get(1) == 1)
    dim = 1;

  // 1D data transfers are always contiguous
  if (dim == 1)
    return true;

  // The slowest index does not need to be of 0 offset and
  // full size => start at
  // * 2 for dim == 2 (slowest index is 1)
  // * 1 for dim == 3 (slowest index is 0)
  for (int i = 4 - dim; i <= 2; ++i) {
    if (offset.get(i) != 0)
      return false;
    if (r.get(i) != allocation_shape.get(i))
      return false;
  }

  return true;
}

class instrumentation_task_guard;

template <class BaseInstrumentation>
class omp_task_timestamp : public BaseInstrumentation {
public:
  friend class instrumentation_task_guard;

  virtual profiler_clock::time_point get_time_point() const override {
    return _time;
  }

  virtual void wait() const override { _signal.wait(); }

private:
  // This should only be called once by the instrumentation_task_guard
  void record_time() {
    assert(!_signal.has_signalled());
    _time = profiler_clock::now();
    _signal.signal();
  }

  profiler_clock::time_point _time;
  mutable signal_channel _signal;
};

using omp_submission_timestamp = simple_submission_timestamp;

using omp_execution_start_timestamp =
    omp_task_timestamp<instrumentations::execution_start_timestamp>;

using omp_execution_finish_timestamp =
    omp_task_timestamp<instrumentations::execution_finish_timestamp>;

class instrumentation_task_guard {
public:
  instrumentation_task_guard(
      std::shared_ptr<omp_execution_start_timestamp> start,
      std::shared_ptr<omp_execution_finish_timestamp> finish)
      : _finish{finish} {
    if (start)
      start->record_time();
  }

  ~instrumentation_task_guard() {
    if (_finish)
      _finish->record_time();
  }

private:
  std::shared_ptr<omp_execution_finish_timestamp> _finish;
};

class omp_instrumentation_setup {
public:
  omp_instrumentation_setup(operation &op, dag_node_ptr node) {
    if (!node)
      return;

    if (node->get_execution_hints()
            .has_hint<
                rt::hints::request_instrumentation_submission_timestamp>()) {

      op.get_instrumentations()
          .add_instrumentation<instrumentations::submission_timestamp>(
              std::make_shared<omp_submission_timestamp>(
                  profiler_clock::now()));
    }
    if (node->get_execution_hints()
            .has_hint<rt::hints::request_instrumentation_start_timestamp>()) {

      _start = std::make_shared<omp_execution_start_timestamp>();

      op.get_instrumentations()
          .add_instrumentation<instrumentations::execution_start_timestamp>(
              _start);
    }
    if (node->get_execution_hints()
            .has_hint<rt::hints::request_instrumentation_finish_timestamp>()) {

      _finish = std::make_shared<omp_execution_finish_timestamp>();

      op.get_instrumentations()
          .add_instrumentation<instrumentations::execution_finish_timestamp>(
              _finish);
    }
  }

  instrumentation_task_guard instrument_task() const {
    return instrumentation_task_guard{_start, _finish};
  }

private:
  std::shared_ptr<omp_execution_start_timestamp> _start;
  std::shared_ptr<omp_execution_finish_timestamp> _finish;
};

#ifdef HIPSYCL_WITH_SSCP_COMPILER

std::size_t get_page_size() {
#ifndef WIN32
  return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#else
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
#endif
}

void *resize_and_align(std::vector<char> &data, std::size_t size,
                       std::size_t alignment) {
  data.resize(size + alignment);
  return reinterpret_cast<void*>(
        next_multiple_of(reinterpret_cast<std::uint64_t>(data.data()),
                         alignment));
}

void *resize_and_strongly_align(std::vector<char> &data, std::size_t size) {
  // compiler/libkernel builtins assume alignment of at least
  // 512 byte boundaries
  std::size_t alignment = std::max(std::size_t{512}, get_page_size());
  return resize_and_align(data, size, alignment);
}

result
launch_kernel_from_so(omp_sscp_executable_object::omp_sscp_kernel *kernel,
                      const rt::range<3> &num_groups,
                      const rt::range<3> &local_size, unsigned shared_memory,
                      void **kernel_args) {
  if (num_groups.size() == 1 && shared_memory == 0) {
    // still need to be able to support group algorithms
    // make thread-local in case we have multiple threads submitting.
    static thread_local std::vector<char> internal_local_memory;
    auto aligned_internal_local_memory = resize_and_strongly_align(
        internal_local_memory, local_size.size() * sizeof(uint64_t));

    omp_sscp_executable_object::work_group_info info{
        num_groups, rt::id<3>{0, 0, 0}, local_size, nullptr,
        aligned_internal_local_memory};
    kernel(&info, kernel_args);
    return make_success();
  }

#ifndef _OPENMP
  HIPSYCL_DEBUG_WARNING << "omp_queue: SSCP kernel launching was built without OpenMP "
                          "support, the kernel will execute sequentially!"
                        << std::endl;
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // get page aligned local memory from heap
    static thread_local std::vector<char> local_memory;
    static thread_local std::vector<char> internal_local_memory;
    auto aligned_local_memory =
        resize_and_strongly_align(local_memory, shared_memory);
    auto aligned_internal_local_memory = resize_and_strongly_align(
        internal_local_memory, local_size.size() * sizeof(uint64_t));
#ifdef _OPENMP
#pragma omp for collapse(3)
#endif
    for (std::size_t k = 0; k < num_groups.get(2); ++k) {
      for (std::size_t j = 0; j < num_groups.get(1); ++j) {
        for (std::size_t i = 0; i < num_groups.get(0); ++i) {
          omp_sscp_executable_object::work_group_info info{
              num_groups, rt::id<3>{i, j, k}, local_size, aligned_local_memory,
              aligned_internal_local_memory};
          kernel(&info, kernel_args);
        }
      }
    }
  }
  return make_success();
}
#endif
} // namespace

omp_queue::omp_queue(omp_backend* be, int dev)
    : _backend_id{be->get_unique_backend_id()}, _sscp_code_object_invoker{this},
      _kernel_cache{kernel_cache::get()} {
  _reflection_map = glue::jit::construct_default_reflection_map(
      be->get_hardware_manager()->get_device(dev));
}

omp_queue::~omp_queue() { _worker.halt(); }

std::shared_ptr<dag_node_event> omp_queue::insert_event() {
  HIPSYCL_DEBUG_INFO << "omp_queue: Inserting event into queue..." << std::endl;

  auto evt = std::make_shared<omp_node_event>();
  auto signal_channel = evt->get_signal_channel();

  _worker([signal_channel] { signal_channel->signal(); });

  return evt;
}

std::shared_ptr<dag_node_event> omp_queue::create_queue_completion_event() {
  return std::make_shared<
      queue_completion_event<std::shared_ptr<signal_channel>, omp_node_event>>(
      this);
}

result omp_queue::submit_memcpy(memcpy_operation &op, const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting memcpy operation..."
                     << std::endl;

  if (!op.source().get_device().is_host() || !op.dest().get_device().is_host()) {
    return register_error(
        __acpp_here(),
        error_info{"omp_queue: OpenMP CPU backend cannot transfer data between "
                   "host and accelerators.",
                   error_type::feature_not_supported});
  }

  void *base_src = op.source().get_base_ptr();
  void *base_dest = op.dest().get_base_ptr();

  assert(base_src);
  assert(base_dest);

  range<3> transferred_range = op.get_num_transferred_elements();
  range<3> src_allocation_shape = op.source().get_allocation_shape();
  range<3> dest_allocation_shape = op.dest().get_allocation_shape();
  id<3> src_offset = op.source().get_access_offset();
  id<3> dest_offset = op.dest().get_access_offset();
  std::size_t src_element_size = op.source().get_element_size();
  std::size_t dest_element_size = op.dest().get_element_size();

  std::size_t total_num_bytes = op.get_num_transferred_bytes();

  bool is_src_contiguous =
      is_contigous(src_offset, transferred_range, src_allocation_shape);
  bool is_dest_contiguous =
      is_contigous(dest_offset, transferred_range, dest_allocation_shape);

  omp_instrumentation_setup instrumentation_setup{op, node};

  _worker([=]() {
    auto instrumentation_guard = instrumentation_setup.instrument_task();

    auto linear_index = [](id<3> id, range<3> allocation_shape) {
      return id[2] + allocation_shape[2] * id[1] +
             allocation_shape[2] * allocation_shape[1] * id[0];
    };

    if (is_src_contiguous && is_dest_contiguous) {
      char *current_src = reinterpret_cast<char *>(base_src);
      char *current_dest = reinterpret_cast<char *>(base_dest);

      current_src +=
          linear_index(src_offset, src_allocation_shape) * src_element_size;
      current_dest +=
          linear_index(dest_offset, dest_allocation_shape) * dest_element_size;

      memcpy(current_dest, current_src, total_num_bytes);
    } else {
      id<3> current_src_offset = src_offset;
      id<3> current_dest_offset = dest_offset;
      std::size_t row_size = transferred_range[2] * src_element_size;

      for (std::size_t surface = 0; surface < transferred_range[0]; ++surface) {
        for (std::size_t row = 0; row < transferred_range[1]; ++row) {

          char *current_src = reinterpret_cast<char *>(base_src);
          char *current_dest = reinterpret_cast<char *>(base_dest);

          current_src +=
              linear_index(current_src_offset, src_allocation_shape) *
              src_element_size;

          current_dest +=
              linear_index(current_dest_offset, dest_allocation_shape) *
              dest_element_size;

          assert(current_src + row_size <=
                 reinterpret_cast<char *>(base_src) +
                     src_allocation_shape.size() * src_element_size);
          assert(current_dest + row_size <=
                 reinterpret_cast<char *>(base_dest) +
                     dest_allocation_shape.size() * dest_element_size);

          memcpy(current_dest, current_src, row_size);

          ++current_src_offset[1];
          ++current_dest_offset[1];
        }
        current_src_offset[1] = src_offset[1];
        current_dest_offset[1] = dest_offset[1];

        ++current_dest_offset[0];
        ++current_src_offset[0];
      }
    }
  });

  return make_success();
}

result omp_queue::submit_kernel(kernel_operation &op, const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting kernel..." << std::endl;

  rt::backend_kernel_launch_capabilities cap;
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);

  const kernel_configuration *config =
      &(op.get_launcher().get_kernel_configuration());

  auto backend_id = _backend_id;
  void* params = this;
  rt::dag_node* node_ptr = node.get();

  omp_instrumentation_setup instrumentation_setup{op, node};
  _worker([=, &op]() {
    auto instrumentation_guard = instrumentation_setup.instrument_task();

    auto err = op.get_launcher().invoke(backend_id, params, cap, node_ptr);
    if(!err.is_success())
      rt::register_error(err);
  });

  return make_success();
}

result omp_queue::submit_sscp_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    std::string_view kernel_name, const rt::hcf_kernel_info *kernel_info,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const kernel_configuration &initial_config) {
#ifdef HIPSYCL_WITH_SSCP_COMPILER
  common::spin_lock_guard lock{_sscp_submission_spin_lock};

  if (!kernel_info) {
    return make_error(
        __acpp_here(),
        error_info{"omp_queue: Could not obtain hcf kernel info for kernel " +
                   std::string{kernel_name}});
  }


  _arg_mapper.construct_mapping(*kernel_info, args, arg_sizes, num_args);

  if (!_arg_mapper.mapping_available()) {
    return make_error(
        __acpp_here(),
        error_info{
            "omp_queue: Could not map C++ arguments to kernel arguments"});
  }

  kernel_adaptivity_engine adaptivity_engine{
      hcf_object, kernel_name, kernel_info, _arg_mapper, num_groups,
      group_size, args,        arg_sizes,   num_args, local_mem_size};

  _config = initial_config;

  _config.append_base_configuration(
      kernel_base_config_parameter::backend_id, backend_id::omp);
  _config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow,
      compilation_flow::sscp);
  _config.append_base_configuration(
      kernel_base_config_parameter::hcf_object_id, hcf_object);

  auto binary_configuration_id =
      adaptivity_engine.finalize_binary_configuration(_config);
  auto code_object_configuration_id = binary_configuration_id;

  auto get_image_and_kernel_names =
      [&](std::vector<std::string> &contained_kernels) -> std::string {
    return adaptivity_engine.select_image_and_kernels(&contained_kernels);
  };

  auto jit_compiler = [&](std::string &compiled_image) -> bool {
    const common::hcf_container *hcf = rt::hcf_cache::get().get_hcf(hcf_object);

    std::vector<std::string> kernel_names;
    std::string selected_image_name = get_image_and_kernel_names(kernel_names);

    // Construct Host translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator =
        compiler::createLLVMToHostTranslator(kernel_names);

    // Lower kernels to binary
    auto err = glue::jit::compile(translator.get(), hcf, selected_image_name,
                                  _config, _reflection_map, compiled_image);

    if (!err.is_success()) {
      register_error(err);
      return false;
    }
    return true;
  };

  auto code_object_constructor =
      [&](const std::string &binary_image) -> code_object * {
    std::vector<std::string> kernel_names;
    get_image_and_kernel_names(kernel_names);

    omp_sscp_executable_object *exec_obj = new omp_sscp_executable_object{
        binary_image, hcf_object, kernel_names, _config};
    result r = exec_obj->get_build_result();

    if (!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    HIPSYCL_DEBUG_INFO
        << "omp_queue: Successfully compiled SSCP kernels to module "
        << exec_obj->get_module() << std::endl;

    return exec_obj;
  };

  const code_object *obj = _kernel_cache->get_or_construct_jit_code_object(
      code_object_configuration_id, binary_configuration_id, jit_compiler,
      code_object_constructor);

  if (!obj) {
    return make_error(__acpp_here(),
                      error_info{"omp_queue: Code object construction failed"});
  }

  auto kernel =
      static_cast<const omp_sscp_executable_object *>(obj)->get_kernel(
          kernel_name);

  return launch_kernel_from_so(kernel, num_groups, group_size, local_mem_size,
                               _arg_mapper.get_mapped_args());

#else
  return make_error(
      __acpp_here(),
      error_info{"omp_queue: SSCP kernel launch was requested, but hipSYCL was "
                 "not built with CPU SSCP support."});
#endif
}

result omp_queue::submit_prefetch(prefetch_operation &op, const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO
      << "omp_queue: Received prefetch submission request, ignoring"
      << std::endl;
  // Yeah, what are you going to do? Prefetching CPU memory on CPU? Go home!
  // (TODO: maybe we should handle the case that we have USM memory from another
  // backend here)

  omp_instrumentation_setup instrumentation_setup{op, node};
  {
    auto instrumentation_guard = instrumentation_setup.instrument_task();
    // empty instrumentation region because of no-op
  }
  return make_success();
}

result omp_queue::submit_memset(memset_operation &op, const dag_node_ptr& node) {
  void *ptr = op.get_pointer();
  std::size_t bytes = op.get_num_bytes();
  int pattern = op.get_pattern();

  if (!ptr) {
    return register_error(
        __acpp_here(),
        error_info{
            "omp_queue: submit_memset(): Invalid argument, pointer is null."});
  }

  omp_instrumentation_setup instrumentation_setup{op, node};
  _worker([=]() {
    auto instrumentation_guard = instrumentation_setup.instrument_task();

    memset(ptr, pattern, bytes);
  });

  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result omp_queue::submit_queue_wait_for(const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting wait for other queue..."
                     << std::endl;
  auto evt = node->get_event();
  if (!evt) {
    return register_error(
        __acpp_here(),
        error_info{"omp_queue: event for synchronization is null.",
                   error_type::invalid_parameter_error});
  }

  _worker([=]() { evt->wait(); });

  return make_success();
}

result omp_queue::wait() {
  _worker.wait();
  return make_success();
}

result omp_queue::query_status(inorder_queue_status &status) {
  status = inorder_queue_status{_worker.queue_size() == 0};
  return make_success();
}

result omp_queue::submit_external_wait_for(const dag_node_ptr& node) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting wait for external node..."
                     << std::endl;

  if (!node) {
    return register_error(
        __acpp_here(),
        error_info{"omp_queue: node for synchronization is null.",
                   error_type::invalid_parameter_error});
  }

  _worker([=]() { node->wait(); });

  return make_success();
}

worker_thread &omp_queue::get_worker() { return _worker; }

device_id omp_queue::get_device() const {
  return device_id{
      backend_descriptor{hardware_platform::cpu, api_platform::omp}, 0};
}

void *omp_queue::get_native_type() const { return nullptr; }

result omp_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, hcf_object_id hcf_object,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, std::string_view kernel_name,
    const rt::hcf_kernel_info *kernel_info,
    const kernel_configuration &config) {

  return _queue->submit_sscp_kernel_from_code_object(
      op, hcf_object, kernel_name, kernel_info, num_groups, group_size,
      local_mem_size, args, arg_sizes, num_args, config);
}

rt::range<3> omp_sscp_code_object_invoker::select_group_size(
    const rt::range<3> &global_range, const rt::range<3> &group_size) const {
  rt::range<3> selected_group_size = group_size;
#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif
  constexpr auto divisor = 1;
  auto z = std::min(
      std::max<std::size_t>(global_range.get(0) / (max_threads * divisor), 16),
      std::min<std::size_t>(global_range.get(0), 1024));
  selected_group_size = rt::range<3>{z, 1, 1};
  return selected_group_size;
}

} // namespace rt
} // namespace hipsycl
