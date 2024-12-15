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
#include "driver_types.h"
#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/cuda/cuda_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/cuda/cuda_queue.hpp"
#include "hipSYCL/runtime/cuda/cuda_backend.hpp"
#include "hipSYCL/runtime/cuda/cuda_event.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/cuda/cuda_code_object.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/ptx/LLVMToPtxFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"

#endif

#include <cuda_runtime_api.h>
#include <cuda_runtime.h> //for make_cudaPitchedPtr
#include <cuda.h> // For kernels launched from modules

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {

void host_synchronization_callback(cudaStream_t stream, cudaError_t status,
                                   void *userData) {
  
  assert(userData);
  dag_node_ptr* node = static_cast<dag_node_ptr*>(userData);
  
  if(status != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_queue callback: CUDA returned error code.",
                              error_code{"CUDA", status}});
  }
  else {
    (*node)->wait();
  }
  delete node;
}


class cuda_instrumentation_guard {
public:
  cuda_instrumentation_guard(cuda_queue *q,
                             operation &op, dag_node* node) 
                             : _queue{q}, _operation{&op}, _node{node} {
    assert(q);
    
    if(!_node)
      return;

    if (_node->get_execution_hints()
            .has_hint<
                rt::hints::request_instrumentation_submission_timestamp>()) {

      op.get_instrumentations()
          .add_instrumentation<instrumentations::submission_timestamp>(
            std::make_shared<cuda_submission_timestamp>(profiler_clock::now()));
    }

    if (_node->get_execution_hints().has_hint<
                rt::hints::request_instrumentation_start_timestamp>()) {

      _task_start = _queue->insert_event();

      op.get_instrumentations()
          .add_instrumentation<instrumentations::execution_start_timestamp>(
              std::make_shared<cuda_execution_start_timestamp>(
                  _queue->get_timing_reference(), _task_start));
    }
  }

  ~cuda_instrumentation_guard() {
    if(!_node)
      return;
    
    if (_node->get_execution_hints()
            .has_hint<rt::hints::request_instrumentation_finish_timestamp>()) {
      std::shared_ptr<dag_node_event> task_finish = _queue->insert_event();

      if(_task_start) {
        _operation->get_instrumentations()
            .add_instrumentation<instrumentations::execution_finish_timestamp>(
                std::make_shared<cuda_execution_finish_timestamp>(
                    _queue->get_timing_reference(), _task_start, task_finish));
      } else {
        _operation->get_instrumentations()
            .add_instrumentation<instrumentations::execution_finish_timestamp>(
                std::make_shared<cuda_execution_finish_timestamp>(
                    _queue->get_timing_reference(), task_finish));
      }
    }
  }

private:
  cuda_queue* _queue;
  operation* _operation;
  dag_node* _node;
  std::shared_ptr<dag_node_event> _task_start;
};

result launch_kernel_from_module(CUmodule module,
                                 std::string_view kernel_name,
                                 const rt::range<3> &grid_size,
                                 const rt::range<3> &block_size,
                                 unsigned shared_memory, cudaStream_t stream,
                                 void **kernel_args) {
  CUfunction f;
  CUresult err = cuModuleGetFunction(&f, module, kernel_name.data());

  if (err != CUDA_SUCCESS) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: could not extract kernel from module",
                                 error_code{"CU", static_cast<int>(err)}});
  }

  err = cuLaunchKernel(f, static_cast<unsigned>(grid_size.get(0)),
                       static_cast<unsigned>(grid_size.get(1)),
                       static_cast<unsigned>(grid_size.get(2)),
                       static_cast<unsigned>(block_size.get(0)),
                       static_cast<unsigned>(block_size.get(1)),
                       static_cast<unsigned>(block_size.get(2)),
                       shared_memory, stream, kernel_args, nullptr);

  if (err != CUDA_SUCCESS) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: could not submit kernel from module",
                                 error_code{"CU", static_cast<int>(err)}});
  }
  
  return make_success();
}
}


void cuda_queue::activate_device() const {
  cuda_device_manager::get().activate_device(_dev.get_id());
}

cuda_queue::cuda_queue(cuda_backend *be, device_id dev, int priority)
    : _dev{dev}, _stream{nullptr},
      _multipass_code_object_invoker{this},
      _sscp_code_object_invoker{this}, _backend{be},
      _kernel_cache{kernel_cache::get()} {
  this->activate_device();

  cudaError_t err;
  if(priority == 0) {
    err = cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
  } else {
    // CUDA API will clamp the priority to the range 
    err = cudaStreamCreateWithPriority(&_stream, cudaStreamNonBlocking, priority);
  }
  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_queue: Couldn't construct backend stream",
                              error_code{"CUDA", err}});
    return;
  }

  _reference_event = host_timestamped_event{this};
}

CUstream_st* cuda_queue::get_stream() const { return _stream; }

cuda_queue::~cuda_queue() {
  auto err = cudaStreamDestroy(_stream);
  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_queue: Couldn't destroy stream",
                              error_code{"CUDA", err}});
  }
}

/// Inserts an event into the stream
std::shared_ptr<dag_node_event> cuda_queue::insert_event() {
  cudaEvent_t evt;
  auto event_creation_result =
      _backend->get_event_pool(_dev)->obtain_event(evt);
  if(!event_creation_result.is_success()) {
    register_error(event_creation_result);
    return nullptr;
  }

  cudaError_t err = cudaEventRecord(evt, this->get_stream());
  if (err != cudaSuccess) {
    register_error(
        __acpp_here(),
        error_info{"cuda_queue: Couldn't record event", error_code{"CUDA", err}});
    return nullptr;
  }

  return std::make_shared<cuda_node_event>(_dev, evt,
                                           _backend->get_event_pool(_dev));
}

std::shared_ptr<dag_node_event> cuda_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<cudaEvent_t, cuda_node_event>>(
      this);
}


result cuda_queue::submit_memcpy(memcpy_operation & op, const dag_node_ptr& node) {

  device_id source_dev = op.source().get_device();
  device_id dest_dev = op.dest().get_device();

  assert(op.source().get_access_ptr());
  assert(op.dest().get_access_ptr());

  cudaMemcpyKind copy_kind = cudaMemcpyHostToDevice;

  if (source_dev.get_full_backend_descriptor().sw_platform == api_platform::cuda) {
    if (dest_dev.get_full_backend_descriptor().sw_platform ==
        api_platform::cuda) {
      assert(source_dev.get_full_backend_descriptor().hw_platform ==
                 dest_dev.get_full_backend_descriptor().hw_platform &&
             "Attempted to execute explicit device<->device copy operation "
             "between devices from different CUDA hardware backends");
      copy_kind = cudaMemcpyDeviceToDevice;
    } else if (dest_dev.get_full_backend_descriptor().hw_platform ==
               hardware_platform::cpu) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else
      assert(false && "Unknown copy destination platform");
  } else if (source_dev.get_full_backend_descriptor().hw_platform ==
             hardware_platform::cpu) {
    if (dest_dev.get_full_backend_descriptor().sw_platform ==
        api_platform::cuda) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (dest_dev.get_full_backend_descriptor().hw_platform ==
        hardware_platform::cpu) {
      copy_kind = cudaMemcpyHostToHost;
    } else
      assert(false && "Unknown copy destination platform");
  } else
    assert(false && "Unknown copy source platform");


  range<3> transfer_range = op.get_num_transferred_elements();

  int dimension = 0;
  if (transfer_range[0] > 1)
    dimension = 3;
  else if (transfer_range[1] > 1)
    dimension = 2;
  else
    dimension = 1;

  // If we transfer the entire buffer, treat it as 1D memcpy for performance.
  // TODO: The same optimization could also be applied for the general case
  // when regions are contiguous
  if (op.get_num_transferred_elements() == op.source().get_allocation_shape() &&
      op.get_num_transferred_elements() == op.dest().get_allocation_shape() &&
      op.source().get_access_offset() == id<3>{} &&
      op.dest().get_access_offset() == id<3>{})
    dimension = 1;

  assert(dimension >= 1 && dimension <= 3);


  cuda_instrumentation_guard instrumentation{this, op, node.get()};

  cudaError_t err = cudaSuccess;
  if (dimension == 1) {
    err = cudaMemcpyAsync(
        op.dest().get_access_ptr(), op.source().get_access_ptr(),
        op.get_num_transferred_bytes(), copy_kind, get_stream());
    
  } else if (dimension == 2) {
    err = cudaMemcpy2DAsync(
        op.dest().get_access_ptr(),
        extract_from_range3<2>(op.dest().get_allocation_shape())[1] *
            op.dest().get_element_size(),
        op.source().get_access_ptr(),
        extract_from_range3<2>(op.source().get_allocation_shape())[1] *
            op.source().get_element_size(),
        extract_from_range3<2>(op.get_num_transferred_elements())[1] *
            op.source().get_element_size(),
        extract_from_range3<2>(op.get_num_transferred_elements())[0], copy_kind,
        get_stream());
    
  } else {
    
    cudaMemcpy3DParms params {};
    params.srcPtr = make_cudaPitchedPtr(op.source().get_access_ptr(),
                                        op.source().get_allocation_shape()[2] *
                                            op.source().get_element_size(),
                                        op.source().get_allocation_shape()[2],
                                        op.source().get_allocation_shape()[1]);
    params.dstPtr = make_cudaPitchedPtr(op.dest().get_access_ptr(),
                                        op.dest().get_allocation_shape()[2] *
                                            op.dest().get_element_size(),
                                        op.dest().get_allocation_shape()[2],
                                        op.dest().get_allocation_shape()[1]);
    params.extent = {op.get_num_transferred_elements()[2] *
                        op.source().get_element_size(),
                    op.get_num_transferred_elements()[1],
                    op.get_num_transferred_elements()[0]};
    params.kind = copy_kind;

    err = cudaMemcpy3DAsync(&params, get_stream());
  }

  if (err != cudaSuccess) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Couldn't submit memcpy",
                                  error_code{"CUDA", err}});
  }
  
  return make_success();
}

result cuda_queue::submit_kernel(kernel_operation &op, const dag_node_ptr& node) {

  this->activate_device();

  rt::backend_kernel_launch_capabilities cap;
  cap.provide_multipass_invoker(&_multipass_code_object_invoker);
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);
  
  cuda_instrumentation_guard instrumentation{this, op, node.get()};
  return op.get_launcher().invoke(backend_id::cuda, this, cap, node.get());
}

result cuda_queue::submit_prefetch(prefetch_operation& op, const dag_node_ptr& node) {
  cuda_instrumentation_guard instrumentation{this, op, node.get()};
#ifndef _WIN32
  cudaError_t err = cudaSuccess;
  
  if (op.get_target().is_host()) {
    err = cudaMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                                        cudaCpuDeviceId, get_stream());
  } else {
    err = cudaMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                                        _dev.get_id(), get_stream());
  }


  if (err != cudaSuccess) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: cudaMemPrefetchAsync() failed",
                                 error_code{"CUDA", err}});
  }
#else
  HIPSYCL_DEBUG_WARNING << "cuda_queue: Ignoring prefetch() hint"
                        << std::endl;
#endif // _WIN32
  return make_success();
}

result cuda_queue::submit_memset(memset_operation &op, const dag_node_ptr& node) {

  cuda_instrumentation_guard instrumentation{this, op, node.get()};
  
  cudaError_t err = cudaMemsetAsync(op.get_pointer(), op.get_pattern(),
                                    op.get_num_bytes(), get_stream());
  

  if (err != cudaSuccess) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: cudaMemsetAsync() failed",
                                 error_code{"CUDA", err}});
  }

  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result cuda_queue::submit_queue_wait_for(const dag_node_ptr& node) {
  auto evt = node->get_event();
  assert(dynamic_is<inorder_queue_event<cudaEvent_t>>(evt.get()));

  inorder_queue_event<cudaEvent_t> *cuda_evt =
      cast<inorder_queue_event<cudaEvent_t>>(evt.get());
  
  auto err = cudaStreamWaitEvent(_stream, cuda_evt->request_backend_event(), 0);
  if (err != cudaSuccess) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: cudaStreamWaitEvent() failed",
                                 error_code{"CUDA", err}});
  }

  return make_success();
}

result cuda_queue::submit_external_wait_for(const dag_node_ptr& node) {

  dag_node_ptr* user_data = new dag_node_ptr;
  assert(user_data);
  *user_data = node;

  auto err = 
      cudaStreamAddCallback(_stream, host_synchronization_callback,
                           reinterpret_cast<void *>(user_data), 0);

  if (err != cudaSuccess) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Couldn't submit stream callback",
                                 error_code{"CUDA", err}});
  }
  
  return make_success();
}

result cuda_queue::wait() {

  auto err = cudaStreamSynchronize(_stream);

  if(err != cudaSuccess) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Couldn't synchronize with stream",
                                 error_code{"CUDA", err}});
  }

  return make_success();
}

result cuda_queue::submit_multipass_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    const std::string &backend_kernel_name, const rt::range<3> &grid_size,
    const rt::range<3> &block_size, unsigned dynamic_shared_mem,
    void **kernel_args, std::size_t num_args) {

  this->activate_device();

  // For now we need to extract HCF in order to get a list of available
  // compilation targets (list of embedded device images in HCF).
  // TODO we could cache this vector to avoid retrieving HCF for every kernel launch
  const common::hcf_container *hcf =
        rt::hcf_cache::get().get_hcf(hcf_object);
  if (!hcf)
    return make_error(
        __acpp_here(),
        error_info{"cuda_queue: Could not access requested HCF object"});

  assert(hcf->root_node());
  std::vector<std::string> available_targets = hcf->root_node()->get_subnodes();
  assert(!available_targets.empty());

  // TODO Select best compiled target based on actual device - currently
  // we just use the first device image no matter which device it was
  // compiled for
  std::string selected_target = available_targets[0];

  int device = _dev.get_id();

  kernel_configuration config;
  config.append_base_configuration(
      kernel_base_config_parameter::backend_id, backend_id::cuda);
  config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow,
      compilation_flow::explicit_multipass);
  config.append_base_configuration(
      kernel_base_config_parameter::hcf_object_id, hcf_object);
  config.append_base_configuration(
      kernel_base_config_parameter::target_arch, selected_target);

  auto binary_configuration_id = config.generate_id();
  auto code_object_configuration_id = binary_configuration_id;
  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_device, device);

  // Will be invoked by the kernel cache in case there is a miss in the kernel
  // cache and we have to construct a new code object
  auto code_object_constructor = [&]() -> code_object* {
    const common::hcf_container::node *tn =
        hcf->root_node()->get_subnode(selected_target);
    if(!tn)
      return nullptr;
    if(!tn->has_binary_data_attached())
      return nullptr;
    
    std::string source_code;
    if(!hcf->get_binary_attachment(tn, source_code)) {
      HIPSYCL_DEBUG_ERROR << "cuda_queue: Could not extract PTX code from "
                              "HCF node; invalid HCF data?"
                          << std::endl;
      return nullptr;
    }

    cuda_executable_object *exec_obj = new cuda_multipass_executable_object{
        hcf_object, selected_target, source_code, device};
    result r = exec_obj->get_build_result();

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = _kernel_cache->get_or_construct_code_object(
      code_object_configuration_id, code_object_constructor);

  if(!obj) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Code object construction failed"});
  }

  CUmodule cumodule = static_cast<const cuda_executable_object*>(obj)->get_module();
  assert(cumodule);

  // Need to find out full backend kernel name. This is necessary because
  // we don't know the *exact* kernel name until we know that we are in the clang 13+
  // name mangling path. It can be that we only have a fragment :(
  std::string full_kernel_name;
  for(const auto& name : obj->supported_backend_kernel_names()) {
    if(name.find(backend_kernel_name) != std::string::npos) {
      full_kernel_name = name;
      break;
    }
  }
  if(full_kernel_name.empty())
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Could not discover full kernel "
                                 "name from partial backend kernel name"});

  return launch_kernel_from_module(cumodule, full_kernel_name, grid_size,
                                   block_size, dynamic_shared_mem, _stream,
                                   kernel_args);
}

result cuda_queue::submit_sscp_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    std::string_view kernel_name, const rt::hcf_kernel_info *kernel_info,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const kernel_configuration &initial_config) {
#ifdef HIPSYCL_WITH_SSCP_COMPILER

  this->activate_device();

  int device = this->_dev.get_id();

  common::spin_lock_guard lock{_sscp_submission_spin_lock};

  cuda_hardware_context *ctx = static_cast<cuda_hardware_context *>(
      this->_backend->get_hardware_manager()->get_device(device));

  unsigned compute_capability = ctx->get_compute_capability();

  if(!kernel_info) {
    return make_error(
        __acpp_here(),
        error_info{"cuda_queue: Could not obtain hcf kernel info for kernel " +
            std::string{kernel_name}});
  }

  
  _arg_mapper.construct_mapping(*kernel_info, args, arg_sizes,
                                            num_args);

  if(!_arg_mapper.mapping_available()) {
    return make_error(
        __acpp_here(),
        error_info{
            "cuda_queue: Could not map C++ arguments to kernel arguments"});
  }

  kernel_adaptivity_engine adaptivity_engine{
      hcf_object, kernel_name, kernel_info, _arg_mapper, num_groups,
      group_size, args,        arg_sizes,   num_args, local_mem_size};

  _config = initial_config;
  _config.append_base_configuration(
      kernel_base_config_parameter::backend_id, backend_id::cuda);
  _config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow,
      compilation_flow::sscp);
  _config.append_base_configuration(
      kernel_base_config_parameter::hcf_object_id, hcf_object);
  
  for(const auto& flag : kernel_info->get_compilation_flags())
    _config.set_build_flag(flag);
  for(const auto& opt : kernel_info->get_compilation_options())
    _config.set_build_option(opt.first, opt.second);
  // TODO This is incorrect, we should attempt to find a better way to determine
  // the right ptx version
  _config.set_build_option(kernel_build_option::ptx_version,
                          compute_capability);
  _config.set_build_option(kernel_build_option::ptx_target_device,
                          compute_capability);

  auto binary_configuration_id = adaptivity_engine.finalize_binary_configuration(_config);
  auto code_object_configuration_id = binary_configuration_id;
  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_device, device);

  auto get_image_and_kernel_names =
      [&](std::vector<std::string> &contained_kernels) -> std::string {
    return adaptivity_engine.select_image_and_kernels(&contained_kernels);
  };

  auto jit_compiler = [&](std::string& compiled_image) -> bool {
    
    std::vector<std::string> kernel_names;
    std::string selected_image_name = get_image_and_kernel_names(kernel_names);

    // Construct PTX translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator = 
      compiler::createLLVMToPtxTranslator(kernel_names);

    // Lower kernels to PTX
    rt::result err;
    if(kernel_names.size() == 1) {
      err = glue::jit::dead_argument_elimination::compile_kernel(
          translator.get(), hcf_object, selected_image_name, _config,
          binary_configuration_id, compiled_image);
    } else {
      err = glue::jit::compile(translator.get(),
        hcf_object, selected_image_name, _config, compiled_image);
    }

    if(!err.is_success()) {
      register_error(err);
      return false;
    }
    return true;
  };

  auto code_object_constructor = [&](const std::string& ptx_image) -> code_object* {

    std::vector<std::string> kernel_names;
    get_image_and_kernel_names(kernel_names);

    std::string target_arch_name = ctx->get_device_arch();

    cuda_sscp_executable_object *exec_obj = new cuda_sscp_executable_object{
        ptx_image, target_arch_name, hcf_object, kernel_names, device, _config};
    result r = exec_obj->get_build_result();

    HIPSYCL_DEBUG_INFO
        << "cuda_queue: Successfully compiled SSCP kernels to module " << exec_obj->get_module()
        << std::endl;

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    if(kernel_names.size() == 1)
      exec_obj->get_jit_output_metadata().kernel_retained_arguments_indices =
          glue::jit::dead_argument_elimination::
              retrieve_retained_arguments_mask(binary_configuration_id);

    return exec_obj;
  };

  const code_object *obj = _kernel_cache->get_or_construct_jit_code_object(
      code_object_configuration_id, binary_configuration_id,
      jit_compiler, code_object_constructor);

  if(!obj) {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Code object construction failed"});
  }

  if(obj->get_jit_output_metadata().kernel_retained_arguments_indices.has_value()) {
    _arg_mapper.apply_dead_argument_elimination_mask(
        obj->get_jit_output_metadata()
            .kernel_retained_arguments_indices.value());
  }
  CUmodule cumodule = static_cast<const cuda_executable_object*>(obj)->get_module();
  assert(cumodule);

  return launch_kernel_from_module(cumodule, kernel_name, num_groups,
                                   group_size, local_mem_size, _stream,
                                   _arg_mapper.get_mapped_args());

#else
  return make_error(
      __acpp_here(),
      error_info{
          "cuda_queue: SSCP kernel launch was requested, but hipSYCL was "
          "not built with CUDA SSCP support."});
#endif
}

device_id cuda_queue::get_device() const {
  return _dev;
}

void *cuda_queue::get_native_type() const {
  return static_cast<void*>(get_stream());
}

cuda_multipass_code_object_invoker::cuda_multipass_code_object_invoker(
    cuda_queue *q)
    : _queue{q} {}

result cuda_queue::query_status(inorder_queue_status &status) {
  auto err = cudaStreamQuery(_stream);
  if(err == cudaSuccess) {
    status = inorder_queue_status{true};
  } else if(err == cudaErrorNotReady) {
    status = inorder_queue_status{false};
  } else {
    return make_error(__acpp_here(),
                      error_info{"cuda_queue: Could not query stream status",
                                 error_code{"CU", static_cast<int>(err)}});
  }

  return make_success();
}

result cuda_multipass_code_object_invoker::submit_kernel(
    const kernel_operation& op,
    hcf_object_id hcf_object,
    const rt::range<3> &num_groups,
    const rt::range<3> &group_size,
    unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const std::string &kernel_name_tag,
    const std::string &kernel_body_name) {

  assert(_queue);

  std::string kernel_name = kernel_body_name;
  if(kernel_name_tag.find("__acpp_unnamed_kernel") == std::string::npos)
    kernel_name = kernel_name_tag;

  return _queue->submit_multipass_kernel_from_code_object(
      op, hcf_object, kernel_name, num_groups, group_size, local_mem_size, args,
      num_args);
}

result cuda_sscp_code_object_invoker::submit_kernel(
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
}
}

