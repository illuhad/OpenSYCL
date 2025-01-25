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
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/ocl/ocl_code_object.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"
#include "hipSYCL/runtime/ocl/ocl_event.hpp"
#include "hipSYCL/runtime/ocl/ocl_queue.hpp"
#include "hipSYCL/runtime/ocl/ocl_hardware_manager.hpp"
#include "hipSYCL/common/spin_lock.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirvFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include <CL/cl.h>

#endif

namespace hipsycl {
namespace rt {


namespace {


result submit_ocl_kernel(cl::Kernel& kernel,
                        cl::CommandQueue& queue,
                        const rt::range<3> &group_size,
                        const rt::range<3> &num_groups, void **kernel_args,
                        const std::size_t *arg_sizes, std::size_t num_args,
                        ocl_usm* usm,
                        const hcf_kernel_info *info,
                        cl::Event* evt_out = nullptr) {

  cl_int err = 0;
  for(std::size_t i = 0; i < num_args; ++i ){
    HIPSYCL_DEBUG_INFO << "ocl_queue: Setting kernel argument " << i
                       << " of size " << arg_sizes[i] << " at " << kernel_args[i]
                       << std::endl;

    err = kernel.setArg(i, static_cast<std::size_t>(arg_sizes[i]), kernel_args[i]);

    if(err != CL_SUCCESS) {
      return make_error(
          __acpp_here(),
          error_info{"ocl_queue: Could not set kernel argument",
                     error_code{"CL", static_cast<int>(err)}});
    }
  }

  // This is necessary for USM pointers, which hipSYCL *always*
  // relies on.
  err = usm->enable_indirect_usm_access(kernel);

  if(err != CL_SUCCESS) {
    return make_error(
          __acpp_here(),
          error_info{"ocl_queue: Could not set indirect access flags",
                     error_code{"CL", static_cast<int>(err)}});
  }

  HIPSYCL_DEBUG_INFO << "ocl_queue: Submitting kernel!" << std::endl;
  rt::range<3> global_size = num_groups * group_size;
  
  cl::NDRange cl_global_size{global_size[0], global_size[1], global_size[2]};
  cl::NDRange cl_local_size{group_size[0], group_size[1], group_size[2]};
  cl::NDRange offset{0, 0, 0};
  if (global_size[2] == 1) {
    cl_global_size = cl::NDRange{global_size[0], global_size[1]};
    cl_local_size = cl::NDRange{group_size[0], group_size[1]};
    offset = cl::NDRange{0, 0};
    if (global_size[1] == 1) {
      cl_global_size = cl::NDRange{global_size[0]};
      cl_local_size = cl::NDRange{group_size[0]};
      offset = cl::NDRange{0};
    }
  }

  err = queue.enqueueNDRangeKernel(kernel, offset, cl_global_size,
                                   cl_local_size, nullptr, evt_out);

  if(err != CL_SUCCESS) {
    return make_error(
        __acpp_here(),
        error_info{"ocl_queue: Kernel launch failed",
                   error_code{"CL", static_cast<int>(err)}});
  }

  return make_success();
}

}

class ocl_hardware_manager;

ocl_queue::ocl_queue(ocl_hardware_manager* hw_manager, std::size_t device_index)
  : _hw_manager{hw_manager}, _device_index{device_index}, _sscp_invoker{this},
    _kernel_cache{kernel_cache::get()} {

  cl_command_queue_properties props = 0;
  ocl_hardware_context *dev_ctx =
      static_cast<ocl_hardware_context *>(hw_manager->get_device(device_index));
  cl::Device cl_dev = dev_ctx->get_cl_device();
  cl::Context cl_ctx = dev_ctx->get_cl_context();

  cl_int err;
  _queue = cl::CommandQueue{cl_ctx, cl_dev, props, &err};
  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_queue: Couldn't construct backend queue",
                              error_code{"CL", err}});
  }

  _reflection_map = glue::jit::construct_default_reflection_map(dev_ctx);
}

ocl_queue::~ocl_queue() {}

std::shared_ptr<dag_node_event> ocl_queue::insert_event() {
  if(!_state.get_most_recent_event()) {
    // Normally, this code path should only be triggered
    // when no work has been submitted to the queue, and so
    // nothing needs to be synchronized with. Thus
    // the returned event should never actually be needed
    // by other nodes in the DAG.
    // However, if some work fails to execute, we can end up
    // with the situation that the "no work submitted yet" situation
    // appears at later stages in the program, when events
    // are expected to work correctly.
    // It is thus safer to enqueue a barrier here.
    cl::Event wait_evt;
    cl_int err = _queue.enqueueBarrierWithWaitList(nullptr, &wait_evt);

    if(err != CL_SUCCESS) {
      register_error(
            __acpp_here(),
            error_info{
                "ocl_queue: enqueueBarrierWithWaitList() failed",
                error_code{"CL", err}});
    }
    register_submitted_op(wait_evt);
    
  }

  return _state.get_most_recent_event();
}

std::shared_ptr<dag_node_event> ocl_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<cl::Event, ocl_node_event>>(
      this);
}

result ocl_queue::submit_memcpy(memcpy_operation &op, const dag_node_ptr&) {

  HIPSYCL_DEBUG_INFO << "ocl_queue: On device "
                     << _hw_manager->get_device_id(_device_index)
                     << ": Processing memcpy request from device "
                     << op.source().get_device() << " to "
                     << op.dest().get_device() << std::endl;

  // TODO We could probably unify some of the logic here between
  // backends

  assert(op.source().get_access_ptr());
  assert(op.dest().get_access_ptr());

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

  auto linear_index = [](id<3> id, range<3> allocation_shape) {
    return id[2] + allocation_shape[2] * id[1] +
           allocation_shape[2] * allocation_shape[1] * id[0];
  };

  cl::Event evt;
  ocl_hardware_context *ocl_ctx = static_cast<ocl_hardware_context *>(
        _hw_manager->get_device(_device_index));
  ocl_usm* usm = ocl_ctx->get_usm_provider();

  if(dimension == 1) {
    
    cl_int err = usm->enqueue_memcpy(_queue, op.dest().get_access_ptr(),
                        op.source().get_access_ptr(),
                        op.get_num_transferred_bytes(), {}, &evt);

    if(err != CL_SUCCESS) {
      return make_error(
          __acpp_here(),
          error_info{"ocl_queue: enqueuing memcpy failed",
                     error_code{"CL", static_cast<int>(err)}});
    }
  } else {
    id<3> src_offset = op.source().get_access_offset();
    id<3> dest_offset = op.dest().get_access_offset();
    std::size_t src_element_size = op.source().get_element_size();
    std::size_t dest_element_size = op.dest().get_element_size();
    range<3> src_allocation_shape = op.source().get_allocation_shape();
    range<3> dest_allocation_shape = op.dest().get_allocation_shape();

    void *base_src = op.source().get_base_ptr();
    void *base_dest = op.dest().get_base_ptr();


    id<3> current_src_offset = src_offset;
    id<3> current_dest_offset = dest_offset;
    std::size_t row_size = transfer_range[2] * src_element_size;

    for (std::size_t surface = 0; surface < transfer_range[0]; ++surface) {
      for (std::size_t row = 0; row < transfer_range[1]; ++row) {

        char *current_src = reinterpret_cast<char *>(base_src);
        char *current_dest = reinterpret_cast<char *>(base_dest);

        current_src += linear_index(current_src_offset, src_allocation_shape) *
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

        cl_int err = usm->enqueue_memcpy(_queue, current_dest, current_src,
                                         row_size, {}, &evt);

        if(err != CL_SUCCESS) {
          return make_error(
              __acpp_here(),
              error_info{"ocl_queue: enqueuing memcpy failed",
                        error_code{"CL", static_cast<int>(err)}});
        }

        ++current_src_offset[1];
        ++current_dest_offset[1];
      }
      current_src_offset[1] = src_offset[1];
      current_dest_offset[1] = dest_offset[1];

      ++current_dest_offset[0];
      ++current_src_offset[0];
    }
  }

  register_submitted_op(evt);
  return make_success();
}

result ocl_queue::submit_kernel(kernel_operation &op, const dag_node_ptr& node) {

  rt::backend_kernel_launch_capabilities cap;
  cap.provide_sscp_invoker(&_sscp_invoker);
  
  // TODO: Instrumentation
  return op.get_launcher().invoke(backend_id::ocl, this, cap, node.get());
}

result ocl_queue::submit_prefetch(prefetch_operation &op, const dag_node_ptr&) {
  ocl_hardware_context *ocl_ctx = static_cast<ocl_hardware_context *>(
        _hw_manager->get_device(_device_index));
  ocl_usm* usm = ocl_ctx->get_usm_provider();

  cl::Event evt;
  cl_int err = 0;
  if(op.get_target().is_host()) {
    err = usm->enqueue_prefetch(_queue, op.get_pointer(), op.get_num_bytes(),
                                CL_MIGRATE_MEM_OBJECT_HOST, {}, &evt);
  } else {
    err = usm->enqueue_prefetch(_queue, op.get_pointer(), op.get_num_bytes(),
                                0, {}, &evt);
  }

  if(err != CL_SUCCESS) {
    return make_error(
          __acpp_here(),
          error_info{"ocl_queue: enqueuing prefetch failed",
                     error_code{"CL", static_cast<int>(err)}});
  }

  register_submitted_op(evt);
  return make_success();
}

result ocl_queue::submit_memset(memset_operation& op, const dag_node_ptr&) {
  ocl_hardware_context *ocl_ctx = static_cast<ocl_hardware_context *>(
        _hw_manager->get_device(_device_index));
  ocl_usm* usm = ocl_ctx->get_usm_provider();

  cl::Event evt;
  cl_int err = usm->enqueue_memset(_queue, op.get_pointer(), op.get_pattern(),
                                   op.get_num_bytes(), {}, &evt);
  if(err != CL_SUCCESS) {
    return make_error(
          __acpp_here(),
          error_info{"ocl_queue: enqueuing memset failed",
                     error_code{"CL", static_cast<int>(err)}});
  }

  register_submitted_op(evt);
  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result ocl_queue::submit_queue_wait_for(const dag_node_ptr& evt) {

  ocl_node_event *ocl_evt =
      static_cast<ocl_node_event *>(evt->get_event().get());
  
  std::vector<cl::Event> events{ocl_evt->get_event()};

  if (_hw_manager->get_context(ocl_evt->get_device()) !=
      _hw_manager->get_context(_hw_manager->get_device_id(_device_index))) {
    return submit_external_wait_for(evt);
  }

  cl::Event wait_evt;
  cl_int err = _queue.enqueueBarrierWithWaitList(&events, &wait_evt);

  if(err != CL_SUCCESS) {
    return make_error(
          __acpp_here(),
          error_info{
              "ocl_queue: enqueueBarrierWithWaitList() failed",
              error_code{"CL", err}});
  }
  register_submitted_op(wait_evt);
  return make_success();
}

result ocl_queue::submit_external_wait_for(const dag_node_ptr& node) {
  ocl_hardware_context* hw_ctx = static_cast<ocl_hardware_context *>(
      _hw_manager->get_device(_device_index));
  cl_int err;
  cl::UserEvent uevt{hw_ctx->get_cl_context(), &err};
  if(err != CL_SUCCESS) {
    return make_error(
          __acpp_here(),
          error_info{
              "ocl_queue: OpenCL user event creation failed",
              error_code{"CL", err}});
  }

  cl::Event barrier_evt;
  std::vector<cl::Event> wait_events{uevt};
  _queue.enqueueBarrierWithWaitList(&wait_events, &barrier_evt);

  register_submitted_op(barrier_evt);
  _host_worker([uevt, node]() mutable{
    node->wait();
    cl_int err = uevt.setStatus(CL_COMPLETE);
    if(err != CL_SUCCESS) {
      register_error(
          __acpp_here(),
          error_info{"ocl_queue: Could not change status of user event",
                     error_code{"CL", err}});
    }
  });

  return make_success();
}

result ocl_queue::wait() {
  cl_int err = _queue.finish();
  if(err != CL_SUCCESS) {
    return make_error(__acpp_here(),
                      error_info{"ocl_queue: Couldn't finish queue",
                                 error_code{"CL", err}});
  }
  return make_success();
}

device_id ocl_queue::get_device() const {
  return _hw_manager->get_device_id(_device_index);
}

/// Return native type if supported, nullptr otherwise
void* ocl_queue::get_native_type() const {
  return nullptr;
}

result ocl_queue::query_status(inorder_queue_status& status) {
  auto evt = _state.get_most_recent_event();
  if(evt) {
    status = inorder_queue_status{evt->is_complete()};
  } else {
    status = inorder_queue_status{true};
  }
  return make_success();
}

ocl_hardware_manager *ocl_queue::get_hardware_manager() const {
  return _hw_manager;
}

result ocl_queue::submit_sscp_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    std::string_view kernel_name, const rt::hcf_kernel_info *kernel_info,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const kernel_configuration &initial_config) {

#ifdef HIPSYCL_WITH_SSCP_COMPILER

  if(!kernel_info) {
    return make_error(
        __acpp_here(),
        error_info{"ocl_queue: Could not obtain hcf kernel info for kernel " +
            std::string{kernel_name}});
  }

  common::spin_lock_guard lock{_sscp_submission_spin_lock};

  _arg_mapper.construct_mapping(*kernel_info, args, arg_sizes, num_args);

  if(!_arg_mapper.mapping_available()) {
    return make_error(
        __acpp_here(),
        error_info{
            "ocl_queue: Could not map C++ arguments to kernel arguments"});
  }

  kernel_adaptivity_engine adaptivity_engine{
      hcf_object, kernel_name, kernel_info, _arg_mapper, num_groups,
      group_size, args,        arg_sizes,   num_args, local_mem_size};

  ocl_hardware_context *hw_ctx = static_cast<ocl_hardware_context *>(
      _hw_manager->get_device(_device_index));
  cl::Context ctx = hw_ctx->get_cl_context();
  cl::Device dev = hw_ctx->get_cl_device();

  _config = initial_config;
  
  _config.append_base_configuration(
      kernel_base_config_parameter::backend_id, backend_id::ocl);
  _config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow,
      compilation_flow::sscp);
  _config.append_base_configuration(
      kernel_base_config_parameter::hcf_object_id, hcf_object);
  
  for(const auto& flag : kernel_info->get_compilation_flags())
    _config.set_build_flag(flag);
  for(const auto& opt : kernel_info->get_compilation_options())
    _config.set_build_option(opt.first, opt.second);

  _config.set_build_option(
      kernel_build_option::spirv_dynamic_local_mem_allocation_size,
      local_mem_size);
  
  // Not all OpenCL implementations support these extensions,
  // however if user code doesn't need them, then the compiler should in theory
  // not generate code that requires them. This should allow us to
  // run on all devices that *can* support this particular kernel.
  //
  // We may have to revisit this handling if there are any issues reported
  // with OpenCL implementations that are not from Intel.
  _config.set_build_flag(
    kernel_build_flag::spirv_enable_intel_llvm_spirv_options);


  // TODO: Enable this if we are on Intel
  // config.set_build_flag(kernel_build_flag::spirv_enable_intel_llvm_spirv_options);

  auto binary_configuration_id = adaptivity_engine.finalize_binary_configuration(_config);
  auto code_object_configuration_id = binary_configuration_id;
  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_device, dev.get());
  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_context, ctx.get());

 

  
  auto jit_compiler = [&](std::string& compiled_image) -> bool {
    
    std::vector<std::string> kernel_names;
    std::string selected_image_name =
        adaptivity_engine.select_image_and_kernels(&kernel_names);

    // Construct SPIR-V translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator = 
      std::move(compiler::createLLVMToSpirvTranslator(kernel_names));
    
    // Lower kernels to SPIR-V
    rt::result err;
    if(kernel_names.size() == 1) {
      err = glue::jit::dead_argument_elimination::compile_kernel(
          translator.get(), hcf_object, selected_image_name, _config,
          binary_configuration_id, _reflection_map, compiled_image);
    } else {
      err =
          glue::jit::compile(translator.get(), hcf_object, selected_image_name,
                             _config, _reflection_map, compiled_image);
    }
    
    if(!err.is_success()) {
      register_error(err);
      return false;
    }
    return true;
  };

  auto code_object_constructor = [&](const std::string& compiled_image) -> code_object* {
    ocl_executable_object *exec_obj = new ocl_executable_object{
        ctx, dev, hcf_object, compiled_image, _config};
    result r = exec_obj->get_build_result();

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    if(exec_obj->supported_backend_kernel_names().size() == 1)
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
                      error_info{"ocl_queue: Code object construction failed"});
  }

  if(obj->get_jit_output_metadata().kernel_retained_arguments_indices.has_value()) {
    _arg_mapper.apply_dead_argument_elimination_mask(
        obj->get_jit_output_metadata()
            .kernel_retained_arguments_indices.value());
  }


  cl::Kernel kernel;
  result res = static_cast<const ocl_executable_object *>(obj)->get_kernel(
      kernel_name, kernel);
  
  if(!res.is_success())
    return res;

  HIPSYCL_DEBUG_INFO << "ocl_queue: Attempting to submit SSCP kernel"
                     << std::endl;

  cl::Event completion_evt;
  auto submission_err = submit_ocl_kernel(
      kernel, _queue, group_size, num_groups, _arg_mapper.get_mapped_args(),
      const_cast<std::size_t *>(_arg_mapper.get_mapped_arg_sizes()),
      _arg_mapper.get_mapped_num_args(), hw_ctx->get_usm_provider(), kernel_info,
      &completion_evt);

  if(!submission_err.is_success())
    return submission_err;

  register_submitted_op(completion_evt);

  return make_success();
#else
  return make_error(
      __acpp_here(),
      error_info{"ocl_queue: SSCP kernel launch was requested, but hipSYCL was "
                 "not built with OpenCL SSCP support."});
#endif
}

void ocl_queue::register_submitted_op(cl::Event evt) {
  this->_state.set_most_recent_event(std::make_shared<ocl_node_event>(
      _hw_manager->get_device_id(_device_index), evt));
}

}
}

