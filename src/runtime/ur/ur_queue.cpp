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
#include "hipSYCL/runtime/ur/ur_queue.hpp"

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"
#include "hipSYCL/runtime/ur/ur_event.hpp"
#include "hipSYCL/runtime/ur/ur_hardware_manager.hpp"
#include "hipSYCL/runtime/util.hpp"

#include "ur_utils.hpp"

#include <cassert>
#include <future>
#include <utility>
#include <vector>

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirvFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"

#endif

using namespace hipsycl::rt;

namespace {

result submit_ur_kernel(const ur_kernel_handle_t &kernel,
                        const ur_queue_handle_t &queue,
                        const range<3> &group_size, const range<3> &num_groups,
                        void **kernel_args, const std::size_t *arg_sizes,
                        const std::size_t num_args, const hcf_kernel_info *info,
                        ur_event_handle_t *evt_out = nullptr) {

  for (std::size_t i = 0; i < num_args; ++i) {
    HIPSYCL_DEBUG_INFO << "ur_queue: Setting kernel argument " << i
                       << " of size " << arg_sizes[i] << " at "
                       << kernel_args[i] << std::endl;

    const auto err =
        urKernelSetArgValue(kernel, i, arg_sizes[i], nullptr, kernel_args[i]);

    if (err != UR_RESULT_SUCCESS) {
      return make_error(__acpp_here(),
                        error_info{"ocl_queue: Could not set kernel argument",
                                   error_code{"CL", static_cast<int>(err)}});
    }
  }

  HIPSYCL_DEBUG_INFO << "ocl_queue: Submitting kernel!" << std::endl;
  range<3> global_size = num_groups * group_size;

  auto cl_global_size =
      std::vector{global_size[0], global_size[1], global_size[2]};
  auto cl_local_size = std::vector{group_size[0], group_size[1], group_size[2]};
  auto offset = std::vector<size_t>{0, 0, 0};

  if (global_size[2] == 1) {
    cl_global_size = {global_size[0], global_size[1]};
    cl_local_size = {group_size[0], group_size[1]};
    offset = {0, 0};
    if (global_size[1] == 1) {
      cl_global_size = {global_size[0]};
      cl_local_size = {group_size[0]};
      offset = {0};
    }
  }

  assert(cl_global_size.size() == cl_local_size.size() &&
         cl_global_size.size() == offset.size() && "Invalid size");

  const auto err = urEnqueueKernelLaunch(
      queue, kernel, cl_global_size.size(), offset.data(),
      cl_global_size.data(), cl_local_size.data(), 0, nullptr, evt_out);

  if (err != UR_RESULT_SUCCESS) {
    return make_error(
        __acpp_here(),
        ur_error_info("ur_queue: urEnqueueKernelLaunch() failed", err));
  }

  return make_success();
}

} // namespace

ur_queue::ur_queue(ur_hardware_manager *hw_manager, const int device_index)
    : _device_index(device_index), _hw_manager(hw_manager),
      _sscp_code_object_invoker{*this} {
  assert(hw_manager != nullptr);

  const ur_hardware_context *hw_context =
      cast<ur_hardware_context>(hw_manager->get_device(device_index));
  assert(hw_context != nullptr);

  constexpr ur_queue_properties_t queue_properties = {};
  const ur_result_t res =
      urQueueCreate(hw_context->get_ur_context(), hw_context->get_ur_device(),
                    &queue_properties, &_queue);
  if (res != UR_RESULT_SUCCESS) {
    register_error(make_error(__acpp_here(),
                              ur_error_info("urQueueCreate() failed", res)));
    return;
  }
}

ur_queue::~ur_queue() {
  const auto err = urQueueRelease(_queue);
  if (err != UR_RESULT_SUCCESS) {
    register_error(make_error(__acpp_here(),
                              ur_error_info("urQueueRelease() failed", err)));
  }
}
std::shared_ptr<dag_node_event> ur_queue::insert_event() {
  if (!_state.get_most_recent_event()) {
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
    ur_event_handle_t wait_evt = nullptr;

    const auto err =
        urEnqueueEventsWaitWithBarrier(_queue, 0, nullptr, &wait_evt);
    if (err != UR_RESULT_SUCCESS) {
      register_error(
          __acpp_here(),
          ur_error_info("ur_queue: urEnqueueEventsWaitWithBarrier() failed",
                        err));
    }

    register_submitted_op(wait_evt);
  }

  return _state.get_most_recent_event();
}

std::shared_ptr<dag_node_event> ur_queue::create_queue_completion_event() {
  return std::make_shared<
      queue_completion_event<ur_event_handle_t, ur_node_event>>(this);
}
result ur_queue::submit_kernel(kernel_operation &op, const dag_node_ptr &node) {

  backend_kernel_launch_capabilities cap;
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);

  // TODO: Instrumentation
  return op.get_launcher().invoke(backend_id::ocl, this, cap, node.get());
};

void ur_queue::register_submitted_op(ur_event_handle_t evt) {
  this->_state.set_most_recent_event(std::make_shared<ur_node_event>(evt));
}

result ur_queue::submit_memcpy(memcpy_operation &op,
                               const dag_node_ptr &node_ptr) {

  assert(op.source().get_access_ptr() != nullptr);
  assert(op.dest().get_access_ptr() != nullptr);

  const auto transfer_range = op.get_num_transferred_elements();

  int dimension = 0;
  if (transfer_range[0] > 1)
    dimension = 3;
  else if (transfer_range[1] > 1)
    dimension = 2;
  else
    dimension = 1;

  const auto src_shape = op.source().get_allocation_shape();
  const auto dst_shape = op.dest().get_allocation_shape();

  const auto src_el_size = op.source().get_element_size();
  const auto dst_el_size = op.dest().get_element_size();

  const auto copy_size = op.get_num_transferred_bytes();

  // If we transfer the entire buffer, treat it as 1D memcpy for performance.
  // TODO: The same optimization could also be applied for the general case
  // when regions are contiguous
  if (transfer_range == src_shape && transfer_range == dst_shape &&
      op.source().get_access_offset() == id<3>{} &&
      op.dest().get_access_offset() == id<3>{})
    dimension = 1;

  assert(dimension >= 1 && dimension <= 3 && "Invalid dimension");

  constexpr bool is_blocking = true;
  constexpr auto num_events_in_waitlist = 0;
  constexpr auto events_in_waitlist = nullptr;

  ur_event_handle_t event = nullptr;
  if (dimension == 1) {

    const auto src_raw_ptr = op.source().get_access_ptr();
    const auto dst_raw_ptr = op.dest().get_access_ptr();

    const ur_result_t res = urEnqueueUSMMemcpy(
        _queue, is_blocking, dst_raw_ptr, src_raw_ptr, copy_size,
        num_events_in_waitlist, events_in_waitlist, &event);

    if (res != UR_RESULT_SUCCESS) {
      return make_error(
          __acpp_here(),
          ur_error_info("ur_queue: urEnqueueUSMMemcpy() failed", res));
    }

  } else if (dimension == 2) {
    const auto src_raw_ptr = op.source().get_access_ptr();
    const auto dst_raw_ptr = op.dest().get_access_ptr();

    const auto dest_row_pitch =
        extract_from_range3<2>(dst_shape)[1] * dst_el_size;
    const auto source_row_pitch =
        extract_from_range3<2>(src_shape)[1] * src_el_size;
    const auto num_bytes_to_copy =
        extract_from_range3<2>(transfer_range)[1] * src_el_size;
    const auto num_rows_to_copy = extract_from_range3<2>(transfer_range)[0];

    const ur_result_t res = urEnqueueUSMMemcpy2D(
        _queue, is_blocking, dst_raw_ptr, dest_row_pitch, src_raw_ptr,
        source_row_pitch, num_bytes_to_copy, num_rows_to_copy,
        num_events_in_waitlist, events_in_waitlist, &event);
    if (res != UR_RESULT_SUCCESS) {
      return make_error(
          __acpp_here(),
          ur_error_info("ur_queue: urEnqueueUSMMemcpy2D() failed", res));
    }

  } else {
    // custom 3D memcpy

    const auto dest_offset = op.dest().get_access_offset();
    const auto src_offset = op.source().get_access_offset();

    void *base_src = op.source().get_base_ptr();
    void *base_dest = op.dest().get_base_ptr();

    const auto row_size = transfer_range[2] * src_el_size;

    auto current_src_offset = src_offset;
    auto current_dest_offset = dest_offset;

    auto linear_index = [](id<3> id, range<3> allocation_shape) {
      return id[2] + allocation_shape[2] * id[1] +
             allocation_shape[2] * allocation_shape[1] * id[0];
    };

    for (std::size_t surface = 0; surface < transfer_range[0]; ++surface) {
      for (std::size_t row = 0; row < transfer_range[1]; ++row) {

        auto current_src = static_cast<char *>(base_src);
        auto current_dst = static_cast<char *>(base_dest);

        current_src +=
            linear_index(current_src_offset, src_shape) * src_el_size;

        current_dst +=
            linear_index(current_dest_offset, dst_shape) * dst_el_size;

        assert(current_src + row_size <=
               static_cast<char *>(base_src) + src_shape.size() * src_el_size);
        assert(current_dst + row_size <=
               static_cast<char *>(base_dest) + dst_shape.size() * dst_el_size);

        const ur_result_t err = urEnqueueUSMMemcpy(
            _queue, is_blocking, current_dst, current_src, row_size,
            num_events_in_waitlist, events_in_waitlist, &event);

        if (err != UR_RESULT_SUCCESS) {
          return make_error(
              __acpp_here(),
              ur_error_info("ur_queue: urEnqueueUSMMemcpy() failed", err));
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

  register_submitted_op(event);
  return make_success();
}
result ur_queue::submit_prefetch(prefetch_operation &op,
                                 const dag_node_ptr &node_ptr) {

  constexpr auto flags = ur_usm_migration_flags_t{};
  constexpr auto num_events_in_wait_list = 0;
  constexpr auto events_in_wait_list = nullptr;

  ur_event_handle_t evt;
  const ur_result_t err =
      urEnqueueUSMPrefetch(_queue, op.get_pointer(), op.get_num_bytes(), flags,
                           num_events_in_wait_list, events_in_wait_list, &evt);

  if (err != UR_RESULT_SUCCESS) {
    return make_error(
        __acpp_here(),
        ur_error_info("ur_queue: urEnqueueUSMPrefetch() failed", err));
  }

  register_submitted_op(evt);
  return make_success();
}
result ur_queue::submit_memset(memset_operation &op, const dag_node_ptr &) {
  constexpr auto num_events_in_wait_list = 0;
  constexpr auto events_in_wait_list = nullptr;

  const auto ptr = op.get_pointer();
  const auto pattern = op.get_pattern();
  const auto size = op.get_num_bytes();

  ur_event_handle_t evt;
  const ur_result_t err =
      urEnqueueUSMFill(_queue, ptr, sizeof(pattern), &pattern, size,
                       num_events_in_wait_list, events_in_wait_list, &evt);
  if (err != UR_RESULT_SUCCESS) {
    return make_error(
        __acpp_here(),
        ur_error_info("ur_queue: urEnqueueUSMFill() failed", err));
  }

  register_submitted_op(evt);
  return make_success();
}

result ur_queue::submit_queue_wait_for(const dag_node_ptr &evt) {

  const auto ur_event = static_cast<ur_node_event *>(evt->get_event().get());
  const auto events = std::vector{ur_event->get_event_handle()};

  ur_event_handle_t wait_evt;
  const ur_result_t err = urEnqueueEventsWaitWithBarrier(
      _queue, events.size(), events.data(), &wait_evt);
  if (err != UR_RESULT_SUCCESS) {
    return make_error(
        __acpp_here(),
        error_info{"ocl_queue: enqueueBarrierWithWaitList() failed",
                   error_code{"CL", err}});
  }

  register_submitted_op(wait_evt);
  return make_success();
}

result ur_queue::submit_external_wait_for(const dag_node_ptr &node) {
  throw std::runtime_error(
      "ur_queue::submit_external_wait_for() not implemented");
}
result ur_queue::wait() {
  const ur_result_t err = urQueueFinish(_queue);
  if (err != UR_RESULT_SUCCESS) {
    return make_error(__acpp_here(),
                      error_info{"ur_queue: Couldn't finish queue",
                                 error_code{"CL", static_cast<int>(err)}});
  }
  return make_success();
}
device_id ur_queue::get_device() const {
  return _hw_manager->get_device_id(_device_index);
}
void *ur_queue::get_native_type() const { return _queue; }

ur_queue_handle_t ur_queue::get_ur_queue() const { return _queue; }
ur_hardware_manager *ur_queue::get_hardware_manager() const {
  return _hw_manager;
}

result ur_queue::query_status(inorder_queue_status &status) {
  const auto evt = _state.get_most_recent_event();
  if (evt != nullptr) {
    status = inorder_queue_status{evt->is_complete()};
  } else {
    status = inorder_queue_status{true};
  }
  return make_success();
}

result ur_queue::submit_sscp_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    std::string_view kernel_name, const hcf_kernel_info *kernel_info,
    const range<3> &num_groups, const range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const kernel_configuration &initial_config) {

#ifndef HIPSYCL_WITH_SSCP_COMPILER
  return make_error(
      __acpp_here(),
      error_info{"ur_queue: SSCP kernel launch was requested, but hipSYCL was "
                 "not built with OpenCL SSCP support."});
#else

  if (kernel_info == nullptr) {
    return make_error(
        __acpp_here(),
        error_info{"Could not obtain hcf kernel info for kernel " +
                   std::string(kernel_name)});
  }

  common::spin_lock_guard lock{_sscp_submission_spin_lock};

  _arg_mapper.construct_mapping(*kernel_info, args, arg_sizes, num_args);

  if (!_arg_mapper.mapping_available()) {
    return make_error(
        __acpp_here(),
        error_info{"Could not map C++ arguments to kernel arguments"});
  }

  kernel_adaptivity_engine adaptivity_engine{
      hcf_object, kernel_name, kernel_info, _arg_mapper, num_groups,
      group_size, args,        arg_sizes,   num_args,    local_mem_size};

  const auto *hw_ctx = dynamic_cast<ur_hardware_context *>(
      _hw_manager->get_device(_device_index));

  const ur_context_handle_t ctx = hw_ctx->get_ur_context();
  const ur_device_handle_t dev = hw_ctx->get_ur_device();

  _config = initial_config;

  _config.append_base_configuration(kernel_base_config_parameter::backend_id,
                                    backend_id::ocl);
  _config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow, compilation_flow::sscp);
  _config.append_base_configuration(kernel_base_config_parameter::hcf_object_id,
                                    hcf_object);

  for (const auto &flag : kernel_info->get_compilation_flags())
    _config.set_build_flag(flag);

  for (const auto &[opt, val] : kernel_info->get_compilation_options())
    _config.set_build_option(opt, val);

  _config.set_build_option(
      kernel_build_option::spirv_dynamic_local_mem_allocation_size,
      local_mem_size);

  // if (hw_ctx->has_intel_extension_profile()) {
  //   _config.set_build_flag(kernel_build_flag::spirv_enable_intel_llvm_spirv_options);
  // }

  // TODO: Enable this if we are on Intel
  // config.set_build_flag(kernel_build_flag::spirv_enable_intel_llvm_spirv_options);

  const auto binary_configuration_id =
      adaptivity_engine.finalize_binary_configuration(_config);
  const auto code_object_configuration_id = binary_configuration_id;

  // kernel_configuration::extend_hash(code_object_configuration_id,
  //                                   kernel_base_config_parameter::runtime_device,
  //                                   hw_ctx.);
  //
  // kernel_configuration::extend_hash(code_object_configuration_id,
  //                                   kernel_base_config_parameter::runtime_context,
  //                                   ctx.get());

  auto jit_compiler = [&](std::string &compiled_image) -> bool {
    std::vector<std::string> kernel_names;
    const std::string selected_image_name =
        adaptivity_engine.select_image_and_kernels(&kernel_names);

    // Construct SPIR-V translator to compile the specified kernels
    const std::unique_ptr<compiler::LLVMToBackendTranslator> translator =
        std::move(compiler::createLLVMToSpirvTranslator(kernel_names));

    // Lower kernels to SPIR-V
    result err;
    if (kernel_names.size() == 1) {
      err = glue::jit::dead_argument_elimination::compile_kernel(
          translator.get(), hcf_object, selected_image_name, _config,
          binary_configuration_id, _reflection_map, compiled_image);
    } else {
      err =
          glue::jit::compile(translator.get(), hcf_object, selected_image_name,
                             _config, _reflection_map, compiled_image);
    }

    if (!err.is_success()) {
      register_error(err);
      return false;
    }

    return true;
  };

  auto code_object_constructor =
      [&](const std::string &compiled_image) -> code_object * {
    auto *exec_obj =
        new ur_executable_object{ctx, dev, hcf_object, compiled_image, _config};

    const result res = exec_obj->get_build_result();
    if (!res.is_success()) {
      register_error(res);
      delete exec_obj;
      return nullptr;
    }

    if (exec_obj->supported_backend_kernel_names().size() == 1)
      exec_obj->get_jit_output_metadata().kernel_retained_arguments_indices =
          glue::jit::dead_argument_elimination::
              retrieve_retained_arguments_mask(binary_configuration_id);

    return exec_obj;
  };

  const code_object *obj = _kernel_cache->get_or_construct_jit_code_object(
      code_object_configuration_id, binary_configuration_id, jit_compiler,
      code_object_constructor);

  if (obj == nullptr) {
    return make_error(__acpp_here(),
                      error_info{"Code object construction failed"});
  }

  if (obj->get_jit_output_metadata()
          .kernel_retained_arguments_indices.has_value()) {
    _arg_mapper.apply_dead_argument_elimination_mask(
        obj->get_jit_output_metadata()
            .kernel_retained_arguments_indices.value());
  }

  ur_kernel_handle_t kernel;
  result res = dynamic_cast<const ur_executable_object *>(obj)->get_kernel(
      kernel_name, kernel);

  if (!res.is_success())
    return res;

  HIPSYCL_DEBUG_INFO << "ur_queue: Submitting SSCP kernel " << kernel_name
                     << std::endl;

  ur_event_handle_t completion_evt;

  auto submission_err = submit_ur_kernel(
      kernel, _queue, group_size, num_groups, _arg_mapper.get_mapped_args(),
      _arg_mapper.get_mapped_arg_sizes(), _arg_mapper.get_mapped_num_args(),
      kernel_info, &completion_evt);

  if (!submission_err.is_success())
    return submission_err;

  register_submitted_op(completion_evt);
  return make_success();
#endif
}