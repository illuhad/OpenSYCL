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
#include "hipSYCL/runtime/ur/ur_code_object.hpp"

#include "hipSYCL/runtime/ur/ur_queue.hpp"
#include "ur_utils.hpp"

using namespace hipsycl::rt;

result ur_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, const hcf_object_id hcf_object,
    const range<3> &num_groups, const range<3> &group_size,
    const unsigned int local_mem_size, void **args, std::size_t *arg_sizes,
    const std::size_t num_args, const std::string_view kernel_name,
    const hcf_kernel_info *kernel_info, const kernel_configuration &config) {

  return _queue.submit_sscp_kernel_from_code_object(
      op, hcf_object, kernel_name, kernel_info, num_groups, group_size,
      local_mem_size, args, arg_sizes, num_args, config);
}

ur_executable_object::ur_executable_object(const ur_context_handle_t ctx,
                                           ur_device_handle_t dev,
                                           const hcf_object_id source,
                                           const std::string &code_image,
                                           const kernel_configuration &config)
    : _source{source}, _ctx{ctx}, _dev{dev}, _id{config.generate_id()} {

  std::vector<char> ir(code_image.size());
  std::memcpy(ir.data(), code_image.data(), code_image.size());

  {
    const auto err =
        urProgramCreateWithIL(_ctx, ir.data(), ir.size(), nullptr, &_program);
    if (err != UR_RESULT_SUCCESS) {
      _build_status = register_error(
          __acpp_here(),
          ur_error_info("Construction of UR program failed", err));
      return;
    }
  }

  {
    const auto err = urProgramBuild(_ctx, _program, nullptr);
    if (err != UR_RESULT_SUCCESS) {
      _build_status = register_error(
          __acpp_here(), ur_error_info("Building of UR program failed", err));
      return;
    }
  }

  // TODO: This is a placeholder. We need to add support for build options.
  throw std::runtime_error{"ur_executable_object: not implemented"}; // TODO
}
result ur_executable_object::get_build_result() const { return _build_status; }
hcf_object_id ur_executable_object::hcf_source() const { return _source; }
backend_id ur_executable_object::managing_backend() const {
  return backend_id::unified_runtime;
}

code_format ur_executable_object::format() const {
  throw std::runtime_error{
      "ur_executable_object::format() not implemented"}; // TODO
}
std::string ur_executable_object::target_arch() const {
  throw std::runtime_error{
      "ur_executable_object::target_arch() not implemented"}; // TODO
}

code_object_state ur_executable_object::state() const {
  return _build_status.is_success() ? code_object_state::executable
                                    : code_object_state::invalid;
}

compilation_flow ur_executable_object::source_compilation_flow() const {
  throw std::runtime_error{"ur_executable_object::source_compilation_flow() "
                           "not implemented"}; // TODO
}

std::vector<std::string>
ur_executable_object::supported_backend_kernel_names() const {
  throw std::runtime_error{"ur_executable_object::supported_backend_kernel_"
                           "names() not implemented"}; // TODO
}
bool ur_executable_object::contains(
    const std::string &backend_kernel_name) const {
  throw std::runtime_error{
      "ur_executable_object::contains() not implemented"}; // TODO
}

void ur_executable_object::load_kernel_handles() {
  throw std::runtime_error{
      "ur_executable_object::load_kernel_handles() not implemented"}; // TODO
}

ur_sscp_executable_object::ur_sscp_executable_object(
    ur_context_handle_t ctx, ur_device_handle_t dev, hcf_object_id source,
    const std::string &spirv_image, const kernel_configuration &config)
    : ur_executable_object{ctx, dev, source, spirv_image, config} {
  // TODO: Implement this
}

compilation_flow ur_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

kernel_configuration::id_type
ur_sscp_executable_object::configuration_id() const {
  return _id;
}

result ur_executable_object::get_kernel(const std::string_view &name,
                                        ur_kernel_handle_t &out) const {
  if (!_build_status.is_success()) {
    return _build_status;
  }

  const auto k_handle = _kernel_handles.find(name);
  if (k_handle == _kernel_handles.end()) {
    return make_error(__acpp_here(), error_info{"Unknown kernel name"});
  }

  out = k_handle->second;
  return make_success();
}