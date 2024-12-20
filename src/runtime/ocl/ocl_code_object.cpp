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
#include "hipSYCL/runtime/ocl/ocl_code_object.hpp"
#include "hipSYCL/common/string_utils.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/ocl/ocl_queue.hpp"

namespace hipsycl {
namespace rt {

result ocl_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, hcf_object_id hcf_object,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned int local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, std::string_view kernel_name,
    const rt::hcf_kernel_info *kernel_info,
    const kernel_configuration &config) {

  assert(_queue);

  return _queue->submit_sscp_kernel_from_code_object(
      op, hcf_object, kernel_name, kernel_info, num_groups, group_size,
      local_mem_size, args, arg_sizes, num_args, config);
}

ocl_executable_object::ocl_executable_object(const cl::Context& ctx, cl::Device& dev,
    hcf_object_id source, const std::string& code_image, const kernel_configuration &config)
: _source{source}, _ctx{ctx}, _dev{dev}, _id{config.generate_id()} {

  std::vector<char> ir(code_image.size());
  std::memcpy(ir.data(), code_image.data(), code_image.size());

  cl_int err = 0;
  _program = cl::Program(_ctx, ir, false, &err);

  if(err != CL_SUCCESS) {
    _build_status = register_error(
        __acpp_here(),
        error_info{"ocl_code_object: Construction of CL program failed",
                   error_code{"CL", static_cast<int>(err)}});
    return;
  }
  
  std::string options_string="-cl-uniform-work-group-size";
  for(const auto& flag : config.build_flags()) {
    if(flag == kernel_build_flag::fast_math) {
      options_string += " -cl-fast-relaxed-math -cl-denorms-are-zero";
    }
  }

  err = _program.build(
      _dev, options_string.c_str());

  if(err != CL_SUCCESS) {
    std::string build_log = "<build log not available>";
    cl_int access_build_log_err =
        _program.getBuildInfo(_dev, CL_PROGRAM_BUILD_LOG, &build_log);

    std::string msg = "ocl_code_object: Building CL program failed.";
    if(access_build_log_err == CL_SUCCESS)
      msg += " Build log: " + build_log;
    
    _build_status = register_error(
        __acpp_here(), error_info{msg,
                                     error_code{"CL", static_cast<int>(err)}});
    return;
  }

  // clCreateKernelsInProgram seems to not work reliably
  //err = _program.createKernels(&kernels);
  std::string concatenated_name_list;
  err = _program.getInfo(CL_PROGRAM_KERNEL_NAMES, &concatenated_name_list);
  
  if(err != CL_SUCCESS) {
    _build_status = register_error(
        __acpp_here(),
        error_info{
            "ocl_code_object: Could not obtain kernel names in program",
            error_code{"CL", static_cast<int>(err)}});
    return;
  }

  _kernel_names =
      common::split_by_delimiter(concatenated_name_list, ';');
  
  std::vector<cl::Kernel> kernels;
  for(const auto& name : _kernel_names) {
    cl::Kernel k{_program, name.c_str(), &err};
    if(err != CL_SUCCESS) {
      _build_status = register_error(
        __acpp_here(),
        error_info{
            "ocl_code_object: Could not construct kernel object for kernel "+name,
            error_code{"CL", static_cast<int>(err)}});
      return;
    }
    _kernel_handles[name] = k;
  }

  _build_status = make_success();
}

ocl_executable_object::~ocl_executable_object() {}

result ocl_executable_object::get_build_result() const {
  return _build_status;
}

code_object_state ocl_executable_object::state() const {
  if(_build_status.is_success())
    return code_object_state::executable;
  return code_object_state::invalid;
}

code_format ocl_executable_object::format() const {
  return code_format::spirv;
}

backend_id ocl_executable_object::managing_backend() const {
  return backend_id::ocl;
}

hcf_object_id ocl_executable_object::hcf_source() const {
  return _source;
}

std::string ocl_executable_object::target_arch() const {
  // TODO might want to return actual device name that we have compiled for?
  return "spirv64";
}

std::vector<std::string>
ocl_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

bool ocl_executable_object::contains(
    const std::string &backend_kernel_name) const {
  return _kernel_handles.find(backend_kernel_name) != _kernel_handles.end();
}

compilation_flow ocl_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

kernel_configuration::id_type
ocl_executable_object::configuration_id() const {
  return _id;
}

cl::Device ocl_executable_object::get_cl_device() const {
  return _dev;
}

cl::Context ocl_executable_object::get_cl_context() const {
  return _ctx;
}
  
result ocl_executable_object::get_kernel(std::string_view name, cl::Kernel& out) const {
  if(!_build_status.is_success())
    return _build_status;
  auto it = _kernel_handles.find(name);
  if(it == _kernel_handles.end())
    return make_error(__acpp_here(),
                      error_info{"ocl_executable_object: Unknown kernel name"});
  out = it->second;
  return make_success();
}

}
}
