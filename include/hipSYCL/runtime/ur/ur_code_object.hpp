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
#ifndef HIPSYCL_UR_CODE_OBJECT_HPP
#define HIPSYCL_UR_CODE_OBJECT_HPP

#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"

#include <string>
#include <ur_api.h>

namespace hipsycl::rt {

class ur_queue;

class ur_sscp_code_object_invoker final : public sscp_code_object_invoker {
public:
  explicit ur_sscp_code_object_invoker(ur_queue &queue) : _queue{queue} {}
  ~ur_sscp_code_object_invoker() override = default;

  result submit_kernel(const kernel_operation &op, hcf_object_id hcf_object,
                       const range<3> &num_groups, const range<3> &group_size,
                       unsigned local_mem_size, void **args,
                       std::size_t *arg_sizes, std::size_t num_args,
                       std::string_view kernel_name,
                       const hcf_kernel_info *kernel_info,
                       const kernel_configuration &config) override;

private:
  ur_queue &_queue;
};

enum class ur_source_format { spirv, native };

class ur_executable_object : public code_object {
public:
  ur_executable_object(ur_context_handle_t ctx, ur_device_handle_t dev,
                       hcf_object_id source, const std::string &code_image,
                       const kernel_configuration &config);
  ~ur_executable_object() override = default;

  code_object_state state() const override;
  code_format format() const override;
  backend_id managing_backend() const override;
  hcf_object_id hcf_source() const override;
  std::string target_arch() const override;
  compilation_flow source_compilation_flow() const override;

  std::vector<std::string> supported_backend_kernel_names() const override;
  bool contains(const std::string &backend_kernel_name) const override;

  result get_build_result() const;

  result get_kernel(const std::string_view &name,
                    ur_kernel_handle_t &out) const;

private:
  hcf_object_id _source;
  code_format _format;
  code_object_state _state;
  result _build_status;

  ur_context_handle_t _ctx;
  ur_device_handle_t _dev;
  ur_program_handle_t _program;

  std::vector<std::string> _kernels;
  std::unordered_map<std::string_view, ur_kernel_handle_t> _kernel_handles;

  void load_kernel_handles();

  kernel_configuration::id_type _id;
};

class ur_sscp_executable_object final : public ur_executable_object {
public:
  ur_sscp_executable_object(ur_context_handle_t ctx, ur_device_handle_t dev,
                            hcf_object_id source,
                            const std::string &spirv_image,
                            const kernel_configuration &config);
  ~ur_sscp_executable_object() override = default;

  compilation_flow source_compilation_flow() const override;
  kernel_configuration::id_type configuration_id() const override;

private:
  kernel_configuration::id_type _id;
};

} // namespace hipsycl::rt

#endif
