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
#ifndef HIPSYCL_OMP_CODE_OBJECT_HPP
#define HIPSYCL_OMP_CODE_OBJECT_HPP

#include <string>
#include <vector>

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/util.hpp"


namespace hipsycl {
namespace rt {

class omp_sscp_executable_object : public code_object {
public:
  // The kernel argument struct providing work-group information.
  struct work_group_info {
    work_group_info(rt::range<3> num_groups, rt::id<3> group_id,
                    rt::range<3> local_size, void *local_memory,
                    void *internal_local_memory)
        : _num_groups(num_groups), _group_id(group_id), _local_size(local_size),
          _local_memory(local_memory),
          _internal_local_memory(internal_local_memory) {}

    rt::range<3> _num_groups;
    rt::range<3> _group_id;
    rt::range<3> _local_size;
    void* _local_memory;
    void* _internal_local_memory;
  };

  using omp_sscp_kernel = void(const work_group_info *, void **);

  omp_sscp_executable_object(const std::string &shared_lib_path,
                             hcf_object_id hcf_source,
                             const std::vector<std::string> &kernel_names,
                             const kernel_configuration &config);

  virtual ~omp_sscp_executable_object();

  virtual result get_build_result() const;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;
  virtual compilation_flow source_compilation_flow() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;
  virtual bool contains(const std::string &backend_kernel_name) const override;

  virtual void *get_module() const;
  virtual omp_sscp_kernel *get_kernel(std::string_view backend_kernel_name) const;

private:
  result build(const std::string &source, const std::vector<std::string> &kernel_names);

  hcf_object_id _hcf;
  kernel_configuration::id_type _id;
  std::string _kernel_cache_path;

  result _build_result;
  void *_module;

  std::vector<std::string> _kernel_names;
  std::unordered_map<std::string_view, omp_sscp_kernel*> _kernels;
};

} // namespace rt
} // namespace hipsycl

#endif
