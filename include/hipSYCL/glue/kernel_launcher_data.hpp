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
#ifndef HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_DATA_HPP
#define HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_DATA_HPP

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/kernel_type.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <vector>
#include <string_view>
#include <functional>

namespace hipsycl {

namespace sycl {
class interop_handle;
}

namespace rt {
class dag_node;
class kernel_configuration;
class backend_kernel_launch_capabilities;
class hcf_kernel_info;
class kernel_operation;
}

namespace glue {

// some kernel launchers use this structure to store
// their data. This avoids having to malloc kernel launcher objects.
//
// This is currently only used by the SSCP kernel launcher. In the future
// we may want to investigate unifying the other kernel launchers into this
// data structure as well, which would allow to share data between them.
struct kernel_launcher_data {
  // will be configured during launch
  void* params = nullptr;

  // data potentially shared by multiple launchers
  rt::kernel_type type;
  // has to be mutable e.g. to initialize accessors (embedded pointers)
  mutable std::vector<uint8_t> kernel_args;
  rt::range<3> global_size; // <- indices must be flipped
  rt::range<3> group_size; // <- indices must be flipped
  unsigned local_mem_size;
  // In case the launch is a custom operation
  std::function<void(rt::kernel_operation*, sycl::interop_handle&)> custom_op;

  using invoker_function_t = rt::result (*)(
      const kernel_launcher_data &launch_config, rt::dag_node *node,
      const rt::kernel_configuration &kernel_config,
      const rt::backend_kernel_launch_capabilities &launch_capabilities,
      void *backend_params);

  // compilation flow-specific fields
  unsigned long long sscp_hcf_object_id;
  const char* sscp_kernel_id = nullptr;
  const rt::hcf_kernel_info* kernel_info = nullptr;
  invoker_function_t sscp_invoker;
};



}
}

#endif
