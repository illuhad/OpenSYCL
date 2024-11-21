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
#ifndef HIPSYCL_ADAPTIVITY_ENGINE_HPP
#define HIPSYCL_ADAPTIVITY_ENGINE_HPP


#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

namespace hipsycl {
namespace rt {

class kernel_adaptivity_engine {
public:
  kernel_adaptivity_engine(
    hcf_object_id hcf_object,
    std::string_view backend_kernel_name,
    const hcf_kernel_info* kernel_info,
    const glue::jit::cxx_argument_mapper& arg_mapper,
    const range<3>& num_groups,
    const range<3>& block_size,
    void** args,
    std::size_t* arg_sizes,
    std::size_t num_args,
    std::size_t local_mem_size);

  kernel_configuration::id_type
  finalize_binary_configuration(kernel_configuration &config);

  std::string select_image_and_kernels(std::vector<std::string>* kernel_names_out);
private:
  hcf_object_id _hcf;
  std::string_view _kernel_name;
  const hcf_kernel_info* _kernel_info;
  const glue::jit::cxx_argument_mapper& _arg_mapper;
  const range<3>& _num_groups;
  const range<3>& _block_size;
  void** _args;
  std::size_t* _arg_sizes;
  std::size_t _num_args;
  std::size_t _local_mem_size;

  int _adaptivity_level;
};

}
}

#endif
