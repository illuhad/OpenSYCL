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

#include "s1_ir_constants.hpp"
#include <cstdlib>


#ifndef ACPP_SSCP_HCF_REGISTRATION_HPP
#define ACPP_SSCP_HCF_REGISTRATION_HPP

// These functions are defined in the AdaptiveCpp runtime (kernel_cache.cpp)
extern "C" void __acpp_register_hcf(const char* hcf, std::size_t size);
extern "C" void __acpp_unregister_hcf(std::size_t hcf_object_id);

namespace hipsycl::glue::sscp {

static const char* get_local_hcf_object() {
  return __acpp_local_sscp_hcf_content;
}
static std::size_t get_local_hcf_size() {
  return __acpp_local_sscp_hcf_object_size;
}
static std::size_t get_local_hcf_id() {
  return __acpp_local_sscp_hcf_object_id;
}

struct static_hcf_registration {
public:
  __attribute__((internal_linkage))
  static_hcf_registration() {
    __acpp_register_hcf(get_local_hcf_object(), get_local_hcf_size());
  }

  __attribute__((internal_linkage))
  ~static_hcf_registration() {
    __acpp_unregister_hcf(get_local_hcf_id());
  }
};

static static_hcf_registration __acpp_register_sscp_hcf_object;
}


#endif
