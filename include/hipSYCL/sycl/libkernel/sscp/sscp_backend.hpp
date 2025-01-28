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

#ifndef ACPP_LIBKERNEL_SSCP_BACKEND_HPP
#define ACPP_LIBKERNEL_SSCP_BACKEND_HPP

// Expose SSCP only when we are targeting it, and we
// are not in a device pass for an SMCP backend. Otherwise,
// it gets  difficult to distinguish the proper SSCP device code
// path since SSCP outlining happens only in the host pass.
#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__) &&                            \
    !defined(ACPP_LIBKERNEL_DEVICE_PASS)

 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP 1
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP 1

 #ifndef ACPP_LIBKERNEL_DEVICE_PASS
  #define ACPP_LIBKERNEL_DEVICE_PASS
 #endif

 #ifdef ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
  #undef ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #endif
 #define ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 1

 #define __acpp_if_target_host(...)                                         \
  if (__acpp_sscp_is_host) {                                                \
    __VA_ARGS__                                                                \
  }
 #define __acpp_if_target_device(...)                                       \
  if (__acpp_sscp_is_device) {                                              \
    __VA_ARGS__                                                                \
  }


 // TODO: Do we need those still?
 #ifndef ACPP_UNIVERSAL_TARGET
  #define ACPP_UNIVERSAL_TARGET
 #endif

 #ifndef ACPP_KERNEL_TARGET
  #define ACPP_KERNEL_TARGET
 #endif

 #ifndef ACPP_HOST_TARGET
  #define ACPP_HOST_TARGET
 #endif

 #include "hipSYCL/glue/llvm-sscp/ir_constants.hpp"
 #include "builtins/core.hpp"
 #include "builtins/linear_id.hpp"
 
#else
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP 0
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP 0
#endif



#endif
