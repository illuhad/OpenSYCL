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
#include "hipSYCL/sycl/libkernel/sscp/builtins/relational.hpp"

#include <cmath>

#define HIPSYCL_SSCP_MAP_HOST_REL_BUILTIN(name)                                \
  HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_##name##_f32(float x) {        \
    return __builtin_##name(x);                                                \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_##name##_f64(double x) {       \
    return __builtin_##name(x);                                                \
  }

HIPSYCL_SSCP_MAP_HOST_REL_BUILTIN(isnan)

HIPSYCL_SSCP_MAP_HOST_REL_BUILTIN(isinf)

HIPSYCL_SSCP_MAP_HOST_REL_BUILTIN(isfinite)

HIPSYCL_SSCP_MAP_HOST_REL_BUILTIN(isnormal)

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_signbit_f32(float x) {
  return std::signbit(x);
}
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_signbit_f64(double x) {
  return std::signbit(x);
}
