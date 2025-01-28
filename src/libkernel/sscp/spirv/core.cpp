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
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"


__acpp_uint64 __spirv_BuiltInLocalInvocationId(int);
__acpp_uint64 __spirv_BuiltInWorkgroupId(int);
__acpp_uint64 __spirv_BuiltInWorkgroupSize(int);
__acpp_uint64 __spirv_BuiltInNumWorkgroups(int);

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x() {
  return __spirv_BuiltInLocalInvocationId(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y() {
  return __spirv_BuiltInLocalInvocationId(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z() {
  return __spirv_BuiltInLocalInvocationId(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x() {
  return __spirv_BuiltInWorkgroupId(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y() {
  return __spirv_BuiltInWorkgroupId(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z() {
  return __spirv_BuiltInWorkgroupId(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x() {
  return __spirv_BuiltInWorkgroupSize(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y() {
  return __spirv_BuiltInWorkgroupSize(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z() {
  return __spirv_BuiltInWorkgroupSize(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x() {
  return __spirv_BuiltInNumWorkgroups(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y() {
  return __spirv_BuiltInNumWorkgroups(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z() {
  return __spirv_BuiltInNumWorkgroups(2);
}
