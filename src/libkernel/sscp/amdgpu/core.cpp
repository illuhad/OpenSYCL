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


extern "C" __acpp_uint64 __ockl_get_global_offset(unsigned);
extern "C" __acpp_uint64 __ockl_get_global_id(unsigned);
extern "C" __acpp_uint64 __ockl_get_local_id(unsigned);
extern "C" __acpp_uint64 __ockl_get_group_id(unsigned);
extern "C" __acpp_uint64 __ockl_get_global_size(unsigned);
extern "C" __acpp_uint64 __ockl_get_local_size(unsigned);
extern "C" __acpp_uint64 __ockl_get_num_groups(unsigned);
extern "C" __acpp_uint32 __ockl_get_work_dim(unsigned);
extern "C" __acpp_uint64 __ockl_get_enqueued_local_size(unsigned);
extern "C" __acpp_uint64 __ockl_get_global_linear_id(unsigned);
extern "C" __acpp_uint64 __ockl_get_local_linear_id(unsigned);

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x() {
  return __ockl_get_local_id(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y() {
  return __ockl_get_local_id(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z() {
  return __ockl_get_local_id(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x() {
  return __ockl_get_group_id(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y() {
  return __ockl_get_group_id(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z() {
  return __ockl_get_group_id(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x() {
  return __ockl_get_local_size(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y() {
  return __ockl_get_local_size(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z() {
  return __ockl_get_local_size(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x() {
  return __ockl_get_num_groups(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y() {
  return __ockl_get_num_groups(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z() {
  return __ockl_get_num_groups(2);
}
