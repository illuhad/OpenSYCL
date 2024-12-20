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

#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

namespace detail {
static inline unsigned int __lane_id() {
  return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}
} // namespace detail

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shl_i8(__acpp_int8 value, __acpp_uint32 delta) {
  return __acpp_sscp_sub_group_shl_i32(value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shl_i16(__acpp_int16 value, __acpp_uint32 delta) {
  return __acpp_sscp_sub_group_shl_i32(value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shl_i32(__acpp_int32 value, __acpp_uint32 delta) {
  auto sg_size = __acpp_sscp_get_subgroup_max_size();
  int self = detail::__lane_id();
  int index = (self + delta);
  index = (int)((self & (sg_size - 1)) + delta) > sg_size ? self : index;

  return __builtin_amdgcn_ds_bpermute(index << 2, value);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shl_i64(__acpp_int64 value, __acpp_uint32 delta) {
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  tmp[0] = __acpp_sscp_sub_group_shl_i32(tmp[0], delta);
  tmp[1] = __acpp_sscp_sub_group_shl_i32(tmp[1], delta);
  __acpp_int64 result =
      (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shr_i8(__acpp_int8 value, __acpp_uint32 delta) {
  return __acpp_sscp_sub_group_shr_i32(value, delta);
}
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shr_i16(__acpp_int16 value, __acpp_uint32 delta) {
  return __acpp_sscp_sub_group_shr_i32(value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shr_i32(__acpp_int32 value, __acpp_uint32 delta) {
  int self = detail::__lane_id();
  int width = __acpp_sscp_get_subgroup_max_size();
  int index = self - delta;
  index = (index < (self & ~(width - 1))) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, value);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shr_i64(__acpp_int64 value, __acpp_uint32 delta) {
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  tmp[0] = __acpp_sscp_sub_group_shr_i32(tmp[0], delta);
  tmp[1] = __acpp_sscp_sub_group_shr_i32(tmp[1], delta);
  __acpp_int64 result =
      (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_permute_i8(__acpp_int8 value, __acpp_int32 mask) {
  return __acpp_sscp_sub_group_permute_i32(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_permute_i16(__acpp_int16 value, __acpp_int32 mask) {
  return __acpp_sscp_sub_group_permute_i32(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_permute_i32(__acpp_int32 value, __acpp_int32 mask) {
  int self = detail::__lane_id();
  int index = self ^ mask;
  return __builtin_amdgcn_ds_bpermute(index << 2, value);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_permute_i64(__acpp_int64 value, __acpp_int32 mask) {
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  tmp[0] = __acpp_sscp_sub_group_permute_i32(tmp[0], mask);
  tmp[1] = __acpp_sscp_sub_group_permute_i32(tmp[1], mask);
  __acpp_int64 result =
      (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_select_i8(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i32(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_select_i16(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i32(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_select_i32(__acpp_int32 value, __acpp_int32 id) {
  int max_subgroup_size = __acpp_sscp_get_subgroup_max_size();
  int index = id % max_subgroup_size;
  return __builtin_amdgcn_ds_bpermute(index << 2, value);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_select_i64(__acpp_int64 value, __acpp_int32 id) {
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  tmp[0] = __acpp_sscp_sub_group_select_i32(tmp[0], id);
  tmp[1] = __acpp_sscp_sub_group_select_i32(tmp[1], id);
  __acpp_int64 result =
      (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}
