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

template <>
__acpp_int8 hipsycl::libkernel::sscp::sg_select<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i8(value, id);
}

template <>
__acpp_int16 hipsycl::libkernel::sscp::sg_select<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i16(value, id);
}

template <>
__acpp_int32 hipsycl::libkernel::sscp::sg_select<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i32(value, id);
}

template <>
__acpp_int64 hipsycl::libkernel::sscp::sg_select<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i64(value, id);
}

template <>
__acpp_int8 hipsycl::libkernel::sscp::sg_shift_left<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i8(value, id);
}

template <>
__acpp_int16 hipsycl::libkernel::sscp::sg_shift_left<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i16(value, id);
}

template <>
__acpp_int32 hipsycl::libkernel::sscp::sg_shift_left<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i32(value, id);
}

template <>
__acpp_int64 hipsycl::libkernel::sscp::sg_shift_left<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i64(value, id);
}


template <>
__acpp_int8 hipsycl::libkernel::sscp::sg_shift_right<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i8(value, id);
}

template <>
__acpp_int16 hipsycl::libkernel::sscp::sg_shift_right<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i16(value, id);
}

template <>
__acpp_int32 hipsycl::libkernel::sscp::sg_shift_right<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i32(value, id);
}

template <>
__acpp_int64 hipsycl::libkernel::sscp::sg_shift_right<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i64(value, id);
}

constexpr unsigned int FULL_MASK = 0xffffffff;

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
  // __acpp_uint32 mask = get_active_mask();
  return __nvvm_shfl_sync_down_i32(FULL_MASK, value, delta, 0x1f);
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
  // __acpp_uint32 mask = get_active_mask();
  return __nvvm_shfl_sync_up_i32(FULL_MASK, value, delta, 0);
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
  // __acpp_uint32 active_thread_mask = get_active_mask();
  return __nvvm_shfl_sync_bfly_i32(FULL_MASK, value, mask, 0x1f);
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
  return __nvvm_shfl_sync_idx_i32(FULL_MASK, value, id, 31);
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
