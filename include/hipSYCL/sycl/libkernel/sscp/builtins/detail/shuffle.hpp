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
#ifndef HIPSYCL_SSCP_DETAIL_SHUFFLE_BUILTINS_HPP
#define HIPSYCL_SSCP_DETAIL_SHUFFLE_BUILTINS_HPP

#include "../builtin_config.hpp"
#include "../shuffle.hpp"

namespace hipsycl::libkernel::sscp {

template <typename T> inline T sg_select(T, __acpp_int32) = delete;

template <> inline __acpp_int8 sg_select<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i8(value, id);
}

template <> inline __acpp_int16 sg_select<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i16(value, id);
}

template <> inline __acpp_int32 sg_select<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i32(value, id);
}

template <> inline __acpp_int64 sg_select<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i64(value, id);
}

template <typename T> T inline sg_shift_left(T, __acpp_int32) = delete;

template <> __acpp_int8 inline sg_shift_left<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i8(value, id);
}

template <> __acpp_int16 inline sg_shift_left<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i16(value, id);
}

template <> __acpp_int32 inline sg_shift_left<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i32(value, id);
}

template <> __acpp_int64 inline sg_shift_left<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shl_i64(value, id);
}

template <typename T> T inline sg_shift_right(T, __acpp_int32) = delete;

template <> __acpp_int8 inline sg_shift_right<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i8(value, id);
}

template <> __acpp_int16 inline sg_shift_right<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i16(value, id);
}

template <> __acpp_int32 inline sg_shift_right<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i32(value, id);
}

template <> __acpp_int64 inline sg_shift_right<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_shr_i64(value, id);
}

} // namespace hipsycl::libkernel::sscp

#endif