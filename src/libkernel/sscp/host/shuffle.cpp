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

template <>
__acpp_int8 __acpp_sscp_sub_group_select<__acpp_int8>(__acpp_int8 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i8(value, id);
}

template <>
__acpp_int16 __acpp_sscp_sub_group_select<__acpp_int16>(__acpp_int16 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i16(value, id);
}

template <>
__acpp_int32 __acpp_sscp_sub_group_select<__acpp_int32>(__acpp_int32 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i32(value, id);
}

template <>
__acpp_int64 __acpp_sscp_sub_group_select<__acpp_int64>(__acpp_int64 value, __acpp_int32 id) {
  return __acpp_sscp_sub_group_select_i64(value, id);
}

#define SUBGROUP_SIZE_ONE_SHUFLLE(int_size, direction)                                             \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##int_size __acpp_sscp_sub_group_##direction##_i##int_size(__acpp_int##int_size value, \
                                                                       __acpp_uint32 delta) {      \
    return delta == 0 ? value : 0;                                                                 \
  }

SUBGROUP_SIZE_ONE_SHUFLLE(8, shl)
SUBGROUP_SIZE_ONE_SHUFLLE(16, shl)
SUBGROUP_SIZE_ONE_SHUFLLE(32, shl)
SUBGROUP_SIZE_ONE_SHUFLLE(64, shl)
SUBGROUP_SIZE_ONE_SHUFLLE(8, shr)
SUBGROUP_SIZE_ONE_SHUFLLE(16, shr)
SUBGROUP_SIZE_ONE_SHUFLLE(32, shr)
SUBGROUP_SIZE_ONE_SHUFLLE(64, shr)

#define SUBGROUP_SIZE_ONE_PERMUTE(int_size)                                                        \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##int_size __acpp_sscp_sub_group_permute_i##int_size(__acpp_int##int_size value,       \
                                                                 __acpp_int32 mask) {              \
    return mask xor 0 ? value : 0;                                                                 \
  }

SUBGROUP_SIZE_ONE_PERMUTE(8)
SUBGROUP_SIZE_ONE_PERMUTE(16)
SUBGROUP_SIZE_ONE_PERMUTE(32)
SUBGROUP_SIZE_ONE_PERMUTE(64)

#define SUBGROUP_SIZE_ONE_SELECT(int_size)                                                         \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_int##int_size __acpp_sscp_sub_group_select_i##int_size(__acpp_int##int_size value,        \
                                                                __acpp_int32 mask) {               \
    return mask == 0 ? value : 0;                                                                  \
  }

SUBGROUP_SIZE_ONE_SELECT(8)
SUBGROUP_SIZE_ONE_SELECT(16)
SUBGROUP_SIZE_ONE_SELECT(32)
SUBGROUP_SIZE_ONE_SELECT(64)
