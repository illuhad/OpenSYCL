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
#ifndef HIPSYCL_SSCP_SCAN_EXCLUSIVE_BUILTINS_HPP
#define HIPSYCL_SSCP_SCAN_EXCLUSIVE_BUILTINS_HPP

#include "builtin_config.hpp"
#include "utils.hpp"
#include "core_typed.hpp"
#include "barrier.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "scan_inclusive.hpp"


#define GROUP_DECL(size,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_work_group_exclusive_scan_##size(__acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init); \

GROUP_DECL(i8, int8);
GROUP_DECL(i16, int16);
GROUP_DECL(i32, int32);
GROUP_DECL(i64, int64);

GROUP_DECL(u8,  uint8);
GROUP_DECL(u16, uint16);
GROUP_DECL(u32, uint32);
GROUP_DECL(u64, uint64);

GROUP_DECL(f16, f16);
GROUP_DECL(f32, f32);
GROUP_DECL(f64, f64);

#undef GROUP_DECL

#define SUBGROUP_DECL(size,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_exclusive_scan_##size(__acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init); \

SUBGROUP_DECL(i8, int8);
SUBGROUP_DECL(i16, int16);
SUBGROUP_DECL(i32, int32);
SUBGROUP_DECL(i64, int64);

SUBGROUP_DECL(u8,  uint8);
SUBGROUP_DECL(u16, uint16);
SUBGROUP_DECL(u32, uint32);
SUBGROUP_DECL(u64, uint64);

SUBGROUP_DECL(f16, f16);
SUBGROUP_DECL(f32, f32);
SUBGROUP_DECL(f64, f64);

#undef SUBGROUP_DECL

template <typename T, typename BinaryOperation> 
T __acpp_subgroup_exclusive_scan_impl(T x, BinaryOperation binary_op, T init) { 
  const __acpp_uint32 lid = __acpp_sscp_get_subgroup_local_id(); 
  const __acpp_uint64 subgroup_size = __acpp_sscp_get_subgroup_max_size();
  x = lid == 0 ? binary_op(x, init) : x;
  auto result_inclusive = __acpp_subgroup_inclusive_scan_impl(x, binary_op);
  auto result = bit_cast<T>(__acpp_sscp_sub_group_select(bit_cast<typename integer_type<T>::type>(result_inclusive), lid-1));
  result = lid%subgroup_size == 0 ? init : result; 
  return result; 
} 


#endif