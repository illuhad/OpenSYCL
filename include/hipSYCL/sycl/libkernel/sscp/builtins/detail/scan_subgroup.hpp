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

#ifndef HIPSYCL_SSCP_DETAIL_SUBGROUP_SCAN_BUILTINS_HPP
#define HIPSYCL_SSCP_DETAIL_SUBGROUP_SCAN_BUILTINS_HPP

#include "../core_typed.hpp"
#include "../subgroup.hpp"
#include "broadcast.hpp"
#include "shuffle.hpp"
#include "utils.hpp"

namespace hipsycl::libkernel::sscp {

template <typename T, typename BinaryOperation>
T sg_inclusive_scan(T x, BinaryOperation binary_op) {
  const __acpp_uint32 lid = __acpp_sscp_get_subgroup_local_id();
  const __acpp_uint32 lrange = __acpp_sscp_get_subgroup_max_size();
  const __acpp_uint64 subgroup_size = __acpp_sscp_get_subgroup_size();
  auto local_x = x;
  for (__acpp_int32 i = 1; i < lrange; i *= 2) {
    __acpp_uint32 next_id = lid - i;
    auto other_x = bit_cast<T>(
        sg_select(bit_cast<typename integer_type<T>::type>(local_x), next_id));
    if (next_id >= 0 && i <= lid)
      local_x = binary_op(local_x, other_x);
  }
  return local_x;
}

template <typename T, typename BinaryOperation>
T sg_exclusive_scan(T x, BinaryOperation binary_op, T init) {
  const __acpp_uint32 lid = __acpp_sscp_get_subgroup_local_id();
  const __acpp_uint64 subgroup_size = __acpp_sscp_get_subgroup_max_size();
  x = lid == 0 ? binary_op(x, init) : x;
  auto result_inclusive = sg_inclusive_scan(x, binary_op);
  auto result = bit_cast<T>(sg_select(
      bit_cast<typename integer_type<T>::type>(result_inclusive), lid - 1));
  result = lid % subgroup_size == 0 ? init : result;
  return result;
}

} // namespace hipsycl::libkernel::sscp

#endif