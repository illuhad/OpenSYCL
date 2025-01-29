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

#ifndef HIPSYCL_SSCP_DETAIL_HOST_SCAN_BUILTINS_HPP
#define HIPSYCL_SSCP_DETAIL_HOST_SCAN_BUILTINS_HPP

#include "../core_typed_linear_id.hpp"
#include "../subgroup.hpp"
#include "broadcast.hpp"
#include "scan_subgroup.hpp"
#include "shuffle.hpp"
#include "utils.hpp"

namespace hipsycl::libkernel::sscp {

template <bool ExclusiveScan, typename OutType, typename MemoryType, typename BinaryOperation>
OutType wg_host_scan(OutType x, BinaryOperation op, MemoryType shrd_mem, OutType init = 0) {
  const __acpp_uint32 wg_lid = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32 wg_size = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32 max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32 sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32 num_subgroups = (wg_size + max_sg_size - 1) / max_sg_size;
  const __acpp_uint32 subgroup_id = wg_lid / max_sg_size;

  const bool last_item_in_sg = (wg_lid % sg_size) == (sg_size - 1);
  OutType local_x;
  if constexpr (ExclusiveScan) {
    if (wg_lid + 1 < wg_size) {
      shrd_mem[wg_lid + 1] = x;
    } else {
      shrd_mem[0] = init;
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);
    local_x = shrd_mem[wg_lid];
  } else {
    shrd_mem[wg_lid] = x;
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);
    local_x = x;
  }

  OutType other_x;
  // TODO: Here we can just call the host inclusive scan
  for (__acpp_int32 i = 1; i < wg_size; i *= 2) {
    __acpp_int32 next_id = wg_lid - i;
    bool is_nextid_valid = (next_id >= 0) && (i <= wg_lid);

    if (is_nextid_valid) {
      other_x = shrd_mem[next_id];
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);

    if (is_nextid_valid) {
      local_x = op(local_x, other_x);
      shrd_mem[wg_lid] = local_x;
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);
  }
  return local_x;
}
} // namespace hipsycl::libkernel::sscp

#endif