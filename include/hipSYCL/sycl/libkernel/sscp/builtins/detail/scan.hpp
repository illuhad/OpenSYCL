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

#ifndef HIPSYCL_SSCP_DETAIL_SCAN_BUILTINS_HPP
#define HIPSYCL_SSCP_DETAIL_SCAN_BUILTINS_HPP

#include "../core_typed.hpp"
#include "../subgroup.hpp"
#include "../utils.hpp"
#include "broadcast.hpp"
#include "shuffle.hpp"

// subgroup_inclusive_scan
template <typename T, typename BinaryOperation>
T subgroup_inclusive_scan(T x, BinaryOperation binary_op) {
  const __acpp_uint32 lid = __acpp_sscp_get_subgroup_local_id();
  const __acpp_uint32 lrange = __acpp_sscp_get_subgroup_max_size();
  const __acpp_uint64 subgroup_size = __acpp_sscp_get_subgroup_size();
  auto local_x = x;
  for (__acpp_int32 i = 1; i < lrange; i *= 2) {
    __acpp_uint32 next_id = lid - i;
    auto other_x = bit_cast<T>(
        __acpp_sscp_sub_group_select(bit_cast<typename integer_type<T>::type>(local_x), next_id));
    if (next_id >= 0 && i <= lid)
      local_x = binary_op(local_x, other_x);
  }
  return local_x;
}

// subgroup_exclusive_scan
template <typename T, typename BinaryOperation>
T subgroup_exclusive_scan(T x, BinaryOperation binary_op, T init) {
  const __acpp_uint32 lid = __acpp_sscp_get_subgroup_local_id();
  const __acpp_uint64 subgroup_size = __acpp_sscp_get_subgroup_max_size();
  x = lid == 0 ? binary_op(x, init) : x;
  auto result_inclusive = subgroup_inclusive_scan(x, binary_op);
  auto result = bit_cast<T>(__acpp_sscp_sub_group_select(
      bit_cast<typename integer_type<T>::type>(result_inclusive), lid - 1));
  result = lid % subgroup_size == 0 ? init : result;
  return result;
}

template <int SharedMemorySize, bool ExclusiveScan, typename OutType, typename MemoryType,
          typename BinaryOperation>
OutType generic_scan_impl(OutType x, BinaryOperation op, MemoryType shrd_mem, OutType init = 0) {

  // The last element of the shared memory is used to store the total sum for exclusive scans.
  const size_t shmem_array_length = SharedMemorySize - 1;

  const __acpp_uint32 wg_lid = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32 wg_size = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32 max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32 sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32 num_subgroups = (wg_size + max_sg_size - 1) / max_sg_size;
  const __acpp_uint32 subgroup_id = wg_lid / max_sg_size;

  const bool last_item_in_sg = (wg_lid % sg_size) == (sg_size - 1);
  OutType sg_scan_result;
  if constexpr (ExclusiveScan) {
    sg_scan_result = subgroup_exclusive_scan(x, op, init);
  } else {
    sg_scan_result = subgroup_inclusive_scan(x, op);
  }

  for (int i = 0; i < (num_subgroups - 1 + shmem_array_length) / shmem_array_length; i++) {
    __acpp_uint32 first_active_thread = i * num_subgroups * max_sg_size;
    __acpp_uint32 last_active_thread = (i + 1) * num_subgroups * max_sg_size;
    last_active_thread = last_active_thread > wg_size ? wg_size : last_active_thread;
    __acpp_uint32 relative_thread_id = wg_lid - first_active_thread;
    if (subgroup_id / shmem_array_length == i) {
      if (last_item_in_sg) {

        if constexpr (ExclusiveScan) {
          shrd_mem[subgroup_id % shmem_array_length] = op(sg_scan_result, x);
        } else {
          shrd_mem[subgroup_id % shmem_array_length] = sg_scan_result;
        }
      }
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);
    // First shmem_array_length number of threads exclusive scan in shared memory
    auto local_x = shrd_mem[relative_thread_id];
    for (__acpp_int32 j = 1; j < shmem_array_length; j *= 2) {
      __acpp_int32 next_id = relative_thread_id - j;
      if (next_id >= 0 && j <= relative_thread_id) {
        if (relative_thread_id < shmem_array_length) {
          auto other_x = shrd_mem[next_id];
          local_x = op(local_x, other_x);
          shrd_mem[relative_thread_id] = local_x;
        }
      }
      __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                     __acpp_sscp_memory_order::relaxed);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);

    if (subgroup_id > 0) {
      auto current_segment_update = shrd_mem[(subgroup_id % shmem_array_length) - 1];
      sg_scan_result = op(current_segment_update, sg_scan_result);
    }
    if (i > 0) {
      sg_scan_result = op(shrd_mem[shmem_array_length], sg_scan_result);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                   __acpp_sscp_memory_order::relaxed);
    shrd_mem[shmem_array_length] = sg_scan_result;
  }
  return sg_scan_result;
}

template <int SharedMemorySize, bool ExclusiveScan, typename OutType, typename MemoryType,
          typename BinaryOperation>
OutType hiplike_scan_impl(OutType x, BinaryOperation op, MemoryType shrd_mem, OutType init = 0) {

  const __acpp_uint32 wg_lid = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32 max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32 sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32 subgroup_id = wg_lid / max_sg_size;

  const bool last_item_in_sg = (wg_lid % sg_size) == (sg_size - 1);
  OutType sg_scan_result;
  if constexpr (ExclusiveScan) {
    sg_scan_result = subgroup_exclusive_scan(x, op, init);
  } else {
    sg_scan_result = subgroup_inclusive_scan(x, op);
  }

  if (last_item_in_sg) {
    if constexpr (ExclusiveScan) {
      shrd_mem[subgroup_id] = op(sg_scan_result, x);
    } else {
      shrd_mem[subgroup_id] = sg_scan_result;
    }
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                 __acpp_sscp_memory_order::relaxed);
  if (subgroup_id == 0) {
    shrd_mem[wg_lid] = subgroup_inclusive_scan(shrd_mem[wg_lid], op);
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                 __acpp_sscp_memory_order::relaxed);
  return subgroup_id > 0 ? op(shrd_mem[subgroup_id - 1], sg_scan_result) : sg_scan_result;
}

template <bool ExclusiveScan, typename OutType, typename MemoryType, typename BinaryOperation>
OutType host_scan_impl(OutType x, BinaryOperation op, MemoryType shrd_mem, OutType init = 0) {
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

#endif