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
#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
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

template <typename OutType, typename BinaryOperation> 
OutType __acpp_group_exclusive_scan_cudalike_impl(OutType x,BinaryOperation op, OutType init){
  ACPP_CUDALIKE_SHMEM_ATTRIBUTE OutType  shrd_mem[32];
  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32        sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32       num_subgroups = (wg_size+max_sg_size-1)/max_sg_size;
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;

  const bool                last_item_in_sg = (wg_lid%sg_size) == (sg_size-1);
  OutType sg_scan_result = __acpp_subgroup_exclusive_scan_impl(x, op, init);

  if(last_item_in_sg){
    shrd_mem[subgroup_id] = op(sg_scan_result,x);
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  if(subgroup_id == 0){
    shrd_mem[wg_lid] =  __acpp_subgroup_exclusive_scan_impl(shrd_mem[wg_lid], op, init);
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  return subgroup_id > 0 ? op(shrd_mem[subgroup_id], sg_scan_result) : sg_scan_result;
}


template <typename OutType, typename BinaryOperation> 
OutType __acpp_group_exclusive_scan_host_impl(OutType x,BinaryOperation op, OutType init){

  OutType* shrd_mem = static_cast<OutType*>(__acpp_sscp_host_get_internal_local_memory());

  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32        sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32       num_subgroups = (wg_size+max_sg_size-1)/max_sg_size;
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;

  const bool                last_item_in_sg = (wg_lid%sg_size) == (sg_size-1);
  
  if(wg_lid+1 < wg_size){
    shrd_mem[wg_lid+1] = x;
  }else{
    shrd_mem[0] = init;
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  OutType local_x = shrd_mem[wg_lid];
  OutType other_x;
  //TODO: Here we can just call the host inclusive scan
  for (__acpp_int32 i = 1; i < wg_size; i *= 2) {  
    __acpp_uint32 next_id = wg_lid -i;
    bool is_nextid_valid = next_id >= 0 && i <= wg_lid;

    if (is_nextid_valid){
      other_x=shrd_mem[next_id]; 
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

    if (is_nextid_valid){
      local_x = op(local_x, other_x); 
      shrd_mem[wg_lid] = local_x;
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  }
  return local_x; 
}

// This is currently only used by the spirv backend,
// TODO: Should be replaced by spirv builtins
template <typename OutType, typename BinaryOperation> 
OutType __acpp_group_exclusive_scan_impl(OutType x,BinaryOperation op, OutType init){
  const constexpr __acpp_uint32 shmem_array_length = 32;
  ACPP_CUDALIKE_SHMEM_ATTRIBUTE OutType shrd_mem[shmem_array_length+1];

  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32        sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32       num_subgroups = (wg_size+max_sg_size-1)/max_sg_size;
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;

  const bool                last_item_in_sg = (wg_lid%sg_size) == (sg_size-1);


  OutType sg_scan_result = __acpp_subgroup_exclusive_scan_impl(x, op, init);

  for(int i = 0; i < (num_subgroups-1+shmem_array_length)/shmem_array_length; i++){
    __acpp_uint32 first_active_thread = i*num_subgroups*max_sg_size;
    __acpp_uint32 last_active_thread  = (i+1)*num_subgroups*max_sg_size;
    last_active_thread  =  last_active_thread > wg_size ? wg_size : last_active_thread;
    __acpp_uint32 relative_thread_id = wg_lid - first_active_thread;
    if(subgroup_id/shmem_array_length == i){
      if(last_item_in_sg){
      // return 1212;

        // return sg_scan_result;
        shrd_mem[subgroup_id%shmem_array_length] = op(sg_scan_result, x);
      }
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
    // First shmem_array_length number of threads exclusive scan in shared memory
      auto local_x = shrd_mem[relative_thread_id];
      for (__acpp_int32 j = 1; j < shmem_array_length; j *= 2) {  
        __acpp_int32 next_id = relative_thread_id -j; 
        if (next_id >= 0 && j <= relative_thread_id){
          if(relative_thread_id < shmem_array_length){
            auto other_x= shrd_mem[next_id];
            local_x = op(local_x, other_x);
            shrd_mem[relative_thread_id] = local_x;
          }
        }
        __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
      }
      __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

    if(subgroup_id > 0){
      auto current_segment_update = shrd_mem[(subgroup_id%shmem_array_length)-1];
      sg_scan_result = op(current_segment_update, sg_scan_result);
    }
    if(i>0){
      sg_scan_result = op(shrd_mem[shmem_array_length], sg_scan_result);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
    shrd_mem[shmem_array_length] = sg_scan_result;
  }
  return sg_scan_result;
}

#endif