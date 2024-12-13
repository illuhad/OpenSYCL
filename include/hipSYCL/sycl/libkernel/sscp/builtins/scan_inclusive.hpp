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
#ifndef HIPSYCL_SSCP_SCAN_INCLUSIVE_BUILTINS_HPP
#define HIPSYCL_SSCP_SCAN_INCLUSIVE_BUILTINS_HPP

#include "builtin_config.hpp"
#include "utils.hpp"
#include "scan_exclusive.hpp"
#include "core_typed.hpp"
#include "barrier.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_inclusive_scan_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_inclusive_scan_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_inclusive_scan_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_inclusive_scan_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_work_group_inclusive_scan_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_work_group_inclusive_scan_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_work_group_inclusive_scan_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_work_group_inclusive_scan_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_work_group_inclusive_scan_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_work_group_inclusive_scan_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_work_group_inclusive_scan_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_inclusive_scan_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_inclusive_scan_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_inclusive_scan_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_inclusive_scan_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_sub_group_inclusive_scan_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_sub_group_inclusive_scan_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_sub_group_inclusive_scan_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_sub_group_inclusive_scan_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_sub_group_inclusive_scan_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_sub_group_inclusive_scan_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_sub_group_inclusive_scan_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);


template <typename T, typename BinaryOperation> 
T __acpp_subgroup_inclusive_scan_impl(T x, BinaryOperation binary_op) { 
  const __acpp_uint32       lid        = __acpp_sscp_get_subgroup_local_id(); 
  const __acpp_uint32       lrange     = __acpp_sscp_get_subgroup_max_size(); 
  const __acpp_uint64       subgroup_size = __acpp_sscp_get_subgroup_size(); 
  auto local_x = x; 
  for (__acpp_int32 i = 1; i < lrange; i *= 2) {  
    __acpp_uint32 next_id = lid -i; 
    auto other_x=bit_cast<T>(__acpp_sscp_sub_group_select(bit_cast<typename integer_type<T>::type>(local_x), next_id)); 
    if (next_id >= 0 && i <= lid) 
        local_x = binary_op(local_x, other_x); 
    } 
  return local_x; 
}

template <typename OutType, typename BinaryOperation> 
OutType __acpp_group_inclusive_scan_cudalike_impl(OutType x,BinaryOperation op){
  ACPP_CUDALIKE_SHMEM_ATTRIBUTE OutType  shrd_mem[32];
  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32        sg_size = __acpp_sscp_get_subgroup_size();
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;
  const bool                last_item_in_sg = (wg_lid%sg_size) == (sg_size-1);

  OutType sg_scan_result = __acpp_subgroup_inclusive_scan_impl(x, op);

  if(last_item_in_sg){
    shrd_mem[subgroup_id] = sg_scan_result;
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  if(subgroup_id == 0){
    shrd_mem[wg_lid] = __acpp_subgroup_inclusive_scan_impl(shrd_mem[wg_lid], op);
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  OutType result = subgroup_id > 0 ? op(shrd_mem[subgroup_id-1], sg_scan_result) : sg_scan_result;
  //WAR dependency in case scan is called in a loop
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  return result;
}

template <typename OutType, typename BinaryOperation> 
OutType __acpp_group_inclusive_scan_host_impl(OutType x,BinaryOperation op){

  OutType* shrd_mem = static_cast<OutType*>(__acpp_sscp_host_get_internal_local_memory());

  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32        sg_size = __acpp_sscp_get_subgroup_size();

  const __acpp_uint32       num_subgroups = (wg_size+max_sg_size-1)/max_sg_size;
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;

  const bool                last_item_in_sg = (wg_lid%sg_size) == (sg_size-1);
  
  shrd_mem[wg_lid] = x;

  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  OutType local_x = x;
  OutType other_x;
  for (__acpp_int32 i = 1; i < wg_size; i *= 2) {  
    __acpp_int32 next_id = wg_lid -i;
    bool is_nextid_valid = (next_id >= 0) && (i <= wg_lid);

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

template <typename OutType, typename BinaryOperation> 
OutType __acpp_group_inclusive_scan_impl(OutType x,BinaryOperation op){
  const constexpr __acpp_uint32 shmem_array_length = 32;
  ACPP_CUDALIKE_SHMEM_ATTRIBUTE OutType  shrd_mem[shmem_array_length+1];

  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_int32        sg_size = __acpp_sscp_get_subgroup_size();
  //const __acpp_int32 first_sg_size = __acpp_sscp_work_group_broadcast(0, sg_size);

  const __acpp_uint32       num_subgroups = (wg_size+max_sg_size-1)/max_sg_size;
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;

  const bool                last_item_in_sg = (wg_lid%sg_size) == (sg_size-1);


  // OutType sg_scan_result = __acpp_subgroup_inclusive_scan_impl(x, op, sg_size);
  OutType sg_scan_result = __acpp_subgroup_inclusive_scan_impl(x, op);
  for(int i = 0; i < (num_subgroups-1+shmem_array_length)/shmem_array_length; i++){
    __acpp_uint32 first_active_thread = i*num_subgroups*max_sg_size;
    __acpp_uint32 last_active_thread  = (i+1)*num_subgroups*max_sg_size;
    last_active_thread  =  last_active_thread > wg_size ? wg_size : last_active_thread;
    __acpp_uint32 relative_thread_id = wg_lid - first_active_thread;
    if(subgroup_id/shmem_array_length == i){
      if(last_item_in_sg){
      // return 1212;

        // return sg_scan_result;
        shrd_mem[subgroup_id%shmem_array_length] = sg_scan_result;
      }
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
    // First shmem_array_length number of threads inclusive scan in shared memory
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