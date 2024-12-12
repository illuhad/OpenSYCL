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
#include "builtin_config.hpp"
#include "core_typed.hpp"
#include "barrier.hpp"
#include "shuffle.hpp"
#include "utils.hpp"

#ifndef HIPSYCL_SSCP_BROADCAST_BUILTINS_HPP
#define HIPSYCL_SSCP_BROADCAST_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_broadcast_i8(__acpp_int32 sender,
                                                      __acpp_int8 x);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_broadcast_i16(__acpp_int32 sender,
                                                        __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_broadcast_i32(__acpp_int32 sender,
                                                        __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_broadcast_i64(__acpp_int32 sender,
                                                        __acpp_int64 x);

template<typename T>
T __acpp_sscp_sub_group_broadcast(T, __acpp_int32) = delete;

template<>
__acpp_int8 __acpp_sscp_sub_group_broadcast<__acpp_int8>(__acpp_int8 value, __acpp_int32 id);

template<>
__acpp_int16 __acpp_sscp_sub_group_broadcast<__acpp_int16>(__acpp_int16 value, __acpp_int32 id);

template<>
__acpp_int32 __acpp_sscp_sub_group_broadcast<__acpp_int32>(__acpp_int32 value, __acpp_int32 idx);

template<>
__acpp_int64 __acpp_sscp_sub_group_broadcast<__acpp_int64>(__acpp_int64 value, __acpp_int32 id);


template<typename T>
T __acpp_sscp_work_group_broadcast(__acpp_int32, T) = delete;

#define TEMPLATE_DECLARATION_WG_BROADCAST(size) \
template<> \
__acpp_int##size __acpp_sscp_work_group_broadcast<__acpp_int##size>(__acpp_int32 id, __acpp_int##size value); \

#define TEMPLATE_DEFINITION_WG_BROADCAST(size) \
template<> \
__acpp_int##size __acpp_sscp_work_group_broadcast<__acpp_int##size>(__acpp_int32 id, __acpp_int##size value){ \
  return __acpp_sscp_work_group_broadcast_i##size(id, value); \
} \

TEMPLATE_DECLARATION_WG_BROADCAST(8)
TEMPLATE_DECLARATION_WG_BROADCAST(16)
TEMPLATE_DECLARATION_WG_BROADCAST(32)
TEMPLATE_DECLARATION_WG_BROADCAST(64)


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_broadcast_i8(__acpp_int32 sender,
                                                     __acpp_int8 x);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_broadcast_i16(__acpp_int32 sender,
                                                       __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_broadcast_i32(__acpp_int32 sender,
                                                       __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_broadcast_i64(__acpp_int32 sender,
                                                       __acpp_int64 x);


#define SUBGROUP_BCAST(fn_suffix,input_type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##input_type __acpp_sscp_sub_group_broadcast_##fn_suffix(__acpp_int32 sender, \
                                                     __acpp_##input_type x){ \
    return __acpp_sscp_sub_group_select_##fn_suffix(x, sender); \
                                                     } \

template<typename T> 
T __acpp_sscp_work_group_broadcas_impl(__acpp_int32 sender, 
                                                     T x){
      #ifndef ACPP_SSCP_OMP_LIBKERNEL
      ACPP_CUDALIKE_SHMEM_ATTRIBUTE T shrd_x[1];
      #else
      T* shrd_x = static_cast<T*>(__acpp_sscp_host_get_internal_local_memory());
      #endif
     

     if(sender == __acpp_sscp_typed_get_local_linear_id<3, int>()){ 
        shrd_x[0] = x; 
     }; 
     __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed); 
     x = shrd_x[0]; 
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed); 
    return x; 
  } 

#define GROUP_BCAST(fn_suffix,input_type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##input_type __acpp_sscp_work_group_broadcast_##fn_suffix(__acpp_int32 sender, \
                                                     __acpp_##input_type x){ \
      return __acpp_sscp_work_group_broadcas_impl(sender, x); \
    } \

#endif
