/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define ACPP_SSCP_OMP_LIBKERNEL

#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include<cstring>

template<>
__acpp_int8 __acpp_sscp_sub_group_broadcast<__acpp_int8>(__acpp_int8 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_broadcast_i8(value, id);
}

template<>
__acpp_int16 __acpp_sscp_sub_group_broadcast<__acpp_int16>(__acpp_int16 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_broadcast_i16(value, id);
}

template<>
__acpp_int32 __acpp_sscp_sub_group_broadcast<__acpp_int32>(__acpp_int32 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_broadcast_i32(value, id);
}

template<>
__acpp_int64 __acpp_sscp_sub_group_broadcast<__acpp_int64>(__acpp_int64 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_broadcast_i64(value, id);
}



template<typename T> 
T __acpp_sscp_work_group_broadcast_host_impl(__acpp_int32 sender, 
                                                     T x){        
     static int shrd_mem;
     #pragma omp threadprivate(shrd_mem)
     if(sender == __acpp_sscp_typed_get_local_linear_id<3, int>()){ 
        shrd_mem = x; 
     }; 
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed); 
    x = shrd_mem; 
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed); 
    return x; 
  } 


#define HOST_GROUP_BCAST(fn_suffix,input_type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##input_type __acpp_sscp_work_group_broadcast_##fn_suffix(__acpp_int32 sender, \
                                                     __acpp_##input_type x){ \
      return __acpp_sscp_work_group_broadcas_impl(sender, x); \
    } \

//__acpp_sscp_work_group_broadcas_impl

TEMPLATE_DEFINITION_WG_BROADCAST(8)
TEMPLATE_DEFINITION_WG_BROADCAST(16)
TEMPLATE_DEFINITION_WG_BROADCAST(32)
TEMPLATE_DEFINITION_WG_BROADCAST(64)

HOST_GROUP_BCAST(i8,int8)
HOST_GROUP_BCAST(i16,int16)
HOST_GROUP_BCAST(i32,int32)
HOST_GROUP_BCAST(i64,int64)

SUBGROUP_BCAST(i8,int8)
SUBGROUP_BCAST(i16,int16)
SUBGROUP_BCAST(i32,int32)
SUBGROUP_BCAST(i64,int64)
