
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

 #include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
 #include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

template<>
__acpp_int8 __acpp_sscp_sub_group_select<__acpp_int8>(__acpp_int8 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_select_i8(value, id);
}

template<>
__acpp_int16 __acpp_sscp_sub_group_select<__acpp_int16>(__acpp_int16 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_select_i16(value, id);
}

template<>
__acpp_int32 __acpp_sscp_sub_group_select<__acpp_int32>(__acpp_int32 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_select_i32(value, id);
}

template<>
__acpp_int64 __acpp_sscp_sub_group_select<__acpp_int64>(__acpp_int64 value, __acpp_int32 id){
  return __acpp_sscp_sub_group_select_i64(value, id);
}   

// __acpp_uint32 get_active_mask(){
//     //__acpp_int64 mask = __nvvm_activemask();
//     __acpp_uint32 subgroup_size = __acpp_sscp_get_subgroup_size();
//     return (1 << subgroup_size)-1;
// }
unsigned int FULL_MASk = 0xffffffff;

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shl_i8(__acpp_int8 value,
                                                __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shl_i32(value, delta);
                                                }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shl_i16(__acpp_int16 value,
                                                  __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shl_i32(value, delta);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shl_i32(__acpp_int32 value,
                                                  __acpp_uint32 delta){
    // __acpp_uint32 mask = get_active_mask();
    return __nvvm_shfl_sync_down_i32(FULL_MASk, value, delta, 0x1f);
                                                  }



// #define __MAKE_SHUFFLES(__FnName, __IntIntrinsic, __FloatIntrinsic, __Mask,    \
//                         __Type)                                                \
//   inline __device__ int __FnName(int __val, __Type __offset,                   \
//                                  int __width = warpSize) {                     \
//     return __IntIntrinsic(__val, __offset,                                     \
//                           ((warpSize - __width) << 8) | (__Mask));             \
//   }   
// __MAKE_SHUFFLES(__shfl_up, __nvvm_shfl_up_i32, __nvvm_shfl_up_f32, 0,
//                 unsigned int);
// __MAKE_SHUFFLES(__shfl_down, __nvvm_shfl_down_i32, __nvvm_shfl_down_f32, 0x1f,
//                 unsigned int);
// __MAKE_SHUFFLES(__shfl_xor, __nvvm_shfl_bfly_i32, __nvvm_shfl_bfly_f32, 0x1f,
//                 int);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shl_i64(__acpp_int64 value,
                                                  __acpp_uint32 delta){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    tmp[0] = __acpp_sscp_sub_group_shl_i32(tmp[0], delta);
    tmp[1] = __acpp_sscp_sub_group_shl_i32(tmp[1], delta);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shr_i8(__acpp_int8 value,
                                                __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shr_i32(value, delta);
                                                }
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shr_i16(__acpp_int16 value,
                                                  __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shr_i32(value, delta);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shr_i32(__acpp_int32 value,
                                                  __acpp_uint32 delta){
    // __acpp_uint32 mask = get_active_mask();
    return __nvvm_shfl_sync_up_i32(FULL_MASk,value, delta, 0);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shr_i64(__acpp_int64 value,
                                                  __acpp_uint32 delta){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    tmp[0] = __acpp_sscp_sub_group_shr_i32(tmp[0], delta);
    tmp[1] = __acpp_sscp_sub_group_shr_i32(tmp[1], delta);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_permute_i8(__acpp_int8 value,
                                                   __acpp_int32 mask){
    return __acpp_sscp_sub_group_permute_i32(value, mask);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_permute_i16(__acpp_int16 value,
                                                     __acpp_int32 mask){
    return __acpp_sscp_sub_group_permute_i32(value, mask);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_permute_i32(__acpp_int32 value,
                                                     __acpp_int32 mask){
    // __acpp_uint32 active_thread_mask = get_active_mask();
    return __nvvm_shfl_sync_bfly_i32(FULL_MASk,value, mask, 0x1f);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_permute_i64(__acpp_int64 value,
                                                     __acpp_int32 mask){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    tmp[0] = __acpp_sscp_sub_group_permute_i32(tmp[0], mask);
    tmp[1] = __acpp_sscp_sub_group_permute_i32(tmp[1], mask);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_select_i8(__acpp_int8 value,
                                                   __acpp_int32 id){
    return __acpp_sscp_sub_group_select_i32(value, id);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_select_i16(__acpp_int16 value,
                                                     __acpp_int32 id){
    return __acpp_sscp_sub_group_select_i32(value, id);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_select_i32(__acpp_int32 value,
                                                     __acpp_int32 id){
    //int max_subgroup_size = __acpp_sscp_get_subgroup_max_size();
    // int index = id%max_subgroup_size;
    // return __builtin_amdgcn_ds_bpermute(index<<2, value);
    // __acpp_uint32 mask = get_active_mask();
    // This doesn't work
    return __nvvm_shfl_sync_idx_i32(FULL_MASk, value, id, 31);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_select_i64(__acpp_int64 value,
                                                     __acpp_int32 id){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    tmp[0] = __acpp_sscp_sub_group_select_i32(tmp[0], id);
    tmp[1] = __acpp_sscp_sub_group_select_i32(tmp[1], id);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }


