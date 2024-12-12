
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

namespace detail {
static inline unsigned int __lane_id(){
    return  __builtin_amdgcn_mbcnt_hi(
        -1, __builtin_amdgcn_mbcnt_lo(-1, 0));
    }
}

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
    auto sg_size = __acpp_sscp_get_subgroup_max_size();
    int self = detail::__lane_id();
    int index = (self + delta);
    index = (int)((self&(sg_size-1))+delta) > sg_size ? self : index;
        
    return __builtin_amdgcn_ds_bpermute(index<<2, value);
                                                  }



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
    int self = detail::__lane_id();
    int width = __acpp_sscp_get_subgroup_max_size();
    int index = self - delta;
    index = (index < (self & ~(width-1)))?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, value);
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
    int self = detail::__lane_id();
    int index = self ^ mask;
    return __builtin_amdgcn_ds_bpermute(index<<2, value);                                                
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
    int max_subgroup_size = __acpp_sscp_get_subgroup_max_size();
    int index = id%max_subgroup_size;
    return __builtin_amdgcn_ds_bpermute(index<<2, value);
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


