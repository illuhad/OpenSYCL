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

#include "hipSYCL/sycl/libkernel/sscp/builtins/scan_exclusive.hpp"



#define SUBGROUP_FLOAT_REDUCTION(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_exclusive_scan_##type(__acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_subgroup_exclusive_scan_impl(x, plus{}, init); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_subgroup_exclusive_scan_impl(x, multiply{}, init); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_subgroup_exclusive_scan_impl(x, min{}, init); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_subgroup_exclusive_scan_impl(x, max{}, init); \
        default: \
            return __acpp_##type{}; \
    } \
} \

SUBGROUP_FLOAT_REDUCTION(f16)
SUBGROUP_FLOAT_REDUCTION(f32)
SUBGROUP_FLOAT_REDUCTION(f64)

#define SUBGROUP_INT_REDUCTION(fn_suffix,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_exclusive_scan_##fn_suffix(__acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_subgroup_exclusive_scan_impl(x, plus{}, init); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_subgroup_exclusive_scan_impl(x, multiply{}, init); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_subgroup_exclusive_scan_impl(x, min{}, init); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_subgroup_exclusive_scan_impl(x, max{}, init); \
        case __acpp_sscp_algorithm_op::bit_and: \
            return __acpp_subgroup_exclusive_scan_impl(x, bit_and{}, init); \
        case __acpp_sscp_algorithm_op::bit_or: \
            return __acpp_subgroup_exclusive_scan_impl(x, bit_or{}, init); \
        case __acpp_sscp_algorithm_op::bit_xor: \
            return __acpp_subgroup_exclusive_scan_impl(x, bit_xor{}, init); \
        case __acpp_sscp_algorithm_op::logical_and: \
            return __acpp_subgroup_exclusive_scan_impl(x, logical_and{}, init); \
        case __acpp_sscp_algorithm_op::logical_or: \
            return __acpp_subgroup_exclusive_scan_impl(x, logical_or{}, init); \
        default: \
            return __acpp_##type{}; \
    } \
} \

SUBGROUP_INT_REDUCTION(i8 ,int8 )
SUBGROUP_INT_REDUCTION(i16,int16)
SUBGROUP_INT_REDUCTION(i32,int32)
SUBGROUP_INT_REDUCTION(i64,int64)
SUBGROUP_INT_REDUCTION(u8 ,uint8 )
SUBGROUP_INT_REDUCTION(u16,uint16)
SUBGROUP_INT_REDUCTION(u32,uint32)
SUBGROUP_INT_REDUCTION(u64,uint64)

#define GROUP_FLOAT_EXCLUSIVE_SCAN(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_work_group_exclusive_scan_##type(__acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_group_exclusive_scan_host_impl(x, plus{}, init); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_group_exclusive_scan_host_impl(x, multiply{}, init); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_group_exclusive_scan_host_impl(x, min{}, init); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_group_exclusive_scan_host_impl(x, max{}, init); \
        default: \
            return __acpp_##type{}; \
    } \
} \

GROUP_FLOAT_EXCLUSIVE_SCAN(f16)
GROUP_FLOAT_EXCLUSIVE_SCAN(f32)
GROUP_FLOAT_EXCLUSIVE_SCAN(f64)

#define GROUP_INT_EXCLUSIVE_SCAN(fn_suffix,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_work_group_exclusive_scan_##fn_suffix(__acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_group_exclusive_scan_host_impl(x, plus{}, init); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_group_exclusive_scan_host_impl(x, multiply{}, init); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_group_exclusive_scan_host_impl(x, min{}, init); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_group_exclusive_scan_host_impl(x, max{}, init); \
        case __acpp_sscp_algorithm_op::bit_and: \
            return __acpp_group_exclusive_scan_host_impl(x, bit_and{}, init); \
        case __acpp_sscp_algorithm_op::bit_or: \
            return __acpp_group_exclusive_scan_host_impl(x, bit_or{}, init); \
        case __acpp_sscp_algorithm_op::bit_xor: \
            return __acpp_group_exclusive_scan_host_impl(x, bit_xor{}, init); \
        case __acpp_sscp_algorithm_op::logical_and: \
            return __acpp_group_exclusive_scan_host_impl(x, logical_and{}, init); \
        case __acpp_sscp_algorithm_op::logical_or: \
            return __acpp_group_exclusive_scan_host_impl(x, logical_or{}, init); \
        default: \
            return __acpp_##type{}; \
    } \
} \

GROUP_INT_EXCLUSIVE_SCAN(i8 ,int8 )
GROUP_INT_EXCLUSIVE_SCAN(i16,int16)
GROUP_INT_EXCLUSIVE_SCAN(i32,int32)
GROUP_INT_EXCLUSIVE_SCAN(i64,int64)
GROUP_INT_EXCLUSIVE_SCAN(u8 ,uint8)
GROUP_INT_EXCLUSIVE_SCAN(u16,uint16)
GROUP_INT_EXCLUSIVE_SCAN(u32,uint32)
GROUP_INT_EXCLUSIVE_SCAN(u64,uint64)


