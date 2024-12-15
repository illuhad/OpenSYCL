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

#include "hipSYCL/sycl/libkernel/sscp/builtins/scan_exclusive.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/scan.hpp"

#define SUBGROUP_FLOAT_REDUCTION(type)                                                             \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_exclusive_scan_##type(__acpp_sscp_algorithm_op op,           \
                                                            __acpp_##type x, __acpp_##type init) { \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return subgroup_exclusive_scan(x, plus{}, init);                                             \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return subgroup_exclusive_scan(x, multiply{}, init);                                         \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return subgroup_exclusive_scan(x, min{}, init);                                              \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return subgroup_exclusive_scan(x, max{}, init);                                              \
    default:                                                                                       \
      __asm__ __volatile__("trap;");                                                               \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

SUBGROUP_FLOAT_REDUCTION(f16)
SUBGROUP_FLOAT_REDUCTION(f32)
SUBGROUP_FLOAT_REDUCTION(f64)

#define SUBGROUP_INT_REDUCTION(fn_suffix, type)                                                    \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_exclusive_scan_##fn_suffix(                                  \
      __acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init) {                          \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return subgroup_exclusive_scan(x, plus{}, init);                                             \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return subgroup_exclusive_scan(x, multiply{}, init);                                         \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return subgroup_exclusive_scan(x, min{}, init);                                              \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return subgroup_exclusive_scan(x, max{}, init);                                              \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return subgroup_exclusive_scan(x, bit_and{}, init);                                          \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return subgroup_exclusive_scan(x, bit_or{}, init);                                           \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return subgroup_exclusive_scan(x, bit_xor{}, init);                                          \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return subgroup_exclusive_scan(x, logical_and{}, init);                                      \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return subgroup_exclusive_scan(x, logical_or{}, init);                                       \
    default:                                                                                       \
      __asm__ __volatile__("trap;");                                                               \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

SUBGROUP_INT_REDUCTION(i8, int8)
SUBGROUP_INT_REDUCTION(i16, int16)
SUBGROUP_INT_REDUCTION(i32, int32)
SUBGROUP_INT_REDUCTION(i64, int64)
SUBGROUP_INT_REDUCTION(u8, uint8)
SUBGROUP_INT_REDUCTION(u16, uint16)
SUBGROUP_INT_REDUCTION(u32, uint32)
SUBGROUP_INT_REDUCTION(u64, uint64)

#define GROUP_FLOAT_EXCLUSIVE_SCAN(type)                                                           \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_exclusive_scan_##type(                                      \
      __acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init) {                          \
    constexpr size_t shmem_array_length = 32;                                                      \
    ACPP_CUDALIKE_SHMEM_ATTRIBUTE __acpp_##type shrd_mem[shmem_array_length];                      \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hiplike_scan_impl<shmem_array_length, true>(x, plus{}, &shrd_mem[0], init);           \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hiplike_scan_impl<shmem_array_length, true>(x, multiply{}, &shrd_mem[0], init);       \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hiplike_scan_impl<shmem_array_length, true>(x, min{}, &shrd_mem[0], init);            \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hiplike_scan_impl<shmem_array_length, true>(x, max{}, &shrd_mem[0], init);            \
    default:                                                                                       \
      __asm__ __volatile__("trap;");                                                               \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

GROUP_FLOAT_EXCLUSIVE_SCAN(f16)
GROUP_FLOAT_EXCLUSIVE_SCAN(f32)
GROUP_FLOAT_EXCLUSIVE_SCAN(f64)

#define GROUP_INT_EXCLUSIVE_SCAN(fn_suffix, type)                                                  \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_exclusive_scan_##fn_suffix(                                 \
      __acpp_sscp_algorithm_op op, __acpp_##type x, __acpp_##type init) {                          \
    constexpr size_t shmem_array_length = 32;                                                      \
    ACPP_CUDALIKE_SHMEM_ATTRIBUTE __acpp_##type shrd_mem[shmem_array_length];                      \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hiplike_scan_impl<shmem_array_length, true>(x, plus{}, &shrd_mem[0], init);           \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hiplike_scan_impl<shmem_array_length, true>(x, multiply{}, &shrd_mem[0], init);       \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hiplike_scan_impl<shmem_array_length, true>(x, min{}, &shrd_mem[0], init);            \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hiplike_scan_impl<shmem_array_length, true>(x, max{}, &shrd_mem[0], init);            \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return hiplike_scan_impl<shmem_array_length, true>(x, bit_and{}, &shrd_mem[0], init);        \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return hiplike_scan_impl<shmem_array_length, true>(x, bit_or{}, &shrd_mem[0], init);         \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return hiplike_scan_impl<shmem_array_length, true>(x, bit_xor{}, &shrd_mem[0], init);        \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return hiplike_scan_impl<shmem_array_length, true>(x, logical_and{}, &shrd_mem[0], init);    \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return hiplike_scan_impl<shmem_array_length, true>(x, logical_or{}, &shrd_mem[0], init);     \
    default:                                                                                       \
      __asm__ __volatile__("trap;");                                                               \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

GROUP_INT_EXCLUSIVE_SCAN(i8, int8)
GROUP_INT_EXCLUSIVE_SCAN(i16, int16)
GROUP_INT_EXCLUSIVE_SCAN(i32, int32)
GROUP_INT_EXCLUSIVE_SCAN(i64, int64)
GROUP_INT_EXCLUSIVE_SCAN(u8, uint8)
GROUP_INT_EXCLUSIVE_SCAN(u16, uint16)
GROUP_INT_EXCLUSIVE_SCAN(u32, uint32)
GROUP_INT_EXCLUSIVE_SCAN(u64, uint64)
