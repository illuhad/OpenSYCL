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

#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/reduction.hpp"

#define ACPP_SUBGROUP_FLOAT_REDUCTION(type)                                                        \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_reduce_##type(__acpp_sscp_algorithm_op op,                   \
                                                    __acpp_##type x) {                             \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::plus>(x);               \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::multiply>(x);           \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::min>(x);                \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::max>(x);                \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_SUBGROUP_FLOAT_REDUCTION(f16)
ACPP_SUBGROUP_FLOAT_REDUCTION(f32)
ACPP_SUBGROUP_FLOAT_REDUCTION(f64)

#define ACPP_SUBGROUP_INT_REDUCTION(fn_suffix, type)                                               \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op,              \
                                                         __acpp_##type x) {                        \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::plus>(x);               \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::multiply>(x);           \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::min>(x);                \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::max>(x);                \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::bit_and>(x);            \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::bit_or>(x);             \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::bit_xor>(x);            \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::logical_and>(x);        \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return hipsycl::libkernel::sscp::sg_reduce<__acpp_sscp_algorithm_op::logical_or>(x);         \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_SUBGROUP_INT_REDUCTION(i8, int8)
ACPP_SUBGROUP_INT_REDUCTION(i16, int16)
ACPP_SUBGROUP_INT_REDUCTION(i32, int32)
ACPP_SUBGROUP_INT_REDUCTION(i64, int64)
ACPP_SUBGROUP_INT_REDUCTION(u8, uint8)
ACPP_SUBGROUP_INT_REDUCTION(u16, uint16)
ACPP_SUBGROUP_INT_REDUCTION(u32, uint32)
ACPP_SUBGROUP_INT_REDUCTION(u64, uint64)

#define ACPP_WORKGROUP_FLOAT_REDUCTION(type)                                                       \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_reduce_##type(__acpp_sscp_algorithm_op op,                  \
                                                     __acpp_##type x) {                            \
    constexpr size_t shmem_array_length = 32;                                                      \
    ACPP_SHMEM_ATTRIBUTE __acpp_##type shrd_mem[shmem_array_length];                               \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::plus{}, &shrd_mem[0]);                                      \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::multiply{}, &shrd_mem[0]);                                  \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::min{}, &shrd_mem[0]);                                       \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::max{}, &shrd_mem[0]);                                       \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_WORKGROUP_FLOAT_REDUCTION(f16)
ACPP_WORKGROUP_FLOAT_REDUCTION(f32)
ACPP_WORKGROUP_FLOAT_REDUCTION(f64)

#define ACPP_WORKGROUP_INT_REDUCTION(fn_suffix, type)                                              \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op,             \
                                                          __acpp_##type x) {                       \
    constexpr size_t shmem_array_length = 32;                                                      \
    ACPP_SHMEM_ATTRIBUTE __acpp_##type shrd_mem[shmem_array_length];                               \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::plus{}, &shrd_mem[0]);                                      \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::multiply{}, &shrd_mem[0]);                                  \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::min{}, &shrd_mem[0]);                                       \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::max{}, &shrd_mem[0]);                                       \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::bit_and{}, &shrd_mem[0]);                                   \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::bit_or{}, &shrd_mem[0]);                                    \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::bit_xor{}, &shrd_mem[0]);                                   \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::logical_and{}, &shrd_mem[0]);                               \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return hipsycl::libkernel::sscp::wg_reduce<shmem_array_length>(                              \
          x, hipsycl::libkernel::sscp::logical_or{}, &shrd_mem[0]);                                \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_WORKGROUP_INT_REDUCTION(i8, int8)
ACPP_WORKGROUP_INT_REDUCTION(i16, int16)
ACPP_WORKGROUP_INT_REDUCTION(i32, int32)
ACPP_WORKGROUP_INT_REDUCTION(i64, int64)
ACPP_WORKGROUP_INT_REDUCTION(u8, uint8)
ACPP_WORKGROUP_INT_REDUCTION(u16, uint16)
ACPP_WORKGROUP_INT_REDUCTION(u32, uint32)
ACPP_WORKGROUP_INT_REDUCTION(u64, uint64)
