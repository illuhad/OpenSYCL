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

#define SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(type)                                                   \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_reduce_##type(__acpp_sscp_algorithm_op op,                   \
                                                    __acpp_##type x) {                             \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::plus>(x);                       \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::multiply>(x);                   \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::min>(x);                        \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::max>(x);                        \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(f16)
SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(f32)
SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(f64)

#define SUBGROUP_INT_SUB_GROUP_REDUCTION(fn_suffix, type)                                          \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op,              \
                                                         __acpp_##type x) {                        \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::plus>(x);                       \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::multiply>(x);                   \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::min>(x);                        \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::max>(x);                        \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::bit_and>(x);                    \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::bit_or>(x);                     \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::bit_xor>(x);                    \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::logical_and>(x);                \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::logical_or>(x);                 \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

SUBGROUP_INT_SUB_GROUP_REDUCTION(i8, int8)
SUBGROUP_INT_SUB_GROUP_REDUCTION(i16, int16)
SUBGROUP_INT_SUB_GROUP_REDUCTION(i32, int32)
SUBGROUP_INT_SUB_GROUP_REDUCTION(i64, int64)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u8, uint8)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u16, uint16)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u32, uint32)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u64, uint64)

#define SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(type)                                                  \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_reduce_##type(__acpp_sscp_algorithm_op op,                  \
                                                     __acpp_##type x) {                            \
    constexpr size_t shmem_array_length = 32;                                                      \
    __acpp_##type *shrd_mem =                                                                      \
        static_cast<__acpp_##type *>(__acpp_sscp_host_get_internal_local_memory());                \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, plus{}, shrd_mem);          \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, multiply{}, shrd_mem);      \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, min{}, shrd_mem);           \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, max{}, shrd_mem);           \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(f16)
SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(f32)
SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(f64)

#define SUBGROUP_INT_WORK_GROUP_REDUCTION(fn_suffix, type)                                         \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op,             \
                                                          __acpp_##type x) {                       \
    constexpr size_t shmem_array_length = 32;                                                      \
    __acpp_##type *shrd_mem =                                                                      \
        static_cast<__acpp_##type *>(__acpp_sscp_host_get_internal_local_memory());                \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, plus{}, shrd_mem);          \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, multiply{}, shrd_mem);      \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, min{}, shrd_mem);           \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, max{}, shrd_mem);           \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, bit_and{}, shrd_mem);       \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, bit_or{}, shrd_mem);        \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, bit_xor{}, shrd_mem);       \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, logical_and{}, shrd_mem);   \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return __acpp_reduce_over_work_group_impl<shmem_array_length>(x, logical_or{}, shrd_mem);    \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

SUBGROUP_INT_WORK_GROUP_REDUCTION(i8, int8)
SUBGROUP_INT_WORK_GROUP_REDUCTION(i16, int16)
SUBGROUP_INT_WORK_GROUP_REDUCTION(i32, int32)
SUBGROUP_INT_WORK_GROUP_REDUCTION(i64, int64)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u8, uint8)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u16, uint16)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u32, uint32)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u64, uint64)
