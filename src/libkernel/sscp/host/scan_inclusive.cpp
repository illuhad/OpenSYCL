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
#define ACPP_SSCP_OMP_LIBKERNEL

#include "hipSYCL/sycl/libkernel/sscp/builtins/scan_inclusive.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/scan_host.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/scan_subgroup.hpp"

HIPSYCL_SSCP_BUILTIN void *__acpp_sscp_host_get_internal_local_memory();

#define ACPP_SUBGROUP_FLOAT_SCAN(type)                                                             \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_inclusive_scan_##type(__acpp_sscp_algorithm_op op,           \
                                                            __acpp_##type x) {                     \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::plus{});     \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::multiply{}); \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::min{});      \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::max{});      \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_SUBGROUP_FLOAT_SCAN(f16)
ACPP_SUBGROUP_FLOAT_SCAN(f32)
ACPP_SUBGROUP_FLOAT_SCAN(f64)

#define ACPP_SUBGROUP_INT_SCAN(fn_suffix, type)                                                    \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_sub_group_inclusive_scan_##fn_suffix(__acpp_sscp_algorithm_op op,      \
                                                                 __acpp_##type x) {                \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::plus{});     \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::multiply{}); \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::min{});      \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::max{});      \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::bit_and{});  \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::bit_or{});   \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x, hipsycl::libkernel::sscp::bit_xor{});  \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x,                                        \
                                                         hipsycl::libkernel::sscp::logical_and{}); \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return hipsycl::libkernel::sscp::sg_inclusive_scan(x,                                        \
                                                         hipsycl::libkernel::sscp::logical_or{});  \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_SUBGROUP_INT_SCAN(i8, int8)
ACPP_SUBGROUP_INT_SCAN(i16, int16)
ACPP_SUBGROUP_INT_SCAN(i32, int32)
ACPP_SUBGROUP_INT_SCAN(i64, int64)
ACPP_SUBGROUP_INT_SCAN(u8, uint8)
ACPP_SUBGROUP_INT_SCAN(u16, uint16)
ACPP_SUBGROUP_INT_SCAN(u32, uint32)
ACPP_SUBGROUP_INT_SCAN(u64, uint64)

#define ACPP_WORKGROUP_FLOAT_SCAN(type)                                                            \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_inclusive_scan_##type(__acpp_sscp_algorithm_op op,          \
                                                             __acpp_##type x) {                    \
    __acpp_##type *shrd_mem =                                                                      \
        static_cast<__acpp_##type *>(__acpp_sscp_host_get_internal_local_memory());                \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::plus{},    \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(                                        \
          x, hipsycl::libkernel::sscp::multiply{}, shrd_mem);                                      \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::min{},     \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::max{},     \
                                                           shrd_mem);                              \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_WORKGROUP_FLOAT_SCAN(f16)
ACPP_WORKGROUP_FLOAT_SCAN(f32)
ACPP_WORKGROUP_FLOAT_SCAN(f64)

#define ACPP_WORKGROUP_INT_SCAN(fn_suffix, type)                                                   \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##type __acpp_sscp_work_group_inclusive_scan_##fn_suffix(__acpp_sscp_algorithm_op op,     \
                                                                  __acpp_##type x) {               \
    __acpp_##type *shrd_mem =                                                                      \
        static_cast<__acpp_##type *>(__acpp_sscp_host_get_internal_local_memory());                \
    switch (op) {                                                                                  \
    case __acpp_sscp_algorithm_op::plus:                                                           \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::plus{},    \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::multiply:                                                       \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(                                        \
          x, hipsycl::libkernel::sscp::multiply{}, shrd_mem);                                      \
    case __acpp_sscp_algorithm_op::min:                                                            \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::min{},     \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::max:                                                            \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::max{},     \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::bit_and:                                                        \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::bit_and{}, \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::bit_or:                                                         \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::bit_or{},  \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::bit_xor:                                                        \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(x, hipsycl::libkernel::sscp::bit_xor{}, \
                                                           shrd_mem);                              \
    case __acpp_sscp_algorithm_op::logical_and:                                                    \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(                                        \
          x, hipsycl::libkernel::sscp::logical_and{}, shrd_mem);                                   \
    case __acpp_sscp_algorithm_op::logical_or:                                                     \
      return hipsycl::libkernel::sscp::wg_host_scan<false>(                                        \
          x, hipsycl::libkernel::sscp::logical_or{}, shrd_mem);                                    \
    default:                                                                                       \
      return __acpp_##type{};                                                                      \
    }                                                                                              \
  }

ACPP_WORKGROUP_INT_SCAN(i8, int8)
ACPP_WORKGROUP_INT_SCAN(i16, int16)
ACPP_WORKGROUP_INT_SCAN(i32, int32)
ACPP_WORKGROUP_INT_SCAN(i64, int64)
ACPP_WORKGROUP_INT_SCAN(u8, uint8)
ACPP_WORKGROUP_INT_SCAN(u16, uint16)
ACPP_WORKGROUP_INT_SCAN(u32, uint32)
ACPP_WORKGROUP_INT_SCAN(u64, uint64)
