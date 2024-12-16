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
#ifndef HIPSYCL_SSCP_SCAN_EXCLUSIVE_BUILTINS_HPP
#define HIPSYCL_SSCP_SCAN_EXCLUSIVE_BUILTINS_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_exclusive_scan_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x,
                                                     __acpp_int8 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_exclusive_scan_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x,
                                                       __acpp_int16 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_exclusive_scan_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x,
                                                       __acpp_int32 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_exclusive_scan_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x,
                                                       __acpp_int64 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_work_group_exclusive_scan_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x,
                                                      __acpp_uint8 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_work_group_exclusive_scan_u16(__acpp_sscp_algorithm_op op,
                                                        __acpp_uint16 x, __acpp_uint16 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_work_group_exclusive_scan_u32(__acpp_sscp_algorithm_op op,
                                                        __acpp_uint32 x, __acpp_uint32 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_work_group_exclusive_scan_u64(__acpp_sscp_algorithm_op op,
                                                        __acpp_uint64 x, __acpp_uint64 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_work_group_exclusive_scan_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x,
                                                     __acpp_f16 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_work_group_exclusive_scan_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x,
                                                     __acpp_f32 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_work_group_exclusive_scan_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x,
                                                     __acpp_f64 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_exclusive_scan_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x,
                                                    __acpp_int8 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_exclusive_scan_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x,
                                                      __acpp_int16 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_exclusive_scan_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x,
                                                      __acpp_int32 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_exclusive_scan_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x,
                                                      __acpp_int64 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_sub_group_exclusive_scan_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x,
                                                     __acpp_uint8 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_sub_group_exclusive_scan_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x,
                                                       __acpp_uint16 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_sub_group_exclusive_scan_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x,
                                                       __acpp_uint32 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_sub_group_exclusive_scan_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x,
                                                       __acpp_uint64 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_sub_group_exclusive_scan_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x,
                                                    __acpp_f16 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_sub_group_exclusive_scan_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x,
                                                    __acpp_f32 init);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_sub_group_exclusive_scan_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x,
                                                    __acpp_f64 init);

#endif