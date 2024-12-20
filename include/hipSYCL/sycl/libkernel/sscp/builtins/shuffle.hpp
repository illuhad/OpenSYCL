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
#ifndef HIPSYCL_SSCP_SHUFFLE_BUILTINS_HPP
#define HIPSYCL_SSCP_SHUFFLE_BUILTINS_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shl_i8(__acpp_int8 value,
                                         __acpp_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shl_i16(__acpp_int16 value,
                                           __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shl_i32(__acpp_int32 value,
                                           __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shl_i64(__acpp_int64 value,
                                           __acpp_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_shl_i8(__acpp_int8 value,
                                          __acpp_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_shl_i16(__acpp_int16 value,
                                            __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_shl_i32(__acpp_int32 value,
                                            __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_shl_i64(__acpp_int64 value,
                                            __acpp_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shr_i8(__acpp_int8 value,
                                         __acpp_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shr_i16(__acpp_int16 value,
                                           __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shr_i32(__acpp_int32 value,
                                           __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shr_i64(__acpp_int64 value,
                                           __acpp_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_shr_i8(__acpp_int8 value,
                                          __acpp_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_shr_i16(__acpp_int16 value,
                                            __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_shr_i32(__acpp_int32 value,
                                            __acpp_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_shr_i64(__acpp_int64 value,
                                            __acpp_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_permute_i8(__acpp_int8 value,
                                             __acpp_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_permute_i16(__acpp_int16 value,
                                               __acpp_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_permute_i32(__acpp_int32 value,
                                               __acpp_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_permute_i64(__acpp_int64 value,
                                               __acpp_int32 mask);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_permute_i8(__acpp_int8 value,
                                              __acpp_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_permute_i16(__acpp_int16 value,
                                                __acpp_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_permute_i32(__acpp_int32 value,
                                                __acpp_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_permute_i64(__acpp_int64 value,
                                                __acpp_int32 mask);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_select_i8(__acpp_int8 value,
                                            __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_select_i16(__acpp_int16 value,
                                              __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_select_i32(__acpp_int32 value,
                                              __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_select_i64(__acpp_int64 value,
                                              __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_select_i8(__acpp_int8 value,
                                             __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_select_i16(__acpp_int16 value,
                                               __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_select_i32(__acpp_int32 value,
                                               __acpp_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_select_i64(__acpp_int64 value,
                                               __acpp_int32 id);

#endif
