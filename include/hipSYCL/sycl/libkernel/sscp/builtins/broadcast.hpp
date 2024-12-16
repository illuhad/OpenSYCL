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
#ifndef HIPSYCL_SSCP_BROADCAST_BUILTINS_HPP
#define HIPSYCL_SSCP_BROADCAST_BUILTINS_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_broadcast_i8(__acpp_int32 sender,
                                                      __acpp_int8 x);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_broadcast_i16(__acpp_int32 sender,
                                                        __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_broadcast_i32(__acpp_int32 sender,
                                                        __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_broadcast_i64(__acpp_int32 sender,
                                                        __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_broadcast_i8(__acpp_int32 sender,
                                                     __acpp_int8 x);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_broadcast_i16(__acpp_int32 sender,
                                                       __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_broadcast_i32(__acpp_int32 sender,
                                                       __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_broadcast_i64(__acpp_int32 sender,
                                                       __acpp_int64 x);


#endif

