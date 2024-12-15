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
#ifndef HIPSYCL_SSCP_DETAIL_SHUFFLE_BUILTINS_HPP
#define HIPSYCL_SSCP_DETAIL_SHUFFLE_BUILTINS_HPP

#include "../builtin_config.hpp"

template<typename T>
T __acpp_sscp_sub_group_select(T, __acpp_int32) = delete;

template<>
__acpp_int8 __acpp_sscp_sub_group_select<__acpp_int8>(__acpp_int8 value, __acpp_int32 id);

template<>
__acpp_int16 __acpp_sscp_sub_group_select<__acpp_int16>(__acpp_int16 value, __acpp_int32 id);

template<>
__acpp_int32 __acpp_sscp_sub_group_select<__acpp_int32>(__acpp_int32 value, __acpp_int32 idx);

template<>
__acpp_int64 __acpp_sscp_sub_group_select<__acpp_int64>(__acpp_int64 value, __acpp_int32 id);

#endif