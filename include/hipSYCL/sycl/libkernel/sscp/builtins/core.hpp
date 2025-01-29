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
#ifndef HIPSYCL_SSCP_BUILTINS_CORE_HPP
#define HIPSYCL_SSCP_BUILTINS_CORE_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z();

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z();

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z();

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y();
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z();

#endif
