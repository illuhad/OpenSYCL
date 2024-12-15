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

#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/broadcast.hpp"

ACPP_TEMPLATE_DEFINITION_WG_BROADCAST(8)
ACPP_TEMPLATE_DEFINITION_WG_BROADCAST(16)
ACPP_TEMPLATE_DEFINITION_WG_BROADCAST(32)
ACPP_TEMPLATE_DEFINITION_WG_BROADCAST(64)

ACPP_WORKGROUP_BCAST(i8,int8)
ACPP_WORKGROUP_BCAST(i16,int16)
ACPP_WORKGROUP_BCAST(i32,int32)
ACPP_WORKGROUP_BCAST(i64,int64)

ACPP_SUBGROUP_BCAST(i8,int8)
ACPP_SUBGROUP_BCAST(i16,int16)
ACPP_SUBGROUP_BCAST(i32,int32)
ACPP_SUBGROUP_BCAST(i64,int64)