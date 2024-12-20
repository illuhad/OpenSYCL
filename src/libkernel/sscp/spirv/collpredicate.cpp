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

 #include "hipSYCL/sycl/libkernel/sscp/builtins/collpredicate.hpp"
 #include "hipSYCL/sycl/libkernel/sscp/builtins/reduction.hpp"
 #include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu/ockl.hpp"


HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_any(bool pred){
    return __acpp_sscp_work_group_reduce_i8(__acpp_sscp_algorithm_op::logical_or, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool pred){
    return __acpp_sscp_work_group_reduce_i8(__acpp_sscp_algorithm_op::logical_and, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool pred){
    bool result_or = __acpp_sscp_work_group_reduce_i8(__acpp_sscp_algorithm_op::logical_or, pred);
    return !result_or;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred){
    return __acpp_sscp_sub_group_reduce_i8(__acpp_sscp_algorithm_op::logical_and, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred){
    return __acpp_sscp_sub_group_reduce_i8(__acpp_sscp_algorithm_op::logical_or, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred){
    bool result_or = __acpp_sscp_sub_group_reduce_i8(__acpp_sscp_algorithm_op::logical_or, pred);
    return !result_or;
}
