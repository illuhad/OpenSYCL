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
#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"

__attribute__((always_inline))
void __acpp_amdgpu_local_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

__attribute__((always_inline)) void
__acpp_amdgpu_mem_fence(__acpp_sscp_memory_scope fence_scope,
                        __acpp_sscp_memory_order order) {

  if(fence_scope == __acpp_sscp_memory_scope::work_group) {
    switch(order) {
    case __acpp_sscp_memory_order::relaxed:
      break;
    case __acpp_sscp_memory_order::acquire:
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
      break;
    case __acpp_sscp_memory_order::release:
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      break;
    case __acpp_sscp_memory_order::acq_rel:
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
      break;
    case __acpp_sscp_memory_order::seq_cst:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
      break;
    }
  } else if(fence_scope == __acpp_sscp_memory_scope::device) {
    switch(order) {
    case __acpp_sscp_memory_order::relaxed:
      break;
    case __acpp_sscp_memory_order::acquire:
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      break;
    case __acpp_sscp_memory_order::release:
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      break;
    case __acpp_sscp_memory_order::acq_rel:
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
      break;
    case __acpp_sscp_memory_order::seq_cst:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
      break;
    }
  } else if(fence_scope == __acpp_sscp_memory_scope::system) {
    switch(order) {
    case __acpp_sscp_memory_order::relaxed:
      break;
    case __acpp_sscp_memory_order::acquire:
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      break;
    case __acpp_sscp_memory_order::release:
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      break;
    case __acpp_sscp_memory_order::acq_rel:
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
      break;
    case __acpp_sscp_memory_order::seq_cst:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
      break;
    }
  }
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope fence_scope,
                               __acpp_sscp_memory_order order) {

  // TODO: Correctly take into account memory order for local_barrier
  __acpp_amdgpu_local_barrier();
  if(fence_scope != __acpp_sscp_memory_scope::work_group) {
    __acpp_amdgpu_mem_fence(fence_scope, order);
  }
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope fence_scope,
                              __acpp_sscp_memory_order order) {

  __acpp_amdgpu_mem_fence(fence_scope, order);
}
