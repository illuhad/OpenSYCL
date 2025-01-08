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

#ifndef HIPSYCL_SSCP_DETAIL_BROADCAST_BUILTINS_HPP
#define HIPSYCL_SSCP_DETAIL_BROADCAST_BUILTINS_HPP

#include "../barrier.hpp"
#include "../broadcast.hpp"
#include "../builtin_config.hpp"
#include "../core_typed.hpp"
#include "../shuffle.hpp"
#include "utils.hpp"

#undef ACPP_TEMPLATE_DECLARATION_WG_BROADCAST

namespace hipsycl::libkernel::sscp {

template <typename T, typename V> T wg_broadcast(__acpp_int32 sender, T x, V shrd_memory) {

  if (sender == __acpp_sscp_typed_get_local_linear_id<3, int>()) {
    shrd_memory[0] = x;
  };
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                 __acpp_sscp_memory_order::relaxed);
  x = shrd_memory[0];
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group,
                                 __acpp_sscp_memory_order::relaxed);
  return x;
}

} // namespace hipsycl::libkernel::sscp

#endif