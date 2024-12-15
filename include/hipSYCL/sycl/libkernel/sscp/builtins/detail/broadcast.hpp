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

#include "../builtin_config.hpp"
#include "../core_typed.hpp"
#include "../barrier.hpp"
#include "../shuffle.hpp"
#include "../utils.hpp"

template <typename T>
T __acpp_sscp_work_group_broadcast(__acpp_int32, T) = delete;

#define ACPP_TEMPLATE_DECLARATION_WG_BROADCAST(size) \
  template <>                                        \
  __acpp_int##size __acpp_sscp_work_group_broadcast<__acpp_int##size>(__acpp_int32 id, __acpp_int##size value);

#define ACPP_TEMPLATE_DEFINITION_WG_BROADCAST(size)                                                            \
  template <>                                                                                                  \
  __acpp_int##size __acpp_sscp_work_group_broadcast<__acpp_int##size>(__acpp_int32 id, __acpp_int##size value) \
  {                                                                                                            \
    return __acpp_sscp_work_group_broadcast_i##size(id, value);                                                \
  }

ACPP_TEMPLATE_DECLARATION_WG_BROADCAST(8)
ACPP_TEMPLATE_DECLARATION_WG_BROADCAST(16)
ACPP_TEMPLATE_DECLARATION_WG_BROADCAST(32)
ACPP_TEMPLATE_DECLARATION_WG_BROADCAST(64)

#define ACPP_SUBGROUP_BCAST(fn_suffix, input_type)                                       \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                        \
  __acpp_##input_type __acpp_sscp_sub_group_broadcast_##fn_suffix(__acpp_int32 sender,   \
                                                                  __acpp_##input_type x) \
  {                                                                                      \
    return __acpp_sscp_sub_group_select_##fn_suffix(x, sender);                          \
  }

template <typename T, typename V>
T __acpp_sscp_work_group_broadcast_impl(__acpp_int32 sender,
                                        T x, V shrd_memory)
{

  if (sender == __acpp_sscp_typed_get_local_linear_id<3, int>())
  {
    shrd_memory[0] = x;
  };
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  x = shrd_memory[0];
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  return x;
}

#define ACPP_WORKGROUP_BCAST(fn_suffix, input_type)                                       \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                         \
  __acpp_##input_type __acpp_sscp_work_group_broadcast_##fn_suffix(__acpp_int32 sender,   \
                                                                   __acpp_##input_type x) \
  {                                                                                       \
    ACPP_CUDALIKE_SHMEM_ATTRIBUTE __acpp_##input_type shrd_x[1];                          \
    return __acpp_sscp_work_group_broadcast_impl(sender, x, &shrd_x[0]);                  \
  }

#endif