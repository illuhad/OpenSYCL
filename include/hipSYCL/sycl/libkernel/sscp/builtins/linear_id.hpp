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


// THIS FILE INCLUDES STANDARD LIBRARY HEADERS AND MUST NOT BE 
// USED IN THE DEFINITION OF SSCP BUILTINS!

#ifndef HIPSYCL_SSCP_BUILTINS_LINEAR_ID_HPP
#define HIPSYCL_SSCP_BUILTINS_LINEAR_ID_HPP

#include "core.hpp"
#include "core_typed.hpp"

#include <stddef.h>

// This is used to implement the optimization in llvm-to-backend to treat
// all queries as fitting into int.
// The implementation is provided by the compiler and does not need to be implemented
// by backends.
HIPSYCL_SSCP_BUILTIN bool
__acpp_sscp_if_global_sizes_fit_in_int();

template<int Dim>
size_t __acpp_sscp_get_global_linear_id() {
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_global_linear_id<Dim, int>();
  } else {
    return __acpp_sscp_typed_get_global_linear_id<Dim, size_t>();
  }
}

template<int Dim>
size_t __acpp_sscp_get_group_linear_id() {
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_group_linear_id<Dim, int>();
  } else {
    return __acpp_sscp_typed_get_group_linear_id<Dim, size_t>();
  }
}

template<int Dim>
size_t __acpp_sscp_get_local_linear_id() {
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_local_linear_id<Dim, int>();
  } else {
    return __acpp_sscp_typed_get_local_linear_id<Dim, size_t>();
  }
}

template<int Dim>
size_t __acpp_sscp_get_global_size() {
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_global_size<Dim, int>();
  } else {
    return __acpp_sscp_typed_get_global_size<Dim, size_t>();
  }
}

template<int Dim>
size_t __acpp_sscp_get_local_size() {
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_local_size<Dim, int>();
  } else {
    return __acpp_sscp_typed_get_local_size<Dim, size_t>();
  }
}

template<int Dim>
size_t __acpp_sscp_get_num_groups() {
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_num_groups<Dim, int>();
  } else {
    return __acpp_sscp_typed_get_num_groups<Dim, size_t>();
  }
}

#endif
