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
#ifndef HIPSYCL_SSCP_BUILTINS_CORE_TYPED_HPP
#define HIPSYCL_SSCP_BUILTINS_CORE_TYPED_HPP

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

template<int Dim, class T>
T __acpp_sscp_typed_get_global_linear_id() {
  if constexpr(Dim == 1) {
    T gid_x = (T)__acpp_sscp_get_group_id_x();
    T lsize_x = (T)__acpp_sscp_get_local_size_x();
    T lid_x = (T)__acpp_sscp_get_local_id_x();

    return gid_x * lsize_x + lid_x;
  } else if constexpr(Dim == 2) {
    T gid_x = (T)__acpp_sscp_get_group_id_x();
    T gid_y = (T)__acpp_sscp_get_group_id_y();
    T lsize_x = (T)__acpp_sscp_get_local_size_x();
    T lsize_y = (T)__acpp_sscp_get_local_size_y();
    T lid_x = (T)__acpp_sscp_get_local_id_x();
    T lid_y = (T)__acpp_sscp_get_local_id_y();
    T ngroups_x = (T)__acpp_sscp_get_num_groups_x();

    T id_x = gid_x * lsize_x + lid_x; 
    T id_y = gid_y * lsize_y + lid_y;

    T global_size_x = lsize_x * ngroups_x;

    return global_size_x * id_y + id_x;
  } else if constexpr(Dim == 3) {
    T gid_x = (T)__acpp_sscp_get_group_id_x();
    T gid_y = (T)__acpp_sscp_get_group_id_y();
    T gid_z = (T)__acpp_sscp_get_group_id_z();
    T lsize_x = (T)__acpp_sscp_get_local_size_x();
    T lsize_y = (T)__acpp_sscp_get_local_size_y();
    T lsize_z = (T)__acpp_sscp_get_local_size_z();
    T lid_x = (T)__acpp_sscp_get_local_id_x();
    T lid_y = (T)__acpp_sscp_get_local_id_y();
    T lid_z = (T)__acpp_sscp_get_local_id_z();
    T ngroups_x = (T)__acpp_sscp_get_num_groups_x();
    T ngroups_y = (T)__acpp_sscp_get_num_groups_y();
    
    T id_x = gid_x * lsize_x + lid_x;
    T id_y = gid_y * lsize_y + lid_y;
    T id_z = gid_z * lsize_z + lid_z;

    T global_size_x = lsize_x * ngroups_x;
    T global_size_y = lsize_y * ngroups_y;

    return global_size_x * global_size_y * id_z + global_size_x * id_y + id_x;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __acpp_sscp_typed_get_local_linear_id() {
  if constexpr(Dim == 1) {
    return (T)__acpp_sscp_get_local_id_x();
  } else if constexpr(Dim == 2) {
    T lid_x = (T)__acpp_sscp_get_local_id_x();
    T lid_y = (T)__acpp_sscp_get_local_id_y();

    T lsize_x = (T)__acpp_sscp_get_local_size_x();

    return lsize_x * lid_y + lid_x;
  } else if constexpr(Dim == 3) {
    T lid_x = (T)__acpp_sscp_get_local_id_x();
    T lid_y = (T)__acpp_sscp_get_local_id_y();
    T lid_z = (T)__acpp_sscp_get_local_id_z();

    T lsize_x = (T)__acpp_sscp_get_local_size_x();
    T lsize_y = (T)__acpp_sscp_get_local_size_y();

    return lsize_x * lsize_y * lid_z + lsize_x * lid_y + lid_x;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __acpp_sscp_typed_get_group_linear_id() {
  if constexpr(Dim == 1) {
    return (T)__acpp_sscp_get_group_id_x();
  } else if constexpr(Dim == 2) {
    T gid_x = (T)__acpp_sscp_get_group_id_x();
    T gid_y = (T)__acpp_sscp_get_group_id_y();

    T ngroups_x = (T)__acpp_sscp_get_num_groups_x();

    return ngroups_x * gid_y + gid_x;
  } else if constexpr(Dim == 3) {
    T gid_x = (T)__acpp_sscp_get_group_id_x();
    T gid_y = (T)__acpp_sscp_get_group_id_y();
    T gid_z = (T)__acpp_sscp_get_group_id_z();

    T ngroups_x = (T)__acpp_sscp_get_num_groups_x();
    T ngroups_y = (T)__acpp_sscp_get_num_groups_y();

    return ngroups_x * ngroups_y * gid_z + ngroups_x * gid_y + gid_x;
  } else {
    return 0;
  }
}

template<int Dim, class T>
T __acpp_sscp_typed_get_global_size() {
  if constexpr(Dim == 1) {
    return (T)__acpp_sscp_get_local_size_x() * (T)__acpp_sscp_get_num_groups_x();
  } else if constexpr(Dim == 2) {
    T size_x = (T)__acpp_sscp_get_local_size_x() * (T)__acpp_sscp_get_num_groups_x();
    T size_y = (T)__acpp_sscp_get_local_size_y() * (T)__acpp_sscp_get_num_groups_y();

    return size_x * size_y;
  } else if constexpr(Dim == 3) {
    T size_x = (T)__acpp_sscp_get_local_size_x() * (T)__acpp_sscp_get_num_groups_x();
    T size_y = (T)__acpp_sscp_get_local_size_y() * (T)__acpp_sscp_get_num_groups_y();
    T size_z = (T)__acpp_sscp_get_local_size_z() * (T)__acpp_sscp_get_num_groups_z();

    return size_x * size_y * size_z;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __acpp_sscp_typed_get_local_size() {
  if constexpr(Dim == 1) {
    return (T)__acpp_sscp_get_local_size_x();
  } else if constexpr(Dim == 2) {
    T size_x = (T)__acpp_sscp_get_local_size_x();
    T size_y = (T)__acpp_sscp_get_local_size_y();

    return size_x * size_y;
  } else if constexpr(Dim == 3) {
    T size_x = (T)__acpp_sscp_get_local_size_x();
    T size_y = (T)__acpp_sscp_get_local_size_y();
    T size_z = (T)__acpp_sscp_get_local_size_z();

    return size_x * size_y * size_z;
  } else {
    return 0;
  }
}


template<int Dim, class T>
T __acpp_sscp_typed_get_num_groups() {
  if constexpr(Dim == 1) {
    return (T)__acpp_sscp_get_num_groups_x();
  } else if constexpr(Dim == 2) {
    T size_x = (T)__acpp_sscp_get_num_groups_x();
    T size_y = (T)__acpp_sscp_get_num_groups_y();

    return size_x * size_y;
  } else if constexpr(Dim == 3) {
    T size_x = (T)__acpp_sscp_get_num_groups_x();
    T size_y = (T)__acpp_sscp_get_num_groups_y();
    T size_z = (T)__acpp_sscp_get_num_groups_z();

    return size_x * size_y * size_z;
  } else {
    return 0;
  }
}


#endif
