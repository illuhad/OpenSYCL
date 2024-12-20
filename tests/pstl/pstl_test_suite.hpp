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

#ifndef HIPSYCL_PSTL_TEST_SUITE_HPP
#define HIPSYCL_PSTL_TEST_SUITE_HPP


struct enable_unified_shared_memory {
  enable_unified_shared_memory() {
#ifndef __ACPP_STDPAR_ASSUME_SYSTEM_USM__
    hipsycl::stdpar::unified_shared_memory::pop_disabled();
#endif
  }

  ~enable_unified_shared_memory() {
#ifndef __ACPP_STDPAR_ASSUME_SYSTEM_USM__
    hipsycl::stdpar::unified_shared_memory::push_disabled();
#endif
  }
};


static thread_local int counter = 0;
struct non_trivial_copy {

  non_trivial_copy(){}

  non_trivial_copy(int val)
  : x{val} {}

  non_trivial_copy(const non_trivial_copy& other){
    x = other.x;
    __acpp_if_target_host(++counter;)
  }

  non_trivial_copy& operator=(const non_trivial_copy& other) {
    x = other.x;
    __acpp_if_target_host(++counter;)
    return *this;
  }

  friend bool operator==(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x == b.x;
  }

  friend bool operator!=(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x != b.x;
  }

  int x;
};

#endif
