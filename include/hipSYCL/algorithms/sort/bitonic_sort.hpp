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

#ifndef ACPP_ALGORITHMS_BITONIC_SORT
#define ACPP_ALGORITHMS_BITONIC_SORT

#include <iterator>
#include <cstdint>
#include "hipSYCL/sycl/queue.hpp"

namespace hipsycl::algorithms::sorting {

namespace detail{


template<class RandomIt, class Size>
RandomIt advance_to(RandomIt first, Size i) {
  std::advance(first, i);
  return first;
}

inline bool can_compare(std::size_t left_id, std::size_t right_id,
                        std::size_t problem_size) {

  return (left_id < right_id) && (left_id < problem_size) &&
         (right_id < problem_size);
}


} //detail

template <class RandomIt, class SizeT, class Barrier, class Compare>
void bitonic_group_sort(RandomIt first, SizeT group_size, SizeT problem_size,
                        SizeT item, Barrier barrier, Compare comp) {

  auto process_pass = [=](SizeT j) {
    for(SizeT a_id = item; a_id < problem_size; a_id += group_size) {
      SizeT b_id = a_id ^ j;
      if(detail::can_compare(a_id, b_id, problem_size)) {
        auto a = *detail::advance_to(first, a_id);
        auto b = *detail::advance_to(first, b_id);
        if(comp(b, a)) {
          *detail::advance_to(first, a_id) = b;
          *detail::advance_to(first, b_id) = a;
        }
      }
    }
    barrier();
  };

  for(SizeT k = 2; (k >> 1) < problem_size; k *= 2) {
    process_pass(k-1);

    for (SizeT j = k >> 1; j > 0; j >>= 1) {
      process_pass(j);
    }
  }
}

template <class RandomIt, class Comparator>
sycl::event bitonic_sort(sycl::queue &q, RandomIt first, RandomIt last,
                         Comparator comp, const std::vector<sycl::event>& deps = {}) {

  std::size_t problem_size = std::distance(first, last);
  sycl::event most_recent_event;
  bool is_first_kernel = true;

  auto launch_kernel = [&](std::size_t j){

    auto k = [=](sycl::id<1> idx) {
      std::size_t a_id = idx.get(0);
      std::size_t b_id = a_id ^ j;
      if(detail::can_compare(a_id, b_id, problem_size)) {
        auto a = *detail::advance_to(first, a_id);
        auto b = *detail::advance_to(first, b_id);
        if(comp(b, a)) {
          *detail::advance_to(first, a_id) = b;
          *detail::advance_to(first, b_id) = a;
        }
      }
    };
    if(is_first_kernel || q.is_in_order())
      most_recent_event = q.parallel_for(problem_size, deps, k);
    else
      most_recent_event = q.parallel_for(problem_size, most_recent_event, k);
    is_first_kernel = false;
  };

  for (std::size_t k = 2; (k >> 1) < problem_size; k *= 2) {
    launch_kernel(k-1);

    for (std::size_t j = k >> 1; j > 0; j >>= 1) {
      launch_kernel(j);
    }
  }

  return most_recent_event;
} // bitonic_sort

}

#endif
