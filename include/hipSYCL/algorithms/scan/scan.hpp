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

#ifndef ACPP_ALGORITHMS_SCAN_HPP
#define ACPP_ALGORITHMS_SCAN_HPP

#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

#include "decoupled_lookback_scan.hpp"
#include <type_traits>

namespace hipsycl::algorithms::scanning {

namespace detail {

inline std::size_t select_scan_work_group_size(sycl::queue& q) {
  std::size_t group_size = 128;
  if(q.get_device().AdaptiveCpp_device_id().get_backend() == sycl::backend::omp) {
    group_size = 1024;
  }
  return group_size;
}

}


template <bool IsInclusive, class T, class BinaryOp,
          class OptionalInitT, class Generator, class Processor>
sycl::event generate_scan_process(sycl::queue &q, util::allocation_group &scratch_allocations,
                 std::size_t problem_size, BinaryOp op,
                 OptionalInitT init, Generator gen, Processor processor,
                 const std::vector<sycl::event> &deps = {}) {
  
  std::size_t group_size = detail::select_scan_work_group_size(q);

  return scanning::decoupled_lookback_scan<IsInclusive, T>(
      q, scratch_allocations, gen, processor, op, problem_size,
      group_size, init, deps);
}

template <bool IsInclusive, class InputIt, class OutputIt, class BinaryOp,
          class OptionalInitT>
sycl::event scan(sycl::queue &q, util::allocation_group &scratch_allocations,
                 InputIt first, InputIt last, OutputIt d_first, BinaryOp op,
                 OptionalInitT init,
                 const std::vector<sycl::event> &deps = {}) {

  auto generator = [=](auto idx, auto effective_group_id, auto effective_global_id,
                 auto problem_size) {
    if(effective_global_id >= problem_size)
      effective_global_id = problem_size - 1;

    InputIt it = first;
    std::advance(it, effective_global_id);
    return *it;
  };
  auto result_processor = [=](auto idx, auto effective_group_id,
                       auto effective_global_id, auto problem_size,
                       auto value) {
    if (effective_global_id < problem_size) {
      OutputIt it = d_first;
      std::advance(it, effective_global_id);
      *it = value;
    }
  };

  std::size_t problem_size = std::distance(first, last);
  using T = std::decay_t<decltype(*first)>;

  return generate_scan_process<IsInclusive, T>(
      q, scratch_allocations, problem_size, op, init, generator,
      result_processor, deps);
}

template <bool IsInclusive, class InputIt, class OutputIt, class UnaryOp,
          class BinaryOp, class OptionalInitT>
sycl::event transform_scan(sycl::queue &q,
                           util::allocation_group &scratch_allocations,
                           InputIt first, InputIt last, OutputIt d_first,
                           UnaryOp unary_op, BinaryOp op, OptionalInitT init,
                           const std::vector<sycl::event> &deps = {}) {

  using T = std::decay_t<decltype(unary_op(*first))>;

  auto generator = [=](auto idx, auto effective_group_id, auto effective_global_id,
                 auto problem_size) {
    if(effective_global_id >= problem_size) {
      if constexpr(std::is_constructible_v<T>) {
        return T{};
      } else {
        // This might be invalid according to a very strict implementation of C++
        // definition of e.g. transform_reduce, since it does not guarantee that
        // unary_op is executed exactly once per element.
        // However, working around this might be fairly costly in case T is not
        // default constructible (Idea: Global variable guarded by an atomic lock
        // which is set by the first thread to have loaded a value), so for
        // now we do "the simple thing". This is probably still better than
        // not offloading in that case.
        return unary_op(*first);
      }
    }

    InputIt it = first;
    std::advance(it, effective_global_id);
    return unary_op(*it);
  };
  auto result_processor = [=](auto idx, auto effective_group_id,
                       auto effective_global_id, auto problem_size,
                       auto value) {
    if (effective_global_id < problem_size) {
      OutputIt it = d_first;
      std::advance(it, effective_global_id);
      *it = value;
    }
  };

  std::size_t problem_size = std::distance(first, last);
  
  return generate_scan_process<IsInclusive, T>(
      q, scratch_allocations, problem_size, op, init, generator,
      result_processor, deps);
}
}

#endif
