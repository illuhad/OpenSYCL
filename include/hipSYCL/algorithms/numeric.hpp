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
#ifndef HIPSYCL_ALGORITHMS_NUMERIC_HPP
#define HIPSYCL_ALGORITHMS_NUMERIC_HPP

#include <cstddef>
#include <iterator>
#include <functional>
#include <limits>

#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include "hipSYCL/sycl/libkernel/functional.hpp"
#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/detail/namespace_compat.hpp"
#include "hipSYCL/algorithms/reduction/reduction_descriptor.hpp"
#include "hipSYCL/algorithms/reduction/reduction_engine.hpp"
#include "hipSYCL/algorithms/scan/scan.hpp"
#include "hipSYCL/algorithms/util/memory_streaming.hpp"


namespace hipsycl::algorithms {

namespace detail {

template<class T, class Op>
struct identity {
  static constexpr bool is_known() { return false; }
};

#define HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(op, value)                    \
  template <class T> struct identity<T, op> {                                  \
    static constexpr bool is_known() { return true; }                          \
    static T get_identity() { return value; }                                  \
  };

// TODO: Need to restrict this to fundamental types
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(std::plus<T>, T{})
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(std::multiplies<T>, T{1})

HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::plus<T>, T{})
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::multiplies<T>, T{1})
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::minimum<T>, std::numeric_limits<T>::max())
HIPSYCL_ALGORITHMS_DEFINE_KNOWN_IDENTITY(sycl::maximum<T>, std::numeric_limits<T>::min())


template<class T, class BinaryOp>
auto get_reduction_operator_configuration(const BinaryOp& op) {
  if constexpr(detail::identity<T, BinaryOp>::is_known()) {
    return reduction::reduction_binary_operator<T, BinaryOp, true>{
        op, detail::identity<T, BinaryOp>::get_identity()};
  } else {
    return reduction::reduction_binary_operator<T, BinaryOp, false>{op};
  }
}


template <class T, class Kernel,
          class BinaryReductionOp>
sycl::event wg_model_reduction(sycl::queue &q,
                               util::allocation_group &scratch_allocations,
                               T *output, T init, std::size_t target_num_groups,
                               std::size_t local_size, std::size_t problem_size,
                               Kernel k, BinaryReductionOp op,
                               const std::vector<sycl::event>& deps = {}) {
  assert(target_num_groups > 0);

  sycl::event last_event;
  auto ndrange_launcher =
      [&](std::size_t num_groups, std::size_t wg_size, std::size_t global_size,
          std::size_t local_mem, auto kernel) {
        last_event = q.submit([&](sycl::handler &cgh) {
          // This is just there to register the appropriate amount of local
          // memory; the reduction engine will access it directly without going
          // through the accessor.
          sycl::local_accessor<char> acc{sycl::range<1>{local_mem}, cgh};
          cgh.parallel_for(sycl::nd_range<1>{wg_size * num_groups, wg_size},
                           kernel);
        });
      };

  auto operator_config = get_reduction_operator_configuration<T>(op);
  auto reduction_descriptor = reduction::reduction_descriptor{
      operator_config, init, output};

  using group_reduction_type =
      reduction::wg_model::group_reductions::generic_local_memory<
          std::decay_t<decltype(reduction_descriptor)>>;
  
  // The reduction engine will update this value with the
  // appropriate amount of local memory for the main kernel.
  std::size_t main_kernel_local_mem = 0;
  reduction::wg_model::group_horizontal_reducer<group_reduction_type>
      horizontal_reducer{
          group_reduction_type{main_kernel_local_mem, local_size}};
  reduction::wg_hierarchical_reduction_engine engine{horizontal_reducer,
                                                     &scratch_allocations};

  util::data_streamer streamer{q.get_device(), problem_size, local_size};

  const std::size_t dispatched_global_size =
      streamer.get_required_global_size();
  auto plan = engine.create_plan(dispatched_global_size, local_size,
                                reduction_descriptor);

  auto main_kernel = engine.make_main_reducing_kernel(
      [=](sycl::nd_item<1> idx, auto &reducer) {

        util::data_streamer::run(problem_size, idx, [&](sycl::id<1> i){
          k(i, reducer);
        });
      },
      plan);

  last_event = q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<char> acc{sycl::range<1>{main_kernel_local_mem}, cgh};
    cgh.depends_on(deps);
    cgh.parallel_for(sycl::nd_range<1>{dispatched_global_size, local_size},
                    main_kernel);
  });

  engine.run_additional_kernels(ndrange_launcher, plan);

  
  return last_event;
}

template <class T, class Kernel, class BinaryReductionOp>
sycl::event
wg_model_reduction(sycl::queue &q, util::allocation_group &scratch_allocations,
                   T *output, T init, std::size_t target_num_groups,
                   std::size_t problem_size, Kernel k, BinaryReductionOp op,
                   const std::vector<sycl::event>& deps = {}) {
  return wg_model_reduction(q, scratch_allocations, output, init,
                                  target_num_groups, 128, problem_size, k, op, deps);
}


template <class T, class Kernel, class BinaryReductionOp>
sycl::event transform_reduce_impl(sycl::queue &q,
                                  util::allocation_group &scratch_allocations,
                                  T *output, T init, std::size_t n, Kernel k,
                                  BinaryReductionOp op,
                                  const std::vector<sycl::event>& deps) {
  sycl::device dev = q.get_device();
  std::size_t num_groups =
      dev.get_info<sycl::info::device::max_compute_units>() * 4;

  return wg_model_reduction(q, scratch_allocations, output, init, num_groups,
                            n, k, op, deps);

}

}

// Note: All transform_reduce variants defined here behave slightly different than STL
// variants:
// * If first==last, returns an event that is complete, even if preceding enqueued operations
//   are not yet complete.
// * If first==last, *out remains untouched and will not contain the init value.
template <class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp,
          class BinaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T *out,
                 T init, BinaryReductionOp reduce,
                 BinaryTransformOp transform,
                 const std::vector<sycl::event>& deps = {}) {
  if(first1 == last1)
    return sycl::event{};
  
  std::size_t n = std::distance(first1, last1);
  auto kernel = [=](sycl::id<1> idx, auto& reducer) {
    auto input_a = first1;
    auto input_b = first2;
    std::advance(input_a, idx[0]);
    std::advance(input_b, idx[0]);
    reducer.combine(transform(*input_a, *input_b));
  };

  return detail::transform_reduce_impl(q, scratch_allocations, out, init, n,
                                       kernel, reduce, deps);
}

template <class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt first, ForwardIt last, T* out, T init,
                 BinaryReductionOp reduce, UnaryTransformOp transform,
                 const std::vector<sycl::event>& deps = {}) {
  if(first == last)
    return sycl::event{};
  
  std::size_t n = std::distance(first, last);
  auto kernel = [=](sycl::id<1> idx, auto& reducer) {
    auto input = first;
    std::advance(input, idx[0]);
    reducer.combine(transform(*input));
  };

  return detail::transform_reduce_impl(q, scratch_allocations, out, init, n,
                                       kernel, reduce, deps);
}

template <class ForwardIt1, class ForwardIt2, class T>
sycl::event transform_reduce(sycl::queue &q,
                             util::allocation_group &scratch_allocations,
                             ForwardIt1 first1, ForwardIt1 last1,
                             ForwardIt2 first2, T *out, T init,
                             const std::vector<sycl::event>& deps = {}) {
  return transform_reduce(q, scratch_allocations, first1, last1, first2, out,
                          init, std::plus<T>{}, std::multiplies<T>{}, deps);
}


template <class ForwardIt, class T, class BinaryOp>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last, T *out, T init,
                   BinaryOp binary_op,
                   const std::vector<sycl::event>& deps = {}) {
  return transform_reduce(q, scratch_allocations, first, last, out, init,
                          binary_op, [](auto x) { return x; }, deps);
}

template <class ForwardIt, class T>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last, T *out, T init,
                   const std::vector<sycl::event>& deps = {}) {
  return reduce(q, scratch_allocations, first, last, out, init, std::plus<T>{}, deps);
}

template <class ForwardIt>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last,
                   typename std::iterator_traits<ForwardIt>::value_type *out,
                   const std::vector<sycl::event>& deps = {}) {
  return reduce(q, scratch_allocations, first, last, out,
                typename std::iterator_traits<ForwardIt>::value_type{}, deps);
}

///////////////////////////// scans /////////////////////////////////////

template <class InputIt, class OutputIt, class BinaryOp>
sycl::event
inclusive_scan(sycl::queue &q, util::allocation_group &scratch_allocations,
               InputIt first, InputIt last, OutputIt d_first, BinaryOp op,
               const std::vector<sycl::event> &deps = {}) {

  return scanning::scan<true>(q, scratch_allocations, first, last, d_first, op,
                              std::nullopt, deps);
}

template <class InputIt, class OutputIt, class BinaryOp, class T>
sycl::event
inclusive_scan(sycl::queue &q, util::allocation_group &scratch_allocations,
               InputIt first, InputIt last, OutputIt d_first, BinaryOp op,
               T init, const std::vector<sycl::event> &deps = {}) {
  return scanning::scan<true>(q, scratch_allocations, first, last, d_first, op,
                              init, deps);
}

template <class InputIt, class OutputIt>
sycl::event inclusive_scan(sycl::queue &q,
                           util::allocation_group &scratch_allocations,
                           InputIt first, InputIt last, OutputIt d_first,
                           const std::vector<sycl::event> &deps = {}) {
  return inclusive_scan(q, scratch_allocations, first, last, d_first,
                        std::plus<>{}, deps);
}

template <class InputIt, class OutputIt, class T, class BinaryOp>
sycl::event
exclusive_scan(sycl::queue &q, util::allocation_group &scratch_allocations,
               InputIt first, InputIt last, OutputIt d_first, T init,
               BinaryOp op, const std::vector<sycl::event> &deps = {}) {
  return scanning::scan<false>(q, scratch_allocations, first, last, d_first, op,
                               init, deps);
}

template <class InputIt, class OutputIt, class T>
sycl::event exclusive_scan(sycl::queue &q,
                           util::allocation_group &scratch_allocations,
                           InputIt first, InputIt last, OutputIt d_first,
                           T init, const std::vector<sycl::event> &deps = {}) {
  return exclusive_scan(q, scratch_allocations, first, last, d_first, init,
                        std::plus<>{}, deps);
}

template <class InputIt, class OutputIt, class BinaryOp, class UnaryOp>
sycl::event transform_inclusive_scan(
    sycl::queue &q, util::allocation_group &scratch_allocations, InputIt first,
    InputIt last, OutputIt d_first, BinaryOp binary_op, UnaryOp unary_op,
    const std::vector<sycl::event> &deps = {}) {
  return scanning::transform_scan<true>(q, scratch_allocations, first, last,
                                        d_first, unary_op, binary_op,
                                        std::nullopt, deps);
}

template <class InputIt, class OutputIt, class BinaryOp, class UnaryOp, class T>
sycl::event transform_inclusive_scan(
    sycl::queue &q, util::allocation_group &scratch_allocations, InputIt first,
    InputIt last, OutputIt d_first, BinaryOp binary_op, UnaryOp unary_op,
    T init, const std::vector<sycl::event> &deps = {}) {
  return scanning::transform_scan<true>(q, scratch_allocations, first, last,
                                        d_first, unary_op, binary_op,
                                        init, deps);
}

template <class InputIt, class OutputIt, class T, class BinaryOp, class UnaryOp>
sycl::event transform_exclusive_scan(
    sycl::queue &q, util::allocation_group &scratch_allocations, InputIt first,
    InputIt last, OutputIt d_first, T init, BinaryOp binary_op,
    UnaryOp unary_op, const std::vector<sycl::event> &deps = {}) {
  return scanning::transform_scan<false>(q, scratch_allocations, first, last,
                                         d_first, unary_op, binary_op, init,
                                         deps);
}

} // algorithms


#endif
