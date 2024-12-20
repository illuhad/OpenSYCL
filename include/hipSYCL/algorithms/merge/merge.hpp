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

#ifndef ACPP_ALGORITHMS_MERGE_HPP
#define ACPP_ALGORITHMS_MERGE_HPP

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

#include "../sort/bitonic_sort.hpp"
#include "merge_path.hpp"

namespace hipsycl::algorithms::merging {

namespace detail {

template <class ForwardIt1, class ForwardIt2, class OutputIt, class Compare, class Size>
void sequential_merge(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                      ForwardIt2 last2, OutputIt out, Compare comp, Size max_num_merged) {

  auto initial_out = out;
  auto copy_remaining = [&](auto first, auto last) {
    for (; first != last && (std::distance(initial_out, out) < max_num_merged);
         ++first, ++out)
      *out = *first;
  };

  for (; first1 != last1 && (std::distance(initial_out, out) < max_num_merged);
       ++out) {
    if(first2 == last2) {
      copy_remaining(first1, last1);
      return;
    } else {
      auto f1 = *first1;
      auto f2 = *first2;
      if(comp(f1, f2)) {
        *out = f1;
        ++first1;
      } else {
        *out = f2;
        ++first2;
      }
    }
  }
  copy_remaining(first2, last2);
}


/// Decomposes the problem into N independent merges of given size, and
/// then runs sequential merge on them. This might be a good strategy on CPU.
///
/// Precondition: distance(fist1, last1) > 0 && distance(first2, last2) > 0.
/// Otherwise we cannot run the merge path algorithm for decomposing the merge.
template <class RandomIt1, class RandomIt2, class OutputIt, class Compare>
void segmented_merge(
    RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2,
    OutputIt out, Compare comp, std::size_t segment_index,
    std::size_t segment_chunk_size) {

  std::size_t p1 = 0;
  std::size_t p2 = 0;

  merge_path::nth_independent_merge_begin(first1, last1, first2, last2, comp,
                                          segment_index,
                                          segment_chunk_size, p1, p2);

  auto chunk_first1 = first1;
  auto chunk_first2 = first2;

  std::advance(chunk_first1, p1);
  std::advance(chunk_first2, p2);

  auto chunk_last1 = chunk_first1;
  auto chunk_last2 = chunk_first2;

  std::advance(chunk_last1, std::min(segment_chunk_size,
                              std::distance(first1, last1) - p1));
  std::advance(chunk_last2, std::min(segment_chunk_size,
                              std::distance(first2, last2) - p2));

  std::size_t chunk_out_offset = segment_index * segment_chunk_size;
  auto chunk_out = out;
  std::advance(chunk_out, chunk_out_offset);

  sequential_merge(chunk_first1, chunk_last1, chunk_first2, chunk_last2,
                    chunk_out, comp, segment_chunk_size);
}

template <class RandomIt1, class RandomIt2, class Compare,
          class IndexT>
void store_segment_begin(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2,
                         RandomIt2 last2, Compare comp,
                         IndexT segment_index,
                         IndexT segment_size, 
                         IndexT* first_out1, IndexT* first_out2,
                         std::size_t offset = 0 // Will be added to result
                         ) {

  auto problem_size1 = std::distance(first1, last1);
  auto problem_size2 = std::distance(first2, last2);

  if(problem_size1 == 0) {
    first_out1[segment_index] = offset + 0;
    first_out2[segment_index] = offset + segment_index * segment_size;
  } else if(problem_size2 == 0) {
    first_out2[segment_index] = offset + 0;
    first_out1[segment_index] = offset + segment_index * segment_size;
  } else {

    IndexT p1 = 0;
    IndexT p2 = 0;

    merge_path::nth_independent_merge_begin(first1, last1, first2, last2, comp,
                                            segment_index,
                                            segment_size, p1, p2);

    first_out1[segment_index] = p1 + offset;
    first_out2[segment_index] = p2 + offset;
  }
}

template <class RandomIt1, class RandomIt2, class OutputIt, class IndexT,
          class Group, class Compare>
void segment_merge_by_group_sort(
    Group grp, // SYCL group object. Group size must correspond to segment size,
               // grp id must correspond to segment index.
    RandomIt1 first1, // Iterators describing the *whole* merge range, not just
                      // this group
    RandomIt1 last1, RandomIt2 first2, RandomIt2 last2, OutputIt out,
    Compare comp, IndexT *segments_begin1, IndexT *segments_begin2,
    IndexT num_segments,
    typename std::iterator_traits<RandomIt1>::value_type *local_mem = nullptr) {

  int lid = grp.get_local_linear_id();
  auto grp_id = grp.get_group_linear_id();
  int grp_size = grp.get_local_linear_range();

  std::size_t segment_begin1 = segments_begin1[grp_id];
  std::size_t segment_begin2 = segments_begin2[grp_id];

  RandomIt1 group_first1 = first1;
  std::advance(group_first1, segment_begin1);
  RandomIt2 group_first2 = first2;
  std::advance(group_first2, segment_begin2);

  std::size_t segment_end1 =
      std::distance(group_first1, last1) + segment_begin1;
  std::size_t segment_end2 =
      std::distance(group_first2, last2) + segment_begin2;
  if(grp_id < num_segments - 1) {
    segment_end1 = segments_begin1[grp_id + 1];
    segment_end2 = segments_begin2[grp_id + 1];
  }

  RandomIt1 group_last1 = first1;
  std::advance(group_last1, segment_end1);

  RandomIt2 group_last2 = first2;
  std::advance(group_last2, segment_end2);

  OutputIt group_out = out;
  std::advance(group_out, grp_id * grp_size);
  
  int input_size1 = std::distance(group_first1, group_last1);
  int input_size2 = std::distance(group_first2, group_last2);
  auto local_problem_size = input_size1 + input_size2;

  auto load = [](auto it, auto idx) {
    std::advance(it, idx);
    return *it;
  };

  auto store = [](auto it, auto idx, auto v) {
    std::advance(it, idx);
    *it = v;
  };

  auto barrier = [&](){
    sycl::group_barrier(grp);
  };

  if(local_mem) {
    if(lid < input_size1)
      local_mem[lid] = load(group_first1, lid);
    if(lid < input_size2)
      local_mem[lid + input_size1] = load(group_first2, lid);
    
    barrier();
    sorting::bitonic_group_sort(local_mem, grp_size, local_problem_size,
                                lid, barrier, comp);
    if(lid < local_problem_size)
      store(group_out, lid, local_mem[lid]);
  } else {
    if(lid < input_size1)
      store(group_out, lid, load(group_first1, lid));
    if(lid < input_size2)
      store(group_out, lid + input_size1, load(group_first2, lid));

    barrier();
    sorting::bitonic_group_sort(group_out, grp_size, local_problem_size,
                                lid, barrier, comp);
  }
}
}

/// Precondition: distance(fist1, last1) > 0 && distance(first2, last2) > 0.
/// Otherwise we cannot run the merge path algorithm for decomposing the merge.
template <class RandomIt1, class RandomIt2, class OutputIt, class Compare>
sycl::event segmented_merge(sycl::queue &q, RandomIt1 first1, RandomIt1 last1,
                            RandomIt2 first2, RandomIt2 last2, OutputIt out,
                            Compare comp,
                            std::size_t segment_chunk_size = 128,
                            const std::vector<sycl::event> &deps = {}) {

  //detail::print_merge_matrix(first1, last1, first2, last2, comp);

  std::size_t problem_size = merge_path::num_independent_merges(
      first1, last1, first2, last2, segment_chunk_size);

  if(problem_size == 0)
    return sycl::event{};

  return q.parallel_for(sycl::range{problem_size}, deps, [=](sycl::id<1> idx) {
    detail::segmented_merge(first1, last1, first2, last2, out, comp, idx.get(0),
                            segment_chunk_size);
  });
}

// Assumes that distance(first1,last1)!=0 && distance(first2,last2)!=0
template <class RandomIt1, class RandomIt2, class OutputIt, class Compare>
sycl::event
hierarchical_hybrid_merge(sycl::queue &q, util::allocation_group &scratch,
                          RandomIt1 first1, RandomIt1 last1, RandomIt2 first2,
                          RandomIt2 last2, OutputIt out, Compare comp,
                          std::size_t segment_chunk_size = 128,
                          const std::vector<sycl::event> &deps = {}) {

  //detail::print_merge_matrix(first1, last1, first2, last2, comp);
  
  std::size_t num_merges = merge_path::num_independent_merges(
      first1, last1, first2, last2, segment_chunk_size);
  std::size_t* segment_start_scratch = scratch.obtain<std::size_t>(2 * num_merges);

  std::size_t* segment_start_scratch1 = segment_start_scratch;
  std::size_t* segment_start_scratch2 = segment_start_scratch + num_merges;

  if(num_merges == 0)
    return sycl::event{};

  sycl::event store_segment_begin_evt =
      q.parallel_for(sycl::range{num_merges}, deps, [=](sycl::id<1> idx) {
        detail::store_segment_begin(
            first1, last1, first2, last2, comp, idx.get(0), segment_chunk_size,
            segment_start_scratch1, segment_start_scratch2);
      });
  
  std::size_t group_size = segment_chunk_size;

  auto deps2 = deps;
  if(!q.is_in_order())
    deps2.push_back(store_segment_begin_evt);

  sycl::event group_sort_evt;
  
  using T = typename std::iterator_traits<RandomIt1>::value_type;
  // TODO: Better to actually check local mem capacity
  if(sizeof(*first1) <= 16) {
    group_sort_evt = q.submit([&](sycl::handler& cgh) {

      sycl::local_accessor<T> local_mem {group_size, cgh};

      cgh.depends_on(deps2);
      cgh.parallel_for(sycl::nd_range<1>{num_merges * group_size, group_size},
                       [=](sycl::nd_item<1> idx) {
                         detail::segment_merge_by_group_sort(
                             idx.get_group(), first1, last1, first2, last2, out,
                             comp, segment_start_scratch1,
                             segment_start_scratch2, num_merges,
                             &(local_mem[0]));
                       });
    });
  } else {
    group_sort_evt = q.parallel_for(
      sycl::nd_range<1>{num_merges * group_size, group_size}, deps2,
      [=](sycl::nd_item<1> idx) {
        detail::segment_merge_by_group_sort(idx.get_group(), first1, last1, first2,
                                    last2, out, comp, segment_start_scratch1,
                                    segment_start_scratch2, num_merges);
      });
  }

  return group_sort_evt;
}
}





#endif
