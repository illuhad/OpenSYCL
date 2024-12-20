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

#ifndef ACPP_ALGORITHMS_INDEX_SEARCH_HPP
#define ACPP_ALGORITHMS_INDEX_SEARCH_HPP

#include <type_traits>

namespace hipsycl::algorithms::binary_searching {

// Same as std::lower_bound, but works in terms of indices
template< class IndexT, class T, class DataGetter,
          class Compare >
constexpr IndexT index_lower_bound( IndexT first, IndexT last,
                                    const T& value, DataGetter load, Compare comp ) {
  using SignedIndexT = typename std::make_signed<IndexT>::type;

  IndexT current;
  SignedIndexT count, step;
  count = last - first;

  while (count > 0) {
    current = first;
    step = count / 2;
    current += step;

    if (comp(load(current), value)) {
      first = ++current;
      count -= step + 1;
    } else
      count = step;
  }

  return first;
}


// Same as std::upper_bound, but works in terms of indices
template< class IndexT, class T, class DataGetter,
          class Compare >
constexpr IndexT index_upper_bound( IndexT first, IndexT last,
                                    const T& value, DataGetter load, Compare comp ) {
  using SignedIndexT = typename std::make_signed<IndexT>::type;

  IndexT current;
  SignedIndexT count, step;
  count = last - first;

  while (count > 0) {
    current = first;
    step = count / 2;
    current += step;

    if (!comp(value, load(current))) {
      first = ++current;
      count -= step + 1;
    } else
      count = step;
  }

  return first;
}

}

#endif
