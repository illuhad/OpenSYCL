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

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>
#include <functional>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_sort, enable_unified_shared_memory)

template <class Policy, class Generator, class Comp = std::less<>>
void test_sort(Policy &&pol, std::size_t problem_size, Generator gen,
               Comp comp = {}) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);
  std::vector<int> host_data = data;

  std::sort(pol, data.begin(), data.end(), comp);
  
  BOOST_CHECK(std::is_sorted(data.begin(), data.end(), comp));
  std::sort(host_data.begin(), host_data.end(), comp);
  BOOST_CHECK(host_data == data);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_sort(std::execution::par_unseq, 0, [](int i){return 0;});
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_sort(std::execution::par_unseq, 1, [](int i){return i;});
}


BOOST_AUTO_TEST_CASE(par_unseq_pow2_descending) {
  test_sort(std::execution::par_unseq, 1024, [](int i){return -i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_pow2_ascending) {
  test_sort(std::execution::par_unseq, 1024, [](int i){return i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_non_pow2_descending) {
  test_sort(std::execution::par_unseq, 1000, [](int i){return -i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_non_pow2_ascending) {
  test_sort(std::execution::par_unseq, 1000, [](int i){return i;});
}

BOOST_AUTO_TEST_SUITE_END()
