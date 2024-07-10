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

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_replace, enable_unified_shared_memory)

template <class Generator>
void test_replace(std::size_t problem_size, Generator gen, int old_val,
                  int new_val) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);
  std::vector<int> host_data = data;

  std::replace(std::execution::par_unseq, data.begin(), data.end(), old_val,
               new_val);
  std::replace(host_data.begin(), host_data.end(), old_val,
               new_val);               
  BOOST_CHECK(data == host_data);
}


BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_replace(0, [](int i){return i;}, 3, 2);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_replace(1, [](int i){return 42;}, 42, 4);
  test_replace(1, [](int i){return 42;}, 2, 4);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_replace(1000, [](int i){return i%10+3;}, 20, 4);
  test_replace(1000, [](int i){return i%10+3;}, -2, 4);
}



BOOST_AUTO_TEST_SUITE_END()
