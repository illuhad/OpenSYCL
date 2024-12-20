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

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_copy_if, enable_unified_shared_memory)


template<class Generator>
void test_copy_if(std::size_t problem_size, Generator&& gen) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = gen(i);
  }

  std::vector<int> dest_device(problem_size);
  std::vector<int> dest_host(problem_size);

  auto p = [](auto x) { return x % 2 == 0; };

  auto ret = std::copy_if(std::execution::par_unseq, data.begin(), data.end(),
                          dest_device.begin(), p);
  auto ret_reference = std::copy_if(data.begin(), data.end(), dest_host.begin(), p);

  BOOST_CHECK(std::distance(dest_device.begin(), ret) ==
              std::distance(dest_host.begin(), ret_reference));

  BOOST_CHECK(dest_device == dest_host);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_copy_if(0, [](int i){return i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_copy_if(1, [](int i){return i+3;});
}

BOOST_AUTO_TEST_CASE(par_unseq_none) {
  test_copy_if(1000, [](int i){return 1;});
}

BOOST_AUTO_TEST_CASE(par_unseq_all) {
  test_copy_if(1000, [](int i){return 2*i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_half) {
  test_copy_if(1000, [](int i){return i;});
}

BOOST_AUTO_TEST_SUITE_END()
