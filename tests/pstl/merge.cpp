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
#include <cstdlib>
#include <execution>
#include <utility>
#include <vector>
#include <functional>
#include <random>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_merge, enable_unified_shared_memory)

template <class Policy, class Generator1, class Generator2,
          class Comp = std::less<>>
void test_merge(Policy &&pol, std::size_t size1, std::size_t size2,
                Generator1 gen1, Generator2 gen2, Comp comp = {}) {
  std::vector<int> data1(size1);
  std::vector<int> data2(size2);
  std::vector<int> out(size1+size2);

  for(int i = 0; i < size1; ++i)
    data1[i] = gen1(i);
  for(int i = 0; i < size2; ++i)
    data2[i] = gen2(i);
  std::vector<int> host_out = out;

  auto ret = std::merge(pol, data1.begin(), data1.end(), data2.begin(), data2.end(),
             out.begin(), comp);
  auto host_ret = std::merge(data1.begin(), data1.end(), data2.begin(), data2.end(),
             host_out.begin(), comp);

  BOOST_CHECK(host_out == out);
  BOOST_CHECK(ret == out.begin() + std::distance(host_out.begin(), host_ret));

  for(int i = 0; i < out.size(); ++i) {
    auto expected = host_out[i];
    auto result = out[i];
    if(result != expected)
      std::cout << i << ": " << expected << " != " << result << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_merge(
      std::execution::par_unseq, 0, 0, [](int i) { return 0; },
      [](int i) { return 0; });
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_merge(
      std::execution::par_unseq, 1, 0, [](int i) { return 2; },
      [](int i) { return 3; });
  test_merge(
      std::execution::par_unseq, 0, 1, [](int i) { return 2; },
      [](int i) { return 3; });
  test_merge(
      std::execution::par_unseq, 1, 1, [](int i) { return 2; },
      [](int i) { return 3; });
}

BOOST_AUTO_TEST_CASE(par_unseq_trivial_merge) {

  auto a = [](int i) { return i; };
  auto b = [](int i) { return i + 1024; };

  test_merge(
      std::execution::par_unseq, 1024, 1024, a, b);

  test_merge(
      std::execution::par_unseq, 1024, 1024, b, a);
}

std::vector<int> generate_sorted_random_numbers(int amount, int seed=123) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist;

  std::vector<int> data;
  for(int i = 0; i < amount; ++i)
    data.push_back(dist(gen));
  std::sort(data.begin(), data.end());
  return data;
}

BOOST_AUTO_TEST_CASE(par_unseq_same_size) {
  std::size_t s1 = 256;
  std::size_t s2 = 256;

  auto v1 = generate_sorted_random_numbers(s1, 123);
  auto v2 = generate_sorted_random_numbers(s2, 42);

  test_merge(
      std::execution::par_unseq, s1, s2, [&](int i) { return v1[i]; },
      [&](int i) { return v2[i]; });
}

BOOST_AUTO_TEST_CASE(par_unseq_same_data) {
  std::size_t s1 = 1024;
  std::size_t s2 = 1024;

  auto v1 = generate_sorted_random_numbers(s1, 123);
  auto v2 = generate_sorted_random_numbers(s2, 123);

  test_merge(
      std::execution::par_unseq, s1, s2, [&](int i) { return v1[i]; },
      [&](int i) { return v2[i]; });
}

BOOST_AUTO_TEST_CASE(par_unseq_v1_larger) {
  std::size_t s1 = 1932;
  std::size_t s2 = 1000;

  auto v1 = generate_sorted_random_numbers(s1, 123);
  auto v2 = generate_sorted_random_numbers(s2, 42);

  test_merge(
      std::execution::par_unseq, s1, s2, [&](int i) { return v1[i]; },
      [&](int i) { return v2[i]; });
}

BOOST_AUTO_TEST_CASE(par_unseq_v2_larger) {
  std::size_t s1 = 1000;
  std::size_t s2 = 1932;

  auto v1 = generate_sorted_random_numbers(s1, 123);
  auto v2 = generate_sorted_random_numbers(s2, 42);

  test_merge(
      std::execution::par_unseq, s1, s2, [&](int i) { return v1[i]; },
      [&](int i) { return v2[i]; });
}

BOOST_AUTO_TEST_SUITE_END()
