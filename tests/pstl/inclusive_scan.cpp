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

#include <cstdint>
#include <numeric>
#include <execution>
#include <utility>
#include <vector>
#include <array>
#include <type_traits>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_inclusive_scan, enable_unified_shared_memory)

template<class T, std::size_t PaddingSize>
struct non_default_constructible {
public:
  static auto make(T x){
    non_default_constructible<T, PaddingSize> t; t.x = x;
    return t;
  }

  T get() const {
    return x;
  }
private:
  non_default_constructible(){}
  T x;
  std::array<uint64_t, (PaddingSize + sizeof(uint64_t) - 1) / sizeof(uint64_t)>
      padding;
};

template<class T>
struct non_default_constructible<T, 0> {
public:
  static auto make(T x){
    non_default_constructible<T,0> t; t.x = x;
    return t;
  }

  T get() const {
    return x;
  }
private:
  non_default_constructible(){}
  T x;
};

template <class T, std::size_t PaddingSize>
bool operator==(const non_default_constructible<T, PaddingSize> &a,
                const non_default_constructible<T, PaddingSize> &b) {
  return a.get() == b.get();
}

template <class T, std::size_t PaddingSize>
bool operator!=(const non_default_constructible<T, PaddingSize> &a,
                const non_default_constructible<T, PaddingSize> &b) {
  return a.get() != b.get();
}

template<class Policy, class Generator, class BinOp, class T>
void test_scan(Policy&& pol, Generator&& gen, T init, BinOp op, std::size_t size) {
  std::vector<T> data;
  for(std::size_t i = 0; i < size; ++i)
    data.push_back(gen(i));

  std::vector<T> reference0;
  if constexpr(std::is_same_v<BinOp, std::plus<>>) {
    reference0 = data;
    std::inclusive_scan(data.begin(), data.end(), reference0.begin());
  }

  std::vector<T> reference1 = data;
  std::inclusive_scan(data.begin(), data.end(), reference1.begin(), op);

  std::vector<T> reference2 = data;
  std::inclusive_scan(data.begin(), data.end(), reference2.begin(), op, init);

  std::vector<T> device_result0 = data;
  if constexpr(std::is_same_v<BinOp, std::plus<>>) {
    BOOST_CHECK(std::inclusive_scan(pol, data.begin(), data.end(),
                                    device_result0.begin()) ==
                device_result0.end());
  }

  std::vector<T> device_result1 = data;
    BOOST_CHECK(std::inclusive_scan(pol, data.begin(), data.end(),
                                    device_result1.begin(), op) ==
                device_result1.end());

  std::vector<T> device_result2 = data;
    BOOST_CHECK(std::inclusive_scan(pol, data.begin(), data.end(),
                                    device_result2.begin(), op, init) ==
                device_result2.end());
  
  if constexpr(std::is_same_v<BinOp, std::plus<>>) {
    BOOST_CHECK(reference0 == device_result0);
  }
  BOOST_CHECK(reference1 == device_result1);
  BOOST_CHECK(reference2 == device_result2);
}

inline auto get_default_generator() {
  return [](std::size_t i) {
    return static_cast<int>(i);
  };
}

template<class T>
inline auto get_non_constructible_generator() {
  return [](std::size_t i) {
    return T::make(i);
  };
}

template<class T>
auto get_non_constructible_bin_op() {
  return [](auto a, auto b){
    return T::make(a.get() + b.get());
  };
}

template<int ProblemSize, class Policy>
void run_all_tests(Policy&& pol) {
  test_scan(std::execution::par_unseq, get_default_generator(), 3,
            std::plus<>{}, ProblemSize);
  test_scan(
      std::execution::par_unseq, get_default_generator(), 3ull,
      [](auto a, auto b) { return a * b; }, ProblemSize);
  
  using non_constructible_t = non_default_constructible<std::size_t, 0>;
  test_scan(std::execution::par_unseq,
            get_non_constructible_generator<non_constructible_t>(), 
            non_constructible_t::make(3ull),
            get_non_constructible_bin_op<non_constructible_t>(), ProblemSize);
  
  using massive_non_constructible_t =
      non_default_constructible<std::size_t, 1024>;
  test_scan(std::execution::par_unseq,
            get_non_constructible_generator<massive_non_constructible_t>(),
            massive_non_constructible_t::make(3ull),
            get_non_constructible_bin_op<massive_non_constructible_t>(),
            ProblemSize);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  run_all_tests<0>(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  run_all_tests<1>(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_incomplete_single_work_group) {
  run_all_tests<127>(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_multiple_groups_incomplete) {
  run_all_tests<1000>(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_large) {
  run_all_tests<1024*1024>(std::execution::par_unseq);
}


BOOST_AUTO_TEST_CASE(par_empty) {
  run_all_tests<0>(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  run_all_tests<1>(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_incomplete_single_work_group) {
  run_all_tests<127>(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_multiple_groups_incomplete) {
  run_all_tests<1000>(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_large) {
  run_all_tests<1024*1024>(std::execution::par);
}




BOOST_AUTO_TEST_SUITE_END()
