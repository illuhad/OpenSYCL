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


#include <numeric>
#include <limits>
#include <type_traits>

#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(marray_tests, reset_device_fixture)

using marray_test_types = boost::mpl::list<
  float,
  double,
  short,
  unsigned,
  int,
  long long,
  unsigned long long>;

template <class T> T get_input_value1(int i) { return static_cast<T>(i + 1); }
template <class T> T get_input_value2(int i) { return static_cast<T>(2*i + i%3); }

template<class T, int N>
void test(sycl::queue& q) {

  sycl::marray<T,N> a, b;

  for(int i = 0; i < N; ++i) {
    a[i] = get_input_value1<T>(i);
    b[i] = get_input_value2<T>(i);
  }

  constexpr std::size_t num_tests = 4;
  sycl::marray<T, N>* output = sycl::malloc_device<sycl::marray<T,N>>(num_tests, q);

  q.single_task([=](){
    output[0] = a + b;
    output[1] = a - b;
    output[2] = a * b;
    output[3] = a / (b + static_cast<T>(1));
  }).wait();

  std::vector<sycl::marray<T,N>> results(num_tests);
  q.copy(output, results.data(), num_tests).wait();
  sycl::free(output, q);

  auto tolerance = boost::test_tools::tolerance(0.0001);
  // divisions may be less precise with OpenCL backend
  // boost.test seems to not work with double tolerance values
  // when the data type of the test is float :(
  auto get_div_tolerance = [&](){
    if constexpr(std::is_same_v<T, float>) {
      return boost::test_tools::tolerance(0.001f);
    } else {
      return tolerance;
    }
  };
    
  
  for(int i = 0; i < N; ++i) {
    BOOST_TEST(results[0][i] == a[i] + b[i], tolerance);
    BOOST_TEST(results[1][i] == a[i] - b[i], tolerance);
    BOOST_TEST(results[2][i] == a[i] * b[i], tolerance);
    
    T expected_div = a[i] / (b[i] + static_cast<T>(1));
    BOOST_TEST(results[3][i] == expected_div, get_div_tolerance());
  }

}

BOOST_AUTO_TEST_CASE_TEMPLATE(marray_ops, T, marray_test_types::type) {
  sycl::queue q;
  
  test<T, 1>(q);
  test<T, 3>(q);
  test<T, 4>(q);
  test<T, 13>(q);
}

template<class T, std::size_t N>
bool verify_marray_content(const sycl::marray<T,N>& x, const std::array<T,N>& ref) {
  for(int i = 0; i < N; ++i)
    if(x[i] != ref[i])
      return false;
  return true;
}

BOOST_AUTO_TEST_CASE(marray_constructor) {
  sycl::marray<int, 1> v1 {3};
  sycl::marray<int, 3> v2 {1};
  sycl::marray<int, 4> v3 {5, v2};
  sycl::marray<int, 4> v4 {v1, v2};
  sycl::marray<int, 8> v5{v2, v1, 0, v2};

  BOOST_TEST(verify_marray_content(v1, {3}));
  BOOST_TEST(verify_marray_content(v2, {1, 1, 1}));
  BOOST_TEST(verify_marray_content(v3, {5, 1, 1, 1}));
  BOOST_TEST(verify_marray_content(v4, {3, 1, 1, 1}));
  BOOST_TEST(verify_marray_content(v5, {1, 1, 1, 3, 0, 1, 1, 1}));
}

#define MARRAY_ALIAS_SAME(type, storage_type, elements)                 \
  (std::is_same_v<sycl::m##type##elements, sycl::marray<storage_type, elements>>)

#define MARRAY_ALIAS_CHECK(type, storage_type)        \
  MARRAY_ALIAS_SAME(type, storage_type, 2) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 3) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 4) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 8) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 16)

BOOST_AUTO_TEST_CASE(marray_aliases) {
  BOOST_CHECK(MARRAY_ALIAS_CHECK(char, int8_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(uchar, uint8_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(short, int16_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(ushort, uint16_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(int, int32_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(uint, uint32_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(long, int64_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(ulong, uint64_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(half, sycl::half));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(float, float));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(double, double));
}

BOOST_AUTO_TEST_CASE(marray_implicit_conversion) {
  constexpr int val = 42;

  const auto f = [val](int x) {
    BOOST_CHECK_EQUAL(x, val);
  };

  sycl::marray<int, 1> arr(val);
  f(arr);
}

#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
// nvc++ seems to have a problem with these tests
BOOST_AUTO_TEST_CASE(marray_constexpr) {
  constexpr sycl::marray arr1{42};
  constexpr sycl::marray<int, 2> arr2{arr1, arr1};
  constexpr sycl::marray<int, 4> arr3{123, arr2, 456};
}
#endif

BOOST_AUTO_TEST_SUITE(marray_builtins)

#define SYCL_BUILTIN(builtin, ...) sycl::builtin(__VA_ARGS__)
#define STD_BUILTIN(builtin, ...) std::builtin(__VA_ARGS__)

#define MARRAY_BUILTINS_UNARY_TEST(builtin, low, high)                                             \
  BOOST_AUTO_TEST_CASE(marray_builtin_test_##builtin) {                                            \
    std::default_random_engine generator;                                                          \
    std::uniform_real_distribution<double> distribution(low, high);                                \
    sycl::queue q;                                                                                 \
    auto *data = sycl::malloc_shared<sycl::marray<double, 32>>(1, q);                              \
    auto *result = sycl::malloc_shared<sycl::marray<double, 32>>(1, q);                            \
    for (int i = 0; i < data[0].size(); ++i)                                                       \
      data[0][i] = distribution(generator);                                                        \
    sycl::queue{}.single_task([=]() { result[0] = SYCL_BUILTIN(builtin, data[0]); }).wait();       \
    for (int i = 0; i < data[0].size(); ++i)                                                       \
      BOOST_TEST(result[0][i] == STD_BUILTIN(builtin, data[0][i]),                                 \
                 boost::test_tools::tolerance(1e-8));                                              \
  }

#define MARRAY_BUILTINS_BINARY_TEST(builtin, low, high)                                            \
  BOOST_AUTO_TEST_CASE(marray_builtin_test_##builtin) {                                            \
    std::default_random_engine generator;                                                          \
    std::uniform_real_distribution<double> distribution(low, high);                                \
    sycl::queue q;                                                                                 \
    auto *data1 = sycl::malloc_shared<sycl::marray<double, 32>>(1, q);                             \
    auto *data2 = sycl::malloc_shared<sycl::marray<double, 32>>(1, q);                             \
    auto *result = sycl::malloc_shared<sycl::marray<double, 32>>(1, q);                            \
    for (int i = 0; i < data1[0].size(); ++i)                                                      \
      data1[0][i] = distribution(generator);                                                       \
    for (int i = 0; i < data2[0].size(); ++i)                                                      \
      data2[0][i] = distribution(generator);                                                       \
    sycl::queue{}                                                                                  \
        .single_task([=]() { result[0] = SYCL_BUILTIN(builtin, data1[0], data2[0]); })             \
        .wait();                                                                                   \
    for (int i = 0; i < data1[0].size(); ++i)                                                      \
      BOOST_TEST(result[0][i] == STD_BUILTIN(builtin, data1[0][i], data2[0][i]),                   \
                 boost::test_tools::tolerance(1e-8));                                              \
  }

#define MARRAY_TEST_MIN std::numeric_limits<double>::lowest()
#define MARRAY_TEST_MAX std::numeric_limits<double>::max()

MARRAY_BUILTINS_UNARY_TEST(acos, -1., 1.)
MARRAY_BUILTINS_UNARY_TEST(acosh, 1., MARRAY_TEST_MAX)
// MARRAY_BUILTINS_UNARY_TEST(acospi) // No acospi in std::
MARRAY_BUILTINS_UNARY_TEST(asin, -1., 1.)
MARRAY_BUILTINS_UNARY_TEST(asinh, MARRAY_TEST_MIN, MARRAY_TEST_MAX)

MARRAY_BUILTINS_UNARY_TEST(atan, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(atanh, -1, 1)
// MARRAY_BUILTINS_UNARY_TEST(atanpi) // No atanpi in std::
MARRAY_BUILTINS_UNARY_TEST(cbrt, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(ceil, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(cos, 0, 2 * 3.14)
MARRAY_BUILTINS_UNARY_TEST(cosh, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_UNARY_TEST(cospi) // No cospi in std::
MARRAY_BUILTINS_UNARY_TEST(erfc, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(erf, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(exp, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(exp2, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_UNARY_TEST(exp10) // No exp10 in std::
MARRAY_BUILTINS_UNARY_TEST(expm1, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(fabs, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(floor, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_UNARY_TEST(lgamma, 0.1, 10)  // TODO: This test fails and the result of
// sycl::lgamma are pretty different from std::lgamma, are they the same?
MARRAY_BUILTINS_UNARY_TEST(log, 0.1, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(log2, 0.1, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(log10, 0.1, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(log1p, -1, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(logb, 0.1, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(rint, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(round, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_UNARY_TEST(rsqrt) // No rsqrt in std::
MARRAY_BUILTINS_UNARY_TEST(sin, 0, 2 * 3.14)
MARRAY_BUILTINS_UNARY_TEST(sinh, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_UNARY_TEST(sinpi) // No sinpi in std::
MARRAY_BUILTINS_UNARY_TEST(sqrt, 0, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(tan, 0, 2 * 3.14)
MARRAY_BUILTINS_UNARY_TEST(tanh, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(tgamma, 0.1, MARRAY_TEST_MAX)
MARRAY_BUILTINS_UNARY_TEST(trunc, MARRAY_TEST_MIN, MARRAY_TEST_MAX)

MARRAY_BUILTINS_BINARY_TEST(atan2, 0.1, 1)
// MARRAY_BUILTINS_BINARY_TEST(atan2pi) // not atan2pi in std::
MARRAY_BUILTINS_BINARY_TEST(copysign, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_BINARY_TEST(fdim, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_BINARY_TEST(fmax, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_BINARY_TEST(fmin, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_BINARY_TEST(fmod, -1, 1)
MARRAY_BUILTINS_BINARY_TEST(hypot, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_BINARY_TEST(maxmag, MARRAY_TEST_MIN, MARRAY_TEST_MAX) // No maxmag in std::
// MARRAY_BUILTINS_BINARY_TEST(minmag, MARRAY_TEST_MIN, MARRAY_TEST_MAX) // No minmag in std::
MARRAY_BUILTINS_BINARY_TEST(nextafter, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
MARRAY_BUILTINS_BINARY_TEST(pow, MARRAY_TEST_MIN, MARRAY_TEST_MAX)
// MARRAY_BUILTINS_BINARY_TEST(powr, MARRAY_TEST_MIN, MARRAY_TEST_MAX) // No powr in std::
MARRAY_BUILTINS_BINARY_TEST(remainder, -1, 1)

BOOST_AUTO_TEST_SUITE_END() // marray_builtins

BOOST_AUTO_TEST_SUITE_END()
