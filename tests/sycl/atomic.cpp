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

#include "sycl_test_suite.hpp"
#include <cstddef>
#include <limits>
#include <type_traits>

using namespace cl;

BOOST_FIXTURE_TEST_SUITE(atomic_tests, reset_device_fixture)

template<class T, class IntT>
T int_to_t(IntT i) {
  T result{};

  if constexpr(std::is_pointer_v<T>) {
    result = reinterpret_cast<T>(i);
  } else {
    result = static_cast<T>(i);
  }

  return result;
}

template<class T>
unsigned long long t_to_int(T x) {
  unsigned long long result = 0ull;

  if constexpr(std::is_pointer_v<T>) {
    result = reinterpret_cast<unsigned long long>(x);
  } else {
    result = static_cast<unsigned long long>(x);
  }
  
  return result;
}

using exchange_test_types =
    boost::mpl::list<int, unsigned int, long, unsigned long, long long,
                     unsigned long long, float, double, int *>;
// mainly compile test
BOOST_AUTO_TEST_CASE_TEMPLATE(load_store_exchange, Type,
                              exchange_test_types::type) {
  sycl::queue q;

  Type initial = int_to_t<Type>(0);
  sycl::buffer<Type> b{&initial, sycl::range{1}};

  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{b, cgh, sycl::read_write};
    cgh.single_task([=]() {
      sycl::atomic_ref<Type, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          r{acc[0]};
      unsigned long long x = t_to_int(r.load());

      // There seems to be a bug in clang related to incorrect
      // code generation at least on CUDA for floating point
      // stores.
      if constexpr(!std::is_floating_point_v<Type>)
        r.store(int_to_t<Type>(x+2));
      
      r.exchange(int_to_t<Type>(x+1));
    });
  });
  sycl::host_accessor hacc{b};

  BOOST_CHECK(t_to_int(hacc[0]) == 1);
}


template <class T, class AtomicOp, class Verifier>
void atomic_device_reduction_test(AtomicOp op, Verifier v,
                                  std::string_view type_name, std::string_view op_name,
                                  std::function<int(int)> init = [](int t) { return -t; }) {
  
  sycl::queue q;

  const std::size_t size = 2048;
  sycl::buffer<T> b{sycl::range{size}};
  {
    sycl::host_accessor hacc{b};
    for(std::size_t i = 0; i < size; ++i) {
      hacc[i] = int_to_t<T>(init(i));
    }
  }
  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{b, cgh, sycl::read_write};
    cgh.parallel_for(sycl::range{size}, [=](sycl::id<1> idx){
      if(idx.get(0) != 0) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                        sycl::memory_scope::device> r{acc[0]};
        if constexpr(std::is_pointer_v<T>) {
          op(r, reinterpret_cast<std::ptrdiff_t>(acc[idx]));
        } else {
          op(r, acc[idx]);
        }
      }
    });
  });
  {
    sycl::host_accessor hacc{b};
    T expected = int_to_t<T>(init(0));
    for(std::size_t i = 1; i < size; ++i) {
      if constexpr(std::is_pointer_v<T>) {
        v(expected, static_cast<std::ptrdiff_t>(init(i)));
      } else {
        v(expected, int_to_t<T>(init(i)));
      }
    }
    BOOST_TEST_CONTEXT("Checking result for " << op_name << " on " << type_name << ": "
        << expected << " (expected) == " << hacc[0] << " (received)") {
      BOOST_CHECK(expected == hacc[0]);
    }
  }
}

BOOST_AUTO_TEST_CASE(fetch_op) {

  auto fetch_add = [](auto& atomic, auto x) {
    return atomic.fetch_add(x);
  };

  auto fetch_sub = [](auto& atomic, auto x) {
    return atomic.fetch_sub(x);
  };

  auto fetch_and = [](auto& atomic, auto x) {
    return atomic.fetch_and(x);
  };

  auto fetch_or = [](auto& atomic, auto x) {
    return atomic.fetch_or(x);
  };

  auto fetch_xor = [](auto& atomic, auto x) {
    return atomic.fetch_xor(x);
  };

  auto fetch_min = [](auto& atomic, auto x) {
    return atomic.fetch_min(x);
  };

  auto fetch_max = [](auto& atomic, auto x) {
    return atomic.fetch_max(x);
  };


  auto fetch_add_verifier = [](auto& v, auto x) {
    v += x;
  };

  auto fetch_sub_verifier = [](auto& v, auto x) {
    v -= x;
  };

  auto fetch_and_verifier = [](auto& v, auto x) {
    v &= x;
  };

  auto fetch_or_verifier = [](auto& v, auto x) {
    v |= x;
  };

  auto fetch_xor_verifier = [](auto& v, auto x) {
    v ^= x;
  };

  auto fetch_min_verifier = [](auto& v, auto x) {
    v = std::min(v,x);
  };

  auto fetch_max_verifier = [](auto& v, auto x) {
    v = std::max(v,x);
  };

#define HIPSYCL_ATOMIC_REF_TEST_T(Type, Op, Verifier)                              \
  atomic_device_reduction_test<Type>(Op, Verifier, #Type, #Op);

#define HIPSYCL_ATOMIC_REF_INTEGER_TEST(Op, Verifier)                              \
  HIPSYCL_ATOMIC_REF_TEST_T(int, Op, Verifier)                                     \
  HIPSYCL_ATOMIC_REF_TEST_T(unsigned int, Op, Verifier)                            \
  HIPSYCL_ATOMIC_REF_TEST_T(long, Op, Verifier)                                    \
  HIPSYCL_ATOMIC_REF_TEST_T(unsigned long, Op, Verifier)                           \
  HIPSYCL_ATOMIC_REF_TEST_T(long long, Op, Verifier)                               \
  HIPSYCL_ATOMIC_REF_TEST_T(unsigned long long, Op, Verifier)

#define HIPSYCL_ATOMIC_REF_FP_TEST(Op, Verifier)                                   \
  HIPSYCL_ATOMIC_REF_TEST_T(float, Op, Verifier);                                  \
  HIPSYCL_ATOMIC_REF_TEST_T(double, Op, Verifier);

#define HIPSYCL_ATOMIC_REF_PTR_TEST(Op, Verifier, Initializer)                     \
  atomic_device_reduction_test<int *>(Op, Verifier, "int*", #Op, Initializer);

#ifndef ACPP_LIBKERNEL_CUDA_NVCXX

  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_sub, fetch_sub_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_or, fetch_or_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_xor, fetch_xor_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_and, fetch_and_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_min, fetch_min_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_max, fetch_max_verifier);

  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_sub, fetch_sub_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_min, fetch_min_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_max, fetch_max_verifier);

  HIPSYCL_ATOMIC_REF_PTR_TEST(fetch_add, fetch_add_verifier, [](auto t) {
    return t+1;
  });

  HIPSYCL_ATOMIC_REF_PTR_TEST(fetch_sub, fetch_sub_verifier, [](auto t) {
    if (t == 0)
      return std::numeric_limits<int>::max();
    else
      return t;
  });

#endif
}

BOOST_AUTO_TEST_SUITE_END()
