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

#include <bitset>
#include <boost/mpl/joint_view.hpp>

#include <cmath>

BOOST_FIXTURE_TEST_SUITE(math_tests, reset_device_fixture)

// list of types classified as "genfloat" in the SYCL standard
using math_test_genfloats = boost::mpl::list<
  float,
  // vec<T,1> is not genfloat according to SYCL 2020. It's unclear
  // if this is an oversight or intentional.
  //cl::sycl::vec<float, 1>
  cl::sycl::vec<float, 2>,
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>,
  cl::sycl::vec<float, 8>,
  cl::sycl::vec<float, 16>,
  double,
  //cl::sycl::vec<double, 1>,
  cl::sycl::vec<double, 2>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>,
  cl::sycl::vec<double, 8>,
  cl::sycl::vec<double, 16>>;



namespace {

  template<typename DT, int D>
  using vec = cl::sycl::vec<DT, D>;

  auto tolerance = boost::test_tools::tolerance(0.0001);

  // utility type traits for generic testing

  template<typename T>
  struct vector_length {
    static constexpr int value = 0;
  };
  template<typename DT, int D>
  struct vector_length<vec<DT, D>> {
    static constexpr int value = D;
  };
  template<typename T>
  constexpr int vector_length_v = vector_length<T>::value;

  template<typename T>
  struct vector_dim {
    static constexpr int value = 0;
  };
  template<typename DT, int D>
  struct vector_dim<vec<DT, D>> {
    static constexpr int value = 1;
  };
  template<typename T>
  constexpr int vector_dim_v = vector_dim<T>::value;

  template<typename T>
  struct vector_elem {
    using type = T;
  };
  template<typename DT, int D>
  struct vector_elem<vec<DT, D>> {
    using type = DT;
  };
  template<typename T>
  using vector_elem_t = typename vector_elem<T>::type;

  template<typename TARGET_DT, typename T>
  struct vector_coerce_elem {
    using type = TARGET_DT;
  };
  template<typename TARGET_DT, typename DT, int D>
  struct vector_coerce_elem<TARGET_DT, vec<DT, D>> {
    using type = vec<TARGET_DT, D>;
  };
  template<typename TARGET_DT, typename T>
  using vector_coerce_elem_t = typename vector_coerce_elem<TARGET_DT, T>::type;

  // utility functions for generic testing

  template <typename DT, int D>
  auto get_math_input(const vec<DT, 16> &v) {
    if constexpr(D==0) {
      return v.template swizzle<0>();
    } else if constexpr(D==2) {
      return vec<DT, 2>{v.template swizzle<0,1>()};
    } else if constexpr(D==3) {
      return vec<DT, 3>{v.template swizzle<0,1,2>()};
    } else if constexpr(D==4) {
      return vec<DT, 4>{v.template swizzle<0,1,2,3>()};
    } else if constexpr(D==8) {
      return vec<DT, 8>{v.template swizzle<0,1,2,3,4,5,6,7>()};
    } else if constexpr(D==16) {
      return v;
    }
  }

  template<typename T>
  auto comp(T v, size_t idx) {
    assert(idx == 0);
    return v;
  }
  template<typename DT, int D>
  auto comp(vec<DT, D> v, size_t idx) {
    assert(idx < D);
    return v[idx];
  }

  // reference functions

  double ref_clamp(double a, double min, double max) { // in C++17 <algorithm>, remove on upgrade
    if(a < min) return min;
    if(a > max) return max;
    return a;
  }

  static constexpr double pi = 3.1415926535897932385;
  double ref_degrees(double v) {
    return 180.0/pi * v;
  }
  double ref_radians(double v) {
    return pi/180.0 * v;
  }

  double ref_mix(double x, double y, double a) {
    return x + (y - x) * a;
  }

  double ref_step(double edge, double x) {
    if(x < edge) return 0.0;
    return 1.0;
  }

  double ref_smoothstep(double edge0, double edge1, double x) {
    BOOST_REQUIRE(edge0 < edge1); // Standard: results are undefined if edge0 >= edge1
    if(x <= edge0) return 0.0;
    if(x >= edge1) return 1.0;
    double t = ref_clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  double ref_sign(double x) {
    if(x > 0.0) return 1.0;
    if(x < 0.0) return -1.0;
    if(std::isnan(x)) return 0.0;
    return x;
  }

  template<typename DT, int D>
  DT ref_dot(vec<DT, D> a, vec<DT, D> b) {
    DT ret = DT{0};
    for(int c = 0; c < D; ++c) {
      ret += comp(a, c) * comp(b, c);
    }
    return ret;
  }
  double ref_dot(double a, double b) {
    return a * b;
  }

  double ref_length(double v) {
    return std::abs(v);
  }
  template<typename DT>
  DT ref_length(vec<DT, 2> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y());
  }
  template<typename DT>
  DT ref_length(vec<DT, 3> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z());
  }
  template<typename DT>
  DT ref_length(vec<DT, 4> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z() + v.w()*v.w());
  }

  template<typename T>
  auto ref_distance(T a, T b) {
    return ref_length(a - b);
  }

  template<typename T>
  auto ref_normalize(T v) {
    return v / ref_length(v);
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_ctz(T x) noexcept {
    if(x==0){return sizeof(T)*CHAR_BIT;}
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    int idx = 0;
    while(!bset[idx]){idx++;}
    return idx;
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_clz(T x) noexcept {
    if(x==0){return sizeof(T)*CHAR_BIT;}
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    int idx = 0;
    while(!bset[sizeof(T)*CHAR_BIT - idx -1]){idx++;}
    return idx;
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_popcount(T x) noexcept {
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    return bset.count();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_binary, T,
                              math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 8;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class math_binary, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::atan2(acc[0], acc[1]);
      acc[i++] = s::copysign(acc[0], acc[1]);
      acc[i++] = s::fmin(acc[0], acc[1]);
      acc[i++] = s::fmax(acc[0], acc[1]);
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      // This triggers ICE in nvc++, no workaround yet.
      acc[i++] = s::fmod(acc[0], acc[1]);
#endif
      acc[i++] = s::fdim(acc[0], acc[1]);
      acc[i++] = s::hypot(acc[0], acc[1]);
      acc[i++] = s::pow(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == std::atan2(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::copysign(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fmin(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fmax(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      BOOST_TEST(comp(acc[i++], c) == std::fmod(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
#endif
      BOOST_TEST(comp(acc[i++], c) == std::fdim(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::hypot(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::pow(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(common_functions, T,
    math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 23;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  constexpr DT input_scalar = 3.5f;
  constexpr DT mix_input_1 = 0.5f;
  constexpr DT mix_input_2 = 0.8f;
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions
  // some of these are tested multiple times to ensure that all overloads are covered
  // (e.g. combinations of vec and scalar input)

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class common_functions, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::fabs(acc[0]);
      acc[i++] = s::clamp(acc[0], acc[1], acc[1] + static_cast<DT>(10));
      acc[i++] = s::clamp(acc[0], input_scalar, static_cast<DT>(input_scalar + 10));
      acc[i++] = s::degrees(acc[0]);
      acc[i++] = s::fma(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mad(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], input_scalar);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::min(acc[0], input_scalar);
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_2});
      acc[i++] = s::mix(acc[0], acc[1], mix_input_1);
      acc[i++] = s::radians(acc[0]);
      acc[i++] = s::step(acc[0], acc[1]);
      acc[i++] = s::step(input_scalar, acc[0]);
      acc[i++] = s::smoothstep(acc[0], acc[0] + static_cast<DT>(10), acc[1]);
      acc[i++] = s::smoothstep(input_scalar, input_scalar + 1, acc[0]);
      acc[i++] = s::sign(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == std::abs(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), comp(acc[1], c), comp(acc[1], c) + 10), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), input_scalar, input_scalar + 10), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_degrees(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fma(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fma(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance); // mad
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), input_scalar), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), input_scalar), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_2), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_radians(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_step(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_step(input_scalar, comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(comp(acc[0], c), comp(acc[0], c) + 10, comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(input_scalar, input_scalar + 1, comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_sign(comp(acc[0], c)), tolerance);
    }
  }
}

// some subset of types classified as "geninteger" in SYCL
using math_test_genints = boost::mpl::list<
  int,
  cl::sycl::vec<int, 2>,
  cl::sycl::vec<int, 3>,
  cl::sycl::vec<int, 16>,
  short,
  cl::sycl::vec<short, 4>,
  unsigned char,
  cl::sycl::vec<unsigned char, 3>,
  unsigned long,
  cl::sycl::vec<unsigned long, 8>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(builtin_int_basic, T, math_test_genints::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 6;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class builtin_int_basic, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::abs(acc[0]);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::ctz(acc[0]);
      acc[i++] = s::clz(acc[0]);
      acc[i++] = s::popcount(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      if constexpr(std::is_signed<DT>::value)
        BOOST_TEST(comp(acc[i++], c) == std::abs(comp(acc[0], c)));
      else
        BOOST_TEST(comp(acc[i++], c) == comp(acc[0], c));
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_ctz(comp(acc[0], c)));
      // It seems that certain LLVM/ROCm versions in CI miscompile this test in SMCP
      // mode. Temporarily disable in HIP SMCP. We still test with SSCP on AMD,
      // and with SMCP on non-AMD devices, including explicit multipass builds
      // were both CUDA and HIP are targeted in a single build.
#ifdef __ACPP_ENABLE_HIP_TARGET__
      bool enable_clz = queue.get_device().get_backend() != s::backend::hip;
#else
      bool enable_clz = true;
#endif
      if(enable_clz){
        BOOST_TEST(comp(acc[i++], c) == ref_clz(comp(acc[0], c)));
      } else {
        i++;
      }
      BOOST_TEST(comp(acc[i++], c) == ref_popcount(comp(acc[0], c)));
    }
  }
}


// types allowed for the "cross" function
using math_test_crossinputs = boost::mpl::list<
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(geometric_cross, T, math_test_crossinputs::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 1;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for (int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class geometric_cross, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::cross(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    int i = 2;
    const auto& res = acc[i++], a = acc[0], b = acc[1];
    BOOST_TEST(res.x() == a.y()*b.z() - a.z()*b.y());
    BOOST_TEST(res.y() == a.z()*b.x() - a.x()*b.z());
    BOOST_TEST(res.z() == a.x()*b.y() - a.y()*b.x());
    if(D==4) BOOST_TEST(comp(res,3) == DT{0});
  }
}

// type classes as per SYCL standard

using math_test_gengeofloats = boost::mpl::list<
  float,
  cl::sycl::vec<float, 2>,
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>>;

using math_test_gengeodoubles = boost::mpl::list<
  double,
  cl::sycl::vec<double, 2>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>>;

using math_test_gengeo = boost::mpl::joint_view<math_test_gengeofloats, math_test_gengeodoubles>;

BOOST_AUTO_TEST_CASE_TEMPLATE(geometric, T, math_test_gengeo::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 4;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class geometric, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::dot(acc[0], acc[1]);
      acc[i++] = s::length(acc[0]);
      acc[i++] = s::distance(acc[0], acc[1]);
      acc[i++] = s::normalize(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    auto dot_ref_result = ref_dot(acc[0], acc[1]);
    auto length_ref_result = ref_length(acc[0]);
    auto distance_ref_result = ref_distance(acc[0], acc[1]);
    auto normalize_ref_result = ref_normalize(acc[0]);
    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(dot_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(length_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(distance_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(comp(normalize_ref_result, c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fast_geometric, T, math_test_gengeofloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 3;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class fast_geometric, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::fast_length(acc[0]);
      acc[i++] = s::fast_distance(acc[0], acc[1]);
      acc[i++] = s::fast_normalize(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    auto length_ref_result = ref_length(acc[0]);
    auto distance_ref_result = ref_distance(acc[0], acc[1]);
    auto normalize_ref_result = ref_normalize(acc[0]);
    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(length_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(distance_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(comp(normalize_ref_result, c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_int, T,
                              math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 1;

  // build inputs and allocate outputs

  s::queue queue;
  s::buffer<T> in{{1}};
  s::buffer<T> out{{FUN_COUNT}};
  {
    auto inputs  = in.get_host_access();
    auto outputs = out.get_host_access();
    s::vec<DT, 16> v{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    inputs[0] = get_math_input<DT, D>(v);
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto inputs  = in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_genfloat_int, D, DT>>([=]() {
      outputs[0] = s::ldexp(inputs[0], 7);
    });
  });

  // check results

  {
    auto inputs  = in.get_host_access();
    auto outputs = out.get_host_access();

    for(int c = 0; c < std::max(D,1); ++c) {
      BOOST_TEST(comp(outputs[0], c) == std::ldexp(comp(inputs[0], c), 7));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_genint, T,
                              math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  using IntType = typename s::detail::builtin_type_traits<T>::template alternative_data_type<int>;

  constexpr int FUN_COUNT = 3;

  // build inputs and allocate outputs

  s::queue queue;
  s::buffer<T> float_in{{1}};
  s::buffer<IntType> int_in{{1}};
  s::buffer<T> out{{FUN_COUNT}};
  {
    auto float_inputs = float_in.get_host_access();
    auto int_inputs = int_in.get_host_access();
    auto outputs = out.get_host_access();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<int, 16> v2{17, -4, -2, 3, 7, -8, 9, -1, 17, -4, -2, 3, 7, -8, 9, -1};
    float_inputs[0] = get_math_input<DT, D>(v1);
    int_inputs[0] = get_math_input<int, D>(v2);
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto float_inputs = float_in.template get_access<s::access::mode::read>(cgh);
    auto int_inputs = int_in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_genfloat_genint, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::ldexp(float_inputs[0], int_inputs[0]);
      outputs[i++] = s::pown(float_inputs[0], int_inputs[0]);
      outputs[i++] = s::rootn(s::fabs(float_inputs[0]), int_inputs[0]);
    });
  });

  // check results

  {
    auto float_inputs = float_in.get_host_access();
    auto int_inputs = int_in.get_host_access();
    auto outputs = out.get_host_access();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == std::ldexp(comp(float_inputs[0], c), comp(int_inputs[0], c)), tolerance);
      BOOST_TEST(comp(outputs[i++], c) == std::pow(comp(float_inputs[0], c), comp(int_inputs[0], c)), tolerance);
      BOOST_TEST(comp(outputs[i++], c) == std::pow(std::fabs(comp(float_inputs[0], c)), 1./comp(int_inputs[0], c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
