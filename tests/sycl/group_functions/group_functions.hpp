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

#ifndef TESTS_GROUP_FUNCTIONS_HH
#define TESTS_GROUP_FUNCTIONS_HH

#include <cstddef>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <type_traits>

#include <sstream>
#include <string>

using namespace cl;

#define HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS



#ifdef TESTS_GROUPFUNCTION_FULL
using test_types =
    boost::mpl::list<char, int, unsigned int, long long, float, double, sycl::vec<int, 1>,
                     sycl::vec<int, 2>, sycl::vec<int, 3>, sycl::vec<int, 4>,
                     sycl::vec<int, 8>, sycl::vec<short, 16>, sycl::vec<long, 3>,
                     sycl::vec<unsigned int, 3>>;
#else
using test_types = boost::mpl::list<char, unsigned int, float, double, sycl::vec<int, 2>>;
#endif

namespace detail {

inline uint32_t get_subgroup_size(const sycl::queue& q) {
  auto sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  assert(sizes.size() > 0);
  return static_cast<uint32_t>(sizes[0]);
}

inline uint32_t get_subgroup_size() {
  return get_subgroup_size(sycl::queue{});
}

template<typename T>
using elementType = std::remove_reference_t<decltype(T{}.s0())>;

template<typename T, int N>
std::string type_to_string(sycl::vec<T, N> v) {
  std::stringstream ss{};

  ss << "(";
  if constexpr (1 <= N)
    ss << +v.s0();
  if constexpr (2 <= N)
    ss << ", " << +v.s1();
  if constexpr (3 <= N)
    ss << ", " << +v.s2();
  if constexpr (4 <= N)
    ss << ", " << +v.s3();
  if constexpr (8 <= N) {
    ss << ", " << +v.s4();
    ss << ", " << +v.s5();
    ss << ", " << +v.s6();
    ss << ", " << +v.s7();
  }
  if constexpr (16 <= N) {
    ss << ", " << +v.s8();
    ss << ", " << +v.s9();
    ss << ", " << +v.sA();
    ss << ", " << +v.sB();
    ss << ", " << +v.sC();
    ss << ", " << +v.sD();
    ss << ", " << +v.sE();
    ss << ", " << +v.sF();
  }
  ss << ")";

  return ss.str();
}

template<typename T>
std::string type_to_string(T x) {
  std::stringstream ss{};
  ss << +x;

  return ss.str();
}

template<typename T, int N>
bool compare_type(sycl::vec<T, N> v1, sycl::vec<T, N> v2) {
  bool ret = true;
  if constexpr (1 <= N)
    ret &= v1.s0() == v2.s0();
  if constexpr (2 <= N)
    ret &= v1.s1() == v2.s1();
  if constexpr (3 <= N)
    ret &= v1.s2() == v2.s2();
  if constexpr (4 <= N)
    ret &= v1.s3() == v2.s3();
  if constexpr (8 <= N) {
    ret &= v1.s4() == v2.s4();
    ret &= v1.s5() == v2.s5();
    ret &= v1.s6() == v2.s6();
    ret &= v1.s7() == v2.s7();
  }
  if constexpr (16 <= N) {
    ret &= v1.s8() == v2.s8();
    ret &= v1.s9() == v2.s9();
    ret &= v1.sA() == v2.sA();
    ret &= v1.sB() == v2.sB();
    ret &= v1.sC() == v2.sC();
    ret &= v1.sD() == v2.sD();
    ret &= v1.sE() == v2.sE();
    ret &= v1.sF() == v2.sF();
  }

  return ret;
}

template<typename T>
bool compare_type(T x1, T x2) {
  return x1 == x2;
}

template<typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
ACPP_KERNEL_TARGET
T initialize_type(T init) {
  return init;
}

template<typename T, typename std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
ACPP_KERNEL_TARGET
T initialize_type(elementType<T> init) {
  constexpr size_t N = T::size();

  if constexpr (std::is_same_v<elementType<T>, bool>)
    return T{init};

  if constexpr (N == 1) {
    return T{init};
  } else if constexpr (N == 2) {
    return T{init, init + 1};
  } else if constexpr (N == 3) {
    return T{init, init + 1, init + 2};
  } else if constexpr (N == 4) {
    return T{init, init + 1, init + 2, init + 3};
  } else if constexpr (N == 8) {
    return T{init, init + 1, init + 2, init + 3, init + 4, init + 5, init + 6, init + 7};
  } else if constexpr (N == 16) {
    return T{init,      init + 1,  init + 2,  init + 3, init + 4,  init + 5,
             init + 6,  init + 7,  init + 8,  init + 9, init + 10, init + 11,
             init + 12, init + 13, init + 14, init + 15};
  }

  return T{};
  static_assert(true, "invalide vector type!");
}

template<typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
ACPP_KERNEL_TARGET
T get_offset(size_t margin, size_t divisor = 1) {
  
  if (std::numeric_limits<T>::max() <= margin) {
    return T{};
  }
  if constexpr (std::is_floating_point_v<T>) {
    return T{};
  }

  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(std::numeric_limits<T>::min() / divisor + margin + 1);
  }
  return static_cast<T>(std::numeric_limits<T>::max() / divisor - margin - 1);
}

template<typename T, typename std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
ACPP_KERNEL_TARGET
T get_offset(size_t margin, size_t divisor = 1) {
  using eT = elementType<T>;
  return initialize_type<T>(get_offset<eT>(margin + 16, divisor));
}

inline void create_bool_test_data(std::vector<char> &buffer, size_t local_size,
                                  size_t global_size) {
  BOOST_REQUIRE(global_size == 4 * local_size);
  BOOST_REQUIRE(local_size + 10 < 2 * local_size);

  // create host_buf 4 different possible configurations:
  // 1: everything except one false
  // 2: everything false
  // 3: everything except one true
  // 4: everything true

  for (size_t i = 0; i < 2 * local_size; ++i)
    buffer[i] = false;
  for (size_t i = 2 * local_size; i < 4 * local_size; ++i)
    buffer[i] = true;

  buffer[10]                  = true;
  buffer[2 * local_size + 10] = false;

  BOOST_REQUIRE(buffer[0] == false);
  BOOST_REQUIRE(buffer[10] == true);
  BOOST_REQUIRE(buffer[local_size] == false);
  BOOST_REQUIRE(buffer[10 + local_size] == false);
  BOOST_REQUIRE(buffer[local_size * 2] == true);
  BOOST_REQUIRE(buffer[10 + local_size * 2] == false);
  BOOST_REQUIRE(buffer[local_size * 3] == true);
  BOOST_REQUIRE(buffer[10 + local_size * 3] == true);
}

template<typename T, int Line>
void check_binary_reduce(std::vector<T> buffer, std::vector<T> input, size_t local_size, size_t global_size,
                         std::vector<bool> expected, std::string name,
                         size_t break_size = 0, size_t offset = 0) {
  std::vector<std::string> cases{"everything except one false", "everything false",
                                 "everything except one true", "everything true"};
  BOOST_REQUIRE(global_size / local_size == expected.size());
  for (size_t i = 0; i < global_size / local_size; ++i) {
    for (size_t j = 0; j < local_size; ++j) {
      // used to stop after first subgroup
      if (break_size != 0 && j == break_size)
        break;

      T computed      = buffer[i * local_size + j + offset];
      T expectedValue = initialize_type<T>(expected[i]);

      BOOST_TEST(compare_type(expectedValue, computed),
                 Line << ":" << type_to_string(computed) << " at position " << j
                      << " instead of " << type_to_string(expectedValue)
                      << " for case: " << cases[i] << " " << name);

      if (!compare_type(expectedValue, computed))
        break;
    }
  }
}

} // namespace detail

template<int N, int M, typename T>
class test_kernel;

template<int CallingLine, typename T, typename DataGenerator, typename TestedFunction,
         typename ValidationFunction>
void test_nd_group_function_1d(size_t elements_per_thread, DataGenerator dg,
                               TestedFunction f, ValidationFunction vf) {
  sycl::queue    queue;
  std::vector<size_t> local_sizes  = {25, 144, 256};
  std::vector<size_t> global_sizes = {100, 576, 1024};
  // currently only groupsizes between 128 and 256 are supported for HIP
  if(queue.get_device().get_backend() == sycl::backend::hip) {
    local_sizes = std::vector<size_t>{256};
    global_sizes = std::vector<size_t>{1024};
  }

  for (int i = 0; i < local_sizes.size(); ++i) {
    size_t local_size  = local_sizes[i];
    size_t global_size = global_sizes[i];
    uint32_t used_sgrp_size = 0;

    std::vector<T> host_buf(elements_per_thread * global_size, T{});

    dg(host_buf, local_size, global_size);

    std::vector<T> original_host_buf(host_buf);

    {
      sycl::buffer<T, 1> buf{host_buf.data(), host_buf.size()};
      sycl::buffer<uint32_t, 1> used_sgrp_size_buffer(&used_sgrp_size, 1);

      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        auto acc = buf.template get_access<mode::read_write>(cgh);
        auto sgpr_size_acc = used_sgrp_size_buffer.template get_access<mode::read_write>(cgh);
        cgh.parallel_for<class test_kernel<1, CallingLine, T>>(
          sycl::nd_range<1>{global_size, local_size},
          [=](sycl::nd_item<1> item) {
          auto g  = item.get_group();
          auto sg = item.get_sub_group();

          T local_value = acc[item.get_global_linear_id()];
          sgpr_size_acc[0] =  sg.get_max_local_range().size();
          f(acc, item.get_global_linear_id(), sg, g, local_value);
        });
      });
    }

    vf(host_buf, original_host_buf,used_sgrp_size, local_size, global_size);
  }
}

template<int CallingLine, typename T, typename DataGenerator, typename TestedFunction,
         typename ValidationFunction>
void test_nd_group_function_2d(size_t elements_per_thread, DataGenerator dg,
                               TestedFunction f, ValidationFunction vf) {
  sycl::queue    queue;

  std::vector<size_t> local_sizes  = {5, 12, 16};
  std::vector<size_t> global_sizes = {10, 24, 32};

  if(queue.get_device().get_backend() == sycl::backend::hip) {
    // currently only groupsizes between 128 and 256 are supported for HIP
    local_sizes = std::vector<size_t>{16};
    global_sizes = std::vector<size_t>{32};
  }

  for (int i = 0; i < local_sizes.size(); ++i) {
    size_t local_size  = local_sizes[i];
    size_t global_size = global_sizes[i];

    std::vector<T> host_buf(elements_per_thread * global_size * global_size, T{});

    dg(host_buf, local_size * local_size, global_size * global_size);

    std::vector<T> original_host_buf(host_buf);

    {
      sycl::buffer<T, 1> buf{host_buf.data(), host_buf.size()};

      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        auto acc = buf.template get_access<mode::read_write>(cgh);

        cgh.parallel_for<class test_kernel<2, CallingLine, T>>(
          sycl::nd_range<2>{sycl::range<2>(global_size, global_size), sycl::range<2>(local_size, local_size)},
          [=](sycl::nd_item<2> item) {
          auto   g                = item.get_group();
          auto   sg               = item.get_sub_group();
          size_t custom_linear_id = item.get_local_linear_id() +
                                    local_size * local_size * item.get_group_linear_id();

          T local_value = acc[custom_linear_id];

          f(acc, custom_linear_id, sg, g, local_value);
        });
      });
    }

    vf(host_buf, original_host_buf,0, local_size * local_size, global_size * global_size);
  }
}

#endif // TESTS_GROUP_FUNCTIONS_HH
