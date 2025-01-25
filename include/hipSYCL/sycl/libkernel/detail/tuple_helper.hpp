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
#ifndef HIPSYCL_TUPLE_HELPER_HPP
#define HIPSYCL_TUPLE_HELPER_HPP

#include <cstddef>
#include <tuple>

// This file provides a way to abstract between tuple-like elements,
// which have a specialization of `std::tuple_size` and provide either
// a member or an ADL get.

namespace hipsycl {
namespace sycl {
namespace detail {


// Type trait to detect a specialization of tuple_size.
template <typename T, typename = void>
struct is_tuple_like : std::false_type {};

// Specialization for types with tuple_size defined
template <typename T>
struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>> : std::true_type {};

template <typename T>
constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

// Helper function to make an ADL get call.
namespace adl {
  // For ADL purposes
  template<typename>
  void get();

  // SFINAE out tuples which do not have an ADL-accessible get
  template<typename Tup, size_t I>
  auto adl_get(Tup && t) noexcept -> decltype(get<I>(std::forward<Tup>(t))) {
      return get<I>(std::forward<Tup>(t));
  }
}

// Template to check if ADL get call exists
template <typename Tup, size_t I, typename = void>
struct is_adl_gettable : std::false_type {};

template <typename Tup, size_t I>
struct is_adl_gettable<Tup, I, std::void_t<decltype(detail::adl::adl_get<Tup, I>(std::declval<Tup>()))>> : std::true_type {};

template <typename Tup, size_t I>
constexpr bool is_adl_gettable_v = is_adl_gettable<Tup, I>::value;

// Dispatch on whether we can get with ADL or with member
template <size_t I, typename Tup>
auto tuple_get(Tup && t) -> std::enable_if_t<is_adl_gettable_v<Tup, I>, decltype(detail::adl::adl_get<Tup, I>(std::forward<Tup>(t)))> {
  return detail::adl::adl_get<Tup, I>(std::forward<Tup>(t));
}

template <size_t I, typename Tup>
auto tuple_get(Tup && t) -> std::enable_if_t<!is_adl_gettable_v<Tup, I>, decltype(t.template get<I>())> {
  return t.template get<I>();
}

}
}
}


#endif
