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
#ifndef HIPSYCL_IS_DEVICE_COPYABLE_HPP
#define HIPSYCL_IS_DEVICE_COPYABLE_HPP

#include <array>
#include <optional>
#include <utility>
#include <tuple>
#include <type_traits>

// AdaptiveCPP does not use this type trait to restrict allowed
// arguments to a kernel - this is simply provided for compatibility.

#define SYCL_DEVICE_COPYABLE 1

namespace hipsycl {
namespace sycl {

template <typename T> struct is_device_copyable;

namespace detail {
template <typename T, typename = void>
struct is_device_copyable_impl : std::is_trivially_copyable<T> {};

template <typename T>
struct is_device_copyable_impl<T, std::enable_if_t<!std::is_same_v<T, std::remove_cv_t<T>>>> : is_device_copyable<std::remove_cv_t<T>> {};
}

template <typename T> struct is_device_copyable : detail::is_device_copyable_impl<T> {};

template<typename T>
inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;

template <typename T>
struct is_device_copyable<std::array<T, 0>> : std::true_type {};

template <typename T, std::size_t N>
struct is_device_copyable<std::array<T, N>> : is_device_copyable<T> {};

template <typename T>
struct is_device_copyable<std::optional<T>> : is_device_copyable<T> {};

template <typename T1, typename T2>
struct is_device_copyable<std::pair<T1, T2>> : std::bool_constant<is_device_copyable_v<T1> && is_device_copyable_v<T2>> {};

template <typename... Ts>
struct is_device_copyable<std::tuple<Ts...>> : std::bool_constant<(... && is_device_copyable_v<Ts>)> {};

}
}

#endif
