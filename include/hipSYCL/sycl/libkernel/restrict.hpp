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
#ifndef ACPP_RESTRICT_HPP
#define ACPP_RESTRICT_HPP

#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

template <class T>
struct __acpp_sscp_emit_param_type_annotation_restrict {
  T value;
};

} // namespace detail

template <class T> class AdaptiveCpp_restrict_ptr {
public:
  template <typename U = T, typename = std::enable_if_t<
                                std::is_default_constructible<U>::value>>
  AdaptiveCpp_restrict_ptr() : _value{} {}

  AdaptiveCpp_restrict_ptr(const T &value) : _value{value} {}

  AdaptiveCpp_restrict_ptr(const AdaptiveCpp_restrict_ptr<T> &other)
      : _value{other._value.value} {}

  AdaptiveCpp_restrict_ptr(sycl::AdaptiveCpp_restrict_ptr<T> &&other) {
    swap(*this, other);
  }

  AdaptiveCpp_restrict_ptr<T> &operator=(const T &value) {
    AdaptiveCpp_restrict_ptr<T> tmp{value};
    swap(*this, tmp);
    return *this;
  }

  AdaptiveCpp_restrict_ptr<T> &operator=(AdaptiveCpp_restrict_ptr<T> other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(AdaptiveCpp_restrict_ptr<T> &first,
                   AdaptiveCpp_restrict_ptr<T> &second) {
    using std::swap;
    swap(first._value.value, second._value.value);
  }

  operator T *() const { return _value.value; }

private:
  detail::__acpp_sscp_emit_param_type_annotation_restrict<T *> _value;
};

} // namespace sycl
} // namespace hipsycl

#endif
