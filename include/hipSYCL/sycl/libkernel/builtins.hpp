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
#ifndef HIPSYCL_BUILTINS_HPP
#define HIPSYCL_BUILTINS_HPP


#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"
#include <type_traits>
#include <cstdlib>

#include "builtin_interface.hpp"

namespace hipsycl::sycl::detail {

template<class T>
struct builtin_type_traits {
  using type = T;

  template<class U>
  using alternative_data_type = U;

  using element_type = T;

  static constexpr int num_elements = 1;

  ACPP_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
  static type& element(type& v, int i) noexcept {
    return v;
  }

  ACPP_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
  static type element(const type& v, int i) noexcept {
    return v;
  }
};

template<class T, int Dim>
struct builtin_type_traits<vec<T, Dim>> {
  using type = vec<T, Dim>;

  template<class U>
  using alternative_data_type = vec<U, Dim>;

  using element_type = T;

  static constexpr int num_elements = Dim;

  ACPP_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
  static T& element(type& v, int i) noexcept {
    return v[i];
  }

  ACPP_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
  static T element(const type& v, int i) noexcept {
    return v[i];
  }
};

template<class T>
ACPP_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
typename builtin_type_traits<T>::element_type
data_element(const T& v, int i) noexcept {
  return builtin_type_traits<T>::element(v, i);
}

template<class T>
ACPP_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
typename builtin_type_traits<T>::element_type&
data_element(T& v, int i) noexcept {
  return builtin_type_traits<T>::element(v, i);
}

template <class T, class IntType>
inline constexpr bool is_genint_alternative_type_v =
    std::is_integral_v<
        typename detail::builtin_type_traits<IntType>::element_type>
        &&detail::builtin_type_traits<IntType>::num_elements ==
    detail::builtin_type_traits<T>::num_elements;
}

namespace hipsycl {
namespace sycl {

namespace detail {

// The spec defines vec aliases in terms of (u)intN_t. However, this
// does not cover fully unsigned char, signed char and char which
// are three distinct types. Additionally, the vec aliases define
// (u)longN as 64 bit types, NOT as vec<(unsigned) long, N> which is
// usually 32 bit.
// We therefore introduce aliases that are in line with what the
// builtins require. There seems to be some ambiguity or even contradiction
// in the spec w.r.t how vec aliases and builtin operand types fit together.
using char2 = vec<char, 2>;
using char3 = vec<char, 3>;
using char4 = vec<char, 4>;
using char8 = vec<char, 8>;
using char16 = vec<char, 16>;

using uchar2 = vec<unsigned char, 2>;
using uchar3 = vec<unsigned char, 3>;
using uchar4 = vec<unsigned char, 4>;
using uchar8 = vec<unsigned char, 8>;
using uchar16 = vec<unsigned char, 16>;

using schar2 = vec<signed char, 2>;
using schar3 = vec<signed char, 3>;
using schar4 = vec<signed char, 4>;
using schar8 = vec<signed char, 8>;
using schar16 = vec<signed char, 16>;

using long2 = vec<long, 2>;
using long3 = vec<long, 3>;
using long4 = vec<long, 4>;
using long8 = vec<long, 8>;
using long16 = vec<long, 16>;

using ulong2 = vec<unsigned long, 2>;
using ulong3 = vec<unsigned long, 3>;
using ulong4 = vec<unsigned long, 4>;
using ulong8 = vec<unsigned long, 8>;
using ulong16 = vec<unsigned long, 16>;

using longlong2 = vec<long long, 2>;
using longlong3 = vec<long long, 3>;
using longlong4 = vec<long long, 4>;
using longlong8 = vec<long long, 8>;
using longlong16 = vec<long long, 16>;

using ulonglong2 = vec<unsigned long long, 2>;
using ulonglong3 = vec<unsigned long long, 3>;
using ulonglong4 = vec<unsigned long long, 4>;
using ulonglong8 = vec<unsigned long long, 8>;
using ulonglong16 = vec<unsigned long long, 16>;

#define HIPSYCL_PP_CONCATENATE_IMPL(a,b) a ## b
#define HIPSYCL_PP_CONCATENATE(a,b) HIPSYCL_PP_CONCATENATE_IMPL(a,b)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_FLOATN(handler, name, builtin_impl_name) \
  handler(sycl::float2, name, builtin_impl_name) \
  handler(sycl::float3, name, builtin_impl_name) \
  handler(sycl::float4, name, builtin_impl_name) \
  handler(sycl::float8, name, builtin_impl_name) \
  handler(sycl::float16, name, builtin_impl_name)
  // TODO mfloat/marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF(handler, name, builtin_impl_name) \
  handler(float, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_FLOATN(handler, name, builtin_impl_name)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_DOUBLEN(handler, name, builtin_impl_name) \
  handler(sycl::double2, name, builtin_impl_name) \
  handler(sycl::double3, name, builtin_impl_name) \
  handler(sycl::double4, name, builtin_impl_name) \
  handler(sycl::double8, name, builtin_impl_name) \
  handler(sycl::double16, name, builtin_impl_name)
  // TODO mfloat/marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD(handler, name, builtin_impl_name) \
  handler(double, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_DOUBLEN(handler, name, builtin_impl_name)

// TODO
#define HIPSYCL_BUILTIN_OVERLOAD_SET_HALFN(handler, name, builtin_impl_name)
// TODO
#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATH(handler, name, builtin_impl_name)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT(handler, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD(handler, name, builtin_impl_name)      \
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF(handler, name, builtin_impl_name)      \
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATH(handler, name, builtin_impl_name)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_SGENFLOAT(handler, name, builtin_impl_name) \
  handler(double, name, builtin_impl_name) \
  handler(float, name, builtin_impl_name)
  // TODO: half

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT(handler, name, builtin_impl_name) \
  handler(float, name, builtin_impl_name)        \
  handler(sycl::float2, name, builtin_impl_name) \
  handler(sycl::float3, name, builtin_impl_name) \
  handler(sycl::float4, name, builtin_impl_name)
  // TODO: mfloat2, mfloat3, mfloat4

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE(handler, name, builtin_impl_name) \
  handler(double, name, builtin_impl_name)        \
  handler(sycl::double2, name, builtin_impl_name) \
  handler(sycl::double3, name, builtin_impl_name) \
  handler(sycl::double4, name, builtin_impl_name)
  // TODO: mdouble2, mdouble3, mdouble4

#define HIPSYCL_BUILTIN_OVERLOAD_SET_CHARN(handler, name, builtin_impl_name) \
  handler(detail::char2, name, builtin_impl_name) \
  handler(detail::char3, name, builtin_impl_name) \
  handler(detail::char4, name, builtin_impl_name) \
  handler(detail::char8, name, builtin_impl_name) \
  handler(detail::char16, name, builtin_impl_name) 
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_SCHARN(handler, name, builtin_impl_name) \
  handler(detail::schar2, name, builtin_impl_name) \
  handler(detail::schar3, name, builtin_impl_name) \
  handler(detail::schar4, name, builtin_impl_name) \
  handler(detail::schar8, name, builtin_impl_name) \
  handler(detail::schar16, name, builtin_impl_name) 
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UCHARN(handler, name, builtin_impl_name) \
  handler(detail::uchar2, name, builtin_impl_name) \
  handler(detail::uchar3, name, builtin_impl_name) \
  handler(detail::uchar4, name, builtin_impl_name) \
  handler(detail::uchar8, name, builtin_impl_name) \
  handler(detail::uchar16, name, builtin_impl_name) 
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_IGENCHAR(handler, name, builtin_impl_name) \
  handler(signed char, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_SCHARN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UGENCHAR(handler, name, builtin_impl_name) \
  handler(unsigned char, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UCHARN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENCHAR(handler, name, builtin_impl_name) \
  handler(char, name, builtin_impl_name)                                       \
  HIPSYCL_BUILTIN_OVERLOAD_SET_CHARN(handler, name, builtin_impl_name)         \
  HIPSYCL_BUILTIN_OVERLOAD_SET_IGENCHAR(handler, name, builtin_impl_name)      \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UGENCHAR(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_SHORTN(handler, name, builtin_impl_name) \
  handler(sycl::short2, name, builtin_impl_name)                              \
  handler(sycl::short3, name, builtin_impl_name)                              \
  handler(sycl::short4, name, builtin_impl_name)                              \
  handler(sycl::short8, name, builtin_impl_name)                              \
  handler(sycl::short16, name, builtin_impl_name) 
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENSHORT(handler, name, builtin_impl_name) \
  handler(short, name, builtin_impl_name)                                       \
  HIPSYCL_BUILTIN_OVERLOAD_SET_SHORTN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_USHORTN(handler, name, builtin_impl_name) \
  handler(sycl::ushort2, name, builtin_impl_name)                              \
  handler(sycl::ushort3, name, builtin_impl_name)                              \
  handler(sycl::ushort4, name, builtin_impl_name)                              \
  handler(sycl::ushort8, name, builtin_impl_name)                              \
  handler(sycl::ushort16, name, builtin_impl_name) 
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UGENSHORT(handler, name, builtin_impl_name) \
  handler(unsigned short, name, builtin_impl_name)                               \
  HIPSYCL_BUILTIN_OVERLOAD_SET_USHORTN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UINTN(handler, name, builtin_impl_name) \
  handler(sycl::uint2, name, builtin_impl_name)                              \
  handler(sycl::uint3, name, builtin_impl_name)                              \
  handler(sycl::uint4, name, builtin_impl_name)                              \
  handler(sycl::uint8, name, builtin_impl_name)                              \
  handler(sycl::uint16, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UGENINT(handler, name, builtin_impl_name) \
  handler(unsigned int, name, builtin_impl_name)                               \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UINTN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_INTN(handler, name, builtin_impl_name) \
  handler(sycl::int2, name, builtin_impl_name)                              \
  handler(sycl::int3, name, builtin_impl_name)                              \
  handler(sycl::int4, name, builtin_impl_name)                              \
  handler(sycl::int8, name, builtin_impl_name)                              \
  handler(sycl::int16, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENINT(handler, name, builtin_impl_name) \
  handler(int, name, builtin_impl_name)                                       \
  HIPSYCL_BUILTIN_OVERLOAD_SET_INTN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_ULONGN(handler, name, builtin_impl_name)   \
  handler(detail::ulong2, name, builtin_impl_name)                              \
  handler(detail::ulong3, name, builtin_impl_name)                              \
  handler(detail::ulong4, name, builtin_impl_name)                              \
  handler(detail::ulong8, name, builtin_impl_name)                              \
  handler(detail::ulong16, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UGENLONG(handler, name, builtin_impl_name) \
  handler(unsigned long int, name, builtin_impl_name)                           \
  HIPSYCL_BUILTIN_OVERLOAD_SET_ULONGN(handler, name, builtin_impl_name)
  // TODO: marray


#define HIPSYCL_BUILTIN_OVERLOAD_SET_LONGN(handler, name, builtin_impl_name)   \
  handler(detail::long2, name, builtin_impl_name)                              \
  handler(detail::long3, name, builtin_impl_name)                              \
  handler(detail::long4, name, builtin_impl_name)                              \
  handler(detail::long8, name, builtin_impl_name)                              \
  handler(detail::long16, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENLONG(handler, name, builtin_impl_name) \
  handler(long int, name, builtin_impl_name)                                   \
  HIPSYCL_BUILTIN_OVERLOAD_SET_LONGN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_ULONGLONGN(handler, name, builtin_impl_name)   \
  handler(detail::ulonglong2, name, builtin_impl_name)                              \
  handler(detail::ulonglong3, name, builtin_impl_name)                              \
  handler(detail::ulonglong4, name, builtin_impl_name)                              \
  handler(detail::ulonglong8, name, builtin_impl_name)                              \
  handler(detail::ulonglong16, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UGENLONGLONG(handler, name, builtin_impl_name) \
  handler(unsigned long long, name, builtin_impl_name)                              \
  HIPSYCL_BUILTIN_OVERLOAD_SET_ULONGLONGN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_LONGLONGN(handler, name, builtin_impl_name)   \
  handler(detail::longlong2, name, builtin_impl_name)                              \
  handler(detail::longlong3, name, builtin_impl_name)                              \
  handler(detail::longlong4, name, builtin_impl_name)                              \
  handler(detail::longlong8, name, builtin_impl_name)                              \
  handler(detail::longlong16, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENLONGLONG(handler, name, builtin_impl_name) \
  handler(long long, name, builtin_impl_name)                              \
  HIPSYCL_BUILTIN_OVERLOAD_SET_LONGLONGN(handler, name, builtin_impl_name)
  // TODO: marray

#define HIPSYCL_BUILTIN_OVERLOAD_SET_IGENLONGINTEGER(handler, name, builtin_impl_name)\
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENLONG(handler, name, builtin_impl_name)              \
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENLONGLONG(handler, name, builtin_impl_name)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_UGENLONGINTEGER(handler, name, builtin_impl_name)\
  HIPSYCL_BUILTIN_OVERLOAD_SET_UGENLONG(handler, name, builtin_impl_name)             \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UGENLONGLONG(handler, name, builtin_impl_name)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER(handler, name, builtin_impl_name)\
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENCHAR(handler, name, builtin_impl_name)         \
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENSHORT(handler, name, builtin_impl_name)        \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UGENSHORT(handler, name, builtin_impl_name)       \
  HIPSYCL_BUILTIN_OVERLOAD_SET_GENINT(handler, name, builtin_impl_name)          \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UGENINT(handler, name, builtin_impl_name)         \
  HIPSYCL_BUILTIN_OVERLOAD_SET_IGENLONGINTEGER(handler, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UGENLONGINTEGER(handler, name, builtin_impl_name)

#define HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER32BIT(handler, name, builtin_impl_name)\
  handler(int32_t, name, builtin_impl_name)                           \
  handler(uint32_t, name, builtin_impl_name)                          \
  HIPSYCL_BUILTIN_OVERLOAD_SET_INTN(handler, name, builtin_impl_name) \
  HIPSYCL_BUILTIN_OVERLOAD_SET_UINTN(handler, name, builtin_impl_name) 

#define HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T(T, name, impl_name)            \
  HIPSYCL_BUILTIN T name(T a, T b, T c) noexcept {                             \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return static_cast<T>(impl_name(detail::data_element(a, 0),              \
                                      detail::data_element(b, 0),              \
                                      detail::data_element(c, 0)));            \
    } else {                                                                   \
      T result;                                                                \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        auto b_i = detail::data_element(b, i);                                 \
        auto c_i = detail::data_element(c, i);                                 \
        detail::data_element(result, i) = impl_name(a_i, b_i, c_i);            \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T(T, name, impl_name)               \
  HIPSYCL_BUILTIN T name(T a, T b) noexcept {                                  \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return static_cast<T>(impl_name(detail::data_element(a, 0),              \
                                      detail::data_element(b, 0)));            \
    } else {                                                                   \
      T result;                                                                \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        auto b_i = detail::data_element(b, i);                                 \
        detail::data_element(result, i) = impl_name(a_i, b_i);                 \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_TGENPTR(T, name, impl_name)         \
  template <access::address_space A>                                           \
  HIPSYCL_BUILTIN T name(T a, const multi_ptr<T, A> &b) noexcept {             \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return impl_name(detail::data_element(a, 0), b.get());                   \
    } else {                                                                   \
      T result;                                                                \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        auto b_i_ptr = &(detail::data_element(*b, i));                         \
        detail::data_element(result, i) = impl_name(a_i, b_i_ptr);             \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_INT(T, name, impl_name)             \
  HIPSYCL_BUILTIN T name(T a, int b) noexcept {                                \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return static_cast<T>(impl_name(detail::data_element(a, 0),              \
                                      detail::data_element(b, 0)));            \
    } else {                                                                   \
      T result;                                                                \
      auto b_0 = detail::data_element(b, 0);                                   \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        detail::data_element(result, i) = impl_name(a_i, b_0);                 \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINT(T, name, impl_name)          \
  template <class IntType,                                                     \
            std::enable_if_t<detail::is_genint_alternative_type_v<T, IntType>, \
                             int> = 0>                                         \
  HIPSYCL_BUILTIN T name(T a, IntType b) noexcept {                            \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return impl_name(detail::data_element(a, 0),                             \
                       detail::data_element(b, 0));                            \
    } else {                                                                   \
      T result;                                                                \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        auto b_i = detail::data_element(b, i);                                 \
        detail::data_element(result, i) = impl_name(a_i, b_i);                 \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINTPTR(T, name, impl_name)       \
  template <class IntType, access::address_space A,                            \
            std::enable_if_t<detail::is_genint_alternative_type_v<T, IntType>, \
                             int> = 0>                                         \
  HIPSYCL_BUILTIN T name(T a, const multi_ptr<IntType, A> &b) noexcept {       \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return impl_name(detail::data_element(a, 0), b.get());                   \
    } else {                                                                   \
      T result;                                                                \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        auto b_i_ptr = &(detail::data_element(*b, i));                         \
        detail::data_element(result, i) = impl_name(a_i, b_i_ptr);             \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_T(T, name, impl_name)                  \
  HIPSYCL_BUILTIN T name(T a) noexcept {                                       \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return static_cast<T>(impl_name(detail::data_element(a, 0)));            \
    } else {                                                                   \
      T result;                                                                \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        detail::data_element(result, i) = impl_name(a_i);                      \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT(T, name, impl_name)          \
  HIPSYCL_BUILTIN                                                              \
  typename detail::builtin_type_traits<T>::alternative_data_type<int> name(    \
      T a) noexcept {                                                          \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return static_cast<                                                      \
          typename detail::builtin_type_traits<T>                              \
                         ::alternative_data_type<int>>                         \
                             (impl_name(detail::data_element(a, 0)));          \
    } else {                                                                   \
      typename detail::builtin_type_traits<T>::alternative_data_type<int>      \
          result;                                                              \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        detail::data_element(result, i) = impl_name(a_i);                      \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT64(T, name, impl_name)        \
  HIPSYCL_BUILTIN                                                              \
  typename detail::builtin_type_traits<T>::alternative_data_type<int64_t> name(\
      T a) noexcept {                                                          \
    if constexpr (std::is_arithmetic_v<T>) {                                   \
      return static_cast<                                                      \
          typename detail::builtin_type_traits<T>                              \
                         ::alternative_data_type<int64_t>>                     \
                             (impl_name(detail::data_element(a, 0)));          \
    } else {                                                                   \
      typename detail::builtin_type_traits<T>::alternative_data_type<int64_t>  \
          result;                                                              \
      for (int i = 0; i < detail::builtin_type_traits<T>::num_elements; ++i) { \
        auto a_i = detail::data_element(a, i);                                 \
        detail::data_element(result, i) = impl_name(a_i);                      \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY(name, impl_name)                \
  template <class T, std::size_t N>                                            \
  HIPSYCL_BUILTIN                                                              \
  marray<T, N> name(marray<T, N> a) noexcept {                                 \
    marray<T, N> result;                                                       \
    for (int i = 0; i < N; ++i)                                                \
      result[i] = impl_name(a[i]);                                             \
    return result;                                                             \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY(name, impl_name)               \
  template <class T, std::size_t N>                                            \
  HIPSYCL_BUILTIN                                                              \
  marray<T, N> name(marray<T, N> a, marray<T, N> b) noexcept {                 \
    marray<T, N> result;                                                       \
    for (int i = 0; i < N; ++i)                                                \
      result[i] = impl_name(a[i], b[i]);                                       \
    return result;                                                             \
  }
  
#define HIPSYCL_DEFINE_BUILTIN(builtin_name, OVERLOAD_SET_GENERATOR,           \
                               FUNCTION_GENERATOR)                             \
  OVERLOAD_SET_GENERATOR(                                                      \
      FUNCTION_GENERATOR, builtin_name,                                        \
      HIPSYCL_PP_CONCATENATE(::hipsycl::sycl::detail::__acpp_,                 \
                             builtin_name))

#define HIPSYCL_DEFINE_NATIVE_BUILTIN(builtin_name, OVERLOAD_SET_GENERATOR,    \
                                      FUNCTION_GENERATOR)                      \
  OVERLOAD_SET_GENERATOR(                                                      \
      FUNCTION_GENERATOR, builtin_name,                                        \
      HIPSYCL_PP_CONCATENATE(::hipsycl::sycl::detail::__acpp_native_,          \
                             builtin_name))

#define HIPSYCL_DEFINE_HALF_BUILTIN(builtin_name, OVERLOAD_SET_GENERATOR,      \
                                      FUNCTION_GENERATOR)                      \
  OVERLOAD_SET_GENERATOR(                                                      \
      FUNCTION_GENERATOR, builtin_name,                                        \
      HIPSYCL_PP_CONCATENATE(::hipsycl::sycl::detail::__acpp_half_,            \
                             builtin_name))

#define HIPSYCL_DEFINE_MARRAY_BUILTIN(builtin_name, FUNCTION_GENERATOR)                            \
  FUNCTION_GENERATOR(builtin_name,                                                                 \
                     HIPSYCL_PP_CONCATENATE(::hipsycl::sycl::detail::__acpp_, builtin_name))

#define HIPSYCL_DEFINE_MARRAY_BUILTIN_NATIVE(builtin_name, FUNCTION_GENERATOR)                     \
  FUNCTION_GENERATOR(                                                                              \
      builtin_name, HIPSYCL_PP_CONCATENATE(::hipsycl::sycl::detail::__acpp_native_, builtin_name))

} // namespace detail

// ********************* math builtins ***************************

HIPSYCL_DEFINE_BUILTIN(acos, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(acosh, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(acospi, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(asin, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(asinh, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(asinpi, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(atan, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(atan2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(atanh, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(atanpi, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(atan2pi, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(cbrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(ceil, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(copysign, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(cos, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(cosh, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(cospi, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(erfc, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(erf, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(exp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(exp2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(exp10, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(expm1, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(fabs, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(fdim, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(floor, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(fma, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)
// fmax/fmin also additionally need to to support (genfloat,sgenfloat)
// arguments. This is a bit nasty since both (genfloat, genfloat) and
// (genfloat, sgenfloat) cover (float,float) so we would end up
// with multiple definitions. Don't define sgenfloat overloads for now -
// this should be fine as vec types can be implicitly constructed
// from scalars anyway.
HIPSYCL_DEFINE_BUILTIN(fmax, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(fmin, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(fmod, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
// fract should accept genfloatptr as second argument, but it's unclear
// if this is really the intention - that would imply that an operation
// on double could write results into a float pointer (and even worse,
// that vector types of multiple dimensions have to interact).
// We assume instead that the second argument should be a pointer to
// the same type as the first argument which seems more reasonable.
HIPSYCL_DEFINE_BUILTIN(fract, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_TGENPTR)
HIPSYCL_DEFINE_BUILTIN(frexp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINTPTR)
HIPSYCL_DEFINE_BUILTIN(hypot, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(ilogb, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT)
HIPSYCL_DEFINE_BUILTIN(ldexp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINT)
HIPSYCL_DEFINE_BUILTIN(ldexp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_INT)
HIPSYCL_DEFINE_BUILTIN(lgamma, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(lgamma_r, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINTPTR)
HIPSYCL_DEFINE_BUILTIN(log, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(log2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(log10, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(log1p, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(logb, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(mad, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)
HIPSYCL_DEFINE_BUILTIN(maxmag, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(minmag, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(modf, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_TGENPTR)

// TODO nan

HIPSYCL_DEFINE_BUILTIN(nextafter, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(pow, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(powr, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(pown, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINT)
HIPSYCL_DEFINE_BUILTIN(remainder, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
// TODO remquo
HIPSYCL_DEFINE_BUILTIN(rint, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(rootn, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINT)
HIPSYCL_DEFINE_BUILTIN(round, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(rsqrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(sin, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(sincos, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_TGENPTR)
HIPSYCL_DEFINE_BUILTIN(sinh, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(sinpi, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(sqrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(tan, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(tanh, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(tgamma, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(trunc, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)

// ************************ marray builtins *****************************
HIPSYCL_DEFINE_MARRAY_BUILTIN(acos, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(acosh, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(acospi, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(asin, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(asinh, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(asinpi, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(atan, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(atan2, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(atanh, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(atanpi, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(atan2pi, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(cbrt, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(ceil, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(copysign, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(cos, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(cosh, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(cospi, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(erfc, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(erf, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(exp, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(exp2, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(exp10, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(expm1, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(fabs, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(fdim, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(floor, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
// HIPSYCL_DEFINE_BUILTIN(fma, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
//                         HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)
HIPSYCL_DEFINE_MARRAY_BUILTIN(fmax,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(fmin,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(fmod,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(fract,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYGENPTR)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(frexp,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYENINTPTR)
HIPSYCL_DEFINE_MARRAY_BUILTIN(hypot,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(ilogb, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY_RET_INT)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(ldexp,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYENINT)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(ldexp,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYNT)
HIPSYCL_DEFINE_MARRAY_BUILTIN(lgamma, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(lgamma_r HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYENINTPTR)
HIPSYCL_DEFINE_MARRAY_BUILTIN(log, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(log2, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(log10, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(log1p, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(logb, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(mad,  HIPSYCL_BUILTIN_GENERATOR_TRINARY_MARRAY_T)
HIPSYCL_DEFINE_MARRAY_BUILTIN(maxmag,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(minmag,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(modf,  HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYGENPTR)

// TODO nan

HIPSYCL_DEFINE_MARRAY_BUILTIN(nextafter, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(pow, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(powr, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
// HIPSYCL_DEFINE_MARRAY_BUILTIN(pown, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAYENINT)
HIPSYCL_DEFINE_MARRAY_BUILTIN(remainder, HIPSYCL_BUILTIN_GENERATOR_BINARY_MARRAY)
// TODO remquo
HIPSYCL_DEFINE_MARRAY_BUILTIN(rint, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
// HIPSYCL_DEFINE_BUILTIN(rootn, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
//                        HIPSYCL_BUILTIN_GENERATOR_BINARY_T_GENINT)
HIPSYCL_DEFINE_MARRAY_BUILTIN(round, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(rsqrt, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(sin, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
// HIPSYCL_DEFINE_BUILTIN(sincos, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
//                        HIPSYCL_BUILTIN_GENERATOR_BINARY_T_TGENPTR)
HIPSYCL_DEFINE_MARRAY_BUILTIN(sinh, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(sinpi, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(sqrt, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(tan, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(tanh, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(tgamma, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)
HIPSYCL_DEFINE_MARRAY_BUILTIN(trunc, HIPSYCL_BUILTIN_GENERATOR_UNARY_MARRAY)

// ********************* native math builtins ***************************

namespace native {

HIPSYCL_DEFINE_NATIVE_BUILTIN(cos, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(divide, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(exp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(exp2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(exp10, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(log, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(log2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(log10, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(powr, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(recip, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(rsqrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(sin, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(sqrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_NATIVE_BUILTIN(tan, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                              HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
}

// ********************* half precision math builtins ***********************
namespace half_precision {

HIPSYCL_DEFINE_HALF_BUILTIN(cos, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(divide, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_HALF_BUILTIN(exp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(exp2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(exp10, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(log, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(log2, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(log10, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(powr, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_HALF_BUILTIN(recip, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(rsqrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(sin, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(sqrt, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_HALF_BUILTIN(tan, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                            HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
}

// ********************** integer functions *******************
#define HIPSYCL_BUILTIN_ENABLE_IF_ADDITIONAL_INT_SCALAR(VecType, ScalarType)   \
  std::enable_if_t<(!std::is_arithmetic_v<VecType> &&                          \
                    std::is_integral_v<typename VecType::element_type> &&      \
                    std::is_integral_v<ScalarType>),                           \
                   int> = 0

HIPSYCL_DEFINE_BUILTIN(abs, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)

// TODO abs_diff
// TODO add_sat
// TODO hadd
// TODO rhadd

HIPSYCL_DEFINE_BUILTIN(clamp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                       HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_ADDITIONAL_INT_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType clamp(const VecType &a, ScalarType minval,
                              ScalarType maxval) {
  using element_type = typename VecType::element_type;
  return clamp(a, VecType{static_cast<element_type>(minval)},
               VecType{static_cast<element_type>(maxval)});
}

HIPSYCL_DEFINE_BUILTIN(ctz, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)

HIPSYCL_DEFINE_BUILTIN(clz, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T)

// TODO mad_hi
// TODO mad_sat

HIPSYCL_DEFINE_BUILTIN(max, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_ADDITIONAL_INT_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType max(const VecType &a, ScalarType b) {
  using element_type = typename VecType::element_type;
  return max(a, VecType{static_cast<element_type>(b)});
}

HIPSYCL_DEFINE_BUILTIN(min, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_ADDITIONAL_INT_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType min(const VecType &a, ScalarType b) {
  using element_type = typename VecType::element_type;
  return min(a, VecType{static_cast<element_type>(b)});
}

// TODO mul_hi
// TODO rotate
// TODO sub_sat
// TODO upsample
// TODO mad24

HIPSYCL_DEFINE_BUILTIN(mul24, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER32BIT,
                        HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)
HIPSYCL_DEFINE_BUILTIN(popcount, HIPSYCL_BUILTIN_OVERLOAD_SET_GENINTEGER,
                        HIPSYCL_BUILTIN_GENERATOR_UNARY_T)

// ********************** common functions *******************
#define HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)      \
  std::enable_if_t<                                                            \
      (!std::is_arithmetic_v<VecType> &&                                       \
       std::is_same_v<ScalarType, typename VecType::element_type> &&           \
       std::is_floating_point_v<ScalarType>),                                  \
      int> = 0


HIPSYCL_DEFINE_BUILTIN(clamp, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                        HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType clamp(const VecType &a, ScalarType minval,
                              ScalarType maxval) {
  return clamp(a, VecType{minval}, VecType{maxval});
}

HIPSYCL_DEFINE_BUILTIN(degrees, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                        HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
HIPSYCL_DEFINE_BUILTIN(radians,
                        HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                        HIPSYCL_BUILTIN_GENERATOR_UNARY_T)

HIPSYCL_DEFINE_BUILTIN(max, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                      HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType max(const VecType &a, ScalarType b) {
  return max(a, VecType{b});
}

HIPSYCL_DEFINE_BUILTIN(min, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                      HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType min(const VecType &a, ScalarType b) {
  return min(a, VecType{b});
}

HIPSYCL_DEFINE_BUILTIN(mix, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                      HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType mix(const VecType &a, const VecType& b, ScalarType c) {
  return mix(a, b, VecType{c});
}

HIPSYCL_DEFINE_BUILTIN(step, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                        HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType step(ScalarType edge, const VecType& x) {
  return step(VecType{edge}, x);
}

HIPSYCL_DEFINE_BUILTIN(smoothstep, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_TRINARY_T_T_T)

template <class VecType, class ScalarType,
          HIPSYCL_BUILTIN_ENABLE_IF_MATCHING_FP_SCALAR(VecType, ScalarType)>
HIPSYCL_BUILTIN VecType smoothstep(ScalarType edge0, ScalarType edge1,
                                   const VecType &x) {
  return smoothstep(VecType{edge0}, VecType{edge1}, x);
}

HIPSYCL_DEFINE_BUILTIN(sign, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOAT,
                        HIPSYCL_BUILTIN_GENERATOR_UNARY_T)
// ********************** geometric functions *******************

// TODO: m[float/double][3/4]
HIPSYCL_BUILTIN float3 cross(float3 a, float3 b) {
  return __acpp_cross3(a, b);
}
HIPSYCL_BUILTIN double3 cross(double3 a, double3 b) {
  return __acpp_cross3(a, b);
}
HIPSYCL_BUILTIN float4 cross(float4 a, float4 b) {
  return __acpp_cross4(a, b);
}
HIPSYCL_BUILTIN double4 cross(double4 a, double4 b) {
  return __acpp_cross4(a, b);
}

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T(T, name, impl_name)     \
  HIPSYCL_BUILTIN typename detail::builtin_type_traits<T>::element_type name(  \
      T a, T b) noexcept {                                                     \
    return impl_name(a, b);                                                    \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T(T, name, impl_name)        \
  HIPSYCL_BUILTIN typename detail::builtin_type_traits<T>::element_type name(  \
      T a) noexcept {                                                          \
    return impl_name(a);                                                       \
  }

HIPSYCL_DEFINE_BUILTIN(dot, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T)
HIPSYCL_DEFINE_BUILTIN(dot, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T)

HIPSYCL_DEFINE_BUILTIN(length, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T)
HIPSYCL_DEFINE_BUILTIN(length, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T)

HIPSYCL_DEFINE_BUILTIN(distance, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T)
HIPSYCL_DEFINE_BUILTIN(distance, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T)

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T_RET_T(T, name, impl_name) \
  HIPSYCL_BUILTIN T name(T a) noexcept { return impl_name(a); }

HIPSYCL_DEFINE_BUILTIN(normalize, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T_RET_T)
HIPSYCL_DEFINE_BUILTIN(normalize, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T_RET_T)

HIPSYCL_DEFINE_BUILTIN(fast_length, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T)
HIPSYCL_DEFINE_BUILTIN(fast_length, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T)

HIPSYCL_DEFINE_BUILTIN(fast_distance, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T)
HIPSYCL_DEFINE_BUILTIN(fast_distance, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_BINARY_REDUCTION_T_T)

HIPSYCL_DEFINE_BUILTIN(fast_normalize, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEOFLOAT,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T_RET_T)
HIPSYCL_DEFINE_BUILTIN(fast_normalize, HIPSYCL_BUILTIN_OVERLOAD_SET_GENGEODOUBLE,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_REDUCTION_T_RET_T)


// ********************** relational functions *******************
HIPSYCL_DEFINE_BUILTIN(isnan, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT)
HIPSYCL_DEFINE_BUILTIN(isnan, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT64)

HIPSYCL_DEFINE_BUILTIN(isinf, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT)
HIPSYCL_DEFINE_BUILTIN(isinf, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT64)

HIPSYCL_DEFINE_BUILTIN(isfinite, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT)
HIPSYCL_DEFINE_BUILTIN(isfinite, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT64)

HIPSYCL_DEFINE_BUILTIN(isnormal, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT)
HIPSYCL_DEFINE_BUILTIN(isnormal, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT64)

HIPSYCL_DEFINE_BUILTIN(signbit, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATF,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT)
HIPSYCL_DEFINE_BUILTIN(signbit, HIPSYCL_BUILTIN_OVERLOAD_SET_GENFLOATD,
                       HIPSYCL_BUILTIN_GENERATOR_UNARY_T_RET_INT64)
}
}

#endif
