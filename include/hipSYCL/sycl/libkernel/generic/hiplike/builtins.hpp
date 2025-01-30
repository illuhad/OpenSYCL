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

#ifndef ACPP_LIBKERNEL_HIPLIKE_BUILTINS_HPP
#define ACPP_LIBKERNEL_HIPLIKE_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"

#include <cstdlib>
#include <cmath>
#include <type_traits>

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl {
namespace detail::hiplike_builtins {

// ********************** math builtins *********************

#define HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(name, impl_name_sp, impl_name_dp)  \
  HIPSYCL_HIPLIKE_BUILTIN float name(float x) { return ::impl_name_sp(x); }    \
  HIPSYCL_HIPLIKE_BUILTIN double name(double x) { return ::impl_name_dp(x); }

#define HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(name, impl_name_sp, impl_name_dp) \
  HIPSYCL_HIPLIKE_BUILTIN float name(float x, float y) {                       \
    return ::impl_name_sp(x, y);                                               \
  }                                                                            \
  HIPSYCL_HIPLIKE_BUILTIN double name(double x, double y) {                    \
    return ::impl_name_dp(x, y);                                               \
  }

#define HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN3(name, impl_name_sp, impl_name_dp) \
  HIPSYCL_HIPLIKE_BUILTIN float name(float x, float y, float z) {              \
    return ::impl_name_sp(x, y, z);                                            \
  }                                                                            \
  HIPSYCL_HIPLIKE_BUILTIN double name(double x, double y, double z) {          \
    return ::impl_name_dp(x, y, z);                                            \
  }


HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_acos, acosf, acos)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_acosh, acoshf, acosh)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_acospi(T x) noexcept {
  return hiplike_builtins::__acpp_acos(x) / M_PI;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_asin, asinf, asin)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_asinh, asinhf, asinh)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_asinpi(T x) noexcept {
  return hiplike_builtins::__acpp_asin(x) / M_PI;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_atan, atanf, atan)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_atan2, atan2f, atan2)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_atanh, atanhf, atanh)


template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atanpi(T x) noexcept {
  return hiplike_builtins::__acpp_atan(x) / M_PI;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atan2pi(T x, T y) noexcept {
  return hiplike_builtins::__acpp_atan2(x, y) / M_PI;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_cbrt, cbrtf, cbrt)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_ceil, ceilf, ceil)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_copysign, copysignf, copysign)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_cos, cosf, cos)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_cosh, coshf, cosh)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_cospi, cospif, cospi)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_erf, erff, erf)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_erfc, erfcf, erfc)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_exp, expf, exp)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_exp2, exp2f, exp2)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_pow, powf, pow)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_exp10(T x) noexcept {
  return hiplike_builtins::__acpp_pow(static_cast<T>(10), x);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_expm1, expm1f, expm1)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_fabs, fabsf, fabs)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_fdim, fdimf, fdim)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_floor, floorf, floor)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN3(__acpp_fma, fmaf, fma)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_fmax, fmaxf, fmax)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_fmin, fminf, fmin)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_fmod, fmodf, fmod)

// Unsupported
template<class T>
HIPSYCL_HIPLIKE_BUILTIN
T __acpp_fract(T x, T* ptr) noexcept;

template<class IntPtr>
HIPSYCL_HIPLIKE_BUILTIN float __acpp_frexp(float x, IntPtr y) noexcept {
  return ::frexpf(x, y);
}

template<class IntPtr>
HIPSYCL_HIPLIKE_BUILTIN double __acpp_frexp(double x, IntPtr y) noexcept {
  return ::frexp(x, y);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_hypot, hypotf, hypot)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_ilogb, ilogbf, ilogb)

template<class IntType>
HIPSYCL_HIPLIKE_BUILTIN float __acpp_ldexp(float x, IntType k) noexcept {
  return ::ldexpf(x, k);
}

template<class IntType>
HIPSYCL_HIPLIKE_BUILTIN double __acpp_ldexp(double x, IntType k) noexcept {
  return ::ldexp(x, k);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_lgamma, lgammaf, lgamma)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_tgamma, tgammaf, tgamma)

template<class T, class IntPtr>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_lgamma_r(T x, IntPtr y) noexcept {
  auto r = hiplike_builtins::__acpp_lgamma(x);
  auto g = hiplike_builtins::__acpp_tgamma(x);
  *y = (g >= 0) ? 1 : -1;
  return r;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_log, logf, log)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_log2, log2f, log2)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_log10, log10f, log10)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_log1p, log1pf, log1p)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_logb, logbf, logb)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_mad(T x, T y, T z) noexcept {
  return hiplike_builtins::__acpp_fma(x, y, z);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_maxmag(T x, T y) noexcept {
  auto abs_x = (x < 0) ? -x : x;
  auto abs_y = (y < 0) ? -y : y;
  if(abs_x == abs_y) return hiplike_builtins::__acpp_fmax(x,y);
  return (abs_x > abs_y) ? x : y;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_minmag(T x, T y) noexcept {
  auto abs_x = (x < 0) ? -x : x;
  auto abs_y = (y < 0) ? -y : y;
  if(abs_x == abs_y) return hiplike_builtins::__acpp_fmin(x,y);
  return (abs_x < abs_y) ? x : y;
}

template<class FloatPtr>
HIPSYCL_HIPLIKE_BUILTIN float __acpp_modf(float x, FloatPtr y) noexcept {
  return ::modff(x, y);
};

template<class FloatPtr>
HIPSYCL_HIPLIKE_BUILTIN double __acpp_modf(double x, FloatPtr y) noexcept {
  return ::modf(x, y);
};

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_nextafter, nextafterf, nextafter)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_powr(T x, T y) noexcept {
  return hiplike_builtins::__acpp_pow(x, y);
}

template<class T, class IntType>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_pown(T x, IntType y) noexcept {
  return hiplike_builtins::__acpp_pow(x, static_cast<T>(y));
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__acpp_remainder, remainderf, remainder)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_rint, rintf, rint)

template<class T, class IntType>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_rootn(T x, IntType y) noexcept {
  return hiplike_builtins::__acpp_pow(x, T{1}/T{y});
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_round, roundf, round)

HIPSYCL_HIPLIKE_BUILTIN float __acpp_rsqrt(float x) noexcept {
  return __frsqrt_rn(x);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_sqrt, sqrtf, sqrt)

HIPSYCL_HIPLIKE_BUILTIN double __acpp_rsqrt(double x) noexcept {
  return 1. / hiplike_builtins::__acpp_sqrt(x);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_sin, sinf, sin)

template<class FloatPtr>
HIPSYCL_HIPLIKE_BUILTIN float __acpp_sincos(float x, FloatPtr cosval) noexcept {
  float sinval;
  ::sincosf(x, &sinval, cosval);
  return sinval;
}

template<class FloatPtr>
HIPSYCL_HIPLIKE_BUILTIN double __acpp_sincos(double x, FloatPtr cosval) noexcept {
  double sinval;
  ::sincos(x, &sinval, cosval);
  return sinval;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_sinh, sinhf, sinh)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_sinpi, sinpif, sinpi)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_tan, tanf, tan)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_tanh, tanhf, tanh)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__acpp_trunc, truncf, trunc)

// ***************** native math builtins ******************

#define HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(name, sp_impl, dp_impl)     \
  HIPSYCL_HIPLIKE_BUILTIN float name(float x) noexcept { return sp_impl(x); }  \
  HIPSYCL_HIPLIKE_BUILTIN double name(double x) noexcept { return dp_impl(x); }

#define HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN2(name, sp_impl, dp_impl)    \
  HIPSYCL_HIPLIKE_BUILTIN float name(float x, float y) noexcept {              \
    return sp_impl(x, y);                                                      \
  }                                                                            \
  HIPSYCL_HIPLIKE_BUILTIN double name(double x, double y) noexcept {           \
    return dp_impl(x, y);                                                      \
  }

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_cos, __cosf,
                                           __acpp_cos)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_native_divide(T x, T y) noexcept {
  return x / y;
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_exp, __expf,
                                           __acpp_exp)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_native_exp2(T x) noexcept {
  return hiplike_builtins::__acpp_exp2(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_native_exp10(T x) noexcept {
  return hiplike_builtins::__acpp_exp10(x);
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_log, __logf,
                                           __acpp_log)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_log2, __log2f,
                                           __acpp_log2)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_log10, __log10f,
                                           __acpp_log10)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN2(__acpp_native_powr, __powf,
                                           __acpp_powr)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_native_recip(T x) noexcept {
  return hiplike_builtins::__acpp_native_divide(T{1}, x);
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_rsqrt, __frsqrt_rn,
                                           __acpp_rsqrt)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_sin, __sinf,
                                           __acpp_sin)

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_native_sqrt(T x) noexcept {
  return hiplike_builtins::__acpp_sqrt(x);
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__acpp_native_tan, __tanf,
                                           __acpp_tan)

// ***************** half precision math builtins ******************


template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_cos(T x) noexcept {
  return hiplike_builtins::__acpp_cos(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_divide(T x, T y) noexcept {
  return x / y;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_exp(T x) noexcept {
  return hiplike_builtins::__acpp_exp(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_exp2(T x) noexcept {
  return hiplike_builtins::__acpp_exp2(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_exp10(T x) noexcept {
  return hiplike_builtins::__acpp_exp10(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_log(T x) noexcept {
  return hiplike_builtins::__acpp_log(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_log2(T x) noexcept {
  return hiplike_builtins::__acpp_log2(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_log10(T x) noexcept {
  return hiplike_builtins::__acpp_log10(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_powr(T x, T y) noexcept {
  return hiplike_builtins::__acpp_powr(x, y);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_recip(T x) noexcept {
  return hiplike_builtins::__acpp_native_divide(T{1}, x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_rsqrt(T x) noexcept {
  return hiplike_builtins::__acpp_rsqrt(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_sin(T x) noexcept {
  return hiplike_builtins::__acpp_sin(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_sqrt(T x) noexcept {
  return hiplike_builtins::__acpp_sqrt(x);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_half_tan(T x) noexcept {
  return hiplike_builtins::__acpp_tan(x);
}

// ***************** integer functions **************

template<class T, std::enable_if_t<std::is_signed_v<T>,int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_abs(T x) noexcept {
  return (x < 0) ? -x : x;
}

template<class T, std::enable_if_t<!std::is_signed_v<T>,int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_abs(T x) noexcept {
  return x;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_max(T x, T y) noexcept {
  return (x > y) ? x : y;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_min(T x, T y) noexcept {
  return (x < y) ? x : y;
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  return hiplike_builtins::__acpp_min(
    hiplike_builtins::__acpp_max(x, minval), maxval);
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) < 4),
              int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_ctz(T x) noexcept {

  //we convert to the unsigned type to avoid the typecast creating 
  //additional ones in front of the value if x is negative
  using Usigned = typename std::make_unsigned<T>::type;

  constexpr T size = CHAR_BIT*sizeof(Usigned);

  return x ? __clz(static_cast<__acpp_int32>(__brev(static_cast<Usigned>(x)))) : size;
  
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 4),
              int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_ctz(T x) noexcept {

  return __clz(static_cast<__acpp_int32>(__brev(static_cast<__acpp_uint32>(x))));
  
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 8),
              int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_ctz(T x) noexcept {

  return __clzll(static_cast<__acpp_int64>(__brevll(static_cast<__acpp_uint64>(x))));

}

template<class T>
__attribute__((noinline))
T __noinline_clz(T a) noexcept {
    return __builtin_clz(a);
}


template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) < 4),
              int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_clz(T x) noexcept {

  //we convert to the unsigned type to avoid the typecast creating 
  //additional ones in front of the value if x is negative
  using Usigned = typename std::make_unsigned<T>::type; 

  constexpr T diff = CHAR_BIT*(sizeof(__acpp_int32) - sizeof(Usigned));
  constexpr T size = CHAR_BIT*sizeof(T);

  auto v = static_cast<__acpp_int32>(static_cast<Usigned>(x));

  // here we force noinline on clz to avoid the if(v) to be optimized away as
  // llvm rightfully assume that clz(0) == bitsize
  // see : https://llvm.org/docs/LangRef.html#llvm-ctlz-intrinsic
  return v ? __noinline_clz(v)-diff : size;
  
}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 4),
              int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_clz(T x) noexcept {

  return __clz(static_cast<__acpp_int32>(x));

}

template <class T,
          std::enable_if_t<
              (std::is_integral_v<T> && sizeof(T) == 8),
              int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_clz(T x) noexcept {

  return __clzll(static_cast<__acpp_int64>(x));

}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_mul24(T x, T y) noexcept {
  return __mul24(x, y);
}


template<class T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) < 4, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_popcount(T x) noexcept {
  //we convert to the unsigned type to avoid the typecast creating
  //additional ones in front of the value if x is negative
  using Usigned = typename std::make_unsigned<T>::type;
  return __popc(static_cast<__acpp_uint32>(static_cast<Usigned>(x)));
}

template<class T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_popcount(T x) noexcept {
  return __popc(static_cast<__acpp_uint32>(x));
}

template<class T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 8, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_popcount(T x) noexcept {
  return __popcll(static_cast<__acpp_uint64>(x));
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  return hiplike_builtins::__acpp_fmin(
    hiplike_builtins::__acpp_fmax(x, minval), maxval);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_degrees(T x) noexcept {
  return (180.f / M_PI) * x;
}

// __acpp_max() and __acpp_min() are handled by the overloads from the
// integer functions

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_mix(T x, T y, T a) noexcept {
  return x + (y - x) * a;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_radians(T x) noexcept {
  return (M_PI / 180.f) * x;
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_step(T edge, T x) noexcept {
  return (x < edge) ? T{0} : T{1};
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_smoothstep(T edge0, T edge1, T x) noexcept {
  T t = hiplike_builtins::__acpp_clamp((x - edge0) / (edge1 - edge0), T{0},
                                          T{1});
  return t * t * (3 - 2 * t);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_sign(T x) noexcept {
  if(std::isnan(x))
    return T{0};
  return (x == T{0}) ? x : ((x > 0) ? T{1} : T{-1});
}

template <typename VecType>
HIPSYCL_HIPLIKE_BUILTIN VecType 
__acpp_cross3(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(),
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x()};
}

template <typename VecType>
HIPSYCL_HIPLIKE_BUILTIN VecType 
__acpp_cross4(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(), 
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x(),
          typename VecType::element_type{0}};
}

// ****************** geometric functions ******************

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_dot(T a, T b) noexcept {
  return a * b;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN typename T::element_type __acpp_dot(T a, T b) noexcept {
  typename T::element_type result = 0;
  for (int i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN typename T::element_type __acpp_length(T a) noexcept {
  auto d = hiplike_builtins::__acpp_dot(a, a);
  return hiplike_builtins::__acpp_sqrt(d);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN auto __acpp_distance(T a, T b) noexcept {
  return hiplike_builtins::__acpp_length(a - b);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_normalize(T a) noexcept {
  // TODO rsqrt might be more efficient
  return a / hiplike_builtins::__acpp_length(a);
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_fast_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_HIPLIKE_BUILTIN typename T::element_type __acpp_fast_length(T a) noexcept {
  auto d = hiplike_builtins::__acpp_dot(a, a);
  return hiplike_builtins::__acpp_half_sqrt(d);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN auto __acpp_fast_distance(T a, T b) noexcept {
  return hiplike_builtins::__acpp_fast_length(a - b);
}

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_fast_normalize(T a) noexcept {
  // TODO use rsqrt
  return a / hiplike_builtins::__acpp_fast_length(a);
}

// ********************** relational functions *********************

#define HIPSYCL_DEFINE_HIPLIKE_REL_BUILTIN(name, impl_name_sp, impl_name_dp)   \
  HIPSYCL_HIPLIKE_BUILTIN int name(float x) { return ::impl_name_sp(x); }      \
  HIPSYCL_HIPLIKE_BUILTIN int name(double x) { return ::impl_name_dp(x); }

HIPSYCL_DEFINE_HIPLIKE_REL_BUILTIN(__acpp_isnan, isnan, isnan);

HIPSYCL_DEFINE_HIPLIKE_REL_BUILTIN(__acpp_isinf, isinf, isinf);

HIPSYCL_DEFINE_HIPLIKE_REL_BUILTIN(__acpp_isfinite, isfinite, isfinite);

HIPSYCL_DEFINE_HIPLIKE_REL_BUILTIN(__acpp_isnormal, __builtin_isnormal, __builtin_isnormal);

HIPSYCL_DEFINE_HIPLIKE_REL_BUILTIN(__acpp_signbit, signbit, signbit);

}
}
}

#endif

#endif
