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
#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu/ockl.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu/ocml.hpp"

template <class T> __amdgpu_private T *to_private(T* gen_pointer) {
  return (__amdgpu_private T *)gen_pointer;
}

#define HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(name, ocml_name)                   \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x) {               \
    return ocml_name##_f32(x);                                                 \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x) {             \
    return ocml_name##_f64(x);                                                 \
  }

#define HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(name, ocml_name)                  \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y) {      \
    return ocml_name##_f32(x, y);                                              \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y) {   \
    return ocml_name##_f64(x, y);                                              \
  }

#define HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN3(name, ocml_name)                  \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y,        \
                                                      float z) {               \
    return ocml_name##_f32(x, y, z);                                           \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y,     \
                                                       double z) {             \
    return ocml_name##_f64(x, y, z);                                           \
  }

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(acos, __ocml_acos)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(acosh, __ocml_acosh)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(acospi, __ocml_acospi)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(asin, __ocml_asin)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(asinh, __ocml_asinh)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(asinpi, __ocml_asinpi)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(atan, __ocml_atan)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(atan2, __ocml_atan2)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(atanh, __ocml_atanh)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(atanpi, __ocml_atanpi)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(atan2pi, __ocml_atan2pi)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(cbrt, __ocml_cbrt)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(ceil, __ocml_ceil)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(copysign, __ocml_copysign)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(cos, __ocml_cos)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(cosh, __ocml_cosh)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(cospi, __ocml_cospi)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(erf, __ocml_erf)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(erfc, __ocml_erfc)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(exp, __ocml_exp)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(exp2, __ocml_exp2)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(exp10, __ocml_exp10)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(pow, __ocml_pow)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(expm1, __ocml_expm1)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(fabs, __ocml_fabs)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(fdim, __ocml_fdim)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(floor, __ocml_floor)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN3(fma, __ocml_fma)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(fmax, __ocml_fmax)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(fmin, __ocml_fmin)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(fmod, __ocml_fmod)

// fmin(x - floor(x), nextafter(genfloat(1.0), genfloat(0.0)) ). floor(x) is returned in iptr.
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fract_f32(float x, float* y ) {
  if(__ockl_is_private_addr(y)) {
    return __ocml_fract_f32(x, to_private(y));
  } else {
    float w;
    auto res = __ocml_fract_f32(x, to_private(&w));
    *y = w;
    return res;
  }
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fract_f64(double x, double* y) {
  if(__ockl_is_private_addr(y)) {
    return __ocml_fract_f64(x, to_private(y));
  } else {
    double w;
    auto res = __ocml_fract_f64(x, to_private(&w));
    *y = w;
    return res;
  }
}

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_frexp_f32(float x, __acpp_int32 *y) {
  if(__ockl_is_private_addr(y))
    return __ocml_frexp_f32(x, to_private(y));
  else {
    __acpp_int32 w;
    float res = __ocml_frexp_f32(x, to_private(&w));
    *y = w;
    return res;
  }
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_frexp_f64(double x, __acpp_int64 *y) {
  __acpp_int32 w;
  double res = __ocml_frexp_f64(x, to_private(&w));
  *y = static_cast<__acpp_int64>(w);
  return res;
}

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(hypot, __ocml_hypot)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(ilogb, __ocml_ilogb)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ldexp_f32(float x, __acpp_int32 k) {
  return __ocml_ldexp_f32(x, k);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ldexp_f64(double x, __acpp_int64 k) {
  return __ocml_ldexp_f64(x, k);
}

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(tgamma, __ocml_tgamma)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(lgamma, __ocml_lgamma)


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_r_f32(float x, __acpp_int32* y ) {
  if(__ockl_is_private_addr(y)) {
    return __ocml_lgamma_r_f32(x, to_private(y));
  } else {
    __acpp_int32 w;
    auto res = __ocml_lgamma_r_f32(x, to_private(&w));
    *y = w;
    return res;
  }
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_r_f64(double x, __acpp_int64* y) {
  __acpp_int32 w;
  auto res = __ocml_lgamma_r_f64(x, to_private(&w));
  *y = w;
  return res;
}

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(log, __ocml_log)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(log2, __ocml_log2)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(log10, __ocml_log10)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(log1p, __ocml_log1p)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(logb, __ocml_logb)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN3(mad, __ocml_mad)

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(maxmag, __ocml_maxmag)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(minmag, __ocml_minmag)


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_modf_f32(float x, float* y ) {
  if(__ockl_is_private_addr(y)) {
    return __ocml_modf_f32(x, to_private(y));
  } else {
    float w;
    auto res = __ocml_modf_f32(x, to_private(&w));
    *y = w;
    return res;
  }
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_modf_f64(double x, double* y) {
  if(__ockl_is_private_addr(y)) {
    return __ocml_modf_f64(x, to_private(y));
  } else {
    double w;
    auto res = __ocml_modf_f64(x, to_private(&w));
    *y = w;
    return res;
  }
}

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(nextafter, __ocml_nextafter)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(powr, __ocml_powr)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pown_f32(float x, __acpp_int32 y) {
  return __ocml_pown_f32(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pown_f64(double x, __acpp_int32 y) {
  return __ocml_pown_f64(x, y);
}

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN2(remainder, __ocml_remainder)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(rint, __ocml_rint)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rootn_f32(float x, __acpp_int32 y) {
  return __ocml_rootn_f32(x, y);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rootn_f64(double x, __acpp_int64 y) {
  return __ocml_rootn_f64(x, static_cast<__acpp_int32>(y));
}

HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(round, __ocml_round)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(rsqrt, __ocml_rsqrt)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(sqrt, __ocml_sqrt)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(sin, __ocml_sin)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(sinh, __ocml_sinh)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(sinpi, __ocml_sinpi)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(tan, __ocml_tan)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(tanh, __ocml_tanh)
HIPSYCL_SSCP_MAP_OCML_FLOAT_BUILTIN(trunc, __ocml_trunc)
