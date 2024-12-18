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
#include <cmath>

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(name)                              \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x) {               \
    return name##f(x);                                                         \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x) {             \
    return name(x);                                                            \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(name)                             \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y) {      \
    return name##f(x, y);                                                      \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y) {   \
    return name(x, y);                                                         \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3(name)                             \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y,        \
                                                      float z) {               \
    return name##f(x, y, z);                                                   \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y,     \
                                                       double z) {             \
    return name(x, y, z);                                                      \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN_NAME(name, float_name,             \
                                                 double_name)                  \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x) {               \
    return float_name(x);                                                      \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x) {             \
    return double_name(x);                                                     \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2_NAME(name, float_name,            \
                                                  double_name)                 \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y) {      \
    return float_name(x, y);                                                   \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y) {   \
    return double_name(x, y);                                                  \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3_NAME(name, float_name,            \
                                                  double_name)                 \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y,        \
                                                      float z) {               \
    return float_name(x, y, z);                                                \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y,     \
                                                       double z) {             \
    return double_name(x, y, z);                                               \
  }

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(acos)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(acosh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(asin)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(asinh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(atan)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(atan2)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(atanh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cbrt)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(ceil)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(copysign)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cos)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cosh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(erf)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(erfc)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(exp)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(exp2)
#ifndef _WIN32
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(exp10)
#endif
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(pow)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(expm1)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(fabs)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fdim)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(floor)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3(fma)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fmax)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fmin)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fmod)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acospi_f32(float x) {
  return acosf(x) / M_PI;
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acospi_f64(double x) {
  return acos(x) / M_PI;
}


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asinpi_f32(float x) {
  return asinf(x) / M_PI;
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asinpi_f64(double x) {
  return asin(x) / M_PI;
}


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atanpi_f32(float x) {
  return atan(x) / M_PI;
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atanpi_f64(double x) {
  return atan(x) / M_PI;
}


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan2pi_f32(float x, float y) {
  return atan2(x,y) / M_PI;
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan2pi_f64(double x, double y) {
  return atan2(x,y) / M_PI;
}


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cospi_f32(float x) {
  return cosf(x) / M_PI;
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cospi_f64(double x) {
  return cos(x) / M_PI;
}

// fmin(x - floor(x), nextafter(genfloat(1.0), genfloat(0.0)) ). floor(x) is returned in iptr.
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fract_f32(float x, float* y) {
  *y = floorf(x);
  return fminf(x - *y, nextafterf(1.f, 0.f));
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fract_f64(double x, double* y) {
  *y = floor(x);
  return fmin(x - *y, nextafter(1., 0.));
}

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_frexp_f32(float x,
                                                    __acpp_int32 *y) {
  return frexpf(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_frexp_f64(double x,
                                                     __acpp_int64 *y) {
  __acpp_int32 w;
  auto res = frexp(x, &w);
  *y = w;
  return res;
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(hypot)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(ilogb)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ldexp_f32(float x,
                                                    __acpp_int32 k) {
  return ldexpf(x, k);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ldexp_f64(double x,
                                                     __acpp_int64 k) {
  return ldexp(x, k);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(tgamma)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(lgamma)


#ifndef _WIN32
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_r_f32(float x, __acpp_int32* y ) {
  return lgammaf_r(x,y);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_r_f64(double x, __acpp_int64* y) {
  __acpp_int32 w;
  auto res = lgamma_r(x,&w);
  *y = w;
  return res;
}
#endif

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log2)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log10)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log1p)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(logb)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3_NAME(mad,fmaf,fma)

#ifndef _WIN32
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2_NAME(maxmag,fmaxmagf,fmaxmag)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2_NAME(minmag,fmaxmagf,fmaxmag)
#endif

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_modf_f32(float x, float* y ) {
  return modff(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_modf_f64(double x, double* y) {
  return modf(x, y);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(nextafter)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2_NAME(powr,powf,pow)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pown_f32(float x, __acpp_int32 y) {
  return __acpp_sscp_pow_f32(x, (float)y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pown_f64(double x,
                                                    __acpp_int32 y) {
  return __acpp_sscp_pow_f64(x, (double)y);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(remainder)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(rint)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rootn_f32(float x,
                                                    __acpp_int32 y) {
  return __acpp_sscp_pow_f32(x, 1.f/static_cast<float>(y));
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rootn_f64(double x,
                                                     __acpp_int64 y) {
  return __acpp_sscp_pow_f64(x, 1./static_cast<double>(y));
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(round)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rsqrt_f32(float x) {
  return 1.f/__acpp_sscp_sqrt_f32(x);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rsqrt_f64(double x) {
  return 1./__acpp_sscp_sqrt_f64(x);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sqrt)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sin)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sinh)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sinpi_f32(float x) {
  return sinf(x) / M_PI;
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sinpi_f64(double x) {
  return sin(x) / M_PI;
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(tan)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(tanh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(trunc)
