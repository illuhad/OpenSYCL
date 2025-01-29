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
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"

#define PI 3.14159265358979323846

#define HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(name, float_version,                \
                                           double_version)                     \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x) {               \
    return float_version(x);                                                   \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x) {             \
    return double_version(x);                                                  \
  }

#define HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(name, float_version,               \
                                            double_version)                    \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y) {      \
    return float_version(x, y);                                                \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y) {   \
    return double_version(x, y);                                               \
  }

#define HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN3(name, float_version,               \
                                            double_version)                    \
  HIPSYCL_SSCP_BUILTIN float __acpp_sscp_##name##_f32(float x, float y,        \
                                                      float z) {               \
    return float_version(x, y, z);                                             \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y,     \
                                                       double z) {             \
    return double_version(x, y, z);                                            \
  }

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(acos, __nv_acosf, __nv_acos)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(acosh, __nv_acoshf, __nv_acosh)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acospi_f32(float x) { return __acpp_sscp_acos_f32(x) / PI; }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acospi_f64(double x) { return __acpp_sscp_acos_f64(x) / PI; }

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(asin, __nv_asinf, __nv_asin)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(asinh, __nv_asinhf, __nv_asinh)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asinpi_f32(float x) { return __acpp_sscp_asin_f32(x) / PI; }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asinpi_f64(double x) { return __acpp_sscp_asin_f64(x) / PI; }

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(atan, __nv_atanf, __nv_atan)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(atan2, __nv_atan2f, __nv_atan2)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(atanh, __nv_atanhf, __nv_atanh)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atanpi_f32(float x) { return __acpp_sscp_atan_f32(x) / PI; }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atanpi_f64(double x) { return __acpp_sscp_atan_f64(x) / PI; }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan2pi_f32(float x, float y) { return __acpp_sscp_atan2_f32(x, y) / PI; }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan2pi_f64(double x, double y) { return __acpp_sscp_atan2_f64(x, y) / PI; }

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(cbrt, __nv_cbrtf, __nv_cbrt)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(ceil, __nv_ceilf, __nv_ceil)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(copysign, __nv_copysignf, __nv_copysign)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(cos, __nv_cosf, __nv_cos)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(cosh, __nv_coshf, __nv_cosh)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(cospi, __nv_cospif, __nv_cospi)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(erf, __nv_erff, __nv_erf)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(erfc, __nv_erfcf, __nv_erfc)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(exp, __nv_expf, __nv_exp)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(exp2, __nv_exp2f, __nv_exp2)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(exp10, __nv_exp10f, __nv_exp10)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(pow, __nv_powf, __nv_pow)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(expm1, __nv_expm1f, __nv_expm1)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(fabs, __nv_fabsf, __nv_fabs)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(fdim, __nv_fdimf, __nv_fdim)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(floor, __nv_floorf, __nv_floor)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN3(fma, __nv_fmaf, __nv_fma)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(fmax, __nv_fmaxf, __nv_fmax)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(fmin, __nv_fminf, __nv_fmin)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(fmod, __nv_fmodf, __nv_fmod)

// fmin(x - floor(x), nextafter(genfloat(1.0), genfloat(0.0)) ). floor(x) is returned in iptr.
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fract_f32(float x, float* y ) {
  *y = __acpp_sscp_floor_f32(x);
  return __acpp_sscp_fmin_f32(x - *y, __acpp_sscp_nextafter_f32(1.f, 0.f));
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fract_f64(double x, double* y) {
  *y = __acpp_sscp_floor_f64(x);
  return __acpp_sscp_fmin_f64(x - *y, __acpp_sscp_nextafter_f64(1.f, 0.f));
}

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_frexp_f32(float x,
                                                    __acpp_int32 *y) {
  return __nv_frexpf(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_frexp_f64(double x,
                                                     __acpp_int64 *y) {
  __acpp_int32 w;
  double res = __nv_frexp(x, &w);
  *y = static_cast<__acpp_int64>(w);
  return res;
}

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(hypot, __nv_hypotf, __nv_hypot)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(ilogb, __nv_ilogbf, __nv_ilogb)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ldexp_f32(float x,
                                                    __acpp_int32 k) {
  return __nv_ldexpf(x, k);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ldexp_f64(double x,
                                                     __acpp_int64 k) {
  return __nv_ldexp(x, k);
}

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(tgamma, __nv_tgammaf, __nv_tgamma)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(lgamma, __nv_lgammaf, __nv_lgamma)


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_r_f32(float x, __acpp_int32* y ) {
  auto r = __acpp_sscp_lgamma_f32(x);
  auto g = __acpp_sscp_tgamma_f32(x);
  *y = (g >= 0) ? 1 : -1;
  return r;
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_r_f64(double x, __acpp_int64* y) {
  auto r = __acpp_sscp_lgamma_f64(x);
  auto g = __acpp_sscp_tgamma_f64(x);
  *y = (g >= 0) ? 1 : -1;
  return r;
}

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(log, __nv_logf, __nv_log)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(log2, __nv_log2f, __nv_log2)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(log10, __nv_log10f, __nv_log10)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(log1p, __nv_log1pf, __nv_log1p)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(logb, __nv_logbf, __nv_logb)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN3(mad, __nv_fmaf, __nv_fma)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_maxmag_f32(float x, float y) {
  auto abs_x = __acpp_sscp_fabs_f32(x);
  auto abs_y = __acpp_sscp_fabs_f32(y);
  if(abs_x == abs_y) return __acpp_sscp_fmax_f32(x,y);
  return (abs_x > abs_y) ? x : y;
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_maxmag_f64(double x, double y) {
  auto abs_x = __acpp_sscp_fabs_f64(x);
  auto abs_y = __acpp_sscp_fabs_f64(y);
  if(abs_x == abs_y) return __acpp_sscp_fmax_f64(x,y);
  return (abs_x > abs_y) ? x : y;
}

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_minmag_f32(float x, float y) {
  auto abs_x = __acpp_sscp_fabs_f32(x);
  auto abs_y = __acpp_sscp_fabs_f32(y);
  if(abs_x == abs_y) return __acpp_sscp_fmin_f32(x,y);
  return (abs_x < abs_y) ? x : y;
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_minmag_f64(double x, double y) {
  auto abs_x = __acpp_sscp_fabs_f64(x);
  auto abs_y = __acpp_sscp_fabs_f64(y);
  if(abs_x == abs_y) return __acpp_sscp_fmin_f64(x,y);
  return (abs_x < abs_y) ? x : y;
}

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_modf_f32(float x, float* y ) { return __nv_modff(x, y); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_modf_f64(double x, double* y) { return __nv_modf(x, y); }

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(nextafter, __nv_nextafterf, __nv_nextafter)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(powr, __nv_powf, __nv_pow)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pown_f32(float x, __acpp_int32 y) {
  return __nv_powif(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pown_f64(double x,
                                                    __acpp_int32 y) {
  return __nv_powi(x, y);
}

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN2(remainder, __nv_remainderf, __nv_remainder)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(rint, __nv_rintf, __nv_rint)

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rootn_f32(float x, __acpp_int32 y) { return __acpp_sscp_pow_f32(x, 1.f/y); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rootn_f64(double x, __acpp_int64 y) {return __acpp_sscp_pow_f64(x, 1./y); }

HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(round, __nv_roundf, __nv_round)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(rsqrt, __nv_rsqrtf, __nv_rsqrt)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(sqrt, __nv_sqrtf, __nv_sqrt)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(sin, __nv_sinf, __nv_sin)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(sinh, __nv_sinhf, __nv_sinh)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(sinpi, __nv_sinpif, __nv_sinpi)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(tan, __nv_tanf, __nv_tan)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(tanh, __nv_tanhf, __nv_tanh)
HIPSYCL_SSCP_MAP_PTX_FLOAT_BUILTIN(trunc, __nv_truncf, __nv_trunc)
