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
#include "hipSYCL/sycl/libkernel/sscp/builtins/native.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_cos_f32(float x) { return __nv_fast_cosf(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_cos_f64(double x) { return __nv_cos(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_divide_f32(float x, float y) { return __nv_fast_fdividef(x, y); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_divide_f64(double x, double y) { return x / y; }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp_f32(float x) { return __nv_fast_expf(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_exp_f64(double x) { return __acpp_sscp_exp_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp2_f32(float x) { return __acpp_sscp_exp2_f32(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_exp2_f64(double x) { return __acpp_sscp_exp2_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp10_f32(float x) { return __nv_fast_exp10f(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_exp10_f64(double x) { return __acpp_sscp_exp10_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log_f32(float x) { return __nv_fast_logf(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_log_f64(double x) { return __acpp_sscp_log_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log2_f32(float x) { return __nv_fast_log2f(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_log2_f64(double x) { return __acpp_sscp_log2_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log10_f32(float x) { return __nv_fast_log10f(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_log10_f64(double x) { return __acpp_sscp_log10_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_powr_f32(float x, float y) { return __nv_fast_powf(x, y); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_powr_f64(double x, double y) { return __acpp_sscp_powr_f64(x, y); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_recip_f32(float x) { return 1.f / x; }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_recip_f64(double x) { return 1. / x; }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_rsqrt_f32(float x) { return __acpp_sscp_rsqrt_f32(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_rsqrt_f64(double x) { return __acpp_sscp_rsqrt_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_sin_f32(float x) { return __nv_fast_sinf(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_sin_f64(double x) { return __acpp_sscp_sin_f64(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_sqrt_f32(float x) { return __nvvm_sqrt_rn_f(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_sqrt_f64(double x) { return __nvvm_sqrt_rn_d(x); }

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_tan_f32(float x) { return __nv_fast_tanf(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_tan_f64(double x) { return __acpp_sscp_tan_f64(x); }
