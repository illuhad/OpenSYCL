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
#ifndef HIPSYCL_COMMON_EXPORT_HPP
#define HIPSYCL_COMMON_EXPORT_HPP

#ifndef _WIN32
#define ACPP_COMMON_EXPORT
#else
#define ACPP_COMMON_EXPORT __declspec(dllexport)
#endif

#endif
