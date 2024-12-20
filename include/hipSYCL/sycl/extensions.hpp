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
#ifndef ACPP_EXTENSIONS_HPP
#define ACPP_EXTENSIONS_HPP

// legacy feature test macros for backwards compatibility

#ifdef HIPSYCL_EXT_ENABLE_ALL
 #define HIPSYCL_EXT_FP_ATOMICS
#endif

#define HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE
#define HIPSYCL_EXT_CUSTOM_PFWI_SYNCHRONIZATION
#define HIPSYCL_EXT_SCOPED_PARALLELISM_V2
#define HIPSYCL_EXT_ENQUEUE_CUSTOM_OPERATION
#define HIPSYCL_EXT_CG_PROPERTY_RETARGET
#define HIPSYCL_EXT_CG_PROPERTY_PREFER_GROUP_SIZE
#define HIPSYCL_EXT_CG_PROPERTY_PREFER_EXECUTION_LANE
#define HIPSYCL_EXT_BUFFER_USM_INTEROP
#define HIPSYCL_EXT_PREFETCH_HOST
#define HIPSYCL_EXT_SYNCHRONOUS_MEM_ADVISE
#define HIPSYCL_EXT_BUFFER_PAGE_SIZE
#define HIPSYCL_EXT_EXPLICIT_BUFFER_POLICIES
#define HIPSYCL_EXT_ACCESSOR_VARIANTS

#ifndef HIPSYCL_STRICT_ACCESSOR_DEDUCTION
 #define HIPSYCL_EXT_ACCESSOR_VARIANT_DEDUCTION
#endif

#define HIPSYCL_EXT_UPDATE_DEVICE
#define HIPSYCL_EXT_QUEUE_WAIT_LIST
#define HIPSYCL_EXT_MULTI_DEVICE_QUEUE
#define HIPSYCL_EXT_COARSE_GRAINED_EVENTS
#define HIPSYCL_EXT_QUEUE_PRIORITY
#define HIPSYCL_EXT_SPECIALIZED
#define HIPSYCL_EXT_DYNAMIC_FUNCTIONS

// current feature test macros

#if defined(ACPP_EXT_ENABLE_ALL) || defined(HIPSYCL_EXT_ENABLE_ALL)
 #define ACPP_EXT_FP_ATOMICS
#endif

#define ACPP_EXT_AUTO_PLACEHOLDER_REQUIRE
#define ACPP_EXT_CUSTOM_PFWI_SYNCHRONIZATION
#define ACPP_EXT_SCOPED_PARALLELISM_V2
#define ACPP_EXT_ENQUEUE_CUSTOM_OPERATION
#define ACPP_EXT_CG_PROPERTY_RETARGET
#define ACPP_EXT_CG_PROPERTY_PREFER_GROUP_SIZE
#define ACPP_EXT_CG_PROPERTY_PREFER_EXECUTION_LANE
#define ACPP_EXT_BUFFER_USM_INTEROP
#define ACPP_EXT_PREFETCH_HOST
#define ACPP_EXT_SYNCHRONOUS_MEM_ADVISE
#define ACPP_EXT_BUFFER_PAGE_SIZE
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#define ACPP_EXT_ACCESSOR_VARIANTS

#if !defined(ACPP_STRICT_ACCESSOR_DEDUCTION) && !defined(HIPSYCL_STRICT_ACCESSOR_DEDUCTION)
 #define ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION
#endif

#define ACPP_EXT_UPDATE_DEVICE
#define ACPP_EXT_QUEUE_WAIT_LIST
#define ACPP_EXT_MULTI_DEVICE_QUEUE
#define ACPP_EXT_COARSE_GRAINED_EVENTS
#define ACPP_EXT_QUEUE_PRIORITY
#define ACPP_EXT_SPECIALIZED
#define ACPP_EXT_DYNAMIC_FUNCTIONS
#define ACPP_EXT_RESTRICT_PTR
#define ACPP_EXT_JIT_COMPILE_IF

// KHR extensions

#define SYCL_KHR_DEFAULT_CONTEXT 1

#endif
