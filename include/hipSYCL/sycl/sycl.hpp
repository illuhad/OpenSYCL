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
#ifndef HIPSYCL_SYCL_HPP
#define HIPSYCL_SYCL_HPP



#define SYCL_IMPLEMENTATION_HIPSYCL
#define SYCL_IMPLEMENTATION_ACPP

#ifdef CL_SYCL_LANGUAGE_VERSION
 #undef CL_SYCL_LANGUAGE_VERSION
#endif
#ifdef SYCL_LANGUAGE_VERSION
 #undef SYCL_LANGUAGE_VERSION
#endif

#define CL_SYCL_LANGUAGE_VERSION 202012L
#define SYCL_LANGUAGE_VERSION 202012L
#define SYCL_FEATURE_SET_FULL

#include "hipSYCL/glue/persistent_runtime.hpp"

#include "extensions.hpp"

#include "libkernel/backend.hpp"
#include "libkernel/bit_cast.hpp"
#include "libkernel/range.hpp"
#include "libkernel/id.hpp"
#include "libkernel/accessor.hpp"
#include "libkernel/nd_item.hpp"
#include "libkernel/multi_ptr.hpp"
#include "libkernel/group.hpp"
#include "libkernel/h_item.hpp"
#include "libkernel/sp_item.hpp"
#include "libkernel/sp_group.hpp"
#include "libkernel/sp_private_memory.hpp"
#include "libkernel/memory_environment.hpp"
#include "libkernel/private_memory.hpp"
#include "libkernel/local_memory.hpp"
#include "libkernel/half.hpp"
#include "libkernel/vec.hpp"
#include "libkernel/marray.hpp"
#include "libkernel/builtins.hpp"
#include "libkernel/atomic.hpp"
#include "libkernel/atomic_ref.hpp"
#include "libkernel/stream.hpp"
#include "libkernel/sub_group.hpp"
#include "libkernel/group_traits.hpp"
#include "libkernel/memory.hpp"
#include "libkernel/group_functions.hpp"
#include "libkernel/group_functions_alias.hpp"
#include "libkernel/functional.hpp"
#include "libkernel/reduction.hpp"

#include "version.hpp"
#include "types.hpp"
#include "exception.hpp"
#include "device_selector.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "queue.hpp"
#include "program.hpp"
#include "kernel.hpp"
#include "buffer.hpp"
#include "usm.hpp"
#include "backend.hpp"
#include "backend_interop.hpp"
#include "interop_handle.hpp"
#include "buffer_explicit_behavior.hpp"
#include "specialized.hpp"
#include "jit.hpp"

// Support SYCL_EXTERNAL for SSCP - we cannot have SYCL_EXTERNAL if accelerated CPU
// is active at the same time :(
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP && !defined(__ACPP_USE_ACCELERATED_CPU__)
  #define SYCL_EXTERNAL [[clang::annotate("hipsycl_sscp_outlining")]]
#endif
// Support SYCL_EXTERNAL for library-only host backend
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST && !defined(__ACPP_USE_ACCELERATED_CPU__) && !defined(SYCL_EXTERNAL)
  #define SYCL_EXTERNAL
#endif
// Support SYCL_EXTERNAL for nvc++
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA && defined(ACPP_LIBKERNEL_CUDA_NVCXX) && !defined(SYCL_EXTERNAL)
  #define SYCL_EXTERNAL
#endif
// TODO: Need to investigate to what extent we can support SYCL_EXTERNAL for cuda and hip multipass targets.

#endif

