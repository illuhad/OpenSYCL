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
#ifndef HIPSYCL_LLVMUTILS_HPP
#define HIPSYCL_LLVMUTILS_HPP

#include <llvm/ADT/StringRef.h>
#if LLVM_VERSION_MAJOR < 16
#define IS_OPAQUE(pointer) (pointer->isOpaquePointerTy())
#define HAS_TYPED_PTR 1
#else
#define IS_OPAQUE(pointer) constexpr(true || pointer) /* Use `pointer` to silence warnings */
#define HAS_TYPED_PTR 0
#endif

namespace hipsycl::llvmutils {

  inline bool starts_with(llvm::StringRef String, llvm::StringRef Prefix) {
#if LLVM_VERSION_MAJOR < 18
    return String.startswith(Prefix);
#else
    return String.starts_with(Prefix);
#endif
  }

  inline bool ends_with(llvm::StringRef String, llvm::StringRef Prefix) {
#if LLVM_VERSION_MAJOR < 18
    return String.endswith(Prefix);
#else
    return String.ends_with(Prefix);
#endif
  }
}// namespace hipsycl::llvmutils

#endif // HIPSYCL_LLVMUTILS_HPP
