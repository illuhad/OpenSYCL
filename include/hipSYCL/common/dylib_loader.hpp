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
#ifndef HIPSYCL_DYLIB_LOADER_HPP
#define HIPSYCL_DYLIB_LOADER_HPP

#include <string>

#include "export.hpp"

#ifdef _WIN32
#define ACPP_SHARED_LIBRARY_EXTENSION "dll"
#else
#define ACPP_SHARED_LIBRARY_EXTENSION "so"
#endif

namespace hipsycl::common {
ACPP_COMMON_EXPORT void *load_library(const std::string &filename, std::string &message);
ACPP_COMMON_EXPORT void *get_symbol_from_library(void *handle, const std::string &symbolName, std::string &message);
ACPP_COMMON_EXPORT void close_library(void *handle, std::string &message);
} // hipsycl::common

#endif

