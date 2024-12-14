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
#include <string_view>

#ifndef _WIN32
#define HIPSYCL_PLUGIN_API_EXPORT extern "C"
#else
#define HIPSYCL_PLUGIN_API_EXPORT extern "C" __declspec(dllexport)
#endif

#ifdef _WIN32
#define HIPSYCL_SHARED_LIBRARY_EXTENSION "dll"
#else
#define HIPSYCL_SHARED_LIBRARY_EXTENSION "so"
#endif

namespace hipsycl {
namespace rt {
namespace detail {
void *load_library(const std::string &filename, std::string_view loader);
void *get_symbol_from_library(void *handle, const std::string &symbolName, std::string_view loader);
void close_library(void *handle, std::string_view loader);

}
}
}

#endif

