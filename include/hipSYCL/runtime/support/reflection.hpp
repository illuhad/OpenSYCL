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
#ifndef HIPSYCL_SUPPORT_REFLECTION_HPP
#define HIPSYCL_SUPPORT_REFLECTION_HPP

#include <mutex>
#include <unordered_map>

namespace hipsycl::rt::support {

class symbol_information {
public:
  static symbol_information& get();

  void register_function_symbol(const void* address, const char* name) {
    std::lock_guard<std::mutex> lock{_mutex};
    _symbol_names[address] = name;
  }

  const char* resolve_symbol_name(const void* address) const {
    std::lock_guard<std::mutex> lock{_mutex};
    auto it = _symbol_names.find(address);
    if(it == _symbol_names.end())
      return nullptr;
    return it->second;
  }
private:
  symbol_information() = default;
  std::unordered_map<const void*, const char*> _symbol_names;
  mutable std::mutex _mutex;
};

}

#endif