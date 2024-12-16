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
#ifndef HIPSYCL_NAME_HANDLING_HPP
#define HIPSYCL_NAME_HANDLING_HPP

#include <string>

namespace hipsycl::compiler {

inline bool isValidCharInSymbolName(char C) {
  return std::isalnum(C) || C == '_' || C == '$' || C == '.';
}

inline void replaceInvalidCharsInSymbolName(std::string &Name) {
  for (auto &C : Name) {
    if (!isValidCharInSymbolName(C)) {
      C = '_';
    }
  }
}

} // namespace hipsycl::compiler
#endif
