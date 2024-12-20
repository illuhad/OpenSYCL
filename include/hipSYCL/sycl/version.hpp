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
#ifndef HIPSYCL_VERSION_HPP
#define HIPSYCL_VERSION_HPP

#include <string>

#include "hipSYCL/common/config.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

static std::string version_string()
{
  std::string zero = (ACPP_VERSION_MINOR < 10 ? "0" : "");
  std::string version = std::to_string(ACPP_VERSION_MAJOR)
      + "." + zero + std::to_string(ACPP_VERSION_MINOR)
      + "." + std::to_string(ACPP_VERSION_PATCH)
      + std::string(ACPP_VERSION_SUFFIX);

  return "AdaptiveCpp " + version;
}

}
}
}

#endif
