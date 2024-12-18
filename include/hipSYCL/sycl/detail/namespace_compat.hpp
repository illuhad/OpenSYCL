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

#ifndef ACPP_NAMESPACE_COMPAT
#define ACPP_NAMESPACE_COMPAT

#ifndef ACPP_NO_SHORT_NAMESPACE
namespace acpp {
  using namespace hipsycl;
}
#endif

namespace adaptivecpp {
  using namespace hipsycl;
}

#endif
