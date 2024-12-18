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
#ifndef ACPP_GLUE_JIT_REFLECTION_MAP_HPP
#define ACPP_GLUE_JIT_REFLECTION_MAP_HPP

#include <unordered_map>
#include <string>
#include <cstdint>

#include "hipSYCL/runtime/hardware.hpp"

namespace hipsycl{
namespace glue {
namespace jit {

using reflection_map = std::unordered_map<std::string, uint64_t>;

inline reflection_map construct_default_reflection_map(rt::hardware_context* ctx) {
  reflection_map rmap;
  rmap["target_vendor_id"] = ctx->get_property(rt::device_uint_property::vendor_id);
  rmap["target_has_independent_forward_progress"] = static_cast<uint64_t>(ctx->has(
      rt::device_support_aspect::work_item_independent_forward_progress));
  rmap["target_arch"] = ctx->get_property(rt::device_uint_property::architecture);
  rmap["target_is_gpu"] = ctx->is_gpu() ? 1 : 0;
  rmap["target_is_cpu"] = ctx->is_cpu() ? 1 : 0;

  rmap["runtime_backend"] = ctx->get_property(rt::device_uint_property::backend_id);
  // compiler_backend is set by the LLVMToBackend infrastructure.
  return rmap;
}

}
}
}

#endif
