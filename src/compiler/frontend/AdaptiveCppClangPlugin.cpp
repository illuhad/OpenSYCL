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
#include "hipSYCL/common/config.hpp"

#include "hipSYCL/compiler/FrontendPlugin.hpp"
#include "hipSYCL/compiler/GlobalsPruningPass.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"

#include "clang/Frontend/FrontendPluginRegistry.h"

namespace hipsycl {
namespace compiler {

static clang::FrontendPluginRegistry::Add<hipsycl::compiler::FrontendASTAction>
    HipsyclFrontendPlugin{"hipsycl_frontend", "enable hipSYCL frontend action"};
}
}
