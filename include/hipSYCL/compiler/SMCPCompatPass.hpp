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
#ifndef ACPP_SMCP_COMPAT_PASS_HPP
#define ACPP_SMCP_COMPAT_PASS_HPP

#include <llvm/IR/PassManager.h>
#include "CompilationState.hpp"

namespace hipsycl {
namespace compiler {

class SMCPCompatPass : public llvm::PassInfoMixin<SMCPCompatPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

}
}

#endif

