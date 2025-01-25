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

#include "hipSYCL/compiler/SMCPCompatPass.hpp"

namespace hipsycl {
namespace compiler {

llvm::PreservedAnalyses SMCPCompatPass::run(llvm::Module& M, llvm::ModuleAnalysisManager& MAM) {
  if (!CompilationStateManager::getASTPassState().isDeviceCompilation())
    return llvm::PreservedAnalyses::all();
  
  // LLVM 18 does not yet support readlane.i32. However,
  // LLVM 18 based ROCm (e.g. ROCm 6.2) may already use this builtin.
#if LLVM_VERSION_MAJOR == 18 && !defined(ROCM_CLANG_VERSION)
  for(auto& F : M) {
    if(F.getName() == "llvm.amdgcn.readlane.i32")
      F.setName("llvm.amdgcn.readlane");
  }
#endif

  return llvm::PreservedAnalyses::none();
}

}
}

