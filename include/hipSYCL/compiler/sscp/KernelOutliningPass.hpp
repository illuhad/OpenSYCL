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
#ifndef HIPSYCL_KERNEL_OUTLINING_PASS_HPP
#define HIPSYCL_KERNEL_OUTLINING_PASS_HPP

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <vector>

namespace hipsycl {
namespace compiler {

class EntrypointPreparationPass : public llvm::PassInfoMixin<EntrypointPreparationPass> {
public:
  EntrypointPreparationPass(bool ExportByDefault = false);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  const std::vector<std::string>& getKernelNames() const {
    return KernelNames;
  }

  const std::vector<std::string>& getOutliningEntrypoints() const {
    return OutliningEntrypoints;
  }

  const std::vector<std::string>& getNonKernelOutliningEntrypoints() const {
    return NonKernelOutliningEntrypoints;
  }

private:
  std::vector<std::string> KernelNames;
  std::vector<std::string> OutliningEntrypoints;
  std::vector<std::string> NonKernelOutliningEntrypoints;
  bool ExportAll;
};

//  Removes all code not belonging to kernels
class KernelOutliningPass : public llvm::PassInfoMixin<KernelOutliningPass>{
public:
  KernelOutliningPass(const std::vector<std::string>& OutliningEntrypoints);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  std::vector<std::string> OutliningEntrypoints;
};

//  Removes all code not belonging to kernels
class KernelArgumentCanonicalizationPass
    : public llvm::PassInfoMixin<KernelArgumentCanonicalizationPass> {
public:
  KernelArgumentCanonicalizationPass(const std::vector<std::string>& KernelNames);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  std::vector<std::string> KernelNames;
};
}
}

#endif