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
#ifndef HIPSYCL_IR_HPP
#define HIPSYCL_IR_HPP

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "CompilationState.hpp"

#include "hipSYCL/common/debug.hpp"

#include <unordered_set>
#include <vector>

namespace hipsycl {
namespace compiler {

struct GlobalsPruningPassLegacy : public llvm::ModulePass {
  static char ID;

  GlobalsPruningPassLegacy() : llvm::ModulePass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL globals pruning pass"; }

  bool runOnModule(llvm::Module &M) override;
};

#if !defined(_WIN32) || defined(ACPP_LLVM_COMPONENT)
class GlobalsPruningPass : public llvm::PassInfoMixin<GlobalsPruningPass> {
public:
  explicit GlobalsPruningPass() {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
#endif // !_WIN32

} // namespace compiler
} // namespace hipsycl

#endif
