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
#include "hipSYCL/compiler/cbs/LoopSplitterInlining.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace {
using namespace hipsycl::compiler::cbs;

bool inlineCallsInBasicBlock(llvm::BasicBlock &BB,
                             const llvm::SmallPtrSet<llvm::Function *, 8> &SplitterCallers,
                             hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  bool Changed = false;
  bool LastChanged;

  do {
    LastChanged = false;
    for (auto &I : BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction()) {
          if (SplitterCallers.find(CallI->getCalledFunction()) != SplitterCallers.end() &&
              !SAA.isSplitterFunc(CallI->getCalledFunction())) {
            LastChanged =
                hipsycl::compiler::utils::checkedInlineFunction(CallI, "[LoopSplitterInlining]");
            if (LastChanged)
              break;
          } else if (SAA.isSplitterFunc(CallI->getCalledFunction()) &&
                     CallI->getCalledFunction()->getName() != BarrierIntrinsicName) {
            HIPSYCL_DEBUG_INFO << "[LoopSplitterInlining] Replace barrier with intrinsic: "
                               << CallI->getCalledFunction()->getName() << "\n";
            hipsycl::compiler::utils::createBarrier(CallI, SAA);
            CallI->eraseFromParent();
            LastChanged = true;
            break;
          }
        }
      }
    }
    if (LastChanged)
      Changed = true;
  } while (LastChanged);

  return Changed;
}

//! \pre all contained functions are non recursive!
// todo: have a recursive-ness termination
bool inlineCallsInFunction(llvm::Function &F,
                           const llvm::SmallPtrSet<llvm::Function *, 8> &SplitterCallers,
                           hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  bool Changed = false;
  bool LastChanged;

  do {
    LastChanged = false;
    for (auto &BB : F) {
      LastChanged = inlineCallsInBasicBlock(BB, SplitterCallers, SAA);
      Changed |= LastChanged;
      if (LastChanged)
        break;
    }
  } while (LastChanged);

  return Changed;
}

// todo: have a recursive-ness termination
bool fillTransitiveSplitterCallers(llvm::Function &F,
                                   const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter,
                                   bool InIntrinsic = false);
bool fillTransitiveSplitterCallers(llvm::ArrayRef<llvm::BasicBlock *> Blocks,
                                   const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter,
                                   bool InIntrinsic) {
  bool Found = false;
  for (auto *BB : Blocks) {
    for (auto &I : *BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction() &&
            fillTransitiveSplitterCallers(*CallI->getCalledFunction(), SAA, FuncsWSplitter, InIntrinsic))
          Found = true;
      }
    }
  }
  return Found;
}

//! \pre \a F is not recursive!
// todo: have a recursive-ness termination
bool fillTransitiveSplitterCallers(llvm::Function &F,
                                   const hipsycl::compiler::SplitterAnnotationInfo &SAA,
                                   llvm::SmallPtrSet<llvm::Function *, 8> &FuncsWSplitter,
                                   bool InIntrinsic) {
  if (SAA.isSplitterFunc(&F)) {
    FuncsWSplitter.insert(&F);
    return true;
  } else if (FuncsWSplitter.find(&F) != FuncsWSplitter.end())
    return true;

  if (F.isDeclaration() && !F.isIntrinsic() && !InIntrinsic) {
    HIPSYCL_DEBUG_WARNING << "[LoopSplitterInlining] " << F.getName() << " is not defined!\n";
  }

  llvm::SmallVector<llvm::BasicBlock *, 8> Blocks;
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });

  if (fillTransitiveSplitterCallers(Blocks, SAA, FuncsWSplitter,
                                    InIntrinsic ||
				    hipsycl::llvmutils::starts_with(F.getName(), "__acpp_sscp"))) {
    FuncsWSplitter.insert(&F);
    return true;
  }
  return false;
}

bool inlineSplitter(llvm::Function &F, hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  bool Changed = false;

  llvm::SmallPtrSet<llvm::Function *, 8> SplitterCallers;
  if (!fillTransitiveSplitterCallers(F, SAA, SplitterCallers)) {
    HIPSYCL_DEBUG_INFO << "[LoopSplitterInlining] transitively no splitter found in kernel."
                       << F.getName() << "\n";
    return Changed;
  }
  Changed |= inlineCallsInFunction(F, SplitterCallers, SAA);

  return Changed;
}
} // namespace

void hipsycl::compiler::LoopSplitterInliningPassLegacy::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool hipsycl::compiler::LoopSplitterInliningPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F))
    return false;

  return inlineSplitter(F, SAA);
}

char hipsycl::compiler::LoopSplitterInliningPassLegacy::ID = 0;
llvm::PreservedAnalyses
hipsycl::compiler::LoopSplitterInliningPass::run(llvm::Function &F,
                                                 llvm::FunctionAnalysisManager &AM) {
  const auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAMProxy.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA) {
    llvm::errs() << "[LoopSplitterInlining] SplitterAnnotationAnalysis not cached.\n";
    return llvm::PreservedAnalyses::all();
  }
  if (!SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  if (!inlineSplitter(F, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
