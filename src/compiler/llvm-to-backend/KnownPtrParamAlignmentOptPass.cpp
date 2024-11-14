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

#include "hipSYCL/compiler/llvm-to-backend/KnownPtrParamAlignmentOptPass.hpp"
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

KnownPtrParamAlignmentOptPass::KnownPtrParamAlignmentOptPass(
    const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &KnownAlignments)
    : KnownPtrParamAlignments{KnownAlignments} {}

llvm::PreservedAnalyses KnownPtrParamAlignmentOptPass::run(llvm::Module &M,
                            llvm::ModuleAnalysisManager &MAM) {
  llvm::Function *AssumeFunc = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::assume);

  for(auto& Entry : KnownPtrParamAlignments) {
    if(auto* F = M.getFunction(Entry.first)) {
      int NumParams = F->getFunctionType()->getNumParams();

      if(!F->isDeclaration()) {
        for(auto& AlignmentInfo : Entry.second) {
          int ParamIndex = AlignmentInfo.first;
          if(ParamIndex < NumParams) {
            llvm::Value* PtrValue = F->getArg(ParamIndex);
            llvm::Constant *True = llvm::ConstantInt::get(M.getContext(), llvm::APInt(1, 1));
            llvm::OperandBundleDef AlignBundle{
                "align", std::vector<llvm::Value *>{
                             PtrValue, llvm::ConstantInt::get(
                                           M.getContext(), llvm::APInt(64, AlignmentInfo.second))}};

            llvm::Instruction *InsertionPoint = &(*F->getEntryBlock().getFirstInsertionPt());
            llvm::CallInst::Create(
                llvm::FunctionCallee{AssumeFunc}, llvm::ArrayRef<llvm::Value *>{True},
                llvm::ArrayRef<llvm::OperandBundleDef>{AlignBundle}, "", InsertionPoint);
          }
        }
      }
    }
  }

  return llvm::PreservedAnalyses::none(); 
}


}
}

