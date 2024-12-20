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
#include "hipSYCL/compiler/llvm-to-backend/DeadArgumentEliminationPass.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace hipsycl {
namespace compiler {

namespace {

void removeUnusedFunctionParameters(llvm::Function *F, llvm::Module &M,
                                    llvm::SmallVector<int> *RetainedParameterIndices,
                                    std::function<void(llvm::Function *, llvm::Function *)>
                                        *ReplacementFunctionAttributeTransfer) {

  std::string FunctionName = F->getName().str();
  F->setName(FunctionName+".old");
  llvm::SmallVector<int> OriginalParameterIndex;
  
  for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if(F->getArg(i)->getNumUses() > 0) {
      OriginalParameterIndex.push_back(i);
    }
  }

  llvm::SmallVector<llvm::Type*, 8> NewParams;
  for(auto Index : OriginalParameterIndex) {
    NewParams.push_back(F->getFunctionType()->getParamType(Index));
  }

  llvm::FunctionType *NewFType = llvm::FunctionType::get(F->getReturnType(), NewParams, false);
  if (auto *NewF = llvm::dyn_cast<llvm::Function>(
        M.getOrInsertFunction(FunctionName, NewFType).getCallee())) {
    
    for(auto Attr : F->getAttributes().getFnAttrs())
      NewF->addFnAttr(Attr);
    for(int i = 0; i < OriginalParameterIndex.size(); ++i){
      for (auto Attr : F->getAttributes().getParamAttrs(OriginalParameterIndex[i]))
        NewF->addParamAttr(i, Attr);
    }
    NewF->setLinkage(F->getLinkage());
    NewF->setCallingConv(F->getCallingConv());

    // Now create function call to old function and inline
    llvm::BasicBlock *BB =
        llvm::BasicBlock::Create(M.getContext(), "", NewF);
    
    llvm::SmallVector<llvm::Value*> CallArgs;
    
    for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
      int NewFArgIndex = -1;
      for(int j = 0; j < OriginalParameterIndex.size(); ++j) {
        if(OriginalParameterIndex[j] == i) {
          NewFArgIndex = j;
          break;
        }
      }
      if(NewFArgIndex == -1)
        CallArgs.push_back(llvm::UndefValue::get(F->getFunctionType()->getParamType(i)));
      else
        CallArgs.push_back(NewF->getArg(NewFArgIndex));
    }
    
    auto *Call = llvm::CallInst::Create(llvm::FunctionCallee(F), CallArgs,
                                        "", BB);
    if(F->getReturnType()->isVoidTy())
      llvm::ReturnInst::Create(M.getContext(), BB);
    else
      llvm::ReturnInst::Create(M.getContext(), Call, BB);

    F->setLinkage(llvm::GlobalValue::InternalLinkage);

    if(RetainedParameterIndices) {
      RetainedParameterIndices->clear();

      for(auto Index : OriginalParameterIndex)
        RetainedParameterIndices->push_back(Index);
    }

    if(ReplacementFunctionAttributeTransfer)
      (*ReplacementFunctionAttributeTransfer)(F, NewF);

    llvm::SmallVector<llvm::CallBase*> CallsToInline;
    for(auto* U : F->users())
      if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U))
        CallsToInline.push_back(CB);
    for(auto* CB : CallsToInline) {
      llvm::InlineFunctionInfo IFI;
      if(llvm::InlineFunction(*CB, IFI).isSuccess()) {
        F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
        F->dropAllReferences();
        F->eraseFromParent();
      }
    }
  }
}
}

DeadArgumentEliminationPass::DeadArgumentEliminationPass(
    llvm::Function *F, llvm::SmallVector<int> *RetainedArgs,
    std::function<void(llvm::Function *, llvm::Function *)>
        *ReplacementFunctionAttributeTransferHandler)
    : TargetFunction{F}, RetainedArguments{RetainedArgs},
      ReplacementFunctionAttributeTransfer{ReplacementFunctionAttributeTransferHandler} {}

llvm::PreservedAnalyses DeadArgumentEliminationPass::run(llvm::Module &M,
                                                        llvm::ModuleAnalysisManager &MAM) {
  
  if(TargetFunction)
    removeUnusedFunctionParameters(TargetFunction, M, RetainedArguments,
                                   ReplacementFunctionAttributeTransfer);
  return llvm::PreservedAnalyses::none();
}
}
}
