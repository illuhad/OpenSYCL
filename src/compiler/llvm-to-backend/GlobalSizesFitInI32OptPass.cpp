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
#include "hipSYCL/compiler/llvm-to-backend/GlobalSizesFitInI32OptPass.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {


// inserts llvm.assume calls to assert that x >= RangeMin && x < RangeMax.
bool insertRangeAssumptionForBuiltinCalls(llvm::Module &M, llvm::StringRef BuiltinName,
                                          long long RangeMin, long long RangeMax, bool MaxIsLessThanEqual) {
  llvm::Function *AssumeIntrinsic = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::assume);
  if(!AssumeIntrinsic)
    return false;

  if(auto* F = M.getFunction(BuiltinName)) {

    auto *ReturnedIntType = llvm::dyn_cast<llvm::IntegerType>(F->getReturnType());
    if(!ReturnedIntType)
      return false;

    for(auto* U : F->users()) {
      if(auto* C = llvm::dyn_cast<llvm::CallInst>(U)) {
        auto* NextInst = C->getNextNonDebugInstruction();

        auto *GreaterEqualMin = llvm::ICmpInst::Create(
            llvm::Instruction::OtherOps::ICmp, llvm::ICmpInst::Predicate::ICMP_SGE, C,
            llvm::ConstantInt::get(M.getContext(),
                                   llvm::APInt(ReturnedIntType->getBitWidth(), RangeMin)),
            "", NextInst);
        
        llvm::ICmpInst::Predicate MaxPredicate = llvm::ICmpInst::ICMP_SLT;
        if(MaxIsLessThanEqual)
          MaxPredicate = llvm::ICmpInst::ICMP_SLE;
        auto *LesserThanMax = llvm::ICmpInst::Create(
            llvm::Instruction::OtherOps::ICmp, MaxPredicate, C,
            llvm::ConstantInt::get(M.getContext(),
                                  llvm::APInt(ReturnedIntType->getBitWidth(), RangeMax)),
            "", NextInst);
        

        llvm::SmallVector<llvm::Value*> CallArgGreaterEqualMin{GreaterEqualMin};
        llvm::SmallVector<llvm::Value*> CallArgLesserThanMax{LesserThanMax};
        llvm::CallInst::Create(llvm::FunctionCallee(AssumeIntrinsic), CallArgGreaterEqualMin,
                                            "", NextInst);
        llvm::CallInst::Create(llvm::FunctionCallee(AssumeIntrinsic), CallArgLesserThanMax,
                                            "", NextInst);
      }
    }
  }

  return true;
}

GlobalSizesFitInI32OptPass::GlobalSizesFitInI32OptPass(bool FitsInInt, int GroupSizeX,
                                                       int GroupSizeY, int GroupSizeZ)
    : GlobalSizesFitInInt{FitsInInt}, KnownGroupSizeX{GroupSizeX}, KnownGroupSizeY{GroupSizeY},
      KnownGroupSizeZ{GroupSizeZ} {}

llvm::PreservedAnalyses GlobalSizesFitInI32OptPass::run(llvm::Module &M,
                                                        llvm::ModuleAnalysisManager &MAM) {
  static const char* IfFitsInIntBuiltinName = "__acpp_sscp_if_global_sizes_fit_in_int";
  if(auto* F = M.getFunction(IfFitsInIntBuiltinName)) {
    // Add definition
    if(F->size() == 0) {
      llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", F);
      llvm::ReturnInst::Create(
          M.getContext(),
          llvm::ConstantInt::get(
              M.getContext(),
              llvm::APInt(F->getReturnType()->getIntegerBitWidth(), GlobalSizesFitInInt ? 1 : 0)),
          BB);
    }
  }

  std::size_t MaxInt = std::numeric_limits<int>::max();
  // This needs to be called regardless of whether the GlobalSizesFitInInt optimization is
  // active.

  if(GlobalSizesFitInInt) {

    if (KnownGroupSizeX > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__acpp_sscp_get_num_groups_x", 0,
                                                MaxInt / KnownGroupSizeX, true);
    }
    if (KnownGroupSizeY > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__acpp_sscp_get_num_groups_y", 0,
                                                MaxInt / KnownGroupSizeY, true);
    }
    if (KnownGroupSizeZ > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__acpp_sscp_get_num_groups_z", 0,
                                                MaxInt / KnownGroupSizeZ, true);
    }


    if (KnownGroupSizeX > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__acpp_sscp_get_group_id_x", 0,
                                                MaxInt / KnownGroupSizeX);
    }
    if (KnownGroupSizeY > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__acpp_sscp_get_group_id_y", 0,
                                                MaxInt / KnownGroupSizeY);
    }
    if (KnownGroupSizeZ > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__acpp_sscp_get_group_id_z", 0,
                                                MaxInt / KnownGroupSizeZ);
    }

  }

  return llvm::PreservedAnalyses::none();
}
}
}
