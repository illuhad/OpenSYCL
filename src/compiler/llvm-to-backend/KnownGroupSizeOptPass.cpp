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
#include "hipSYCL/compiler/llvm-to-backend/KnownGroupSizeOptPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/GlobalSizesFitInI32OptPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>


namespace hipsycl {
namespace compiler {

namespace {


bool applyKnownGroupSize(llvm::Module &M, int KnownGroupSize,
                         llvm::StringRef GetGroupSizeBuiltinName,
                         llvm::StringRef GetLocalIdBuiltinName) {

  // First create replacement functions for GetGroupSizeBuiltinName
  // which directly return the known group size, and replace all
  // uses.
  if(auto* GetGroupSizeF = M.getFunction(GetGroupSizeBuiltinName)) {
    std::string NewFunctionName = std::string{GetGroupSizeBuiltinName}+"_known_size";

    auto *NewGetGroupSizeF = llvm::dyn_cast<llvm::Function>(
        M.getOrInsertFunction(NewFunctionName, GetGroupSizeF->getFunctionType(),
                              GetGroupSizeF->getAttributes())
            .getCallee());
    if(!NewGetGroupSizeF)
      return false;

    if(!NewGetGroupSizeF->hasFnAttribute(llvm::Attribute::AlwaysInline))
      NewGetGroupSizeF->addFnAttr(llvm::Attribute::AlwaysInline);

    llvm::BasicBlock *BB =
        llvm::BasicBlock::Create(M.getContext(), "", NewGetGroupSizeF);

    auto *ReturnedIntType = llvm::dyn_cast<llvm::IntegerType>(GetGroupSizeF->getReturnType());
    if(!ReturnedIntType)
      return false;

    llvm::Constant *ReturnedValue = llvm::ConstantInt::get(
        M.getContext(), llvm::APInt(ReturnedIntType->getBitWidth(), KnownGroupSize));

    llvm::ReturnInst::Create(M.getContext(), ReturnedValue, BB);

    GetGroupSizeF->replaceNonMetadataUsesWith(NewGetGroupSizeF);
  }

  // Insert __builtin_assume(0 <= local_id); __builtin_assume(local_id < group_size);
  // for every call to GetLocalIdBuiltinName
  if(!insertRangeAssumptionForBuiltinCalls(M, GetLocalIdBuiltinName, 0, KnownGroupSize))
    return false;

  return true;
}

}

KnownGroupSizeOptPass::KnownGroupSizeOptPass(int GroupSizeX, int GroupSizeY, int GroupSizeZ)
    : KnownGroupSizeX{GroupSizeX}, KnownGroupSizeY{GroupSizeY}, KnownGroupSizeZ{GroupSizeZ} {}


llvm::PreservedAnalyses KnownGroupSizeOptPass::run(llvm::Module &M,
                                                        llvm::ModuleAnalysisManager &MAM) {
  if (KnownGroupSizeX > 0) {
    applyKnownGroupSize(M, KnownGroupSizeX, "__acpp_sscp_get_local_size_x",
                        "__acpp_sscp_get_local_id_x");
  }

  if (KnownGroupSizeY > 0) {
    applyKnownGroupSize(M, KnownGroupSizeY, "__acpp_sscp_get_local_size_y",
                        "__acpp_sscp_get_local_id_y");
  }

  if (KnownGroupSizeZ > 0) {
    applyKnownGroupSize(M, KnownGroupSizeZ, "__acpp_sscp_get_local_size_z",
                        "__acpp_sscp_get_local_id_z");
  }

  return llvm::PreservedAnalyses::none();

}


}
}
