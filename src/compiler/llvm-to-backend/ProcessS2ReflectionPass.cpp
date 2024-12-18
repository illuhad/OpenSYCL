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

#include "hipSYCL/compiler/llvm-to-backend/ProcessS2ReflectionPass.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Constants.h>

#include <algorithm>
#include <cctype>

namespace hipsycl {
namespace compiler {

namespace {

void handleReflectionFunction(llvm::Module& M, llvm::Function& F, uint64_t Value) {
  F.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  auto *ReplacementValue = llvm::ConstantInt::get(
      M.getContext(), llvm::APInt{F.getReturnType()->getIntegerBitWidth(), Value});
  
  llvm::SmallVector<llvm::CallBase*> CallsToRemove;
  for(auto* U : F.users()) {
    if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U)){
      CB->replaceNonMetadataUsesWith(ReplacementValue);
      CallsToRemove.push_back(CB);
    }
  }
  for (auto *C : CallsToRemove) {
    C->replaceAllUsesWith(llvm::UndefValue::get(C->getType()));
    C->dropAllReferences();
    C->eraseFromParent();
  }
}

std::string getQueryName(llvm::StringRef FunctionName, const std::string& Prefix) {
  auto Pos = FunctionName.find(Prefix);
  if(Pos == std::string::npos)
    return {};

  return FunctionName.str().substr(Pos+Prefix.length());
}

}

ProcessS2ReflectionPass::ProcessS2ReflectionPass(
    const std::unordered_map<std::string, uint64_t> &Fields) {

  for(const auto& KV : Fields) {
    std::string CanonicalizedKey = KV.first;

    std::transform(CanonicalizedKey.begin(), CanonicalizedKey.end(), CanonicalizedKey.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for(auto& c : CanonicalizedKey)
      if(!std::isalnum(c) && c != '_')
        c='_';

    SupportedFields[CanonicalizedKey] = KV.second;
  }
}

llvm::PreservedAnalyses ProcessS2ReflectionPass::run(llvm::Module& M, llvm::ModuleAnalysisManager& MAM) {


  auto processReflectionCalls = [&](const std::string &QueryPrefix,
                                    const std::string &KnowsQueryPrefix) {
    for(auto& F : M) {
      // Note: The order of the if/else branch here assumes that
      // QueryPrefix is a substring of KnowsQueryPrefix!
      if(llvmutils::starts_with(F.getName(), KnowsQueryPrefix)) {
        auto QueryName = getQueryName(F.getName(), KnowsQueryPrefix);
        auto It = SupportedFields.find(QueryName);
        if(It != SupportedFields.end())
          handleReflectionFunction(M, F, 1);
        else
          handleReflectionFunction(M, F, 0);
      } else if(llvmutils::starts_with(F.getName(), QueryPrefix)) {
        auto QueryName = getQueryName(F.getName(), QueryPrefix);
        auto It = SupportedFields.find(QueryName);
        if(It != SupportedFields.end())
          handleReflectionFunction(M, F, It->second);
      } 
    }
  };

  processReflectionCalls("__acpp_sscp_jit_reflect_", "__acpp_sscp_jit_reflect_knows_");
  processReflectionCalls("__acpp_sscp_s2_reflect_", "__acpp_sscp_s2_reflect_knows_");

  return llvm::PreservedAnalyses::none();
}

}
}