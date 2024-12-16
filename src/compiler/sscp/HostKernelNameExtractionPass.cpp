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
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/NameHandling.hpp"
#include "hipSYCL/compiler/sscp/HostKernelNameExtractionPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>

namespace hipsycl {
namespace compiler {

namespace {

static const char* SSCPExtractKernelNameIdentifier = "__acpp_sscp_extract_kernel_name";
static const char* SSCPExtractKernelNameIdentifierAnnotation = "acpp_sscp_extract_kernel_name";
}

llvm::PreservedAnalyses HostKernelNameExtractionPass::run(llvm::Module &M,
                                                          llvm::ModuleAnalysisManager &MAM) {


  llvm::SmallVector<llvm::Function*> SSCPKernelNameExtractionFunctions;
  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F) {
      if(Annotation.compare(SSCPExtractKernelNameIdentifierAnnotation) == 0){
        SSCPKernelNameExtractionFunctions.push_back(F);
      }
    }
  });
  for(auto* F : SSCPKernelNameExtractionFunctions){
    for(auto U : F->users()) {
      if(auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
        if(CI->arg_size() == 2) {
          // Arg 0 should be function pointer to the kernel entry point.
          std::string KernelName;
          if (llvm::Function *KernelFunc = llvm::dyn_cast<llvm::Function>(CI->getOperand(0))) {
            KernelName = KernelFunc->getName();
#ifdef _WIN32
            replaceInvalidCharsInSymbolName(KernelName);
#endif
          } else {
            HIPSYCL_DEBUG_WARNING << "HostKernelNameExtractionPass: Could not find kernel name "
                                      "for __acpp_sscp_extract_kernel_name invocation: "
                                  << F->getName() << "\n";
          }
          
          // Arg 1 is the access to the global __acpp_sscp_kernel_name
          // variable. This might result in a ConstantExpr to do a getelementptr
          // Instruction.
          // We need to extract the global variable that is accessed, so that
          // we can access it.
          llvm::GlobalVariable* GV = nullptr;
          if(llvm::ConstantExpr* CE = llvm::dyn_cast<llvm::ConstantExpr>(CI->getArgOperand(1))) {
            for(auto O = CE->op_begin(); O != CE->op_end(); ++O) {
              if(llvm::GlobalVariable* G = llvm::dyn_cast<llvm::GlobalVariable>(O)) {
                GV = G;
              }
            }
          } else if (llvm::GlobalVariable *G =
                          llvm::dyn_cast<llvm::GlobalVariable>(CI->getArgOperand(1))) {
            GV = G;
          }

          if(!GV) {
            HIPSYCL_DEBUG_WARNING
                << "HostKernelNameExtractionPass: Could not find target global variable "
                    "for __acpp_sscp_extract_kernel_name invocation: "
                << F->getName() << "\n";
          } else {
            
            IRConstant IRC{M, *GV};
            // If there are multiple calls (this can happen during e.g. stdpar malloc2usm callgraph
            // duplication, don't set again if we have already been set)
            if(!IRC.isInitialized()) {
              HIPSYCL_DEBUG_INFO << "HostKernelNameExtractionPass: Exposing kernel name "
                                << KernelName << " in global symbol " << GV->getName() << "\n";
            
              // Now set GV to the kernel name
              IRC.set<std::string>(KernelName + '\0');
            }
          }
        
        } else {
          HIPSYCL_DEBUG_WARNING
              << "HostKernelNameExtractionPass: found __acpp_sscp_extract_kernel_name "
                  "invocation, but incorrect number of arguments ("
              << CI->arg_size() << ", should be 2)\n";
        }
      }
    }
  }

  for(llvm::Function* F : SSCPKernelNameExtractionFunctions) {
    // These functions are no longer needed
    llvm::SmallVector<llvm::CallBase*> KernelNameExtractionCalls;
    for(auto U : F->users()) {
      if(auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
        KernelNameExtractionCalls.push_back(CI);
      } else if (!llvm::isa<llvm::ConstantStruct>(U)) {
        HIPSYCL_DEBUG_WARNING
            << "HostKernelNameExtractionPass: found user of __acpp_sscp_extract_kernel_name() "
               "that is not a function call\n";
      }
    }

    for(auto Call : KernelNameExtractionCalls) {
      Call->replaceAllUsesWith(llvm::UndefValue::get(Call->getType()));
      Call->eraseFromParent();
    }

    F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
    F->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}

}
}
