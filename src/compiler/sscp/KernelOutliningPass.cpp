
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include <cstddef>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Constants.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/IR/Comdat.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace hipsycl {
namespace compiler {

namespace {

bool isUsedInFunctions(llvm::SmallPtrSet<llvm::User*, 16>& VisitedUsers, llvm::User* User) {
  if(llvm::isa<llvm::Function>(User))
      return true;
  if(llvm::Instruction* I = llvm::dyn_cast<llvm::Instruction>(User)){
    if(I->getFunction())
      return true;
  }

  if(VisitedUsers.contains(User))
    return false;
  VisitedUsers.insert(User);

  for(auto* U : User->users()) {
    if(isUsedInFunctions(VisitedUsers, U))
      return true;
  }
  return false;
}

template<class FunctionSetT>
void descendCallGraphAndAdd(llvm::Function* F, llvm::CallGraph& CG, FunctionSetT& Set){
  if(!F || Set.contains(F))
    return;

  Set.insert(F);
  llvm::CallGraphNode* CGN = CG.getOrInsertFunction(F);
  if(!CGN)
    return;
  for(unsigned i = 0; i < CGN->size(); ++i){
    descendCallGraphAndAdd((*CGN)[i]->getFunction(), CG, Set);
  }
}

// Check whether F is used by an instruction from any function contained in
// a set S
template<class Set>
bool isCalledFromAnyFunctionOfSet(llvm::Function* F, const Set& S) {
  for(auto* U : F->users()) {
    if(auto* I = llvm::dyn_cast<llvm::Instruction>(U)) {
      auto* UsingFunc = I->getFunction();
      if(UsingFunc && S.contains(UsingFunc)) {
        return true;
      }
    }
  }
  return false;
}

bool isKernelArgumentStruct(llvm::Type* T) {
  if(!T)
    return false;
  if(!T->isAggregateType())
    return false;
  return T->getStructName().contains("::__sscp_dispatch::");
}

// Attempts to determine the pointee type of a pointer function argument of a kernel
// by investigating users, and looking for instructions that provide
// this information, such as getelementptr.
// This is particularly necessary due to LLVM's move to opaque pointers,
// where pointer types are no longer associated with the pointee type.
class FunctionArgPointeeTypeInfererence {
public:
  llvm::Type* run(llvm::Function* F, int ArgNo) {
    VisitedUsers.clear();
    if(llvm::Value* Arg = F->getArg(ArgNo)) {
      if(llvm::dyn_cast<llvm::PointerType>(Arg->getType())) {

        // If either byval or byref attributes are present, we can just look up
        // the pointee type directly.
        if(F->hasParamAttribute(ArgNo, llvm::Attribute::ByVal))
          return F->getParamAttribute(ArgNo, llvm::Attribute::ByVal).getValueAsType();

        if(F->hasParamAttribute(ArgNo, llvm::Attribute::ByRef))
          return F->getParamAttribute(ArgNo, llvm::Attribute::ByRef).getValueAsType();

        // Otherwise, we need to investigate uses of the argument to check
        // for clues regarding the pointee type.
        llvm::SmallDenseMap<llvm::Type *, int> Scores;
        scanAllocas(F, Scores);
        rankUsers(F->getArg(ArgNo), Scores);

        llvm::Type* BestTy = nullptr;
        int BestScore = std::numeric_limits<int>::min();
        for(auto S : Scores) {
          if(S.second > BestScore) {
            BestTy = S.first;
            BestScore = S.second;
          }
        }
        return BestTy;
      } else {
        return Arg->getType();
      }
    }
    return nullptr;
  }

private:
  void scanAllocas(llvm::Function *F, llvm::SmallDenseMap<llvm::Type *, int> &Scores) {
    for(auto& BB : *F) {
      for(auto& I : BB) {
        if(auto* AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
          if(isKernelArgumentStruct(AI->getAllocatedType())) {
            Scores[AI->getAllocatedType()] = 0;
          }
        }
      }
    }
  }
  void rankUsers(llvm::Value *Parent, llvm::SmallDenseMap<llvm::Type *, int> &Scores,
                 int CurrentScore = 0) {
    if(VisitedUsers.contains(Parent)) {
      return;
    }
    VisitedUsers.insert(Parent);

    if(!Parent)
      return;
    for(auto* Current : Parent->users()) {
      if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Current)) {
        Scores[LI->getType()] = CurrentScore - 2;
      } else if (auto GEPI = llvm::dyn_cast<llvm::GetElementPtrInst>(Current)) {
        Scores[GEPI->getSourceElementType()] = CurrentScore - 1;
      } else if (auto EEI = llvm::dyn_cast<llvm::ExtractElementInst>(Current)) {
        Scores[EEI->getVectorOperand()->getType()] = CurrentScore - 2;
      } else if (llvm::dyn_cast<llvm::AddrSpaceCastInst>(Current)) {
        // Follow address space casts, we don't care about pointer address spaces
        rankUsers(Current, Scores, CurrentScore);
      } else if(auto CI = llvm::dyn_cast<llvm::CallBase>(Current)) {
        // Ugh, the value is forwarded as an argument into some other function, need
        // to continue looking there...

        // First, check if we have any interesting allocas in the called function
        scanAllocas(CI->getCalledFunction(), Scores);

        // Next, follow the argument that was passed in there
        for (int i = 0; i < CI->getCalledFunction()->getFunctionType()->getNumParams(); ++i) {
          if(CI->getArgOperand(i) == Parent) {
            auto Arg = CI->getCalledFunction()->getArg(i);
            // Never, ever take into account the callee argument. This should never happen,
            // but if it does, it will go terribly because we will take into account users of functions,
            // not arguments anymore.
            if(!llvm::isa<llvm::Function>(Arg))
              rankUsers(Arg, Scores, CurrentScore);
          }
        }
      }
    }
  }

  llvm::SmallSet<llvm::Value*, 32> VisitedUsers;
};

// In principle, we expect our kernels to have one argument: A struct containing
// the user lambda that has the ByVal LLVM attribute.
// In practice, there might be deviations from this because, as a single-pass compiler,
// the code we are given is affected by the host ABI and ABI-specific argument passing tricks.
// For example, on x86 if the struct is smaller than 16 bytes, clang will already have expanded it.
// Another issue is that when the struct is non-trivially copyable (which can happen if
// it captures e.g. std::tuple) clang will not emit the ByVal attribute. This is particularly
// problematic in combination with opaque pointers, where we might not know the pointee type
// otherwise.
// This function attempts to fix those issues. In particular, if there is a struct argument,
// it tries to canonicalize it such that it has the ByVal attribute. This is expected
// by the AggregateArgumentExpansionPass later on.
void canonicalizeKernelParameters(llvm::Function* F, llvm::Module& M) {
  // If we have a different number of parameters than 1, we can assume
  // that clang has pre-expanded the struct for us to raw primitive types or data pointers. In that
  // case, there is nothing to do because those types can be used directly inside kernels.
  if(F->getFunctionType()->getNumParams() == 1) {
    auto* Type = F->getArg(0)->getType();

    // If it is not pointer type, we are dealing with a value that was preexpanded
    // by clang, e.g. if the struct just captures a single constant (such kernels, while
    // not really useful, are still allowed).
    if(Type->isPointerTy()) {
      // We should not have ByRef attribute, as this does not make
      // sense for kernel parameters. But maybe the kernel stub in
      // the kernel launcher was changed to const &. Just change to ByVal and be done.
      if(F->hasParamAttribute(0, llvm::Attribute::ByRef)) {
        auto *PointeeType = F->getParamAttribute(0, llvm::Attribute::ByRef).getValueAsType();
        F->removeParamAttr(0, llvm::Attribute::ByRef);
        F->addParamAttr(0, llvm::Attribute::getWithByValType(M.getContext(), PointeeType));
      } else if(!F->hasParamAttribute(0, llvm::Attribute::ByVal)) {
        // Now it gets interesting: We have a single pointer and no ByVal.
        // Following explanations are possible:
        // 1.) We are dealing with a struct, but ByVal is missing, e.g.
        //     because the struct is not trivially copyable, and clang
        //     therefore has not emitted ByVal due to ABI.
        // 2.) The user has captured just a single USM pointer
        // 3.) clang has pre-expanded the struct but there was only a single pointer inside.

        // We need to proceed by obtaining more information about the pointee type.
        FunctionArgPointeeTypeInfererence ArgPTI;
        llvm::Type* PointeeType = ArgPTI.run(F, 0);
        if(!PointeeType) {
          // We could not infer pointee type - this happens e.g.
          // when the parameter is not used inside the kernel.
          // So chances are, we can just ignore this case
          // as a kernel that executes without side effects
          // also does not care about its parameter conventions.
          HIPSYCL_DEBUG_INFO
              << "canonicalizeKernelParameters: Could not infer argument pointee type of kernel "
              << F->getName() << "\n";
        } else {
          // If we are not dealing with an aggregate type, we can just treat
          // the pointer as e.g. a regular USM pointer.
          if(PointeeType->isAggregateType()) {
            // It could also be a USM pointer to an aggregate, so check
            // against type name for our dedicated kernel types
            if(isKernelArgumentStruct(PointeeType)) {
              HIPSYCL_DEBUG_INFO
                  << "canonicalizeKernelParameters: Attaching ByVal to argument of kernel "
                  << F->getName() << "\n";
              F->addParamAttr(0, llvm::Attribute::getWithByValType(M.getContext(), PointeeType));
            }
          }
        }
      } /* else {} We have ByVal attribute, so all is well. */
    }
  }
}

}

EntrypointPreparationPass::EntrypointPreparationPass(bool ExportByDefault)
: ExportAll{ExportByDefault} {}

llvm::PreservedAnalyses
EntrypointPreparationPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static constexpr const char* SSCPKernelMarker = "hipsycl_sscp_kernel";
  static constexpr const char* SSCPOutliningMarker = "hipsycl_sscp_outlining";

  llvm::SmallSet<std::string, 16> Kernels;


  llvm::DenseSet<llvm::Function*> MarkedFunctions;
  auto MarkThisFunctionForOutlining = [&](llvm::Function* F) {
    HIPSYCL_DEBUG_INFO << "Found SSCP outlining entrypoint: " << F->getName() << "\n";
    // Make kernel have external linkage to avoid having everything optimized away
    F->setLinkage(llvm::GlobalValue::ExternalLinkage);

    // If we have a definition, we need to perform outlining.
    // Otherwise, we would need to treat the function as imported --
    // however this cannot really happen as clang does not codegen our
    // attribute((annotate("hipsycl_sscp_outlining"))) for declarations
    // without definition.
    if(F->size() > 0 && !MarkedFunctions.contains(F)) {
      this->OutliningEntrypoints.push_back(F->getName().str());
      MarkedFunctions.insert(F);
    }
  };

  utils::findFunctionsWithStringAnnotationsWithArg(M, [&](llvm::Function* F, llvm::StringRef Annotation, llvm::Constant* Argument){
    if(F) {
      if(Annotation.compare(SscpKernelDimensionName) == 0){
        HIPSYCL_DEBUG_INFO << "Found kernel dim annotation: " << F->getName() << " with arg: " << * Argument << "\n";
        // annotate the actual kernel with the dimension
        for (auto &U : F->uses()) {
          if (auto *CI = llvm::dyn_cast<llvm::CallInst>(U.getUser())) {
            if (CI->getCalledFunction() == F) {

              auto DimVal = llvm::cast<llvm::Constant>(Argument->getOperand(0));
              if (DimVal->getNumOperands() > 0) {
                DimVal = llvm::cast<llvm::Constant>(DimVal->getOperand(0));
              }

              llvm::SmallVector<llvm::Metadata *, 4> Operands;
              Operands.push_back(llvm::ValueAsMetadata::get(CI->getFunction()));
              Operands.push_back(llvm::MDString::get(M.getContext(), SscpKernelDimensionName));
              Operands.push_back(llvm::ValueAsMetadata::getConstant(DimVal));

              M.getOrInsertNamedMetadata(SscpAnnotationsName)
                  ->addOperand(llvm::MDTuple::get(M.getContext(), Operands));
            }
          }
        }
      }
      if(Annotation.compare(SSCPKernelMarker) == 0) {
        HIPSYCL_DEBUG_INFO << "Found SSCP kernel: " << F->getName() << "\n";
        this->KernelNames.push_back(F->getName().str());
        Kernels.insert(F->getName().str());
      }

      if(Annotation.compare(SSCPOutliningMarker) == 0) {
        MarkThisFunctionForOutlining(F);
      }
    }
  });

  if(ExportAll) {
    for(auto& F: M) {
      if (!F.isIntrinsic() && F.getLinkage() != llvm::GlobalValue::LinkageTypes::InternalLinkage)
        MarkThisFunctionForOutlining(&F);
    }
  }


  for(const auto& EP : OutliningEntrypoints) {
    if(!Kernels.contains(EP)) {
      NonKernelOutliningEntrypoints.push_back(EP);
    }
  }

  return llvm::PreservedAnalyses::none();
}

KernelOutliningPass::KernelOutliningPass(const std::vector<std::string>& OutliningEPs)
: OutliningEntrypoints{OutliningEPs} {}

llvm::PreservedAnalyses
KernelOutliningPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  // Some backends (e.g. PTX) don't like aliases. We need to replace
  // them early on, because it can get difficult to handle them once
  // we have removed what their aliasees.
  llvm::SmallVector<llvm::GlobalAlias*, 16> AliasesToRemove;
  for(auto& A : M.aliases())
    AliasesToRemove.push_back(&A);
  // Need separate iteration, so that we don't erase stuff from the list
  // we are iterating over.
  for(auto* A : AliasesToRemove) {
    if(A) {
      if(A->getAliasee())
        A->replaceAllUsesWith(A->getAliasee());
      A->eraseFromParent();
    }
  }

  llvm::SmallPtrSet<llvm::Function*, 16> SSCPEntrypoints;
  for(const auto& EntrypointName : OutliningEntrypoints) {
    llvm::Function* F = M.getFunction(EntrypointName);

    if(F) {
      SSCPEntrypoints.insert(F);
    }
  }
  llvm::SmallPtrSet<llvm::Function*, 16> DeviceFunctions;

  llvm::CallGraph CG{M};
  for(auto F: SSCPEntrypoints)
    descendCallGraphAndAdd(F, CG, DeviceFunctions);

  // for(auto* F : DeviceFunctions) {
    //HIPSYCL_DEBUG_INFO << "SSCP Kernel outlining: Function is device function: "
    //                   << F->getName().str() << "\n";
  // }

  llvm::SmallVector<llvm::Function*, 16> PureHostFunctions;
  for(auto& F: M) {
    // Called Intrinsics don't show up in our device functions list,
    // so we need to treat them specially
    if(F.isIntrinsic()) {
      if(!isCalledFromAnyFunctionOfSet(&F, DeviceFunctions)) {
        PureHostFunctions.push_back(&F);
      }
    } else if(!DeviceFunctions.contains(&F)) {
      PureHostFunctions.push_back(&F);
    }
  }

  for(auto F : PureHostFunctions) {
    if(F) {
      bool SafeToRemove = !isCalledFromAnyFunctionOfSet(F, DeviceFunctions);
      if(!SafeToRemove) {
        HIPSYCL_DEBUG_WARNING << "KernelOutliningPass: Attempted to remove " << F->getName()
                              << ", but it is still used by functions marked as device functions.\n";
      }
      // Better safe than sorry!
      if(SafeToRemove) {
        F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
        F->eraseFromParent();
      }
    }
  }

  llvm::SmallVector<llvm::GlobalVariable*, 16> UnneededGlobals;
  for(auto& G: M.globals()) {
    G.removeDeadConstantUsers();
    // Throw away globals that are either totally unused or don't have any
    // use inside functions (either directly or indirectly). The latter is
    // to remove globals with circular dependencies, which can cause JIT failures.
    llvm::SmallPtrSet<llvm::User*, 16> VisitedUsers;
    if(G.getNumUses() == 0 || !isUsedInFunctions(VisitedUsers, &G))
      UnneededGlobals.push_back(&G);
  }
  for(auto& G : UnneededGlobals) {
    G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
    G->eraseFromParent();
  }
  llvm::GlobalOptPass GO;
  GO.run(M, AM);
  return llvm::PreservedAnalyses::none();
}

KernelArgumentCanonicalizationPass::KernelArgumentCanonicalizationPass(
    const std::vector<std::string> &KernelNames)
    : KernelNames{KernelNames} {}

llvm::PreservedAnalyses KernelArgumentCanonicalizationPass::run(llvm::Module &M,
                                                                llvm::ModuleAnalysisManager &AM) {
  for (const auto &K : KernelNames) {
    if (auto *F = M.getFunction(K)) {
      canonicalizeKernelParameters(F, M);
    }
  }
  return llvm::PreservedAnalyses::none();
}
}
}
