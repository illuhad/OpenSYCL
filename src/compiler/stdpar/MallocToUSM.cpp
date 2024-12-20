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
#include "hipSYCL/compiler/stdpar/MallocToUSM.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"



#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Instructions.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Transforms/Utils/Cloning.h>



namespace hipsycl {
namespace compiler {

namespace {

std::string addABITag(llvm::StringRef OriginalName, llvm::StringRef ABITag) {
  auto makeFallBackName = [&](){
    return OriginalName.str()+"_"+ABITag.str();
  };

  auto FirstNumber = OriginalName.find_first_of("0123456789");
  if(FirstNumber == std::string::npos)
    return makeFallBackName();
  
  int NumCharacters = std::atoi(OriginalName.data() + FirstNumber);
  
  auto NameStart = OriginalName.find_first_not_of("0123456789", FirstNumber);
  if(NameStart == std::string::npos)
    return makeFallBackName();
  
  auto InsertionPoint = NameStart + NumCharacters;

  std::string Result = OriginalName.str();
  if(InsertionPoint > Result.size())
    return makeFallBackName();
  
  std::string ABITagIdentifer = "B"+std::to_string(ABITag.size()) + ABITag.str();
  Result.insert(InsertionPoint, ABITagIdentifer);
  return Result;
}

bool NameStartsWithItaniumIdentifier(llvm::StringRef Name, llvm::StringRef Identifier) {
  auto FirstNumber = Name.find_first_of("0123456789");
  auto IdentifierPos = Name.find(std::to_string(Identifier.size())+Identifier.str());

  if (FirstNumber == std::string::npos || IdentifierPos == std::string::npos)
    return false;
  
  return FirstNumber == IdentifierPos;
}


bool isRestrictedToRegularMalloc(llvm::Function* F) {
  llvm::StringRef Name = F->getName();
  if(!llvmutils::starts_with(Name, "_Z"))
    return false;
  
  if(NameStartsWithItaniumIdentifier(Name, "hipsycl"))
    return true;
  
  return false;
}

bool isStdFunction(llvm::Function* F) {
  llvm::StringRef Name = F->getName();
  if(llvmutils::starts_with(Name, "_ZNSt") ||
     llvmutils::starts_with(Name, "_ZSt") ||
     llvmutils::starts_with(Name, "_ZNKSt"))
    return true;
  return false;
}

template<class SetT>
void collectAllCallees(llvm::CallGraph& CG, llvm::Function* F, SetT& Out) {
  if(Out.contains(F))
    return;
  
  // Functions that are available_externally and have their address taken
  // we need to discard, as they won't be emitted within this module.
  if(F->getLinkage() == llvm::GlobalValue::AvailableExternallyLinkage) {
    if(F->hasAddressTaken()) {
      return;
    }
  }
  
  Out.insert(F);
  
  llvm::CallGraphNode* CGN = CG.getOrInsertFunction(F);
  if(CGN) {
    for(unsigned i = 0; i < CGN->size(); ++i){
      auto* CalledFunction = (*CGN)[i]->getFunction();
      if(CalledFunction) {
        collectAllCallees(CG, CalledFunction, Out);
      }
    }
  }
}

template <class CallerMapT, class SetT>
void collectAllCallersFromSet(const CallerMapT &CM, llvm::Function *F, const SetT &Input,
                              SetT &DiscardedOut, SetT &Out) {
  if(!F)
    return;
  
  if(Out.contains(F) || DiscardedOut.contains(F) || !Input.contains(F)) {
    DiscardedOut.insert(F);
    return;
  }

  auto It = CM.find(F);
  if(It == CM.end()) {
    DiscardedOut.insert(F);
    return;
  }

  Out.insert(F);

  for(auto* Caller : It->getSecond()) {
    collectAllCallersFromSet(CM, Caller, Input, DiscardedOut, Out);
  }
}
}

llvm::PreservedAnalyses MallocToUSMPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static constexpr const char* AllocIdentifier = "hipsycl_stdpar_alloc";
  static constexpr const char* FreeIdentifier = "hipsycl_stdpar_free";
  llvm::SmallPtrSet<llvm::Function*, 16> ManagedAllocFunctions;
  llvm::SmallPtrSet<llvm::Function*, 16> ManagedFreeFunctions;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F) {
      if(Annotation.compare(AllocIdentifier) == 0) {
        HIPSYCL_DEBUG_INFO
            << "[stdpar] MallocToUSM: Found new memory allocation function definition: "
            << F->getName() << "\n";
        ManagedAllocFunctions.insert(F);
        
      }
      if(Annotation.compare(FreeIdentifier) == 0) {
        ManagedFreeFunctions.insert(F);
      }
    }
  });


  llvm::CallGraph CG{M};
  
  for(auto& F: M)
    CG.addToCallGraph(&F);
  

  llvm::DenseMap<llvm::Function*, llvm::SmallPtrSet<llvm::Function*, 16>> FunctionCallers;

  for(auto& F: M) {
    llvm::CallGraphNode* CGN = CG.getOrInsertFunction(&F);
    if(CGN) {
      for(unsigned i = 0; i < CGN->size(); ++i){
        auto* CalledFunction = (*CGN)[i]->getFunction();
        if(CalledFunction) {
          FunctionCallers[CalledFunction].insert(&F);
        }
      }
    }
  }

  static constexpr const char* ForcedRegularMallocABITag = "hipsycl_stdpar_regular_alloc";
  static constexpr const char* USMABITag = "hipsycl_stdpar_usm_alloc";

  // First, find all functions that define entrypoints to subgraphs of
  // the callgraph where no malloc hijacking should occur.
  llvm::SmallPtrSet<llvm::Function*, 16> RestrictedEntrypoints;
  for(auto& F: M) {
    if(isRestrictedToRegularMalloc(&F) && !F.isDeclaration()) {
      RestrictedEntrypoints.insert(&F);
    }
  }

  // Find all functions used from those entrypoints
  llvm::SmallPtrSet<llvm::Function*, 16> RestrictedSubCallgraph;
  for(auto* F: RestrictedEntrypoints)
    collectAllCallees(CG, F, RestrictedSubCallgraph);
  
  // Functions that are used in a branch of the call graph
  // that ends up doing memory management need to be duplicated
  // so that we can force memory management to not be hijacked there.
  // So, first find all functions of the RestrictedSubCallgraph
  // that are part of a possible call stack leading to a memory management function call.
  llvm::SmallPtrSet<llvm::Function*, 16> PrunedRestrictedSubCallgraph;
  for(auto* F : ManagedAllocFunctions) {
    llvm::SmallPtrSet<llvm::Function*, 16> DiscardedFunctions;
    collectAllCallersFromSet(FunctionCallers, F, RestrictedSubCallgraph, DiscardedFunctions,
                             PrunedRestrictedSubCallgraph);
  }
  // Due to the way collectAllCallersFromSet is currently implemented, it might also return the
  // alloc functions in the set, which we don't need or want.
  for(auto* F: ManagedAllocFunctions) {
    if(PrunedRestrictedSubCallgraph.contains(F))
      PrunedRestrictedSubCallgraph.erase(F);
  }
  for(auto* F: ManagedFreeFunctions) {
    if(PrunedRestrictedSubCallgraph.contains(F))
      PrunedRestrictedSubCallgraph.erase(F);
  }

  llvm::SmallDenseMap<llvm::Function*, llvm::Function*> DuplicatedFunctions;
  llvm::SmallPtrSet<llvm::Function*, 16> ReplacementSubCallgraph;
  for(auto* F: PrunedRestrictedSubCallgraph) {
    llvm::ValueToValueMapTy VMap;
    llvm::Function* NewF = llvm::CloneFunction(F, VMap);
    // This is safe because when generating the callgraph, we already
    // exclude available_externally functions that have their address taken.
    if(NewF->getLinkage() == llvm::GlobalValue::AvailableExternallyLinkage)
      NewF->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    
    NewF->setName(addABITag(NewF->getName(), ForcedRegularMallocABITag));

    DuplicatedFunctions[F] = NewF;
    ReplacementSubCallgraph.insert(NewF);
  }

  // For every call to the original function, we need to put in a call
  // to the new function if it is in a restricted entry point or a new function.
  for(auto& Entry : DuplicatedFunctions) {
    llvm::Function* OriginalFunction = Entry.getFirst();
    llvm::Function* NewFunction = Entry.getSecond();
    OriginalFunction->replaceUsesWithIf(NewFunction, [&](llvm::Use& U) -> bool{
      // if user is function call
      if(auto* I = llvm::dyn_cast<llvm::Instruction>(U.getUser())) {
        if(auto* BB = I->getParent()) {
          // Obtain function that calls OriginalFunction
          if(auto* Caller = BB->getParent()) {
            // return true if the function call happens in one of the duplicated functions,
            // or in the entrypoints.
            return ReplacementSubCallgraph.contains(Caller) || RestrictedEntrypoints.contains(Caller) ||
                   ManagedAllocFunctions.contains(Caller) || ManagedFreeFunctions.contains(Caller);
          }
        }
      }
      return false;
    });
  }

  for(auto* MemoryF : ManagedAllocFunctions) {
    // Rename the function to avoid ODR issues during linking
    std::string OriginalName = MemoryF->getName().str();
    MemoryF->setName(addABITag(MemoryF->getName(), USMABITag));

    // Create decl for the original, unmodified memory management function
    llvm::Function *UnmodifiedFuncDecl =
        llvm::Function::Create(MemoryF->getFunctionType(), MemoryF->getLinkage(), OriginalName, M);
    UnmodifiedFuncDecl->setVisibility(MemoryF->getVisibility());
    UnmodifiedFuncDecl->setAttributes(MemoryF->getAttributes());
    
    MemoryF->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    MemoryF->setVisibility(llvm::GlobalValue::DefaultVisibility);

    // Replace all uses from within the restricted function call graph
    MemoryF->replaceUsesWithIf(UnmodifiedFuncDecl, [&](llvm::Use& U) -> bool {
      if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U.getUser())) {
        if(auto* BB = CB->getParent()) {
          if(auto* F = BB->getParent()) {
            if(ReplacementSubCallgraph.contains(F) || RestrictedEntrypoints.contains(F)) {
              HIPSYCL_DEBUG_INFO << "[stdpar] MallocToUSM: Forcing regular allocation in "
                                 << F->getName().str() << "\n";
              return true;
            }
          }
        }
      }
      return false;
    });
  }

  // Internalize memory management definitions
  for(auto* F: ManagedFreeFunctions)
    F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);

  // Ideally, we could insert an ABI tag for every function that uses USM, such that external
  // libraries do not ODR-resolve symbols to functions using USM when the libraries have not
  // been compiled by us and then cannot free the USM pointers.
  // However, this does not work because we cannot know whether other translation or
  // linkage units expect symbols defined here with their regular mangled names for linking.
  //
  // What we can however do is to do this for STL functions. These are most prone to this issue anyway
  // because chances are that they are used in many different places.
  // For STL functions we know  that every user will also have their own definition available,
  // if we also have a definition here, so adding an ABI tag is safe.
  for(auto& F : M) {
    // Only do this for functions that do not potentially have an ABI tag already.
    if (!ManagedAllocFunctions.contains(&F) && !ManagedFreeFunctions.contains(&F) &&
        !RestrictedEntrypoints.contains(&F) && !ReplacementSubCallgraph.contains(&F)) {
      // Only consider functions in std:: that have a definition
      if(!F.isDeclaration() && isStdFunction(&F)) {
        F.setName(addABITag(F.getName(), USMABITag));
        // There are certain functions around basic_string that have available_externally linkage,
        // meaning that although we have a definition, the linker will try to link them to external
        // symbols.
        // We need to internalize these symbols and ensure that our definition is used
        // to obtain the correct USM behavior, and avoid linking issues when adding the ABI tag.
        if(F.getLinkage() == llvm::GlobalValue::AvailableExternallyLinkage) {
          F.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
        }
      }
    }
  }

  return llvm::PreservedAnalyses::none();
}
}
}
