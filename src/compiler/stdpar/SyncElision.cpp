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
#include "hipSYCL/compiler/stdpar/SyncElision.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Attributes.h>
#include <llvm/Support/Casting.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Instruction.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>


namespace hipsycl {
namespace compiler {

namespace {

template <class Handler>
bool descendInstructionUseTree(llvm::Instruction *I, Handler &&H,
                               llvm::Instruction *Parent = nullptr) {
  if(H(I, Parent)) {
    for(auto* U : I->users()) {
      if(auto* UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if(!descendInstructionUseTree(UI, H, I))
          return false;
      } else {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

using InstToInstListMapT =
    llvm::SmallDenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 16>>;

// Identifies store instructions that might be related for argument handling:
// We identify these instructions by looking for allocas in the function. If that alloca
// is only used by getelementptr, stores, and calls to stdpar functions, chances are
// these instructions are only relevant for constructing stdpar arguments.
//
// The result is a map from encountered store instructions to other instructions referencing the same
// memory.
void identifyStoresPotentiallyForStdparArgHandling(
    llvm::Function *F, const llvm::SmallPtrSet<llvm::Function *, 16> &StdparFunctions,
    InstToInstListMapT &Out) {
  for(auto& BB : *F) {
    for(auto& I : BB) {
      if(llvm::isa<llvm::AllocaInst>(&I)) {
        llvm::SmallVector<llvm::Instruction*, 16> Users;

        bool onlyUsedInAllowedInstructions = descendInstructionUseTree(
            &I, [&](llvm::Instruction *Current, llvm::Instruction *Parent) {
              if (llvm::isa<llvm::AllocaInst>(Current) ||
                  llvm::isa<llvm::GetElementPtrInst>(Current)) {
                Users.push_back(Current);
                return true;
              } else if(auto *SI = llvm::dyn_cast<llvm::StoreInst>(Current)) {
                // For store instructions, we enforce that the previous instruction in
                // the use chain must be the pointer operand, not the value operand.
                if(SI->getValueOperand() != Parent) {
                  Users.push_back(Current);
                  return true;
                }
              } else if (auto *CB = llvm::dyn_cast<llvm::CallBase>(Current)) {
                if (StdparFunctions.contains(CB->getCalledFunction())) {
                  Users.push_back(Current);
                  return true;
                } else if(llvmutils::starts_with(CB->getCalledFunction()->getName(), "llvm.lifetime")) {
                  return true;
                }
              }

              return false;
            });

        if(onlyUsedInAllowedInstructions) {
          for(auto* U: Users) {
            if(auto* SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
              for(auto* U : Users) {
                Out[SI].push_back(U);
              }  
            }
          }
        }
      }
    }
  }
}

void identifyStoresPotentiallyForStdparArgHandling(
    const llvm::SmallVector<llvm::Instruction *, 16> &StdparCallPositions,
    const llvm::SmallPtrSet<llvm::Function *, 16> &StdparFunctions,
    InstToInstListMapT &Out) {
  llvm::SmallPtrSet<llvm::Function*, 16> InvolvedFunctions;

  for(auto* I : StdparCallPositions) {
    if(I) {
      InvolvedFunctions.insert(I->getParent()->getParent());
    }
  }

  for(auto* F: InvolvedFunctions) {
    identifyStoresPotentiallyForStdparArgHandling(F, StdparFunctions, Out);
  }
}

bool instructionAccessesMemory(llvm::Instruction* I) {
  if (llvm::isa<llvm::StoreInst>(I) || llvm::isa<llvm::LoadInst>(I) ||
      llvm::isa<llvm::AtomicRMWInst>(I) || llvm::isa<llvm::AtomicCmpXchgInst>(I) ||
      llvm::isa<llvm::FenceInst>(I))
    return true;

  return false;
}

bool functionDoesNotAccessMemory(llvm::Function* F){
  if(!F)
    return true;
  if(F->isIntrinsic()) {
    if(llvmutils::starts_with(F->getName(), "llvm.lifetime")){
      return true;
    }
  }
  // We could improve this logic massively: E.g. a function which does not have ptr arguments,
  // does not have gobal variable users, contains no inttoptr instructions, and only calls functions
  // which satisfy these criteria could be assumed to not access memory.
  return false;
}

// returns whether To is in the same BB as From, and succeeds it in the instruction list.
bool isSucceedingInBB(llvm::Instruction* From, llvm::Instruction* To) {
  if(From->getParent() == To->getParent()) {
    for(auto* I = From; I != nullptr; I = I->getNextNonDebugInstruction()) {
      if(I == To)
        return true;
    }
  } 
  return false;
}

template <unsigned N>
bool allAreSucceedingInBB(llvm::Instruction* From,
                          const llvm::SmallVector<llvm::Instruction *, N> &To) {
  for(auto* I : To) {
    if(!isSucceedingInBB(From, I))
      return false;
  }
  return true;
}

constexpr const char* BarrierBuiltinName = "__acpp_stdpar_optional_barrier";
constexpr const char* EntrypointMarker = "hipsycl_stdpar_entrypoint";

template<class Handler>
void forEachStdparFunction(llvm::Module& M, Handler&& H){
  utils::findFunctionsWithStringAnnotations(M,  [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F) {
      if(Annotation.compare(EntrypointMarker) == 0) {
        H(F);
      }
    }
  });
}

template <class Handler>
void forEachReachableInstructionRequiringSync(
    llvm::Instruction *Start, const llvm::SmallPtrSet<llvm::Function *, 16> &StdparFunctions,
    const InstToInstListMapT& PotentialStoresForStdparArgs,
    llvm::SmallPtrSet<llvm::BasicBlock*, 16> &CompletelyVisitedBlocks,
    Handler &&H) {

  if(!Start)
    return;

  llvm::Instruction* Current = Start;
  if(CompletelyVisitedBlocks.contains(Current->getParent())) {
    return;
  }

  llvm::Instruction* FirstInst = &(*Current->getParent()->getFirstInsertionPt());
  if(Current == FirstInst) {
    CompletelyVisitedBlocks.insert(Current->getParent());
  }

  while(Current) {
    if(auto* CB = llvm::dyn_cast<llvm::CallBase>(Current)) {
      llvm::Function* CalledF = CB->getCalledFunction();
      if(CalledF->getName() == BarrierBuiltinName) {
        // basic block already contains barrier; nothing to do
        return;
      }

      // If we have found a call to an stdpar function, we can skip it --
      // after all, the whole point is to not sync after every stdpar call.
      // For all other calls, we need a sync because we currently
      // do not take control flow beyond our own function into account.
      // We can also safely ignore functions for which we know that they
      // do not access memory
      bool CanSkipFunctionCall =
          StdparFunctions.contains(CalledF) || functionDoesNotAccessMemory(CalledF);

      if(!CanSkipFunctionCall) {
        H(Current);
        return;
      }
    } else if(instructionAccessesMemory(Current)) {
      bool isSkippableStore = false;
      if(llvm::isa<llvm::StoreInst>(Current)) {
        // Check if the store is perhaps only used to setup arguments of stdpar calls
        // (e.g. to assemble kernel lambdas)
        auto It = PotentialStoresForStdparArgs.find(Current);
        if(It != PotentialStoresForStdparArgs.end()) {
          // Store is skippable, if the referenced memory is used by stdpar function calls
          // which succeed the store in the control flow.
          llvm::SmallVector<llvm::Instruction*, 16> StdparCallsUsingMemory;
          for(auto* I : It->getSecond()) {
            if(auto *CB = llvm::dyn_cast<llvm::CallBase>(I)){
              if(StdparFunctions.contains(CB->getCalledFunction())) {
                StdparCallsUsingMemory.push_back(CB);
              }
            }
          }
          if(allAreSucceedingInBB(Current, StdparCallsUsingMemory)) {
            isSkippableStore = true;
          }
        }
      }

      if(!isSkippableStore) {
        H(Current);
        return;
      } else {
        HIPSYCL_DEBUG_INFO
            << "[stdpar] SyncElision: Detected store that does not block barrier movement\n";
      }
    } else if(Current->isTerminator()){
      // If this terminator causes control flow to exit from this function, we need
      // to insert synchronization.
      // TODO: Look again at exception handling instructions in more detail
      if (llvm::isa<llvm::ReturnInst>(Current) || llvm::isa<llvm::InvokeInst>(Current) ||
          llvm::isa<llvm::CallBrInst>(Current) || llvm::isa<llvm::ResumeInst>(Current)) {
        H(Current);
        return;
      }
    }
    Current = Current->getNextNonDebugInstruction();
  }
  // We have reached the end of this BB - so we need to look
  // at all its successors in the CFG
  llvm::BasicBlock* BB = Start->getParent();
  for(int i = 0; i < BB->getTerminator()->getNumSuccessors(); ++i) {
    llvm::BasicBlock* Successor = BB->getTerminator()->getSuccessor(i);
    if(Successor->size() > 0) {
      llvm::Instruction* FirstI = &(*Successor->getFirstInsertionPt());
      forEachReachableInstructionRequiringSync(
          FirstI, StdparFunctions, PotentialStoresForStdparArgs, CompletelyVisitedBlocks, H);
    }
  }
}
}


llvm::PreservedAnalyses SyncElisionInliningPass::run(llvm::Module& M, llvm::ModuleAnalysisManager& AM) {

  auto InlineEachCaller = [](llvm::Function *F) {
    if(!F)
      return;
    for (auto *U : F->users()) {
      if (auto *I = llvm::dyn_cast<llvm::CallBase>(U)) {
        if (auto *BB = I->getParent()) {
          if (auto *Caller = BB->getParent()) {
            if (Caller != F && !Caller->hasFnAttribute(llvm::Attribute::AlwaysInline)) {
              Caller->addFnAttr(llvm::Attribute::AlwaysInline);
            }
          }
        }
      }
    }
  };

  forEachStdparFunction(M, [&](llvm::Function* F){
    InlineEachCaller(F);
  });

  return llvm::PreservedAnalyses::all();
}

llvm::PreservedAnalyses SyncElisionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  llvm::SmallPtrSet<llvm::Function*, 16> StdparFunctions;
  forEachStdparFunction(M, [&](llvm::Function *F) {
    HIPSYCL_DEBUG_INFO << "[stdpar] SyncElision: Found stdpar call: " << F->getName() << "\n";
    if (F->hasFnAttribute(llvm::Attribute::NoInline)) {
      F->removeFnAttr(llvm::Attribute::NoInline);
    }
    StdparFunctions.insert(F);
  });

  if(auto* SyncF = M.getFunction(BarrierBuiltinName)) {
    SyncF->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    if (SyncF->hasFnAttribute(llvm::Attribute::NoInline)) {
      SyncF->removeFnAttr(llvm::Attribute::NoInline);
    }

    llvm::SmallPtrSet<llvm::Instruction*, 16> SyncCallsToRemove;
    llvm::SmallVector<llvm::Instruction*, 16> StdparCallPositions;
    
    for(auto* U : SyncF->users()) {
      if(auto* I = llvm::dyn_cast<llvm::CallBase>(U)){
        
        llvm::Function* Caller = I->getParent()->getParent();
        if(StdparFunctions.contains(Caller)) {
          
          for(auto* CallerU : Caller->users()) {
            if(auto* CB = llvm::dyn_cast<llvm::CallBase>(CallerU)) {
              HIPSYCL_DEBUG_INFO << "[stdpar] SyncElision: Found stdpar call in potential need of "
                                    "synchronization: Call to "
                                 << Caller->getName() << " in function "
                                 << CB->getParent()->getParent()->getName() << "\n";
              SyncCallsToRemove.insert(I);
              StdparCallPositions.push_back(CB);
            }
          }
        } else {
          HIPSYCL_DEBUG_WARNING << "[stdpar] SyncElision: Encountered call to " << BarrierBuiltinName
                                << " in function that is not stdpar entrypoint\n";
        }
      }
    }
    
    // Remove synchronization calls present in stdpar function definitions
    for(auto* I : SyncCallsToRemove) {
      I->eraseFromParent();
    }

    // It can frequently happen that we have store instructions between two stdpar calls.
    // These store instructions can prevent synchronization elision, even if they are just
    // used to set up stdpar arguments (e.g., construct lambda objects).
    // To counter this, we try to identify instructions that are purely used for argument handling,
    // and do not interact with the stdpar kernel itself.
    InstToInstListMapT InstructionsPotentiallyForStdparArgHandling;
    identifyStoresPotentiallyForStdparArgHandling(
        StdparCallPositions, StdparFunctions, InstructionsPotentiallyForStdparArgHandling);

    for(auto* I : StdparCallPositions) {
      // For the start of our search, we need be move to the next instruction following
      // the stdpar call.
      // If the stdpar call is mapped to an InvokeInst (which is tpyically the case),
      // it does not have a next instruction.
      //
      // It is important to have this logic here, and not e.g. when collecting StdparCallPositions,
      // because the appropriate start position might be altered by other barrier insertions
      // in earlier iterations!
      llvm::SmallVector<llvm::Instruction*, 8> StartPositions;
      if(I->isTerminator()) {
        for(int i = 0; i < I->getNumSuccessors(); ++i) {
          StartPositions.push_back(&*(I->getSuccessor(i)->getFirstInsertionPt()));
        }
      } else {
        StartPositions.push_back(I->getNextNonDebugInstruction());
      }
      for(auto* Start : StartPositions) {

        llvm::SmallPtrSet<llvm::BasicBlock*, 16> VisitedBlocks;
        forEachReachableInstructionRequiringSync(
            Start, StdparFunctions, InstructionsPotentiallyForStdparArgHandling, VisitedBlocks,
            [&](llvm::Instruction *InsertSyncBefore) {
              HIPSYCL_DEBUG_INFO << "[stdpar] SyncElision: Inserting synchronization in function "
                                << InsertSyncBefore->getParent()->getParent()->getName() << "\n";
              llvm::CallInst::Create(SyncF->getFunctionType(), SyncF, "", InsertSyncBefore);
            });
      }
    }
  }

  return llvm::PreservedAnalyses::none();
}
}
}
