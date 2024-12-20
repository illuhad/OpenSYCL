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

#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>

namespace hipsycl::compiler::utils {
using namespace hipsycl::compiler::cbs;

void replaceUsesOfGVWith(llvm::Function &F, llvm::StringRef GlobalVarName, llvm::Value *To, llvm::StringRef LogPrefix) {
  auto M = F.getParent();
  auto GV = M->getGlobalVariable(GlobalVarName);
  if (!GV)
    return;

  HIPSYCL_DEBUG_INFO << LogPrefix << "RUOGVW: " << *GV << " with " << *To << "\n";
  llvm::SmallVector<llvm::Instruction *> ToErase;
  for (auto U : GV->users()) {
    if (auto I = llvm::dyn_cast<llvm::LoadInst>(U); I && I->getFunction() == &F) {
      HIPSYCL_DEBUG_INFO << LogPrefix << "RUOGVW: " << *I << " with " << *To << "\n";
      I->replaceAllUsesWith(To);
    }
  }
  for (auto I : ToErase)
    I->eraseFromParent();
}

llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B,
                          llvm::Function &F) {
  DT.reset();
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);
  return LI.getLoopFor(B);
}

bool isBarrier(const llvm::Instruction *I, const SplitterAnnotationInfo &SAA) {
  if (const auto *CI = llvm::dyn_cast<llvm::CallInst>(I))
    return CI->getCalledFunction() && SAA.isSplitterFunc(CI->getCalledFunction());
  return false;
}

bool blockHasBarrier(const llvm::BasicBlock *BB,
                     const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  return std::any_of(BB->begin(), BB->end(), [&SAA](const auto &I) { return isBarrier(&I, SAA); });
}

// Returns true in case the given basic block starts with a barrier,
// that is, contains a branch instruction after possible PHI nodes.
bool startsWithBarrier(const llvm::BasicBlock *BB,
                       const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  return isBarrier(BB->getFirstNonPHI(), SAA);
}

// Returns true in case the given basic block ends with a barrier,
// that is, contains only a branch instruction after a barrier call.
bool endsWithBarrier(const llvm::BasicBlock *BB,
                     const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  const llvm::Instruction *T = BB->getTerminator();
  assert(T);
  return BB->size() > 1 && T->getPrevNode() && isBarrier(T->getPrevNode(), SAA);
}

bool hasOnlyBarrier(const llvm::BasicBlock *BB,
                    const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  return endsWithBarrier(BB, SAA) && BB->size() == 2;
}

// Returns true in case the given function is a kernel with work-group
// barriers inside it.
bool hasBarriers(const llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  for (auto &BB : F) {
    if (blockHasBarrier(&BB, SAA)) {

      // Ignore the implicit entry and exit barriers.
      if (hasOnlyBarrier(&BB, SAA) && &BB == &F.getEntryBlock())
        continue;

      if (hasOnlyBarrier(&BB, SAA) && BB.getTerminator()->getNumSuccessors() == 0)
        continue;

      return true;
    }
  }
  return false;
}

llvm::CallInst *createBarrier(llvm::Instruction *InsertBefore, SplitterAnnotationInfo &SAA) {
  llvm::Module *M = InsertBefore->getParent()->getParent()->getParent();

  if (InsertBefore != &InsertBefore->getParent()->front() &&
      isBarrier(InsertBefore->getPrevNode(), SAA))
    return llvm::cast<llvm::CallInst>(InsertBefore->getPrevNode());
  llvm::Function *F = llvm::cast<llvm::Function>(
      M->getOrInsertFunction(BarrierIntrinsicName, llvm::Type::getVoidTy(M->getContext()))
          .getCallee());

  F->addFnAttr(llvm::Attribute::NoDuplicate);
  F->setLinkage(llvm::GlobalValue::LinkOnceAnyLinkage);
  SAA.addSplitter(F);

  return llvm::CallInst::Create(F, "", InsertBefore);
}

bool checkedInlineFunction(llvm::CallBase *CI, llvm::StringRef PassPrefix, int NoInlineDebugLevel) {
  if (CI->getCalledFunction()->isIntrinsic() ||
      CI->getCalledFunction()->getName() == BarrierIntrinsicName)
    return false;

  // needed to be valid for success log
  const auto CalleeName = CI->getCalledFunction()->getName().str();

  llvm::InlineFunctionInfo IFI;

  llvm::InlineResult ILR = llvm::InlineFunction(*CI, IFI);
  if (!ILR.isSuccess()) {
    HIPSYCL_DEBUG_STREAM(NoInlineDebugLevel, (NoInlineDebugLevel >= HIPSYCL_DEBUG_LEVEL_INFO
                                                  ? HIPSYCL_DEBUG_PREFIX_INFO
                                                  : HIPSYCL_DEBUG_PREFIX_WARNING))
        << PassPrefix << " failed to inline function <" << CalleeName << ">: '"
        << ILR.getFailureReason() << "'\n";

    return false;
  }

  HIPSYCL_DEBUG_INFO << PassPrefix << " inlined function <" << CalleeName << ">\n";
  return true;
}

bool isAnnotatedParallel(llvm::Loop *TheLoop) { // from llvm for debugging.
  llvm::MDNode *DesiredLoopIdMetadata = TheLoop->getLoopID();

  if (!DesiredLoopIdMetadata)
    return false;

  llvm::MDNode *ParallelAccesses =
      llvm::findOptionMDForLoop(TheLoop, "llvm.loop.parallel_accesses");
  llvm::SmallPtrSet<llvm::MDNode *, 4> ParallelAccessGroups; // For scalable 'contains' check.
  if (ParallelAccesses) {
    for (const llvm::MDOperand &MD : llvm::drop_begin(ParallelAccesses->operands(), 1)) {
      llvm::MDNode *AccGroup = llvm::cast<llvm::MDNode>(MD.get());
      assert(llvm::isValidAsAccessGroup(AccGroup) && "List item must be an access group");
      ParallelAccessGroups.insert(AccGroup);
    }
  }

  // The loop branch contains the parallel loop metadata. In order to ensure
  // that any parallel-loop-unaware optimization pass hasn't added loop-carried
  // dependencies (thus converted the loop back to a sequential loop), check
  // that all the memory instructions in the loop belong to an access group that
  // is parallel to this loop.
  for (llvm::BasicBlock *BB : TheLoop->blocks()) {
    for (llvm::Instruction &I : *BB) {
      if (!I.mayReadOrWriteMemory())
        continue;

      if (llvm::MDNode *AccessGroup = I.getMetadata(llvm::LLVMContext::MD_access_group)) {
        auto ContainsAccessGroup = [&ParallelAccessGroups](llvm::MDNode *AG) -> bool {
          if (AG->getNumOperands() == 0) {
            assert(llvm::isValidAsAccessGroup(AG) && "Item must be an access group");
            return ParallelAccessGroups.count(AG);
          }

          for (const llvm::MDOperand &AccessListItem : AG->operands()) {
            llvm::MDNode *AccGroup = llvm::cast<llvm::MDNode>(AccessListItem.get());
            assert(llvm::isValidAsAccessGroup(AccGroup) && "List item must be an access group");
            if (ParallelAccessGroups.count(AccGroup))
              return true;
          }
          return false;
        };

        if (ContainsAccessGroup(AccessGroup))
          continue;
      }
      auto ReturnFalse = [&I]() {
        HIPSYCL_DEBUG_EXECUTE_WARNING(llvm::outs()
                                          << HIPSYCL_DEBUG_PREFIX_WARNING << "loop not parallel: ";
                                      I.print(llvm::outs()); llvm::outs() << "\n";)
        return false;
      };
      // The memory instruction can refer to the loop identifier metadata
      // directly or indirectly through another list metadata (in case of
      // nested parallel loops). The loop identifier metadata refers to
      // itself so we can check both cases with the same routine.
      llvm::MDNode *LoopIdMD = I.getMetadata(llvm::LLVMContext::MD_mem_parallel_loop_access);

      if (!LoopIdMD)
        return ReturnFalse();

      if (!llvm::is_contained(LoopIdMD->operands(), DesiredLoopIdMetadata))
        return ReturnFalse();
    }
  }
  return true;
}

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F, llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup) {
  // findOptionMDForLoopID also checks if there's a loop id, so this is fine
  if (auto *ParAccesses =
          llvm::findOptionMDForLoopID(L->getLoopID(), "llvm.loop.parallel_accesses")) {
    llvm::SmallVector<llvm::Metadata *, 4> AccessGroups{
        ParAccesses->op_begin(), ParAccesses->op_end()}; // contains .parallel_accesses
    AccessGroups.push_back(MDAccessGroup);
    auto *NewParAccesses = llvm::MDNode::get(F->getContext(), AccessGroups);

    const auto *const PIt =
        std::find(L->getLoopID()->op_begin(), L->getLoopID()->op_end(), ParAccesses);
    auto PIdx = std::distance(L->getLoopID()->op_begin(), PIt);
    L->getLoopID()->replaceOperandWith(PIdx, NewParAccesses);
  } else {
    auto *NewParAccesses = llvm::MDNode::get(
        F->getContext(),
        {llvm::MDString::get(F->getContext(), "llvm.loop.parallel_accesses"), MDAccessGroup});
    L->setLoopID(llvm::makePostTransformationMetadata(F->getContext(), L->getLoopID(), {},
                                                      {NewParAccesses}));
  }
}

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup) {
  if (auto *PresentMD = I->getMetadata(llvm::LLVMContext::MD_access_group)) {
    llvm::SmallVector<llvm::Metadata *, 4> MDs;
    if (PresentMD->getNumOperands() == 0)
      MDs.push_back(PresentMD);
    else
      MDs.append(PresentMD->op_begin(), PresentMD->op_end());
    MDs.push_back(MDAccessGroup);
    auto *CombinedMDAccessGroup = llvm::MDNode::getDistinct(I->getContext(), MDs);
    I->setMetadata(llvm::LLVMContext::MD_access_group, CombinedMDAccessGroup);
  } else
    I->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
}

llvm::SmallPtrSet<llvm::BasicBlock *, 8> getBasicBlocksInWorkItemLoops(const llvm::LoopInfo &LI) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> BBSet;
  for (auto *WIL : utils::getLoopsInPreorder(LI))
    if (isWorkItemLoop(*WIL))
      for (auto *BB : WIL->blocks())
        if (BB != WIL->getLoopLatch() && BB != WIL->getHeader() && BB != WIL->getExitBlock())
          BBSet.insert(BB);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(HIPSYCL_DEBUG_INFO << "WorkItemLoop BBs:\n"; for (auto *BB
                                                                                  : BBSet) {
    HIPSYCL_DEBUG_INFO << "  " << BB->getName() << "\n";
  })
  return BBSet;
}

bool isWorkItemLoop(const llvm::Loop &L) {
  return llvm::findOptionMDForLoop(&L, hipsycl::compiler::MDKind::WorkItemLoop);
}

bool isInWorkItemLoop(const llvm::Loop &L) {
  llvm::Loop *PL = L.getParentLoop();
  while (PL) {
    if (isWorkItemLoop(*PL))
      return true;
    PL = PL->getParentLoop();
  }
  return false;
}

bool isInWorkItemLoop(const llvm::Region &R, const llvm::LoopInfo &LI) {
  if (auto *L = LI.getLoopFor(R.getEntry()))
    return isWorkItemLoop(*L) || isInWorkItemLoop(*L);
  return false;
}

llvm::Loop *getOneWorkItemLoop(const llvm::LoopInfo &LI) {
  for (auto *L : LI) {
    if (isWorkItemLoop(*L))
      return L;
  }
  return nullptr;
}

llvm::BasicBlock *getWorkItemLoopBodyEntry(const llvm::Loop *WILoop) {
  llvm::BasicBlock *Entry = nullptr;
  assert(!llvm::successors(WILoop->getHeader()).empty() && "WILoop must have a body!");
  for (auto *Succ : llvm::successors(WILoop->getHeader())) {
    if (Succ != WILoop->getExitBlock()) {
      Entry = Succ;
      break;
    }
  }
  return Entry;
}

llvm::SmallVector<llvm::Loop *, 4> getLoopsInPreorder(const llvm::LoopInfo &LI) {
  return const_cast<llvm::LoopInfo &>(LI).getLoopsInPreorder();
}

llvm::BasicBlock *simplifyLatch(const llvm::Loop *L, llvm::BasicBlock *Latch, llvm::LoopInfo &LI,
                                llvm::DominatorTree &DT) {
  assert(L->getCanonicalInductionVariable() && "must be canonical loop!");
  llvm::Value *InductionValue = L->getCanonicalInductionVariable()->getIncomingValueForBlock(Latch);
  auto *InductionInstr = llvm::cast<llvm::Instruction>(InductionValue);
  return llvm::SplitBlock(Latch, InductionInstr, &DT, &LI, nullptr, Latch->getName() + ".latch");
}

llvm::BasicBlock *splitEdge(llvm::BasicBlock *Root, llvm::BasicBlock *&Target, llvm::LoopInfo *LI,
                            llvm::DominatorTree *DT) {
  auto *NewBlockAtEdge = llvm::SplitEdge(Root, Target, DT, LI, nullptr);

  assert(NewBlockAtEdge->getTerminator()->getSuccessor(0) == Target &&
         "NewBlockAtEdge must be predecessor to Target");
  return NewBlockAtEdge;
}

void promoteAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT,
                    llvm::AssumptionCache &AC) {
  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  while (true) {
    WL.clear();
    for (auto &I : *EntryBlock) {
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (llvm::isAllocaPromotable(Alloca))
          WL.push_back(Alloca);
      }
    }
    if (WL.empty())
      break;
    llvm::PromoteMemToReg(WL, DT, &AC);
  }
}

llvm::Instruction *getBrCmp(const llvm::BasicBlock &BB) {
  if (auto *BI = llvm::dyn_cast_or_null<llvm::BranchInst>(BB.getTerminator()))
    if (BI->isConditional()) {
      if (auto *CmpI = llvm::dyn_cast<llvm::ICmpInst>(BI->getCondition()))
        return CmpI;
      if (auto *SelectI = llvm::dyn_cast<llvm::SelectInst>(BI->getCondition()))
        return SelectI;
    }
  return nullptr;
}

void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::Loop &L, llvm::Value *Idx,
                     const llvm::DominatorTree &DT) {
  assert(Idx && "Valid WI-Index required");

  auto *MDAlloca = llvm::MDNode::get(
      EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), MDKind::LoopState)});

  auto &LoopBlocks = L.getBlocksSet();
  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (Alloca->getMetadata(hipsycl::compiler::MDKind::Arrayified))
        continue; // already arrayificated
      if (!std::all_of(Alloca->user_begin(), Alloca->user_end(), [&LoopBlocks](llvm::User *User) {
            auto *Inst = llvm::dyn_cast<llvm::Instruction>(User);
            return Inst && LoopBlocks.contains(Inst->getParent());
          }))
        continue;
      WL.push_back(Alloca);
    }
  }

  for (auto *I : WL) {
    llvm::IRBuilder AllocaBuilder{I};
    llvm::Type *T = I->getAllocatedType();
    if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
      auto ArrSize = ArrSizeC->getLimitedValue();
      if (ArrSize > 1) {
        T = llvm::ArrayType::get(T, ArrSize);
        HIPSYCL_DEBUG_WARNING << "Caution, alloca was array\n";
      }
    }

    auto *Alloca = AllocaBuilder.CreateAlloca(
        T, AllocaBuilder.getInt32(hipsycl::compiler::NumArrayElements), I->getName() + "_alloca");
    Alloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
    Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

    llvm::Instruction *GepIp = nullptr;
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!GepIp || DT.dominates(UI, GepIp))
          GepIp = UI;
      }
    }
    if (GepIp) {
      llvm::IRBuilder LoadBuilder{GepIp};
      auto *GEP = llvm::cast<llvm::GetElementPtrInst>(LoadBuilder.CreateInBoundsGEP(
          Alloca->getAllocatedType(), Alloca, Idx, I->getName() + "_gep"));
      GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      I->replaceAllUsesWith(GEP);
      I->eraseFromParent();
    }
  }
}

/// Arrayification of work item private values

// Create a new alloca of size \a NumElements at \a IPAllocas.
// The type is taken from \a ToArrayify.
// At \a InsertionPoint, a store is added that stores the \a ToArrayify
// value to the alloca element at \a Idx.
llvm::AllocaInst *arrayifyValue(llvm::Instruction *IPAllocas, llvm::Value *ToArrayify,
                                llvm::Instruction *InsertionPoint, llvm::Value *Idx,
                                llvm::Value *NumElements, llvm::MDTuple *MDAlloca) {
  assert(Idx && "Valid WI-Index required");

  if (!MDAlloca)
    MDAlloca = llvm::MDNode::get(IPAllocas->getContext(),
                                 {llvm::MDString::get(IPAllocas->getContext(), MDKind::LoopState)});

  auto *T = ToArrayify->getType();
  llvm::IRBuilder AllocaBuilder{IPAllocas};
  auto *Alloca = AllocaBuilder.CreateAlloca(T, NumElements, ToArrayify->getName() + "_alloca");
  if (NumElements)
    Alloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
  Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

  llvm::IRBuilder WriteBuilder{InsertionPoint};
  llvm::Value *StoreTarget = Alloca;
  if (NumElements) {
    auto *GEP = llvm::cast<llvm::GetElementPtrInst>(WriteBuilder.CreateInBoundsGEP(
        Alloca->getAllocatedType(), Alloca, Idx, ToArrayify->getName() + "_gep"));
    GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);
    StoreTarget = GEP;
  }
  WriteBuilder.CreateStore(ToArrayify, StoreTarget);
  return Alloca;
}

// see arrayifyValue. The store is inserted after the \a ToArrayify instruction
llvm::AllocaInst *arrayifyInstruction(llvm::Instruction *IPAllocas, llvm::Instruction *ToArrayify,
                                      llvm::Value *Idx, llvm::Value *NumElements,
                                      llvm::MDTuple *MDAlloca) {
  llvm::Instruction *InsertionPoint = &*(++ToArrayify->getIterator());
  if (llvm::isa<llvm::PHINode>(ToArrayify))
    InsertionPoint = ToArrayify->getParent()->getFirstNonPHI();

  return arrayifyValue(IPAllocas, ToArrayify, InsertionPoint, Idx, NumElements, MDAlloca);
}

// load from the \a Alloca at \a Idx, if array alloca, otherwise just load the
// alloca value
llvm::LoadInst *loadFromAlloca(llvm::AllocaInst *Alloca, llvm::Value *Idx,
                               llvm::Instruction *InsertBefore, const llvm::Twine &NamePrefix) {
  assert(Idx && "Valid WI-Index required");
  auto *MDAlloca = Alloca->getMetadata(hipsycl::compiler::MDKind::Arrayified);

  llvm::IRBuilder LoadBuilder{InsertBefore};
  llvm::Value *LoadFrom = Alloca;
  if (Alloca->isArrayAllocation()) {
    auto *GEP = llvm::cast<llvm::GetElementPtrInst>(LoadBuilder.CreateInBoundsGEP(
        Alloca->getAllocatedType(), Alloca, Idx, NamePrefix + "_lgep"));
    GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);
    LoadFrom = GEP;
  }
  auto *Load = LoadBuilder.CreateLoad(Alloca->getAllocatedType(), LoadFrom, NamePrefix + "_load");
  return Load;
}

// get the work-item state alloca a load reads from (through GEPs..)
llvm::AllocaInst *getLoopStateAllocaForLoad(llvm::LoadInst &LInst) {
  llvm::AllocaInst *Alloca = nullptr;
  if (auto *GEPI = llvm::dyn_cast<llvm::GetElementPtrInst>(LInst.getPointerOperand())) {
    Alloca = llvm::dyn_cast<llvm::AllocaInst>(GEPI->getPointerOperand());
  } else {
    Alloca = llvm::dyn_cast<llvm::AllocaInst>(LInst.getPointerOperand());
  }
  if (Alloca && Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
    return Alloca;
  return nullptr;
}

// bring along the llvm.dbg.value intrinsics when cloning values
void copyDgbValues(llvm::Value *From, llvm::Value *To, llvm::Instruction *InsertBefore) {
  llvm::SmallVector<llvm::DbgValueInst *, 1> DbgValues;
  llvm::findDbgValues(DbgValues, From);
  if (!DbgValues.empty()) {
    auto *DbgValue = DbgValues.back();
    llvm::DIBuilder DbgBuilder{*InsertBefore->getParent()->getParent()->getParent()};
    DbgBuilder.insertDbgValueIntrinsic(To, DbgValue->getVariable(), DbgValue->getExpression(),
                                       DbgValue->getDebugLoc(), InsertBefore);
  }
}

void dropDebugLocation(llvm::Instruction &I) {
  I.dropLocation();
}

void dropDebugLocation(llvm::BasicBlock *BB) {
  for (auto &I : *BB) {
    auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
    if (!CI || !llvm::isDbgInfoIntrinsic(CI->getIntrinsicID())) {
      dropDebugLocation(I);
    }
  }
}

} // namespace hipsycl::compiler::utils
