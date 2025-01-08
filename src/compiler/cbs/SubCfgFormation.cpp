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
#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/cbs/UniformityAnalysis.hpp"

#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Regex.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include <cstddef>
#include <functional>
#include <numeric>

#define DEBUG_SUBCFG_FORMATION

namespace {
using namespace hipsycl::compiler;
using namespace hipsycl::compiler::cbs;

static const std::array<char, 3> DimName{'x', 'y', 'z'};

// gets the load inside F from the global variable called VarName
llvm::Instruction *getLoadForGlobalVariable(llvm::Function &F, llvm::StringRef VarName) {
  auto SizeT = F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
  auto *GV = F.getParent()->getOrInsertGlobal(VarName, SizeT);
  for (auto U : GV->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->getFunction() == &F)
      return LI;
  }

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  return Builder.CreateLoad(SizeT, GV);
}

llvm::LoadInst *mergeGVLoadsInEntry(llvm::Function &F, llvm::StringRef VarName) {
  auto SizeT = F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
  auto *GV = F.getParent()->getOrInsertGlobal(VarName, SizeT);

  llvm::LoadInst *FirstLoad = nullptr;
  llvm::SmallVector<llvm::LoadInst *, 4> Loads;
  for (auto U : GV->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->getFunction() == &F) {
      if (!FirstLoad)
        FirstLoad = LI;
      else
        Loads.push_back(LI);
    }
  }

  if (FirstLoad) {
    FirstLoad->moveBefore(&F.getEntryBlock().front());
    for (auto *LI : Loads) {
      LI->replaceAllUsesWith(FirstLoad);
      LI->eraseFromParent();
    }
    return FirstLoad;
  }

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  auto *Load = Builder.CreateLoad(GV->getType(), GV, "cbs.load." + GV->getName());
  return Load;
}

// parses the range dimensionality from the mangled kernel name
std::size_t getRangeDim(llvm::Function &F, bool IsSscp) {
  auto FName = F.getName();
  // todo: fix with MS mangling
  llvm::Regex Rgx("iterate_nd_range_ompILi([1-3])E");
  llvm::SmallVector<llvm::StringRef, 4> Matches;
  if (Rgx.match(FName, &Matches))
    return std::stoull(static_cast<std::string>(Matches[1]));

  auto GetValue = [](llvm::Constant *Const) -> std::size_t {
    if (auto CI = llvm::dyn_cast<llvm::ConstantInt>(Const))
      return CI->getZExtValue();
    if (auto ZI = llvm::dyn_cast<llvm::ConstantAggregateZero>(Const))
      return 0;
    if (auto CS = llvm::dyn_cast<llvm::ConstantAggregate>(Const)) {
      return llvm::cast<llvm::ConstantInt>(CS->getOperand(0))->getZExtValue();
    }
    llvm_unreachable("[SubCFG] Could not interpret kernel dimensionality value!");
  };

  if (!IsSscp) {
    std::size_t Dim = 0;
    utils::findFunctionsWithStringAnnotationsWithArg(*F.getParent(), [&](llvm::Function *AnnotF,
                                                                         llvm::StringRef Annotation,
                                                                         llvm::Constant *Argument) {
      if (AnnotF == &F) {
        if (Annotation.compare(CbsKernelDimensionName) == 0) {
          if (auto GV = llvm::dyn_cast<llvm::GlobalVariable>(Argument)){
            Dim = GetValue(GV->getInitializer());
          } else
            Dim = GetValue(Argument);
        }
      }
    });
    return Dim;
  } else if (auto MD = F.getParent()->getNamedMetadata(SscpAnnotationsName)) {
    for (auto OP : MD->operands()) {
      if (OP->getNumOperands() == 3 &&
          llvm::cast<llvm::MDString>(OP->getOperand(1))->getString() == SscpKernelDimensionName) {
        if (&F == llvm::dyn_cast<llvm::Function>(
                      llvm::cast<llvm::ValueAsMetadata>(OP->getOperand(0))->getValue())) {
          auto ConstMD = llvm::cast<llvm::ConstantAsMetadata>(OP->getOperand(2))->getValue();
          return GetValue(ConstMD);
        }
      }
    }
  }
  llvm_unreachable("[SubCFG] Could not deduce kernel dimensionality!");
}

// searches for llvm.var.annotation and returns the value that is annotated by it, as well the
// annotation instruction
std::pair<llvm::Value *, llvm::Instruction *>
getLocalSizeArgumentFromAnnotation(llvm::Function &F) {
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *UI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (hipsycl::llvmutils::starts_with(UI->getCalledFunction()->getName(),
                                            "llvm.var.annotation")) {
          HIPSYCL_DEBUG_INFO << *UI << '\n';
          llvm::GlobalVariable *AnnotateStr = nullptr;
          if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(UI->getOperand(1));
              CE && CE->getOpcode() == llvm::Instruction::GetElementPtr) {
            if (auto *AnnoteStr = llvm::dyn_cast<llvm::GlobalVariable>(CE->getOperand(0)))
              AnnotateStr = AnnoteStr;
          } else if (auto *AnnoteStr =
                         llvm::dyn_cast<llvm::GlobalVariable>(UI->getOperand(1))) // opaque-ptr
            AnnotateStr = AnnoteStr;

          if (AnnotateStr) {
            if (auto *Data =
                    llvm::dyn_cast<llvm::ConstantDataSequential>(AnnotateStr->getInitializer())) {
              if (Data->isString() &&
                  hipsycl::llvmutils::starts_with(Data->getAsString(),
                                                  "hipsycl_nd_kernel_local_size_arg")) {
                if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(UI->getOperand(0)))
                  return {BC->getOperand(0), UI};
                return {UI->getOperand(0), UI};
              }
            }
          }
        }

  assert(false && "Didn't find annotated argument!");
  return {nullptr, nullptr};
}

// identify the local size values by the store to it
void fillStores(llvm::Value *V, int Idx, llvm::SmallVector<llvm::Value *, 3> &LocalSize) {
  if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(V)) {
    LocalSize[Idx] = Store->getOperand(0);
  } else if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(V)) {
    for (auto *BCU : BC->users()) {
      fillStores(BCU, Idx, LocalSize);
    }
  } else if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
    auto *IdxV = GEP->indices().begin() + (GEP->getNumIndices() - 1);
    auto *IdxC = llvm::cast<llvm::ConstantInt>(IdxV);
    for (auto *GU : GEP->users()) {
      fillStores(GU, IdxC->getSExtValue(), LocalSize);
    }
  }
}

// reinterpret single argument as array if neccessary and load scalar size values into LocalSize
void loadSizeValuesFromArgument(llvm::Function &F, int Dim, llvm::Value *LocalSizeArg,
                                const llvm::DataLayout &DL,
                                llvm::SmallVector<llvm::Value *, 3> &LocalSize, bool IsSscp) {
  // local_size is just an array of size_t's..
  auto SizeTSize = DL.getLargestLegalIntTypeSizeInBits();
  auto *SizeT = DL.getLargestLegalIntType(F.getContext());

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  llvm::Value *LocalSizePtr = nullptr;
  if (!LocalSizeArg->getType()->isArrayTy()) {
#if HAS_TYPED_PTR
    auto PtrTy = llvm::Type::getIntNPtrTy(F.getContext(), SizeTSize);
#else
    auto PtrTy = llvm::PointerType::get(F.getContext(), 0);
#endif
    LocalSizePtr = Builder.CreatePointerCast(LocalSizeArg, PtrTy, "local_size.cast");
  }
  for (unsigned int I = 0; I < Dim; ++I) {
    auto CurDimName = DimName[IsSscp ? Dim - I - 1 : I];
    if (LocalSizeArg->getType()->isArrayTy()) {
      LocalSize[I] =
          Builder.CreateExtractValue(LocalSizeArg, {I}, "local_size." + llvm::Twine{CurDimName});
    } else {
      auto *LocalSizeGep =
          Builder.CreateInBoundsGEP(SizeT, LocalSizePtr, Builder.getIntN(SizeTSize, I),
                                    "local_size.gep." + llvm::Twine{CurDimName});
      HIPSYCL_DEBUG_INFO << *LocalSizeGep << "\n";

      LocalSize[I] =
          Builder.CreateLoad(SizeT, LocalSizeGep, "local_size." + llvm::Twine{CurDimName});
    }
  }
}

// get the wg size values for the loop bounds
llvm::SmallVector<llvm::Value *, 3> getLocalSizeValues(llvm::Function &F, int Dim,
                                                       bool isSscpKernel) {
  if (isSscpKernel) {
    llvm::SmallVector<llvm::Value *, 3> LocalSize(Dim);
    for (int I = 0; I < Dim; ++I) {
      auto Load = getLoadForGlobalVariable(F, LocalSizeGlobalNames[Dim - I - 1]);
      Load->moveBefore(F.getEntryBlock().getTerminator());
      LocalSize[I] = Load;
    }
    return LocalSize;
  }

  auto &DL = F.getParent()->getDataLayout();
  auto [LocalSizeArg, Annotation] = getLocalSizeArgumentFromAnnotation(F);

  llvm::SmallVector<llvm::Value *, 3> LocalSize(Dim);
  HIPSYCL_DEBUG_INFO << *LocalSizeArg << "\n";

  if (!llvm::dyn_cast<llvm::Argument>(LocalSizeArg))
    for (auto *U : LocalSizeArg->users())
      fillStores(U, 0, LocalSize);
  else
    loadSizeValuesFromArgument(F, Dim, LocalSizeArg, DL, LocalSize, false);

  Annotation->eraseFromParent();
  return LocalSize;
}

std::unique_ptr<hipsycl::compiler::RegionImpl>
getRegion(llvm::Function &F, const llvm::LoopInfo &LI, llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  return std::unique_ptr<hipsycl::compiler::RegionImpl>{
      new hipsycl::compiler::FunctionRegion(F, Blocks)};
}

// calculate uniformity analysis
hipsycl::compiler::VectorizationInfo
getVectorizationInfo(llvm::Function &F, hipsycl::compiler::Region &R, llvm::LoopInfo &LI,
                     llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT, size_t Dim) {
  hipsycl::compiler::VectorizationInfo VecInfo{F, R};
  // seed varyingness
  for (size_t D = 0; D < Dim - 1; ++D) {
    VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, LocalIdGlobalNames[D]),
                           hipsycl::compiler::VectorShape::cont());
  }
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, LocalIdGlobalNames[Dim - 1]),
                         hipsycl::compiler::VectorShape::cont());

  hipsycl::compiler::VectorizationAnalysis VecAna{VecInfo, LI, DT, PDT};
  VecAna.analyze();
  return VecInfo;
}

// create the wi-loops around a kernel or subCFG, LastHeader input should be the load block,
// ContiguousIdx may be any identifyable value (load from undef)
void createLoopsAround(llvm::Function &F, llvm::BasicBlock *AfterBB,
                       const llvm::ArrayRef<llvm::Value *> &LocalSize, int EntryId,
                       llvm::ValueToValueMapTy &VMap,
                       llvm::SmallVector<llvm::BasicBlock *, 3> &Latches,
                       llvm::BasicBlock *&LastHeader, llvm::Value *&ContiguousIdx, bool IsSscp) {
  const auto &DL = F.getParent()->getDataLayout();
  auto *LoadBB = LastHeader;
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};

  const size_t Dim = LocalSize.size();

  std::array<llvm::StringRef, 3> LocalIdGlobalNamesRotated;
  std::array<char, 3> DimNameRotated;
  if (IsSscp)
    for (size_t D = 0; D < Dim; ++D) {
      LocalIdGlobalNamesRotated[D] = LocalIdGlobalNames[Dim - D - 1];
      DimNameRotated[D] = DimName[Dim - D - 1];
    }
  else
    for (size_t D = 0; D < Dim; ++D) {
      LocalIdGlobalNamesRotated[D] = LocalIdGlobalNames[D];
      DimNameRotated[D] = DimName[D];
    }

  // from innermost to outermost: create loops around the LastHeader and use AfterBB as dummy exit
  // to be replaced by the outer latch later
  llvm::SmallVector<llvm::PHINode *, 3> IndVars;
  for (int D = Dim - 1; D >= 0; --D) {
    const std::string Suffix =
        (llvm::Twine{DimNameRotated[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    auto *Header = llvm::BasicBlock::Create(LastHeader->getContext(), "header." + Suffix + "b",
                                            LastHeader->getParent(), LastHeader);

    Builder.SetInsertPoint(Header, Header->getFirstInsertionPt());

    auto *WIIndVar =
        Builder.CreatePHI(DL.getLargestLegalIntType(F.getContext()), 2, "indvar." + Suffix);
    WIIndVar->addIncoming(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 0),
                          &F.getEntryBlock());
    IndVars.push_back(WIIndVar);
    Builder.CreateBr(LastHeader);

    auto *Latch = llvm::BasicBlock::Create(F.getContext(), "latch." + Suffix + "b", &F);
    Builder.SetInsertPoint(Latch, Latch->getFirstInsertionPt());
    auto *IncIndVar =
        Builder.CreateAdd(WIIndVar, Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 1),
                          "addInd." + Suffix, true, false);
    WIIndVar->addIncoming(IncIndVar, Latch);

    auto *LoopCond = Builder.CreateICmpULT(IncIndVar, LocalSize[D], "exit.cond." + Suffix);
    Builder.CreateCondBr(LoopCond, Header, AfterBB);
    Latches.push_back(Latch);
    LastHeader = Header;
  }

  std::reverse(Latches.begin(), Latches.end());
  std::reverse(IndVars.begin(), IndVars.end());

  for (size_t D = 1; D < Dim; ++D) {
    Latches[D]->getTerminator()->replaceSuccessorWith(AfterBB, Latches[D - 1]);
    IndVars[D]->replaceIncomingBlockWith(&F.getEntryBlock(), IndVars[D - 1]->getParent());
  }

  auto *MDWorkItemLoop = llvm::MDNode::get(
      F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::WorkItemLoop)});
  auto *LoopID =
      llvm::makePostTransformationMetadata(F.getContext(), nullptr, {}, {MDWorkItemLoop});
  Latches[Dim - 1]->getTerminator()->setMetadata("llvm.loop", LoopID);
  VMap[AfterBB] = Latches[Dim - 1];

  // add contiguous ind var calculation to load block
  Builder.SetInsertPoint(IndVars[Dim - 1]->getParent(), ++IndVars[Dim - 1]->getIterator());
  llvm::Value *Idx = IndVars[0];
  for (size_t D = 1; D < Dim; ++D) {
    const std::string Suffix =
        (llvm::Twine{DimNameRotated[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    Idx = Builder.CreateMul(Idx, LocalSize[D], "idx.mul." + Suffix, true);
    Idx = Builder.CreateAdd(IndVars[D], Idx, "idx.add." + Suffix, true);

    VMap[mergeGVLoadsInEntry(F, LocalIdGlobalNamesRotated[D])] = IndVars[D];
  }

  // todo: replace `ret` with branch to innermost latch

  VMap[mergeGVLoadsInEntry(F, LocalIdGlobalNamesRotated[0])] = IndVars[0];

  // in case code references all dimensions, we need to set the remaining dimensions to 0
  for (size_t D = Dim; D < 3; ++D) {
    auto ID = mergeGVLoadsInEntry(F, LocalIdGlobalNames[D]);
    ID->replaceAllUsesWith(Builder.getIntN(Idx->getType()->getIntegerBitWidth(), 0));
    ID->eraseFromParent();
  }

  VMap[ContiguousIdx] = Idx;
  ContiguousIdx = Idx;
}

class SubCFG {

  using BlockVector = llvm::SmallVector<llvm::BasicBlock *, 8>;
  BlockVector Blocks_;
  BlockVector NewBlocks_;
  size_t EntryId_;
  llvm::BasicBlock *EntryBarrier_;
  llvm::SmallDenseMap<llvm::BasicBlock *, size_t> ExitIds_;
  llvm::AllocaInst *LastBarrierIdStorage_;
  llvm::Value *ContIdx_;
  llvm::BasicBlock *EntryBB_;
  llvm::BasicBlock *ExitBB_;
  llvm::BasicBlock *LoadBB_;
  llvm::BasicBlock *PreHeader_;
  size_t Dim;

  llvm::BasicBlock *
  createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                   llvm::BasicBlock *After, llvm::BasicBlock *TargetBB);

  void loadMultiSubCfgValues(
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap);
  void loadUniformAndRecalcContValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createLoadBB(llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createUniformLoadBB(llvm::BasicBlock *OuterMostHeader);

public:
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
         const SplitterAnnotationInfo &SAA, llvm::Value *IndVar, size_t Dim);

  SubCFG(const SubCFG &) = delete;
  SubCFG &operator=(const SubCFG &) = delete;

  SubCFG(SubCFG &&) = default;
  SubCFG &operator=(SubCFG &&) = default;

  BlockVector &getBlocks() noexcept { return Blocks_; }
  const BlockVector &getBlocks() const noexcept { return Blocks_; }

  BlockVector &getNewBlocks() noexcept { return NewBlocks_; }
  const BlockVector &getNewBlocks() const noexcept { return NewBlocks_; }

  size_t getEntryId() const noexcept { return EntryId_; }

  llvm::BasicBlock *getEntry() noexcept { return EntryBB_; }
  llvm::BasicBlock *getExit() noexcept { return ExitBB_; }
  llvm::BasicBlock *getLoadBB() noexcept { return LoadBB_; }
  llvm::Value *getContiguousIdx() noexcept { return ContIdx_; }

  void replicate(llvm::Function &F,
                 const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
                     &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                 llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, bool IsSscp);

  void arrayifyMultiSubCfgValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements,
      VectorizationInfo &VecInfo);
  void fixSingleSubCfgValues(
      llvm::DominatorTree &DT,
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
      llvm::Value *ReqdArrayElements, VectorizationInfo &VecInfo);

  void print() const;
  void removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const;
  llvm::SmallVector<llvm::Instruction *, 16>
  topoSortInstructions(const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const;
};

// create new exiting block writing the exit's id to LastBarrierIdStorage_
llvm::BasicBlock *
SubCFG::createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                         llvm::BasicBlock *After, llvm::BasicBlock *TargetBB) {
  HIPSYCL_DEBUG_INFO << "Create new exit with ID: " << BarrierPair.second << " at "
                     << After->getName() << "\n";

  auto *Exit = llvm::BasicBlock::Create(After->getContext(),
                                        After->getName() + ".subcfg.exit" +
                                            llvm::Twine{BarrierPair.second} + "b",
                                        After->getParent(), TargetBB);

  auto &DL = Exit->getParent()->getParent()->getDataLayout();
  llvm::IRBuilder Builder{Exit, Exit->getFirstInsertionPt()};
  Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), BarrierPair.second),
                      LastBarrierIdStorage_);
  Builder.CreateBr(TargetBB);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

// identify a new SubCFG using DFS starting at EntryBarrier
SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
               const SplitterAnnotationInfo &SAA, llvm::Value *IndVar, size_t Dim)
    : EntryId_(BarrierIds.lookup(EntryBarrier)), EntryBarrier_(EntryBarrier),
      LastBarrierIdStorage_(LastBarrierIdStorage), ContIdx_(IndVar),
      EntryBB_(EntryBarrier->getSingleSuccessor()), LoadBB_(nullptr), PreHeader_(nullptr),
      Dim(Dim) {
  assert(ContIdx_ && "Must have found __acpp_cbs_local_id_{x,y,z}");

  llvm::SmallVector<llvm::BasicBlock *, 4> WL{EntryBarrier};
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();

    llvm::SmallVector<llvm::BasicBlock *, 2> Succs{llvm::succ_begin(BB), llvm::succ_end(BB)};
    for (auto *Succ : Succs) {
      if (std::find(Blocks_.begin(), Blocks_.end(), Succ) != Blocks_.end())
        continue;

      if (!utils::hasOnlyBarrier(Succ, SAA)) {
        WL.push_back(Succ);
        Blocks_.push_back(Succ);
      } else {
        size_t BId = BarrierIds.lookup(Succ);
        assert(BId != 0 && "Exit barrier block not found in map");
        ExitIds_.insert({Succ, BId});
      }
    }
  }
}

void SubCFG::print() const {
  HIPSYCL_DEBUG_INFO << "SubCFG entry barrier: " << EntryId_ << "\n";
  HIPSYCL_DEBUG_INFO << "SubCFG block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : Blocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG exits: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto ExitIt
                                  : ExitIds_) {
    llvm::outs() << ExitIt.first->getName() << " (" << ExitIt.second << "), ";
  } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG new block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : NewBlocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
}

void addRemappedDenseMapKeys(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &OrgInstAllocaMap,
    const llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &NewInstAllocaMap) {
  for (auto &InstAllocaPair : OrgInstAllocaMap) {
    if (auto *NewInst =
            llvm::dyn_cast_or_null<llvm::Instruction>(VMap.lookup(InstAllocaPair.first)))
      NewInstAllocaMap.insert({NewInst, InstAllocaPair.second});
  }
}

// clone all BBs of the subcfg, create wi-loop structure around and fixup values
void SubCFG::replicate(
    llvm::Function &F, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
    llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, bool IsSscp) {
  llvm::ValueToValueMapTy VMap;

  // clone blocks
  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", &F);
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      if (auto ExitIt = ExitIds_.find(Succ); ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, AfterBB));
      }
    }
  }

  LoadBB_ = createLoadBB(VMap);

  VMap[EntryBarrier_] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  llvm::BasicBlock *LastHeader = LoadBB_;
  llvm::Value *Idx = ContIdx_;

  createLoopsAround(F, AfterBB, LocalSize, EntryId_, VMap, Latches, LastHeader, Idx, IsSscp);

  PreHeader_ = createUniformLoadBB(LastHeader);
  LastHeader->replacePhiUsesWith(&F.getEntryBlock(), PreHeader_);

  print();

  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap);
  loadUniformAndRecalcContValues(BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = Latches[0];
  ContIdx_ = Idx;
}

// remove incoming PHI blocks that no longer actually have an edge to the PHI
void SubCFG::removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const {
  for (auto *BB : BlocksToRemap) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> Predecessors{llvm::pred_begin(BB), llvm::pred_end(BB)};
    for (auto &I : *BB) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
        llvm::SmallVector<llvm::BasicBlock *, 4> IncomingBlocksToRemove;
        for (size_t IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues(); ++IncomingIdx) {
          auto *IncomingBB = Phi->getIncomingBlock(IncomingIdx);
          if (!Predecessors.contains(IncomingBB))
            IncomingBlocksToRemove.push_back(IncomingBB);
        }
        for (auto *IncomingBB : IncomingBlocksToRemove) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Remove incoming block " << IncomingBB->getName()
                             << " from PHI " << *Phi << "\n";
          Phi->removeIncomingValue(IncomingBB);
          HIPSYCL_DEBUG_INFO << "[SubCFG] Removed incoming block " << IncomingBB->getName()
                             << " from PHI " << *Phi << "\n";
        }
      }
    }
  }
}

// check if a contiguous value can be tracked back to only uniform values and the wi-loop indvar
// currently cannot track back the value through PHI nodes.
bool dontArrayifyContiguousValues(
    llvm::Instruction &I,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements, llvm::Value *IndVar,
    VectorizationInfo &VecInfo) {
  // is cont indvar
  if (VecInfo.isPinned(I))
    return true;

  llvm::SmallVector<llvm::Instruction *, 4> WL;
  llvm::SmallPtrSet<llvm::Instruction *, 8> UniformValues;
  llvm::SmallVector<llvm::Instruction *, 8> ContiguousInsts;
  llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;
  HIPSYCL_DEBUG_INFO << "[SubCFG] Cont value? " << I << " IndVar: " << *IndVar << "\n";
  WL.push_back(&I);
  while (!WL.empty()) {
    auto *WLValue = WL.pop_back_val();
    if (auto *WLI = llvm::dyn_cast<llvm::Instruction>(WLValue))
      for (auto *V : WLI->operand_values()) {
        HIPSYCL_DEBUG_INFO << "[SubCFG] Considering: " << *V << "\n";

        if (V == IndVar || VecInfo.isPinned(*V) || llvm::isa<llvm::Constant>(V))
          continue;
        // todo: fix PHIs
        if (LookedAt.contains(V))
          return false;
        LookedAt.insert(V);

        // collect cont and uniform source values
        if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
          if (VecInfo.getVectorShape(*OpI).isContiguousOrStrided()) {
            WL.push_back(OpI);
            ContiguousInsts.push_back(OpI);
          } else if (!UniformValues.contains(OpI))
            UniformValues.insert(OpI);
        }
      }
  }
  for (auto *UI : UniformValues) {
    HIPSYCL_DEBUG_INFO << "[SubCFG] UniValue to store: " << *UI << "\n";
    if (BaseInstAllocaMap.lookup(UI))
      continue;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Store required uniform value to single element alloca " << I
                       << "\n";
    auto *Alloca = utils::arrayifyInstruction(AllocaIP, UI, IndVar, nullptr);
    BaseInstAllocaMap.insert({UI, Alloca});
    VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
  }
  ContInstReplicaMap.insert({&I, ContiguousInsts});
  return true;
}

// creates array allocas for values that are identified as spanning multiple subcfgs
void SubCFG::arrayifyMultiSubCfgValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements,
    VectorizationInfo &VecInfo) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> OtherCFGBlocks;
  for (auto &Cfg : SubCFGs) {
    if (&Cfg != this)
      OtherCFGBlocks.insert(Cfg.Blocks_.begin(), Cfg.Blocks_.end());
  }

  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (&I == ContIdx_)
        continue;
      if (InstAllocaMap.lookup(&I))
        continue;
      // if any use is in another subcfg
      if (utils::anyOfUsers<llvm::Instruction>(&I, [&OtherCFGBlocks, &I](auto *UI) {
            return (UI->getParent() != I.getParent() ||
                    UI->getParent() == I.getParent() && UI->comesBefore(&I)) &&
                   OtherCFGBlocks.contains(UI->getParent());
          })) {
        // load from an alloca, just widen alloca
        if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(&I))
          if (auto *Alloca = utils::getLoopStateAllocaForLoad(*LInst)) {
            InstAllocaMap.insert({&I, Alloca});
            continue;
          }
        // GEP from already widened alloca: reuse alloca
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (GEP->hasMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            InstAllocaMap.insert({&I, llvm::cast<llvm::AllocaInst>(GEP->getPointerOperand())});
            continue;
          }

        auto Shape = VecInfo.getVectorShape(I);
#ifndef HIPSYCL_NO_PHIS_IN_SPLIT
        // if value is uniform, just store to 1-wide alloca
        if (Shape.isUniform()) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Value uniform, store to single element alloca " << I
                             << "\n";
          auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContIdx_, nullptr);
          InstAllocaMap.insert({&I, Alloca});
          VecInfo.setVectorShape(*Alloca, VectorShape::uni());
          continue;
        }
#endif
        // if contiguous, and can be recalculated, don't arrayify but store
        // uniform values and insts required for recalculation
        if (Shape.isContiguousOrStrided()) {
          if (dontArrayifyContiguousValues(I, BaseInstAllocaMap, ContInstReplicaMap, AllocaIP,
                                           ReqdArrayElements, ContIdx_, VecInfo)) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying " << I << "\n";
            continue;
          }
        }
        // create wide alloca and store the value
        auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContIdx_, ReqdArrayElements);
        InstAllocaMap.insert({&I, Alloca});
        VecInfo.setVectorShape(*Alloca, Shape);
      }
    }
  }
}

void remapInstruction(llvm::Instruction *I, llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<llvm::Value *, 8> WL{I->value_op_begin(), I->value_op_end()};
  for (auto *V : WL) {
    if (VMap.count(V))
      I->replaceUsesOfWith(V, VMap[V]);
  }
  HIPSYCL_DEBUG_INFO << "[SubCFG] remapped Inst " << *I << "\n";
}

// inserts loads from the loop state allocas for varying values that were identified as
// multi-subcfg values
void SubCFG::loadMultiSubCfgValues(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap) {
  llvm::Value *NewContIdx = VMap[ContIdx_];
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::IRBuilder Builder{LoadTerm};

  for (auto &InstAllocaPair : InstAllocaMap) {
    // If def not in sub CFG but a use of it is in the sub CFG
    if (std::find(Blocks_.begin(), Blocks_.end(), InstAllocaPair.first->getParent()) ==
        Blocks_.end()) {
      if (utils::anyOfUsers<llvm::Instruction>(InstAllocaPair.first, [this](llvm::Instruction *UI) {
            return std::find(NewBlocks_.begin(), NewBlocks_.end(), UI->getParent()) !=
                   NewBlocks_.end();
          })) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(InstAllocaPair.first))
          if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                GEP->getType(), GEP->getPointerOperand(), NewContIdx, GEP->getName() + "c"));
            NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
            VMap[InstAllocaPair.first] = NewGEP;
            continue;
          }
        auto *IP = LoadTerm;
        if (!InstAllocaPair.second->isArrayAllocation())
          IP = UniformLoadTerm;
        HIPSYCL_DEBUG_INFO << "[SubCFG] Load from Alloca " << *InstAllocaPair.second << " in "
                           << IP->getParent()->getName() << "\n";
        auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewContIdx, IP,
                                           InstAllocaPair.first->getName());
        utils::copyDgbValues(InstAllocaPair.first, Load, IP);
        VMap[InstAllocaPair.first] = Load;
      }
    }
  }
}

// Inserts loads for the multi-subcfg values that were identified as uniform
// inside the wi-loop preheader. Additionally clones the instructions that were
// identified as contiguous \a ContInstReplicaMap inside the LoadBB_ to restore
// the contiguous value just from the uniform values and the wi-idx.
void SubCFG::loadUniformAndRecalcContValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap) {
  llvm::ValueToValueMapTy UniVMap;
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::Value *NewContIdx = VMap[this->ContIdx_];
  UniVMap[this->ContIdx_] = NewContIdx;

  // copy local id load value to univmap
  for (size_t D = 0; D < this->Dim; ++D) {
    auto *Load = getLoadForGlobalVariable(*this->LoadBB_->getParent(), LocalIdGlobalNames[D]);
    UniVMap[Load] = VMap[Load];
  }

  // load uniform values from allocas
  for (auto &InstAllocaPair : BaseInstAllocaMap) {
    auto *IP = UniformLoadTerm;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Load base value from Alloca " << *InstAllocaPair.second
                       << " in " << IP->getParent()->getName() << "\n";
    auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewContIdx, IP,
                                       InstAllocaPair.first->getName());
    utils::copyDgbValues(InstAllocaPair.first, Load, IP);
    UniVMap[InstAllocaPair.first] = Load;
  }

  // get a set of unique contiguous instructions
  llvm::SmallPtrSet<llvm::Instruction *, 16> UniquifyInsts;
  for (auto &Pair : ContInstReplicaMap) {
    UniquifyInsts.insert(Pair.first);
    for (auto &Target : Pair.second)
      UniquifyInsts.insert(Target);
  }

  auto OrderedInsts = topoSortInstructions(UniquifyInsts);

  llvm::SmallPtrSet<llvm::Instruction *, 16> InstsToRemap;
  // clone the contiguous instructions to restore the used values
  for (auto *I : OrderedInsts) {
    if (UniVMap.count(I))
      continue;

    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction and operands of: " << *I << " to "
                       << LoadTerm->getParent()->getName() << "\n";
    auto *IClone = I->clone();
    IClone->insertBefore(LoadTerm);
    InstsToRemap.insert(IClone);
    UniVMap[I] = IClone;
    if (VMap.count(I) == 0)
      VMap[I] = IClone;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction: " << *IClone << "\n";
  }

  // finally remap the singular instructions to use the other cloned contiguous instructions /
  // uniform values
  for (auto *IToRemap : InstsToRemap)
    remapInstruction(IToRemap, UniVMap);
}
llvm::SmallVector<llvm::Instruction *, 16> SubCFG::topoSortInstructions(
    const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const {
  llvm::SmallVector<llvm::Instruction *, 16> OrderedInsts(UniquifyInsts.size());
  std::copy(UniquifyInsts.begin(), UniquifyInsts.end(), OrderedInsts.begin());

  auto IsUsedBy = [](llvm::Instruction *LHS, llvm::Instruction *RHS) {
    for (auto *U : LHS->users()) {
      if (U == RHS)
        return true;
    }
    return false;
  };
  for (int I = 0; I < OrderedInsts.size(); ++I) {
    int InsertAt = I;
    for (int J = OrderedInsts.size() - 1; J > I; --J) {
      if (IsUsedBy(OrderedInsts[J], OrderedInsts[I])) {
        InsertAt = J;
        break;
      }
    }
    if (InsertAt != I) {
      auto *Tmp = OrderedInsts[I];
      for (int J = I + 1; J <= InsertAt; ++J) {
        OrderedInsts[J - 1] = OrderedInsts[J];
      }
      OrderedInsts[InsertAt] = Tmp;
      --I;
    }
  }
  return OrderedInsts;
}

llvm::BasicBlock *SubCFG::createUniformLoadBB(llvm::BasicBlock *OuterMostHeader) {
  auto *LoadBB = llvm::BasicBlock::Create(OuterMostHeader->getContext(),
                                          "uniloadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          OuterMostHeader->getParent(), OuterMostHeader);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(OuterMostHeader);
  return LoadBB;
}

llvm::BasicBlock *SubCFG::createLoadBB(llvm::ValueToValueMapTy &VMap) {
  auto *NewEntry = llvm::cast<llvm::BasicBlock>(static_cast<llvm::Value *>(VMap[EntryBB_]));
  auto *LoadBB = llvm::BasicBlock::Create(NewEntry->getContext(),
                                          "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          NewEntry->getParent(), NewEntry);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(NewEntry);
  return LoadBB;
}

// if the kernel contained a loop, it is possible, that values inside a single
// subcfg don't dominate their uses inside the same subcfg. This function
// identifies and fixes those values.
void SubCFG::fixSingleSubCfgValues(
    llvm::DominatorTree &DT,
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
    llvm::Value *ReqdArrayElements, VectorizationInfo &VecInfo) {

  auto *AllocaIP = LoadBB_->getParent()->getEntryBlock().getTerminator();
  auto *LoadIP = LoadBB_->getTerminator();
  auto *UniLoadIP = PreHeader_->getTerminator();
  llvm::IRBuilder Builder{LoadIP};

  llvm::DenseMap<llvm::Instruction *, llvm::Instruction *> InstLoadMap;

  for (auto *BB : NewBlocks_) {
    llvm::SmallVector<llvm::Instruction *, 16> Insts{};
    std::transform(BB->begin(), BB->end(), std::back_inserter(Insts), [](auto &I) { return &I; });
    for (auto *Inst : Insts) {
      auto &I = *Inst;
      for (auto *OPV : I.operand_values()) {
        // check if all operands dominate the instruction -> otherwise we have to fix it
        if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV); OPI && !DT.dominates(OPI, &I)) {
          if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(Inst)) {
            // if a PHI node, we have to check that the incoming values dominate the terminators
            // of the incoming block..
            bool FoundIncoming = false;
            for (auto &Incoming : Phi->incoming_values()) {
              if (OPV == Incoming.get()) {
                auto *IncomingBB = Phi->getIncomingBlock(Incoming);
                if (DT.dominates(OPI, IncomingBB->getTerminator())) {
                  FoundIncoming = true;
                  break;
                }
              }
            }
            if (FoundIncoming)
              continue;
          }
          HIPSYCL_DEBUG_INFO << "Instruction not dominated " << I << " operand: " << *OPI << "\n";

          if (auto *Load = InstLoadMap.lookup(OPI))
            // if the already inserted Load does not dominate I, we must create another load.
            if (DT.dominates(Load, &I)) {
              I.replaceUsesOfWith(OPI, Load);
              continue;
            }

          if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(OPI))
            if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
              auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                  GEP->getType(), GEP->getPointerOperand(), ContIdx_, GEP->getName() + "c"));
              NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
              I.replaceUsesOfWith(OPI, NewGEP);
              InstLoadMap.insert({OPI, NewGEP});
              continue;
            }

          llvm::AllocaInst *Alloca = nullptr;
          if (auto *RemAlloca = RemappedInstAllocaMap.lookup(OPI))
            Alloca = RemAlloca;
          if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(OPI))
            Alloca = utils::getLoopStateAllocaForLoad(*LInst);
          if (!Alloca) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] No alloca, yet for " << *OPI << "\n";
            Alloca = utils::arrayifyInstruction(
                AllocaIP, OPI, ContIdx_,
                VecInfo.getVectorShape(I).isUniform() ? nullptr : ReqdArrayElements);
            VecInfo.setVectorShape(*Alloca, VecInfo.getVectorShape(I));
          }

          auto Idx = ContIdx_;
#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          // in split loop, OPI might be used multiple times, get the user, dominating this user
          // and insert load there
          llvm::Instruction *NewIP = &I;
          for (auto *U : OPI->users()) {
            if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U); UI && DT.dominates(UI, NewIP)) {
              NewIP = UI;
            }
          }
#else
          // doesn't happen if we keep the PHIs
          auto *NewIP = LoadIP;
          if (!Alloca->isArrayAllocation()) {
            NewIP = UniLoadIP;
            Idx = llvm::ConstantInt::get(ContIdx_->getType(), 0);
          }
#endif

          auto *Load = utils::loadFromAlloca(Alloca, Idx, NewIP, OPI->getName());
          utils::copyDgbValues(OPI, Load, NewIP);

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          I.replaceUsesOfWith(OPI, Load);
          InstLoadMap.insert({OPI, Load});
#else
          // if a loop is conditionally split, the first block in a subcfg might have another
          // incoming edge, need to insert a PHI node then
          const auto NumPreds = std::distance(llvm::pred_begin(BB), llvm::pred_end(BB));
          if (!llvm::isa<llvm::PHINode>(I) && NumPreds > 1 &&
              std::find(llvm::pred_begin(BB), llvm::pred_end(BB), LoadBB_) != llvm::pred_end(BB)) {
            Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
            auto *PHINode = Builder.CreatePHI(Load->getType(), NumPreds, I.getName());
            for (auto *PredBB : llvm::predecessors(BB))
              if (PredBB == LoadBB_)
                PHINode->addIncoming(Load, PredBB);
              else
                PHINode->addIncoming(OPV, PredBB);

            I.replaceUsesOfWith(OPI, PHINode);
            InstLoadMap.insert({OPI, PHINode});
          } else {
            I.replaceUsesOfWith(OPI, Load);
            InstLoadMap.insert({OPI, Load});
          }
#endif
        }
      }
    }
  }
}

llvm::BasicBlock *createUnreachableBlock(llvm::Function &F) {
  auto *Default = llvm::BasicBlock::Create(F.getContext(), "cbs.while.default", &F);
  llvm::IRBuilder Builder{Default, Default->getFirstInsertionPt()};
  Builder.CreateUnreachable();
  return Default;
}

// create the actual while loop around the subcfgs and the switch instruction to
// select the next subCFG based on the value in \a LastBarrierIdStorage
llvm::BasicBlock *generateWhileSwitchAround(llvm::BasicBlock *PreHeader, llvm::BasicBlock *OldEntry,
                                            llvm::BasicBlock *Exit,
                                            llvm::AllocaInst *LastBarrierIdStorage,
                                            std::vector<SubCFG> &SubCFGs) {
  auto &F = *PreHeader->getParent();
  auto &M = *F.getParent();
  const auto &DL = M.getDataLayout();

  auto *WhileHeader = llvm::BasicBlock::Create(PreHeader->getContext(), "cbs.while.header",
                                               PreHeader->getParent(), OldEntry);
  llvm::IRBuilder Builder{WhileHeader, WhileHeader->getFirstInsertionPt()};
  auto *LastID = Builder.CreateLoad(LastBarrierIdStorage->getAllocatedType(), LastBarrierIdStorage,
                                    "cbs.while.last_barr.load");
  auto *Switch = Builder.CreateSwitch(LastID, createUnreachableBlock(F), SubCFGs.size());
  for (auto &Cfg : SubCFGs) {
    Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), Cfg.getEntryId()),
                    Cfg.getEntry());
    Cfg.getEntry()->replacePhiUsesWith(PreHeader, WhileHeader);
    Cfg.getExit()->getTerminator()->replaceSuccessorWith(Exit, WhileHeader);
  }
  Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), ExitBarrierId), Exit);

  Builder.SetInsertPoint(PreHeader->getTerminator());
  Builder.CreateStore(
      llvm::ConstantInt::get(LastBarrierIdStorage->getAllocatedType(), EntryBarrierId),
      LastBarrierIdStorage);
  PreHeader->getTerminator()->replaceSuccessorWith(OldEntry, WhileHeader);
  return WhileHeader;
}

// drops all lifetime intrinsics - they are misinforming ASAN otherwise (and are
// not really fixable at the right scope..)
void purgeLifetime(SubCFG &Cfg) {
  llvm::SmallVector<llvm::Instruction *, 8> ToDelete;
  for (auto *BB : Cfg.getNewBlocks())
    for (auto &I : *BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction())
          if (CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
              CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_end)
            ToDelete.push_back(CI);

  for (auto *I : ToDelete)
    I->eraseFromParent();
}

// fills \a Hull with all transitive users of \a Alloca
void fillUserHull(llvm::AllocaInst *Alloca, llvm::SmallVectorImpl<llvm::Instruction *> &Hull) {
  llvm::SmallVector<llvm::Instruction *, 8> WL;
  std::transform(Alloca->user_begin(), Alloca->user_end(), std::back_inserter(WL),
                 [](auto *U) { return llvm::cast<llvm::Instruction>(U); });
  llvm::SmallPtrSet<llvm::Instruction *, 32> AlreadySeen;
  while (!WL.empty()) {
    auto *I = WL.pop_back_val();
    AlreadySeen.insert(I);
    Hull.push_back(I);
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!AlreadySeen.contains(UI))
          if (UI->mayReadOrWriteMemory() || UI->getType()->isPointerTy())
            WL.push_back(UI);
      }
    }
  }
}

// checks if all uses of an alloca are in just a single subcfg (doesn't have to be arrayified!)
bool isAllocaSubCfgInternal(llvm::AllocaInst *Alloca, const std::vector<SubCFG> &SubCfgs,
                            const llvm::DominatorTree &DT) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> UserBlocks;
  {
    llvm::SmallVector<llvm::Instruction *, 32> Users;
    fillUserHull(Alloca, Users);
    utils::PtrSetWrapper<decltype(UserBlocks)> Wrapper{UserBlocks};
    std::transform(Users.begin(), Users.end(), std::inserter(Wrapper, UserBlocks.end()),
                   [](auto *I) { return I->getParent(); });
  }

  for (auto &SubCfg : SubCfgs) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> SubCfgSet{SubCfg.getNewBlocks().begin(),
                                                       SubCfg.getNewBlocks().end()};
    if (std::any_of(UserBlocks.begin(), UserBlocks.end(),
                    [&SubCfgSet](auto *BB) { return SubCfgSet.contains(BB); }) &&
        !std::all_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet, Alloca](auto *BB) {
          if (SubCfgSet.contains(BB)) {
            return true;
          }
          HIPSYCL_DEBUG_INFO << "[SubCFG] BB not in subcfgset: " << BB->getName()
                             << " for alloca: ";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alloca->print(llvm::outs()); llvm::outs() << "\n";)
          return false;
        }))
      return false;
  }

  return true;
}

// Widens the allocas in the entry block to array allocas.
// Replace uses of the original alloca with GEP that indexes the new alloca with
// \a Idx.
void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT,
                     std::vector<SubCFG> &SubCfgs, llvm::Value *ReqdArrayElements,
                     VectorizationInfo &VecInfo) {
  auto *MDAlloca = llvm::MDNode::get(
      EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), MDKind::LoopState)});

  llvm::SmallPtrSet<llvm::BasicBlock *, 32> SubCfgsBlocks;
  for (auto &SubCfg : SubCfgs)
    SubCfgsBlocks.insert(SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end());

  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
        continue; // already arrayified
      if (utils::anyOfUsers<llvm::Instruction>(Alloca, [&SubCfgsBlocks](llvm::Instruction *UI) {
            return !SubCfgsBlocks.contains(UI->getParent());
          }))
        continue;
      if (!isAllocaSubCfgInternal(Alloca, SubCfgs, DT))
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

    auto *Alloca = AllocaBuilder.CreateAlloca(T, ReqdArrayElements, I->getName() + "_alloca");
    Alloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
    Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

    for (auto &SubCfg : SubCfgs) {
      auto *GepIp = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();

      llvm::IRBuilder LoadBuilder{GepIp};
      auto *GEP = llvm::cast<llvm::GetElementPtrInst>(LoadBuilder.CreateInBoundsGEP(
          Alloca->getAllocatedType(), Alloca, SubCfg.getContiguousIdx(), I->getName() + "_gep"));
      GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      llvm::replaceDominatedUsesWith(I, GEP, DT, SubCfg.getLoadBB());
    }
    I->eraseFromParent();
  }
}

void moveAllocasToEntry(llvm::Function &F, llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  llvm::SmallVector<llvm::AllocaInst *, 4> AllocaWL;
  for (auto *BB : Blocks)
    for (auto &I : *BB)
      if (auto *AllocaInst = llvm::dyn_cast<llvm::AllocaInst>(&I))
        AllocaWL.push_back(AllocaInst);
  for (auto *I : AllocaWL)
    I->moveBefore(F.getEntryBlock().getTerminator());
}

llvm::DenseMap<llvm::BasicBlock *, size_t>
getBarrierIds(llvm::BasicBlock *Entry, llvm::SmallPtrSetImpl<llvm::BasicBlock *> &ExitingBlocks,
              llvm::ArrayRef<llvm::BasicBlock *> Blocks, const SplitterAnnotationInfo &SAA) {
  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;
  // mark exit barrier with the corresponding id:
  for (auto *BB : ExitingBlocks)
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[Entry] = EntryBarrierId;

  // store all other barrier blocks with a unique id:
  size_t BarrierId = 1;
  for (auto *BB : Blocks)
    if (Barriers.find(BB) == Barriers.end() && utils::hasOnlyBarrier(BB, SAA))
      Barriers.insert({BB, BarrierId++});
  return Barriers;
}

void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                 llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, bool IsSscp) {
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  const std::size_t Dim = getRangeDim(F, IsSscp);
  HIPSYCL_DEBUG_INFO << "[SubCFG] Kernel is " << Dim << "-dimensional\n";

  const auto LocalSize = getLocalSizeValues(F, Dim, IsSscp);

  auto *Entry = &F.getEntryBlock();

  llvm::IRBuilder Builder{Entry->getTerminator()};
  llvm::Value *ReqdArrayElements = LocalSize[0];
  for (size_t D = 1; D < LocalSize.size(); ++D)
    ReqdArrayElements = Builder.CreateMul(ReqdArrayElements, LocalSize[D]);

  std::vector<llvm::BasicBlock *> Blocks;
  Blocks.reserve(std::distance(F.begin(), F.end()));
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });

  // non-entry block Allocas are considered broken, move to entry.
  moveAllocasToEntry(F, Blocks);

  auto RImpl = getRegion(F, LI, Blocks);
  hipsycl::compiler::Region R{*RImpl};
  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, Dim);

  llvm::SmallPtrSet<llvm::BasicBlock *, 2> ExitingBlocks;
  R.getEndingBlocks(ExitingBlocks);

  if (ExitingBlocks.empty()) {
    HIPSYCL_DEBUG_ERROR << "[SubCFG] Invalid kernel! No kernel exits!\n";
    llvm_unreachable("[SubCFG] Invalid kernel! No kernel exits!\n");
  }

  auto Barriers = getBarrierIds(Entry, ExitingBlocks, Blocks, SAA);

  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  auto *LastBarrierIdStorage =
      Builder.CreateAlloca(DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");

  // get a common (pseudo) index value to be replaced by the actual index later
  Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
  auto *IndVarT = getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
  llvm::Instruction *IndVar =
      Builder.CreateLoad(IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)));
  // kept for simple reenabling of more advanced uniformity analysis
  VecInfo.setPinnedShape(*IndVar, hipsycl::compiler::VectorShape::cont());

  // create subcfgs
  std::vector<SubCFG> SubCFGs;
  for (auto &BIt : Barriers) {
    HIPSYCL_DEBUG_INFO << "Create SubCFG from " << BIt.first->getName() << "(" << BIt.first
                       << ") id: " << BIt.second << "\n";
    if (BIt.second != ExitBarrierId)
      SubCFGs.emplace_back(BIt.first, LastBarrierIdStorage, Barriers, SAA, IndVar, Dim);
  }

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> BaseInstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> InstContReplicaMap;

  for (auto &Cfg : SubCFGs)
    Cfg.arrayifyMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, SubCFGs,
                                  F.getEntryBlock().getTerminator(), ReqdArrayElements, VecInfo);

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> RemappedInstAllocaMap;
  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    Cfg.replicate(F, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap,
                  *ExitingBlocks.begin(), LocalSize, IsSscp);
    purgeLifetime(Cfg);
  }

  llvm::BasicBlock *WhileHeader = nullptr;
  WhileHeader =
      generateWhileSwitchAround(&F.getEntryBlock(), F.getEntryBlock().getSingleSuccessor(),
                                *ExitingBlocks.begin(), LastBarrierIdStorage, SubCFGs);

  llvm::removeUnreachableBlocks(F);

  DT.recalculate(F);
  arrayifyAllocas(&F.getEntryBlock(), DT, SubCFGs, ReqdArrayElements, VecInfo);

  for (auto &Cfg : SubCFGs) {
    Cfg.fixSingleSubCfgValues(DT, RemappedInstAllocaMap, ReqdArrayElements, VecInfo);
  }

  IndVar->eraseFromParent();

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  assert(!llvm::verifyFunction(F, &llvm::errs()) && "Function verification failed");

  // simplify while loop to get single latch that isn't marked as wi-loop to prevent
  // misunderstandings.
  auto *WhileLoop = utils::updateDtAndLi(LI, DT, WhileHeader, F);
  llvm::simplifyLoop(WhileLoop, &DT, &LI, nullptr, nullptr, nullptr, false);
}

void createLoopsAroundKernel(llvm::Function &F, llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                             llvm::PostDominatorTree &PDT, bool IsSscp) {

  auto *Body = llvm::SplitBlock(&F.getEntryBlock(), &*F.getEntryBlock().getFirstInsertionPt(), &DT,
                                &LI, nullptr, "wibody", true);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG());

  Body = Body->getSingleSuccessor();

  llvm::SmallVector<llvm::BasicBlock *, 4> ExitBBs;
  llvm::BasicBlock *ExitBB = llvm::BasicBlock::Create(F.getContext(), "exit", &F);
  llvm::IRBuilder<> Bld{ExitBB};
  Bld.CreateRetVoid();
  for (auto &BB : F) {
    if (BB.getTerminator()->getNumSuccessors() == 0 &&
        !llvm::isa<llvm::UnreachableInst>(BB.getTerminator()) && &BB != ExitBB) {
      auto *oldTerm = BB.getTerminator();
      Bld.SetInsertPoint(oldTerm);
      Bld.CreateBr(ExitBB);
      oldTerm->eraseFromParent();
    }
  }

  llvm::SmallVector<llvm::BasicBlock *, 8> Blocks{};
  Blocks.reserve(std::distance(F.begin(), F.end()));
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });

  moveAllocasToEntry(F, Blocks);

  const auto Dim = getRangeDim(F, IsSscp);

  // insert dummy induction variable that can be easily identified and replaced later
  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  auto *IndVarT = getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
  llvm::Value *Idx =
      Builder.CreateLoad(IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)));

  auto LocalSize = getLocalSizeValues(F, Dim, IsSscp);

  llvm::ValueToValueMapTy VMap;
  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  auto *LastHeader = Body;

  createLoopsAround(F, ExitBB, LocalSize, 0, VMap, Latches, LastHeader, Idx, IsSscp);

  F.getEntryBlock().getTerminator()->setSuccessor(0, LastHeader);
  llvm::remapInstructionsInBlocks(Blocks, VMap);

  // remove uses of the undefined global id variables
  for (int D = 0; D < 3; ++D)
    if (auto *Load =
            llvm::cast_or_null<llvm::LoadInst>(mergeGVLoadsInEntry(F, LocalIdGlobalNames[D]))) {
      if (D >= Dim)
        Load->replaceAllUsesWith(llvm::ConstantInt::get(Load->getType(), 0));
      Load->eraseFromParent();
    }
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG())
}

} // namespace

namespace hipsycl::compiler {
void SubCfgFormationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequiredTransitive<llvm::DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<llvm::PostDominatorTreeWrapperPass>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool SubCfgFormationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F) || getRangeDim(F, false) == 0)
    return false;

  HIPSYCL_DEBUG_INFO << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  if (utils::hasBarriers(F, SAA))
    formSubCfgs(F, LI, DT, PDT, SAA, false);
  else
    createLoopsAroundKernel(F, DT, LI, PDT, false);

  return true;
}

char SubCfgFormationPassLegacy::ID = 0;

llvm::PreservedAnalyses SubCfgFormationPass::run(llvm::Function &F,
                                                 llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());

  if (!SAA || !SAA->isKernelFunc(&F) || getRangeDim(F, IsSscp_) == 0)
    return llvm::PreservedAnalyses::all();

  HIPSYCL_DEBUG_INFO << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  if (utils::hasBarriers(F, *SAA))
    formSubCfgs(F, LI, DT, PDT, *SAA, IsSscp_);
  else
    createLoopsAroundKernel(F, DT, LI, PDT, IsSscp_);

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
