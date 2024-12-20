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
#include "hipSYCL/compiler/sscp/StdAtomicRemapperPass.hpp"
#include "hipSYCL/common/debug.hpp"


#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/AtomicOrdering.h>

#include <string>

namespace hipsycl {
namespace compiler {

namespace {

// These enums need to align with the builtin memory scope/order/address-space conventions.
enum class memory_scope : int {
  work_item,
  sub_group,
  work_group,
  device,
  system
};

enum class memory_order : int
{
  relaxed,
  acquire,
  release,
  acq_rel,
  seq_cst
};

enum class address_space : int
{
  global_space,
  local_space,
  constant_space,
  private_space,
  generic_space
};

enum class rmw_op : int { op_and, op_or, op_xor, op_add, op_sub, op_min, op_max };
enum class rmw_data_type : int { signed_int, unsigned_int, floating_type };

memory_order llvmOrderingToAcppOrdering(llvm::AtomicOrdering AO) {
  if (AO == llvm::AtomicOrdering::NotAtomic || AO == llvm::AtomicOrdering::Unordered ||
      AO == llvm::AtomicOrdering::Monotonic)
    return memory_order::relaxed;
  // TODO memory_order_consume?
  else if(AO == llvm::AtomicOrdering::Acquire)
    return memory_order::acquire;
  else if(AO == llvm::AtomicOrdering::Release)
    return memory_order::release;
  else if(AO == llvm::AtomicOrdering::AcquireRelease)
    return memory_order::acq_rel;
  else
    return memory_order::seq_cst;
}

llvm::Type* getPointerType(llvm::Type* PointeeT, int AddressSpace) {
#if LLVM_VERSION_MAJOR < 16
    return llvm::PointerType::get(PointeeT, AddressSpace);
#else
    return llvm::PointerType::get(PointeeT->getContext(), AddressSpace);
#endif
}

template<class IntType>
llvm::Value* getIntConstant(llvm::Module& M, IntType Value) {
  return llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), static_cast<uint64_t>(Value));
}

bool isSizeSupportedByBuiltins(llvm::Module& M, llvm::Type* T) {
  auto Size = M.getDataLayout().getTypeSizeInBits(T);
  return Size == 8 || Size == 16 || Size == 32 || Size == 64;
}

bool needsBitcastsForIntAtomics(llvm::Module& M, llvm::Type* T) {
  return isSizeSupportedByBuiltins(M, T) && T->isIntegerTy();
}

llvm::Value *bitcastToIntN(llvm::Module &M, llvm::Value *V, int BitSize,
                           llvm::Instruction *InsertBefore) {
  auto *TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
  return new llvm::BitCastInst(V, TargetType, "", InsertBefore);
}

llvm::Value* ptrcastToIntNPtr(llvm::Module &M, llvm::Value *V, int BitSize,
                           llvm::Instruction *InsertBefore) {
  auto *TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
  return new llvm::BitCastInst(
      V, getPointerType(TargetType, V->getType()->getPointerAddressSpace()), "", InsertBefore);
}

bool llvmBinOpToAcppBinOp(llvm::AtomicRMWInst::BinOp Op, rmw_op& Out) {
  
  using LLVMBinOp = llvm::AtomicRMWInst::BinOp;
  if(Op == LLVMBinOp::Add)
    Out = rmw_op::op_add;
  else if(Op == LLVMBinOp::And)
    Out = rmw_op::op_and;
  else if(Op == LLVMBinOp::FAdd)
    Out = rmw_op::op_add;
#if LLVM_VERSION_MAJOR > 14
  else if(Op == LLVMBinOp::FMax)
    Out = rmw_op::op_max;
  else if(Op == LLVMBinOp::FMin)
    Out = rmw_op::op_min;
#endif
  else if(Op == LLVMBinOp::FSub)
    Out = rmw_op::op_sub;
  else if(Op == LLVMBinOp::Max)
    Out = rmw_op::op_max;
  else if(Op == LLVMBinOp::Min)
    Out = rmw_op::op_min;
  else if(Op == LLVMBinOp::Or)
    Out = rmw_op::op_or;
  else if(Op == LLVMBinOp::Sub)
    Out = rmw_op::op_sub;
  else if(Op == LLVMBinOp::UMax)
    Out = rmw_op::op_max;
  else if(Op == LLVMBinOp::UMin)
    Out = rmw_op::op_min;
  else if(Op == LLVMBinOp::Xor)
    Out = rmw_op::op_xor;
  else
    return false;

  return true;
}

// ----------------------------------------------------------------------------
// These functions obtain SSCP builtin declarations in IR
// ----------------------------------------------------------------------------

llvm::Function* getAtomicStoreBuiltin(llvm::Module& M, int BitSize) {
  std::string BuiltinName = "__acpp_sscp_atomic_store_i"+std::to_string(BitSize);


  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);

  return static_cast<llvm::Function *>(M.getOrInsertFunction(BuiltinName,
                                                             llvm::Type::getVoidTy(M.getContext()),
                                                             I32Ty, I32Ty, I32Ty, PtrTy, OpTy)
                                           .getCallee());
}


llvm::Function *getAtomicLoadBuiltin(llvm::Module &M, int BitSize) {
  std::string BuiltinName = "__acpp_sscp_atomic_load_i"+std::to_string(BitSize);


  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);
  
  return static_cast<llvm::Function *>(
      M.getOrInsertFunction(BuiltinName, OpTy, I32Ty, I32Ty, I32Ty, PtrTy).getCallee());
}

llvm::Function* getAtomicExchangeBuiltin(llvm::Module& M, int BitSize) {
  std::string BuiltinName = "__acpp_sscp_atomic_exchange_i"+std::to_string(BitSize);

  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);
  
  return static_cast<llvm::Function *>(
      M.getOrInsertFunction(BuiltinName, OpTy, I32Ty, I32Ty, I32Ty, PtrTy, OpTy).getCallee());
}

llvm::Function* getAtomicCmpExchangeBuiltin(llvm::Module& M, bool Strong, int BitSize) {
  std::string BuiltinName;
  if(Strong)
    BuiltinName = "__acpp_sscp_cmp_exch_strong_i"+std::to_string(BitSize);
  else
    BuiltinName = "__acpp_sscp_cmp_exch_weak_i"+std::to_string(BitSize);

  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);

  return static_cast<llvm::Function *>(
      M.getOrInsertFunction(BuiltinName, llvm::Type::getInt1Ty(M.getContext()),
        I32Ty, I32Ty, I32Ty, I32Ty, PtrTy, PtrTy, OpTy)
          .getCallee());
}

llvm::Function* getAtomicFetchOpBuiltin(llvm::Module& M, rmw_op Op, int BitSize, rmw_data_type TypeCategory) {
  std::string OpName;
  if(Op == rmw_op::op_add)
    OpName = "add";
  else if(Op == rmw_op::op_and)
    OpName = "and";
  else if(Op == rmw_op::op_max)
    OpName = "max";
  else if(Op == rmw_op::op_min)
    OpName = "min";
  else if(Op == rmw_op::op_or)
    OpName = "or";
  else if(Op == rmw_op::op_sub)
    OpName = "sub";
  else if(Op == rmw_op::op_xor)
    OpName = "xor";
  else
    return nullptr;

  std::string BuiltinName = "__acpp_sscp_atomic_fetch_" + OpName + "_";
  if(TypeCategory == rmw_data_type::floating_type)
    BuiltinName += "f"+std::to_string(BitSize);
  else if(TypeCategory == rmw_data_type::signed_int)
    BuiltinName += "i"+std::to_string(BitSize);
  else if(TypeCategory == rmw_data_type::unsigned_int)
    BuiltinName += "u"+std::to_string(BitSize);

  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = nullptr;
  if(TypeCategory == rmw_data_type::floating_type) {
    if(BitSize == 32)
      OpTy = llvm::Type::getFloatTy(M.getContext());
    else if(BitSize == 64)
      OpTy = llvm::Type::getDoubleTy(M.getContext());
    else
      return nullptr;
  } else {
    OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  }

  llvm::Type* PtrTy = getPointerType(OpTy, 0);
  return static_cast<llvm::Function *>(
      M.getOrInsertFunction(BuiltinName, OpTy,
        I32Ty, I32Ty, I32Ty, PtrTy, OpTy)
          .getCallee());
}

// ----------------------------------------------------------------------------
// These functions generate calls to corresponding SSCP builtin declarations
// ----------------------------------------------------------------------------


llvm::Value *createAtomicStore(llvm::Module &M, llvm::Value *Value, llvm::Value *Addr,
                               memory_order Order, memory_scope Scope,
                               llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(Value->getType());
  if(!isSizeSupportedByBuiltins(M, Value->getType()))
    return nullptr;

  if(needsBitcastsForIntAtomics(M, Value->getType())) {
    auto* TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
    Value = bitcastToIntN(M, Value, BitSize, InsertBefore);
    Addr = ptrcastToIntNPtr(M, Addr, BitSize, InsertBefore);
  }

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, Order),
    getIntConstant(M, Scope),
    Addr,
    Value
  };

  llvm::Function *Builtin = getAtomicStoreBuiltin(M, BitSize);
  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}


llvm::Value *createAtomicLoad(llvm::Module &M, llvm::Type* DataType, llvm::Value *Addr,
                               memory_order Order, memory_scope Scope,
                               llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(DataType);
  if(!isSizeSupportedByBuiltins(M, DataType))
    return nullptr;

  if(needsBitcastsForIntAtomics(M, DataType)) {
    auto* TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
    Addr = ptrcastToIntNPtr(M, Addr, BitSize, InsertBefore);
  }

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, Order),
    getIntConstant(M, Scope),
    Addr,
  };

  llvm::Function *Builtin = getAtomicLoadBuiltin(M, BitSize);
  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}

llvm::Value *createAtomicExchange(llvm::Module &M, llvm::Value* Value, llvm::Value *Addr,
                               memory_order Order, memory_scope Scope,
                               llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(Value->getType());
  llvm::Type* DataType = Value->getType();
  if(!isSizeSupportedByBuiltins(M, DataType))
    return nullptr;

  if(needsBitcastsForIntAtomics(M, DataType)) {
    auto* TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
    Value = bitcastToIntN(M, Value, BitSize, InsertBefore);
    Addr = ptrcastToIntNPtr(M, Addr, BitSize, InsertBefore);
  }

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, Order),
    getIntConstant(M, Scope),
    Addr,
    Value
  };

  llvm::Function *Builtin = getAtomicExchangeBuiltin(M, BitSize);
  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}

llvm::Value *createAtomicCmpExchange(llvm::Module &M, bool IsStrong, llvm::Value *Value,
                                     llvm::Value *Addr, llvm::Value *ExpectedAddr,
                                     memory_order SuccessOrder, memory_order FailureOrder,
                                     memory_scope Scope, llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(Value->getType());
  llvm::Type* DataType = Value->getType();
  if(!isSizeSupportedByBuiltins(M, DataType))
    return nullptr;

  if(needsBitcastsForIntAtomics(M, DataType)) {
    auto* TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
    Value = bitcastToIntN(M, Value, BitSize, InsertBefore);
    Addr = ptrcastToIntNPtr(M, Addr, BitSize, InsertBefore);
    ExpectedAddr = ptrcastToIntNPtr(M, ExpectedAddr, BitSize, InsertBefore);
  }

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, SuccessOrder),
    getIntConstant(M, FailureOrder),
    getIntConstant(M, Scope),
    Addr,
    ExpectedAddr,
    Value
  };

  llvm::Function *Builtin = getAtomicCmpExchangeBuiltin(M, IsStrong, BitSize);
  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}

llvm::Value *createAtomicFetchOp(llvm::Module &M, llvm::AtomicRMWInst::BinOp LLVMOp,
                                 llvm::Value *Value, llvm::Value *Addr, memory_order Order,
                                 memory_scope Scope, llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(Value->getType());
  llvm::Type* DataType = Value->getType();
  if(!isSizeSupportedByBuiltins(M, DataType))
    return nullptr;

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, Order),
    getIntConstant(M, Scope),
    Addr,
    Value
  };
  
  bool IsFloat = DataType->isFloatingPointTy();

  rmw_op Op;
  if(!llvmBinOpToAcppBinOp(LLVMOp, Op))
    return nullptr;

  if(IsFloat) {
    if(Op == rmw_op::op_and || Op == rmw_op::op_or || Op == rmw_op::op_xor)
      return nullptr;
  }

  rmw_data_type TypeCategory;
  if(IsFloat)
    TypeCategory = rmw_data_type::floating_type;
  else {
    if (LLVMOp == llvm::AtomicRMWInst::BinOp::UMax || LLVMOp == llvm::AtomicRMWInst::BinOp::UMin)
      TypeCategory = rmw_data_type::unsigned_int;
    else
      TypeCategory = rmw_data_type::signed_int;
  }

  llvm::Function *Builtin = getAtomicFetchOpBuiltin(M, Op, BitSize, TypeCategory);
  if(!Builtin)
    return nullptr;

  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}
}

llvm::PreservedAnalyses StdAtomicRemapperPass::run(llvm::Module &M,
                                                   llvm::ModuleAnalysisManager &MAM) {

  llvm::SmallVector<llvm::StoreInst*> AtomicStores;
  llvm::SmallVector<llvm::LoadInst*> AtomicLoads;
  llvm::SmallVector<llvm::AtomicRMWInst*> AtomicExchanges;
  llvm::SmallVector<llvm::AtomicCmpXchgInst*> AtomicCmpExchanges;
  llvm::SmallVector<llvm::AtomicRMWInst*> AtomicFetchOps;

  for(auto& F : M) {
    for(auto& BB : F){
      for(auto& I : BB) {
        if(auto* SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
          if(SI->isAtomic())
            AtomicStores.push_back(SI);
        } else if(auto* LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
          if(LI->isAtomic())
            AtomicLoads.push_back(LI);
        } else if(auto* RMWI = llvm::dyn_cast<llvm::AtomicRMWInst>(&I)) {
          if(RMWI->getOperation() == llvm::AtomicRMWInst::BinOp::Xchg) {
            AtomicExchanges.push_back(RMWI);
          } else {
            AtomicFetchOps.push_back(RMWI);
          }
        } else if(auto *CI = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&I)) {
          AtomicCmpExchanges.push_back(CI);
        }
      }
    }
  }

  llvm::SmallVector<llvm::Instruction*> ReplacedInstructions;
  for(auto* AS : AtomicStores) {
    llvm::AtomicOrdering Order = AS->getOrdering();
    if (auto *NewI =
            createAtomicStore(M, AS->getValueOperand(), AS->getPointerOperand(),
                              llvmOrderingToAcppOrdering(Order), memory_scope::system, AS)) {
      AS->replaceNonMetadataUsesWith(NewI);
      ReplacedInstructions.push_back(AS);
    }
  }
  for(auto* AL : AtomicLoads) {
    llvm::AtomicOrdering Order = AL->getOrdering();
    if (auto *NewI =
            createAtomicLoad(M, AL->getType(), AL->getPointerOperand(),
                              llvmOrderingToAcppOrdering(Order), memory_scope::system, AL)) {
      AL->replaceNonMetadataUsesWith(NewI);
      ReplacedInstructions.push_back(AL);
    }
  }
  for(auto* XI : AtomicExchanges) {
    llvm::AtomicOrdering Order = XI->getOrdering();
    if (auto *NewI =
            createAtomicExchange(M, XI->getValOperand(), XI->getPointerOperand(),
                              llvmOrderingToAcppOrdering(Order), memory_scope::system, XI)) {
      XI->replaceNonMetadataUsesWith(NewI);
      ReplacedInstructions.push_back(XI);
    }
  }
  for(auto* CI : AtomicCmpExchanges) {
    llvm::AtomicOrdering SuccessOrder = CI->getSuccessOrdering();
    llvm::AtomicOrdering FailureOrder = CI->getFailureOrdering();

    // Create alloca and store expected value - this we can later use
    // as pointer argument for the builtin
    llvm::AllocaInst* ExpectedAI = new llvm::AllocaInst(CI->getCompareOperand()->getType(), 0, "", CI);
    llvm::StoreInst *ExpectedStore = new llvm::StoreInst(CI->getCompareOperand(), ExpectedAI, CI);
    if (auto *NewI = createAtomicCmpExchange(
            M, !CI->isWeak(), CI->getNewValOperand(), CI->getPointerOperand(), ExpectedAI,
            llvmOrderingToAcppOrdering(SuccessOrder), llvmOrderingToAcppOrdering(FailureOrder),
            memory_scope::system, CI)) {

      llvm::Value* RetVal = llvm::UndefValue::get(CI->getType());
      llvm::Value *ExpectedLoad =
          new llvm::LoadInst(CI->getCompareOperand()->getType(), ExpectedAI, "", CI);
      
      llvm::SmallVector<unsigned int> InsertExpectedArgs{0};
      auto* I1 = llvm::InsertValueInst::Create(
          RetVal, ExpectedLoad, llvm::ArrayRef<unsigned int>{InsertExpectedArgs}, "", CI);
      InsertExpectedArgs = {1};
      auto* I2 = llvm::InsertValueInst::Create(
          I1, NewI, llvm::ArrayRef<unsigned int>{InsertExpectedArgs}, "", CI);
      
      CI->replaceNonMetadataUsesWith(I2);
      ReplacedInstructions.push_back(CI);
    }
  }
  for(auto* FI : AtomicFetchOps) {
    llvm::AtomicOrdering Order = FI->getOrdering();

    if (auto *NewI =
            createAtomicFetchOp(M, FI->getOperation(), FI->getValOperand(), FI->getPointerOperand(),
                                llvmOrderingToAcppOrdering(Order), memory_scope::system, FI)) {
      FI->replaceNonMetadataUsesWith(NewI);
      ReplacedInstructions.push_back(FI);
    }
  }

  // Clean up
  for(auto* I : ReplacedInstructions) {
    I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
    I->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}
}
}

