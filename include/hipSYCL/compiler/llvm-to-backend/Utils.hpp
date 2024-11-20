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
#ifndef HIPSYCL_LLVM_TO_BACKEND_UTILS_HPP
#define HIPSYCL_LLVM_TO_BACKEND_UTILS_HPP

#include <atomic>

#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Attributes.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Casting.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Passes/PassBuilder.h>

namespace hipsycl {
namespace compiler {

template<class F>
class AtScopeExit {
public:
  AtScopeExit(F&& f)
  : Handler(f) {}

  ~AtScopeExit() {Handler();}
private:
  std::function<void()> Handler;
};

struct PassHandler {
  llvm::PassBuilder* PassBuilder;
  llvm::ModuleAnalysisManager* ModuleAnalysisManager;
};

inline llvm::Error loadModuleFromString(const std::string &LLVMIR, llvm::LLVMContext &ctx,
                                        std::unique_ptr<llvm::Module> &out) {

  auto buff = llvm::MemoryBuffer::getMemBuffer(LLVMIR, "", false);
  auto BC = llvm::parseBitcodeFile(*buff.get(), ctx);

  if(auto err = BC.takeError()) {
    return err;
  }

  out = std::move(BC.get());

  return llvm::Error::success();
}

template<class F>
inline auto withPassBuilder(F&& handler) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  return handler(PB, LAM, FAM, CGAM, MAM);
}

template<class F>
inline auto withPassBuilderAndMAM(F&& handler) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  return handler(PB, MAM);
}


class KernelFunctionParameterRewriter {
public:
  // Attribute that should be attached to by-value
  // pointer function arguments
  enum class ByValueArgAttribute {
    ByVal, ByRef
  };

  /// \param Attr Whether by-value arguments that are passed as pointers
  /// should have byval or byref attribute. Note: At least one of them
  /// must already be present.
  /// \param DesiredByValueArgAddressSpace Address space that pointers
  /// to by-value arguments should have
  /// \param DesiredPointerAddressSpace Address space that actual pointers
  /// should have
  /// \param WrapPointers If true, wrap pointer arguments in their own byval/byref
  /// struct. This can be used to hide the pointer from the runtime backend, which
  /// may be desirable if it does overzealous pointer validation (e.g. level zero)
  KernelFunctionParameterRewriter(ByValueArgAttribute Attr, unsigned DesiredByValueArgAddressSpace,
                                  unsigned DesiredPointerAddressSpace, bool WrapPointers = false)
      : Attr{Attr}, ByValueArgAddressSpace{DesiredByValueArgAddressSpace},
        PointerAddressSpace{DesiredPointerAddressSpace}, WrapPointers{WrapPointers} {}

  void run(llvm::Module &M, const std::vector<std::string> &KernelNames,
           llvm::ModuleAnalysisManager &MAM) {
    for(auto KernelName : KernelNames) {
      if(auto* F = M.getFunction(KernelName)) {
        run(M, F);
      }
    }
    MAM.invalidate(M, llvm::PreservedAnalyses::none());
  }

  void run(llvm::Module& M, llvm::Function* F) {
    auto* OldFType = F->getFunctionType();
    std::string FunctionName = F->getName().str();
    F->setName(FunctionName+"_PreKernelParameterRewriting");
    F->setLinkage(llvm::GlobalValue::InternalLinkage);

    llvm::SmallVector<llvm::Type*, 8> Params;
    // If pointer wrapping is enabled, maps the parameter index
    // to the generated pointer wrapper type, if available.
    llvm::SmallDenseMap<int, llvm::Type*> WrapperTypes;

    for(int i = 0; i < OldFType->getNumParams(); ++i){
      llvm::Type* CurrentParamType = OldFType->getParamType(i);

      if(llvm::PointerType* PT = llvm::dyn_cast<llvm::PointerType>(CurrentParamType)) {
        bool HasByRefAttr = F->hasParamAttribute(i, llvm::Attribute::ByRef);
        bool HasByValAttr = F->hasParamAttribute(i, llvm::Attribute::ByVal);

        llvm::Type* NewT = nullptr;

        if(!HasByRefAttr && !HasByValAttr) {
          // Regular pointer
          if(shouldWrapPointer(*F, i)) {
            llvm::Type* WrapperType;
            NewT = getWrappedGlobalPointerType(M, PT, WrapperType);
            WrapperTypes[i] = WrapperType;
          } else {
#if LLVM_VERSION_MAJOR < 17
            NewT = llvm::PointerType::getWithSamePointeeType(PT, PointerAddressSpace);
#else
            NewT = llvm::PointerType::get(PT->getContext(), PointerAddressSpace);
#endif
          }
        } else {
          // ByVal or ByRef - this probably means that
          // some struct is passed into the kernel by value.
          // (the attribute will be handled later)
#if LLVM_VERSION_MAJOR < 17
          NewT = llvm::PointerType::getWithSamePointeeType(PT, ByValueArgAddressSpace);
#else
          NewT = llvm::PointerType::get(PT->getContext(), ByValueArgAddressSpace);
#endif
        }
        Params.push_back(NewT);
      
      } else {
        Params.push_back(CurrentParamType);
      }
    }


    llvm::FunctionType *FType = llvm::FunctionType::get(F->getReturnType(), Params, false);
    if (auto *NewF = llvm::dyn_cast<llvm::Function>(
            M.getOrInsertFunction(FunctionName, FType, F->getAttributes()).getCallee())) {

      // Fix ByVal and ByRef attributes (different backends require different
      // attributes for kernel arguments)
      for(int i = 0; i < NewF->getFunctionType()->getNumParams(); ++i) {
        bool HasByRefAttr = NewF->hasParamAttribute(i, llvm::Attribute::ByRef);
        bool HasByValAttr = NewF->hasParamAttribute(i, llvm::Attribute::ByVal);
        // If byval/byref are present, ensure the right one is set
        if(HasByRefAttr || HasByValAttr) {
          llvm::Attribute::AttrKind PresentAttr =
              HasByRefAttr ? llvm::Attribute::ByRef : llvm::Attribute::ByVal;
          if(PresentAttr != getDesiredByValueArgAttribute()) {
            llvm::Type* ValT = F->getParamAttribute(i, PresentAttr).getValueAsType();
            assert(ValT);

            NewF->removeParamAttr(i, PresentAttr);
            addByValueArgAttribute(M, *NewF, i, ValT);          
          }
        // Otherwise we might be dealing with a pointer that needs to be wrapped in
        // a by-value struct
        } else if (WrapperTypes.find(i) != WrapperTypes.end()) {
          llvm::Type* WrapperT = WrapperTypes[i];
          addByValueArgAttribute(M, *NewF, i, WrapperT);
        }
      }

      // Now create function call to old function
      llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", NewF);
      llvm::SmallVector<llvm::Value*, 8> CallArgs;

      for(int i = 0; i < Params.size(); ++i) {
        // By default, just pass in the function argument ...
        llvm::Value* CallArg = NewF->getArg(i);
        // ... but if pointers are involved, we may have to insert additional
        // instructions, such as address space casts.
        if (llvm::PointerType *NewPT =
                llvm::dyn_cast<llvm::PointerType>(NewF->getArg(i)->getType())) {
          assert(llvm::isa<llvm::PointerType>(F->getArg(i)->getType()));
          llvm::PointerType *OldPT = llvm::dyn_cast<llvm::PointerType>(F->getArg(i)->getType());

          // Check if this is a wrapped pointer
          auto it = WrapperTypes.find(i);
          if(it != WrapperTypes.end()) {
            llvm::Type* WrapperType = it->second;
            // Wrapped pointer, so we need an additional getelementptr(ptr,0,0) instruction.
            auto Zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 0);
            llvm::SmallVector<llvm::Value*> GEPIndices{Zero, Zero};
            auto *GEPInst = llvm::GetElementPtrInst::CreateInBounds(
              WrapperType, NewF->getArg(i), llvm::ArrayRef<llvm::Value *>{GEPIndices}, "", BB);
            
            
            auto* WrappedTy = GEPInst->getResultElementType();
            auto* LoadInst = new llvm::LoadInst{WrappedTy, GEPInst, "", BB};

            auto* WrappedPointerTy = llvm::dyn_cast<llvm::PointerType>(WrappedTy);
            if(WrappedPointerTy && (WrappedPointerTy->getAddressSpace() != OldPT->getAddressSpace())) {
              auto *ASCastInst = new llvm::AddrSpaceCastInst{LoadInst, OldPT, "", BB};
              CallArg = ASCastInst;
            } else {
              CallArg = LoadInst;
            }
            
          } else {
            // We are dealing with a USM pointer
            if (NewPT->getAddressSpace() != OldPT->getAddressSpace()) {
              auto *ASCastInst = new llvm::AddrSpaceCastInst{NewF->getArg(i), OldPT, "", BB};
              CallArg = ASCastInst;
            }
            // The else branch is unnecessary, because by default we just
            // pass in the original function argument.
          }
        } 
        
        CallArgs.push_back(CallArg);
      }
  
      assert(CallArgs.size() == F->getFunctionType()->getNumParams());
      for(int i = 0; i < CallArgs.size(); ++i) {
        assert(CallArgs[i]->getType() == F->getFunctionType()->getParamType(i));
      }

      llvm::CallInst::Create(llvm::FunctionCallee(F), CallArgs, "", BB);
      llvm::ReturnInst::Create(M.getContext(), BB);

      if(!F->hasFnAttribute(llvm::Attribute::AlwaysInline))
        F->addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }
private:

  bool shouldWrapPointer(llvm::Function& F, int Param) {
    bool HasByRefAttr = F.hasParamAttribute(Param, llvm::Attribute::ByRef);
    bool HasByValAttr = F.hasParamAttribute(Param, llvm::Attribute::ByVal);
    if(!HasByRefAttr && !HasByValAttr) {
      if(F.getFunctionType()->getParamType(Param)->isPointerTy()) {
        return WrapPointers;
      }
    }
    return false;
  }


  llvm::Attribute::AttrKind getDesiredByValueArgAttribute() const {
    return (Attr == ByValueArgAttribute::ByRef) ? llvm::Attribute::ByRef : llvm::Attribute::ByVal;
  }

  void addByValueArgAttribute(llvm::Module &M, llvm::Function &F, int i, llvm::Type *ValT) const {
    if (getDesiredByValueArgAttribute() == llvm::Attribute::ByRef)
      F.addParamAttr(i, llvm::Attribute::getWithByRefType(M.getContext(), ValT));
    else
      F.addParamAttr(i, llvm::Attribute::getWithByValType(M.getContext(), ValT));
  }

  llvm::Type *getWrappedGlobalPointerType(llvm::Module &M, llvm::PointerType *OriginalPointerType,
                                          llvm::Type *&WrapperType) {
    WrapperType = getWrapperType(M, OriginalPointerType);
#if LLVM_VERSION_MAJOR < 16
    return llvm::PointerType::get(WrapperType, ByValueArgAddressSpace);
#else
    return llvm::PointerType::get(M.getContext(), ByValueArgAddressSpace);
#endif
  }

  llvm::Type *getWrapperType(llvm::Module &M, llvm::PointerType *OriginalPointerType) {
    static std::atomic<std::size_t> WrapperCounter = 0;

    llvm::Type *WrappedType =
#if LLVM_VERSION_MAJOR < 17
        llvm::PointerType::getWithSamePointeeType(OriginalPointerType, PointerAddressSpace);
#else
        llvm::PointerType::get(OriginalPointerType->getContext(), PointerAddressSpace);
#endif
    
    auto it = PointerWrapperTypes.find(WrappedType);
    if(it != PointerWrapperTypes.end())
      return it->second;
    
    std::string Name = "__acpp_sscp_pointer_wrapper." + std::to_string(++WrapperCounter);
    llvm::SmallVector<llvm::Type*> Elements {WrappedType};
    
    llvm::Type* NewType = llvm::StructType::create(M.getContext(), Elements, Name);

    PointerWrapperTypes[WrappedType] = NewType;

    return NewType;
  }

  ByValueArgAttribute Attr;
  unsigned ByValueArgAddressSpace;
  unsigned PointerAddressSpace;
  bool WrapPointers;
  llvm::SmallDenseMap<llvm::Type*, llvm::Type*> PointerWrapperTypes;
};


}
}

#endif
