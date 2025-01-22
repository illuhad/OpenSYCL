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
#include "hipSYCL/compiler/llvm-to-backend/ptx/LLVMToPtx.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Program.h>
#include <memory>
#include <cassert>
#include <string>
#include <system_error>
#include <vector>
#include <array>

namespace hipsycl {
namespace compiler {

namespace {



class LibdevicePath {
public:
  static bool get(std::string& Out) {
    static LibdevicePath P;

    if(P.IsFound)
      Out = P.Path;
    return P.IsFound;
  }
private:
  LibdevicePath() {
    IsFound = findLibdevice(Path);

    if(IsFound) {
      HIPSYCL_DEBUG_INFO << "LLVMToPtx: Found libdevice: " << Path << "\n";
    } else {
      HIPSYCL_DEBUG_INFO << "LLVMToPtx: Could not find CUDA libdevice!\n";
    }
  }

  bool findLibdevice(std::string& Out) {
    
    std::string CUDAPath = HIPSYCL_CUDA_PATH;
    std::vector<std::string> SubDir {"nvvm", "libdevice"};
    std::string BitcodeDir = common::filesystem::join_path(CUDAPath, SubDir);

    try {
      auto Files = common::filesystem::list_regular_files(BitcodeDir);
      for(const auto& F : Files) {
        if (F.find("libdevice.") != std::string::npos && F.find(".bc") != std::string::npos) {
          Out = F;
          return true;
        }
      }
    }catch(...) { /* false will be returned anyway at this point */ }

    return false;
  }

  std::string Path;
  bool IsFound;
};

void setNVVMReflectParameter(llvm::Module& M, llvm::StringRef Name, int Value) {
  llvm::SmallVector<llvm::Metadata*, 4> Metadata;
  Metadata.push_back(llvm::ValueAsMetadata::getConstant(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 4)));
  Metadata.push_back(llvm::MDString::get(M.getContext(), "nvvm-reflect-" + std::string{Name}));
  Metadata.push_back(llvm::ValueAsMetadata::getConstant(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), Value)));

  M.getModuleFlagsMetadata()->addOperand(llvm::MDTuple::get(M.getContext(), Metadata)); 
}

void setFTZMode(llvm::Module& M, int Mode) {
  setNVVMReflectParameter(M, "ftz", Mode);
}

void setPrecDiv(llvm::Module& M, int Mode) {
  setNVVMReflectParameter(M, "prec-div", Mode);
}

void setPrecSqrt(llvm::Module& M, int Mode) {
  setNVVMReflectParameter(M, "prec-sqrt", Mode);
}


using IntrinsicMapping = std::array<const char*, 2>;
// These intrinsics seem to not be handled correctly by NVPTX backend,
// so replace them with our own builtins.
static constexpr std::array IntrinsicReplacementMap = {
  IntrinsicMapping{"llvm.pow.f32", "__acpp_sscp_pow_f32"},
  IntrinsicMapping{"llvm.pow.f64", "__acpp_sscp_pow_f64"},
  IntrinsicMapping{"llvm.exp.f32", "__acpp_sscp_exp_f32"},
  IntrinsicMapping{"llvm.exp.f64", "__acpp_sscp_exp_f64"},
  IntrinsicMapping{"llvm.exp2.f32", "__acpp_sscp_exp2_f32"},
  IntrinsicMapping{"llvm.exp2.f64", "__acpp_sscp_exp2_f64"},
  IntrinsicMapping{"llvm.exp10.f32", "__acpp_sscp_exp10_f32"},
  IntrinsicMapping{"llvm.exp10.f64", "__acpp_sscp_exp10_f64"},
  IntrinsicMapping{"llvm.cos.f32", "__acpp_sscp_cos_f32"},
  IntrinsicMapping{"llvm.cos.f64", "__acpp_sscp_cos_f64"},
  IntrinsicMapping{"llvm.sin.f32", "__acpp_sscp_sin_f32"},
  IntrinsicMapping{"llvm.sin.f64", "__acpp_sscp_sin_f64"},
  // tan seems fine
  IntrinsicMapping{"llvm.log.f32", "__acpp_sscp_log_f32"},
  IntrinsicMapping{"llvm.log.f64", "__acpp_sscp_log_f64"},
  IntrinsicMapping{"llvm.log2.f32", "__acpp_sscp_log2_f32"},
  IntrinsicMapping{"llvm.log2.f64", "__acpp_sscp_log2_f64"},
  IntrinsicMapping{"llvm.log10.f32", "__acpp_sscp_log10_f32"},
  IntrinsicMapping{"llvm.log10.f64", "__acpp_sscp_log10_f64"},
  // asin seems fine (presumably acos and atan as well)
  // sqrt seems fine
};

void replaceBrokenLLVMIntrinsics(llvm::Module& M) {
  for(auto& RM : IntrinsicReplacementMap) {
    if(auto* F = M.getFunction(RM[0])) {
      llvm::Function* Replacement = M.getFunction(RM[1]);

      if(!Replacement) {
        Replacement = llvm::Function::Create(F->getFunctionType(),
                                             llvm::GlobalValue::ExternalLinkage, RM[1], M);
        F->replaceAllUsesWith(Replacement);
      }
    }
  }
}

}

LLVMToPtxTranslator::LLVMToPtxTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::ptx), KN, KN},
      KernelNames{KN} {}

bool LLVMToPtxTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  std::string Triple = "nvptx64-nvidia-cuda";
  std::string DataLayout =
      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-"
      "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

  M.setTargetTriple(Triple);
  M.setDataLayout(DataLayout);

  // Initialize libdevice parameters. These values are < 0 in case no explicit
  // setting has been done.
  if(FlushDenormalsToZero < 0)
    FlushDenormalsToZero = IsFastMath ? 1 : 0;
  if(PreciseDiv < 0)
    PreciseDiv = IsFastMath ? 0 : 1;
  if(PreciseSqrt < 0)
    PreciseSqrt = IsFastMath ? 0 : 1;

  setFTZMode(M, FlushDenormalsToZero);
  setPrecDiv(M, PreciseDiv);
  setPrecSqrt(M, PreciseSqrt);

  AddressSpaceMap ASMap = getAddressSpaceMap();
  
  KernelFunctionParameterRewriter ParamRewriter{
      // PTX wants ByVal attribute for all aggregates passed in by-value
      KernelFunctionParameterRewriter::ByValueArgAttribute::ByVal,
      // Those pointers to by-value data can be in generic AS
      ASMap[AddressSpace::Generic],
      // Actual pointers should be in global memory
      ASMap[AddressSpace::Global]};
  
  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);

  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)) {
      applyKernelProperties(F);
    }
  }

  replaceBrokenLLVMIntrinsics(M);

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-ptx-full.bc"});
  
  std::string LibdeviceFile;
  if(!LibdevicePath::get(LibdeviceFile)) {
    this->registerError("LLVMToPtx: Could not find CUDA libdevice bitcode library");
    return false;
  }

  AddressSpaceInferencePass ASIPass {ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  // It seems there is an issue with debug info in PTX, so strip it for now
  // TODO: We should attempt to find out what exactly is causing the problem
  // so that code still can be debugged on NVIDIA GPUs.
  llvm::StripDebugInfo(M);

  if(!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;
  if(!this->linkBitcodeFile(M, LibdeviceFile, Triple, DataLayout))
    return false;

  return true;
}

bool LLVMToPtxTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {

  auto InputFile = llvm::sys::fs::TempFile::create("acpp-sscp-ptx-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("acpp-sscp-ptx-%%%%%%.s");

  if (auto Err = InputFile.takeError()) {
    this->registerError("LLVMToPtx: Could not create temp file: "+InputFile->TmpName);
    return false;
  }

  if (auto Err = OutputFile.takeError()) {
    this->registerError("LLVMToPtx: Could not create temp file: " + OutputFile->TmpName);
    return false;
  }

  AtScopeExit DestroyInputFile([&]() {
    if (auto Err = InputFile->discard())
      ;
  });
  AtScopeExit DestroyOutputFile([&]() {
    if (auto Err = OutputFile->discard())
      ;
  });

  std::error_code EC;
  llvm::raw_fd_ostream InputStream{InputFile->FD, false};
  
  llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
  InputStream.flush();

  std::string ClangPath = HIPSYCL_CLANG_PATH;

  std::string PtxVersionArg = "+ptx" + std::to_string(PtxVersion);
  std::string PtxTargetArg = "sm_" + std::to_string(PtxTarget);
  llvm::SmallVector<llvm::StringRef, 16> Invocation{ClangPath,
                                                    "-cc1",
                                                    "-triple",
                                                    "nvptx64-nvidia-cuda",
                                                    "-target-feature",
                                                    PtxVersionArg,
                                                    "-target-cpu",
                                                    PtxTargetArg,
                                                    "-O3",
                                                    "-S",
                                                    "-x",
                                                    "ir",
                                                    "-o",
                                                    OutputFile->TmpName,
                                                    InputFile->TmpName};
  if(IsFastMath)
    Invocation.push_back("-ffast-math");

  std::string ArgString;
  for(const auto& S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToPtx: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(
      ClangPath, Invocation);
  
  if(R != 0) {
    this->registerError("LLVMToPtx: clang invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }
  
  auto ReadResult =
      llvm::MemoryBuffer::getFile(OutputFile->TmpName, -1);
  
  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToPtx: Could not read result file"+Err.message());
    return false;
  }
  
  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToPtxTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if(Option == "ptx-version") {
    this->PtxVersion = std::stoi(Value);
    return true;
  } else if(Option == "ptx-target-device") {
    this->PtxTarget = std::stoi(Value);
    return true;
  }

  return false;
}

bool LLVMToPtxTranslator::applyBuildFlag(const std::string& Option) {
  if(Option == "ptx-ftz") {
    this->FlushDenormalsToZero = 1;
    return true;
  } else if(Option == "ptx-approx-div") {
    this->PreciseDiv = 0;
    return true;
  } else if(Option == "ptx-approx-sqrt") {
    this->PreciseSqrt = 0;
    return true;
  }
  return false;
}

bool LLVMToPtxTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  for(const auto& Name : KernelNames)
    if(F.getName() == Name)
      return true;
  return false;
}

AddressSpaceMap LLVMToPtxTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 5;
  ASMap[AddressSpace::Constant] = 4;
  // NVVM wants to have allocas in address space 0
  ASMap[AddressSpace::AllocaDefault] = 0;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 4;

  return ASMap;
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToPtxTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToPtxTranslator>(KernelNames);
}

void LLVMToPtxTranslator::migrateKernelProperties(llvm::Function* From, llvm::Function* To) {
  llvm::Module& M = *From->getParent();
  
  if(auto* MD = M.getNamedMetadata("nvvm.annotations")) {
    MD->eraseFromParent();
  }
  for (int i = 0; i < From->getFunctionType()->getNumParams(); ++i)
    if (From->getArg(i)->hasAttribute(llvm::Attribute::ByVal))
      From->getArg(i)->removeAttr(llvm::Attribute::ByVal);

  From->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  for(const auto& KN : KernelNames) {
    if(KN != To->getName() && KN != From->getName())
      if(auto* F = M.getFunction(KN))
        applyKernelProperties(F);
  }
  applyKernelProperties(To);
}

void LLVMToPtxTranslator::applyKernelProperties(llvm::Function* F) {
  llvm::Module& M = *F->getParent();

  llvm::SmallVector<llvm::Metadata*, 4> Operands;
  Operands.push_back(llvm::ValueAsMetadata::get(F));
  Operands.push_back(llvm::MDString::get(M.getContext(), "kernel"));
  Operands.push_back(llvm::ValueAsMetadata::getConstant(
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 1)));


  M.getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDTuple::get(M.getContext(), Operands));

  if(KnownGroupSizeX > 0 && KnownGroupSizeY > 0 && KnownGroupSizeZ > 0) {

    llvm::SmallVector<llvm::Metadata*, 7> KnownGroupSizeOperands;
    KnownGroupSizeOperands.push_back(llvm::ValueAsMetadata::get(F));
    
    KnownGroupSizeOperands.push_back(llvm::MDString::get(M.getContext(), "maxntidx"));
    KnownGroupSizeOperands.push_back(llvm::ValueAsMetadata::getConstant(
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeX)));

    KnownGroupSizeOperands.push_back(llvm::MDString::get(M.getContext(), "maxntidy"));
    KnownGroupSizeOperands.push_back(llvm::ValueAsMetadata::getConstant(
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeY)));
    
    KnownGroupSizeOperands.push_back(llvm::MDString::get(M.getContext(), "maxntidz"));
    KnownGroupSizeOperands.push_back(llvm::ValueAsMetadata::getConstant(
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeZ)));
    
    M.getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDTuple::get(M.getContext(), KnownGroupSizeOperands));
  }

  F->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
}


}
}
