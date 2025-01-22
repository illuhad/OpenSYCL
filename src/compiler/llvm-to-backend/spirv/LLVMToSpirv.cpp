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
#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirv.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
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

namespace hipsycl {
namespace compiler {

namespace {

static const char* DynamicLocalMemArrayName = "__acpp_sscp_spirv_dynamic_local_mem";

void appendIntelLLVMSpirvOptions(llvm::SmallVector<std::string>& out) {
  llvm::SmallVector<std::string> Args {"-spirv-max-version=1.3",
      "-spirv-debug-info-version=ocl-100",
      "-spirv-allow-extra-diexpressions",
      "-spirv-allow-unknown-intrinsics=llvm.genx.",
      "-spirv-ext=-all,+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_"
      "KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls,+SPV_KHR_expect_assume,+SPV_INTEL_"
      "subgroups,+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,+SPV_INTEL_"
      "fpga_loop_controls,+SPV_INTEL_fpga_memory_attributes,+SPV_INTEL_fpga_memory_accesses,+SPV_"
      "INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes,+SPV_INTEL_"
      "function_pointers,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes,+SPV_INTEL_inline_"
      "assembly,+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_float_controls2,+SPV_INTEL_"
      "vector_compute,+SPV_INTEL_fast_composite,+SPV_INTEL_fpga_buffer_location,+SPV_INTEL_joint_"
      "matrix,+SPV_INTEL_arbitrary_precision_fixed_point,+SPV_INTEL_arbitrary_precision_floating_"
      "point,+SPV_INTEL_arbitrary_precision_floating_point,+SPV_INTEL_variable_length_array,+SPV_"
      "INTEL_fp_fast_math_mode,+SPV_INTEL_fpga_cluster_attributes,+SPV_INTEL_loop_fuse,+SPV_INTEL_"
      "long_constant_composite,+SPV_INTEL_fpga_invocation_pipelining_attributes,+SPV_INTEL_fpga_"
      "dsp_control,+SPV_INTEL_arithmetic_fence,+SPV_INTEL_runtime_aligned,"
      "+SPV_INTEL_optnone,+SPV_INTEL_token_type,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_"
      "matrix,+SPV_INTEL_hw_thread_queries,+SPV_INTEL_memory_access_aliasing,+SPV_EXT_relaxed_printf_string_address_space"
  };
  for(const auto& S : Args) {
    out.push_back(S);
  }
}

bool setDynamicLocalMemoryCapacity(llvm::Module& M, unsigned numBytes) {
  llvm::GlobalVariable* GV = M.getGlobalVariable(DynamicLocalMemArrayName);

  if(!GV) {
    // If non-zero number of bytes are needed, not finding the global variable is
    // an error.
    return numBytes == 0;
  }

  if(numBytes > 0) {
    unsigned AddressSpace = GV->getAddressSpace();
    unsigned numInts = (numBytes + 4 - 1) / 4;
    llvm::Type* T = llvm::ArrayType::get(llvm::Type::getInt32Ty(M.getContext()), numInts);

    llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
        M, T, false, llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(T),
        GV->getName() + ".resized", nullptr, llvm::GlobalVariable::ThreadLocalMode::NotThreadLocal,
        AddressSpace);

    NewVar->setAlignment(GV->getAlign());
    llvm::Value* V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());
    GV->replaceAllUsesWith(V);
    GV->eraseFromParent();
  }
  return true;
}

bool removeDynamicLocalMemorySupport(llvm::Module& M) {
  llvm::GlobalVariable* GV = M.getGlobalVariable(DynamicLocalMemArrayName);
  if(GV) {
    GV->replaceAllUsesWith(llvm::ConstantPointerNull::get(GV->getType()));
    GV->eraseFromParent();
  }
  return true;
}

void assignSPIRCallConvention(llvm::Function *F) {
  if (F->getCallingConv() != llvm::CallingConv::SPIR_FUNC)
    F->setCallingConv(llvm::CallingConv::SPIR_FUNC);

  // All callers must use spir_func calling convention
  for (auto U : F->users()) {
    if (auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
      CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    }
  }
}
}

LLVMToSpirvTranslator::LLVMToSpirvTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::spirv), KN, KN},
      KernelNames{KN} {}

bool LLVMToSpirvTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {
  
  M.setTargetTriple("spir64-unknown-unknown");
  M.setDataLayout("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
                  "1024-A4-n8:16:32:64");

  AddressSpaceMap ASMap = getAddressSpaceMap();
  KernelFunctionParameterRewriter ParamRewriter{
    // llvm-spirv wants ByVal attribute for all aggregates passed in by-value
    KernelFunctionParameterRewriter::ByValueArgAttribute::ByVal,
    // Those pointers to by-value data should be in private AS
    ASMap[AddressSpace::Private],
    // Actual pointers should be in global memory
    ASMap[AddressSpace::Global],
    // We need to wrap pointer types
    true};

  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);

  for(auto KernelName : KernelNames) {
    HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Setting up kernel " << KernelName << "\n";
    if(auto* F = M.getFunction(KernelName)) {
      applyKernelProperties(F);
    }
  }

  for(auto& F : M) {
    if(F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL){
      // All functions must be marked as spir_func
      assignSPIRCallConvention(&F);
    }
  }

  std::string BuiltinBitcodeFile = 
    common::filesystem::join_path(common::filesystem::get_install_directory(),
      {"lib", "hipSYCL", "bitcode", "libkernel-sscp-spirv-full.bc"});

  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile, M.getTargetTriple(), M.getDataLayoutStr()))
    return false;

  // Set up local memory
  if(DynamicLocalMemSize > 0) {
    HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Configuring kernel for " << DynamicLocalMemSize
                       << " bytes of local memory\n";
    if(!setDynamicLocalMemoryCapacity(M, DynamicLocalMemSize)) {
      HIPSYCL_DEBUG_WARNING
          << "Could not set dynamic local memory size; this could imply that local memory "
             "requested by the application is not actually used inside kernels\n";
    }
  } else {
    HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Removing dynamic local memory support from module\n";
    removeDynamicLocalMemorySupport(M);
  }


  AddressSpaceInferencePass ASIPass{ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  // llvm-spirv translator does not like llvm.lifetime.start/end operate on generic
  // pointers. TODO: We should only remove them when we actually need to, and attempt
  // to fix them otherwise.
  llvm::SmallVector<llvm::CallBase*, 16> Calls;
  for(auto& F : M) {
    for(auto& BB : F) {
      for(auto& I : BB) {
        if(llvm::CallBase* CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
          // llvm-spirv translator does not like llvm.lifetime.start/end operate on generic
          // pointers.
          auto* CalledF = CB->getCalledFunction();
          if (llvmutils::starts_with(CalledF->getName(), "llvm.lifetime.start") ||
              llvmutils::starts_with(CalledF->getName(), "llvm.lifetime.end")) {
            if(CB->getNumOperands() > 1 && CB->getArgOperand(1)->getType()->isPointerTy())
              if (CB->getArgOperand(1)->getType()->getPointerAddressSpace() ==
                  ASMap[AddressSpace::Generic])
                Calls.push_back(CB);
          }
        }
      }
    }
  }
  for(auto CB : Calls) {
    CB->replaceAllUsesWith(llvm::UndefValue::get(CB->getType()));
    CB->eraseFromParent();
  }

  // It seems there is an issue with debug info in llvm-spirv, so strip it for now
  // TODO: We should attempt to find out what exactly is causing the problem
  llvm::StripDebugInfo(M);

  return true;
}

bool LLVMToSpirvTranslator::translateToBackendFormat(llvm::Module &FlavoredModule, std::string &out) {

  auto InputFile = llvm::sys::fs::TempFile::create("acpp-sscp-spirv-%%%%%%.bc");
  auto OutputFile = llvm::sys::fs::TempFile::create("acpp-sscp-spirv-%%%%%%.spv");
  
  std::string OutputFilename = OutputFile->TmpName;
  
  auto E = InputFile.takeError();
  if(E){
    this->registerError("LLVMToSpirv: Could not create temp file: "+InputFile->TmpName);
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

  std::string LLVMSpirVTranslator = hipsycl::common::filesystem::join_path(
      hipsycl::common::filesystem::get_install_directory(), HIPSYCL_RELATIVE_LLVMSPIRV_PATH);


  llvm::SmallVector<std::string> Args{
      "-o=" + OutputFilename
  };
  if(UseIntelLLVMSpirvArgs)
    appendIntelLLVMSpirvOptions(Args);
  else {
    Args.push_back("-spirv-max-version=1.3");
    Args.push_back("-spirv-ext=+SPV_EXT_relaxed_printf_string_address_space");
  }

  llvm::SmallVector<llvm::StringRef, 16> Invocation{LLVMSpirVTranslator};
  for(const auto& A : Args)
    Invocation.push_back(A);
  Invocation.push_back(InputFile->TmpName);

  std::string ArgString;
  for(const auto& S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToSpirv: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(
      LLVMSpirVTranslator, Invocation);

  if(R != 0) {
    this->registerError("LLVMToSpirv: llvm-spirv invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }
  
  auto ReadResult =
      llvm::MemoryBuffer::getFile(OutputFile->TmpName, -1);
  
  if(auto Err = ReadResult.getError()) {
    this->registerError("LLVMToSpirv: Could not read result file"+Err.message());
    return false;
  }
  
  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToSpirvTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if(Option == "spirv-dynamic-local-mem-allocation-size") {
    this->DynamicLocalMemSize = static_cast<unsigned>(std::stoi(Value));
    return true;
  }

  return false;
}

bool LLVMToSpirvTranslator::applyBuildFlag(const std::string& Flag) {
  if(Flag == "spirv-enable-intel-llvm-spirv-options") {
    UseIntelLLVMSpirvArgs = true;
    return true;
  }
  return false;
}

bool LLVMToSpirvTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  return F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
}

AddressSpaceMap LLVMToSpirvTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;

  // By default, llvm-spirv translator uses the mapping where
  // ASMap[AddressSpace::Generic] = 4;
  // ASMap[AddressSpace::Private] = 0;
  // We currently require a different mapping where the default address
  // space is the generic address space, which requires a patched llvm-spirv.
  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 4;
  ASMap[AddressSpace::Constant] = 2;
  ASMap[AddressSpace::AllocaDefault] = 4;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  // we cannot put constant globals into constant AS because
  // llvm-spirv translator does not allow AS cast from constant to generic
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 1;

  return ASMap;
}

bool LLVMToSpirvTranslator::optimizeFlavoredIR(llvm::Module& M, PassHandler& PH) {
  bool Result = LLVMToBackendTranslator::optimizeFlavoredIR(M, PH);
  if(!Result)
    return false;

  // Optimizations may introduce the freeze instruction, which is not supported
  // by llvm-spirv.
  // See https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/1140
  // We adopt the workaround proposed there.

  llvm::SmallVector<llvm::Instruction*> InstsToRemove;
  for(auto& F : M) {
    for(auto& BB : F) {
      for(auto& I : BB) {
        if(auto* FI = llvm::dyn_cast<llvm::FreezeInst>(&I)) {
          FI->replaceAllUsesWith(FI->getOperand(0));
          FI->dropAllReferences();
          InstsToRemove.push_back(FI);
        }
      }
    }
  }
  for(auto* I : InstsToRemove)
    I->eraseFromParent();
  
  return Result;
}

void LLVMToSpirvTranslator::migrateKernelProperties(llvm::Function* From, llvm::Function* To) {
  removeKernelProperties(From);
  applyKernelProperties(To);
}

void LLVMToSpirvTranslator::applyKernelProperties(llvm::Function* F) {
  F->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

  llvm::Module& M = *F->getParent();

  if (KnownGroupSizeX != 0 && KnownGroupSizeY != 0 && KnownGroupSizeZ != 0) {
    llvm::SmallVector<llvm::Metadata *> MDs;
    MDs.push_back(llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeX)));
    MDs.push_back(llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeY)));
    MDs.push_back(llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeZ)));

    static const char *ReqdWGSize = "reqd_work_group_size";
    F->setMetadata(ReqdWGSize, llvm::MDNode::get(M.getContext(), MDs));
  }
}

void LLVMToSpirvTranslator::removeKernelProperties(llvm::Function* F) {
  assignSPIRCallConvention(F);
  for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if(F->getArg(i)->hasAttribute(llvm::Attribute::ByVal)) {
      F->getArg(i)->removeAttr(llvm::Attribute::ByVal);
    }
  }
  F->clearMetadata();
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToSpirvTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToSpirvTranslator>(KernelNames);
}

}
}
