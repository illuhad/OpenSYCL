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
#include "hipSYCL/compiler/llvm-to-backend/host/LLVMToHost.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/HostKernelWrapperPass.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include "hipSYCL/runtime/dylib_loader.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR < 16
#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
#else
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#endif

#include <cassert>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace hipsycl {
namespace compiler {

LLVMToHostTranslator::LLVMToHostTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::host), KN, KN},
      KernelNames{KN} {}

bool LLVMToHostTranslator::toBackendFlavor(llvm::Module &M, PassHandler &PH) {

  for (auto KernelName : KernelNames) {
    if (auto *F = M.getFunction(KernelName)) {

      llvm::SmallVector<llvm::Metadata *, 4> Operands;
      Operands.push_back(llvm::ValueAsMetadata::get(F));
      Operands.push_back(llvm::MDString::get(M.getContext(), "kernel"));
      Operands.push_back(llvm::ValueAsMetadata::getConstant(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 1)));

      M.getOrInsertNamedMetadata(SscpAnnotationsName)
          ->addOperand(llvm::MDTuple::get(M.getContext(), Operands));

      F->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    }
  }

  std::string BuiltinBitcodeFileName = "libkernel-sscp-host-full.bc";
  if(IsFastMath)
    BuiltinBitcodeFileName = "libkernel-sscp-host-fast-full.bc";
  std::string BuiltinBitcodeFile =
      common::filesystem::join_path(common::filesystem::get_install_directory(),
                                    {"lib", "hipSYCL", "bitcode", BuiltinBitcodeFileName});

  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  llvm::ModulePassManager MPM;
  // TODO: Clean me up.
  // PH.ModuleAnalysisManager->clear(); // for some reason we need to reset the analyses... otherwise
  //                                    // we get a crash at IPSCCP

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  PH.PassBuilder->registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager &MAM) {
    MAM.registerPass([] { return SplitterAnnotationAnalysis{}; });
  });
  // PH.PassBuilder->registerModuleAnalyses(*PH.ModuleAnalysisManager);

  PH.PassBuilder->registerModuleAnalyses(MAM);
  PH.PassBuilder->registerCGSCCAnalyses(CGAM);
  PH.PassBuilder->registerFunctionAnalyses(FAM);
  PH.PassBuilder->registerLoopAnalyses(LAM);
  PH.PassBuilder->crossRegisterProxies(LAM, FAM, CGAM, MAM);
  registerCBSPipeline(MPM, hipsycl::compiler::OptLevel::O3, true);
  HIPSYCL_DEBUG_INFO << "LLVMToHostTranslator: Done registering\n";

  llvm::FunctionPassManager FPM;
  FPM.addPass(HostKernelWrapperPass{KnownLocalMemSize});
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));

  // MPM.printPipeline(llvm::outs(),[](llvm::StringRef Name){return Name;});
  MPM.run(M, MAM);
  HIPSYCL_DEBUG_INFO << "LLVMToHostTranslator: Done toBackendFlavor\n";
  return true;
}

bool LLVMToHostTranslator::translateToBackendFormat(llvm::Module &FlavoredModule,
                                                    std::string &out) {

  llvm::SmallVector<char> InputFile;
  int InputFD;
  // don't use fs::TempFile, as we can't unlock the file for the clang invocation later... (Windows)
  if(auto E = llvm::sys::fs::createTemporaryFile("acpp-sscp-host", "bc", InputFD, InputFile, llvm::sys::fs::OF_None)){
    this->registerError("LLVMToHost: Could not create temp input file" + E.message());
    return false;
  }
  llvm::StringRef InputFileName = InputFile.data();

  AtScopeExit RemoveInputFile([&](){auto Err = llvm::sys::fs::remove(InputFileName);});

  llvm::SmallVector<char> OutputFile;
  if(auto E = llvm::sys::fs::createTemporaryFile("acpp-sscp-host", ACPP_SHARED_LIBRARY_EXTENSION, OutputFile, llvm::sys::fs::OF_None)){
    this->registerError("LLVMToHost: Could not create temp input file" + E.message());
    return false;
  }
  llvm::StringRef OutputFileName = OutputFile.data();
  AtScopeExit RemoveOutputFile([&](){auto Err = llvm::sys::fs::remove(OutputFileName);});

  {
    llvm::raw_fd_ostream InputStream{InputFD, true};

    llvm::WriteBitcodeToFile(FlavoredModule, InputStream);
    
    if(InputStream.error()) {HIPSYCL_DEBUG_ERROR << "Error while writing" << InputStream.error().message() << '\n'; }
    InputStream.flush();
    if(InputStream.error()) {HIPSYCL_DEBUG_ERROR << "Error while flushing" << InputStream.error().message() << '\n'; }
  }

  const std::string CpuFlag = HIPSYCL_HOST_CPU_FLAG;

  llvm::SmallVector<llvm::StringRef, 16> Invocation{getClangPath(),
                                                    "-O3",
                                                    CpuFlag,
                                                    "-x",
                                                    "ir",
                                                    "-shared",
                                                    "-Wno-pass-failed",
                                                    #ifndef _WIN32
                                                    "-fPIC",
                                                    #endif
                                                    "-o",
                                                    OutputFileName,
                                                    InputFileName
                                                    };

  std::string ArgString;
  for (const auto &S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToHost: Invoking " << ArgString << "\n";

  int R = llvm::sys::ExecuteAndWait(ClangPath, Invocation);

  if (R != 0) {
    this->registerError("LLVMToHost: clang invocation failed with exit code " + std::to_string(R));
    return false;
  }

  auto ReadResult = llvm::MemoryBuffer::getFile(OutputFileName, -1);

  if (auto Err = ReadResult.getError()) {
    this->registerError("LLVMToHost: Could not read result file" + Err.message());
    return false;
  }

  out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToHostTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  return false;
}

bool LLVMToHostTranslator::isKernelAfterFlavoring(llvm::Function &F) {
  for (const auto &Name : KernelNames)
    if (F.getName() == Name)
      return true;
  return false;
}

AddressSpaceMap LLVMToHostTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;
  // Zero initialize for CPU.. we don't have address spaces
  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 0;
  ASMap[AddressSpace::Local] = 0;
  ASMap[AddressSpace::Private] = 0;
  ASMap[AddressSpace::Constant] = 0;
  ASMap[AddressSpace::AllocaDefault] = 0;
  ASMap[AddressSpace::GlobalVariableDefault] = 0;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 0;

  return ASMap;
}

ACPP_BACKEND_API_EXPORT std::unique_ptr<LLVMToBackendTranslator>
createLLVMToHostTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToHostTranslator>(KernelNames);
}

void LLVMToHostTranslator::migrateKernelProperties(llvm::Function *From, llvm::Function *To) {
  assert(false && "migrateKernelProperties is unsupport for LLVMToHost");
}

} // namespace compiler
} // namespace hipsycl
