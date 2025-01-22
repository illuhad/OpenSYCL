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
#include "hipSYCL/compiler/llvm-to-backend/NameHandling.hpp"
#include "hipSYCL/compiler/sscp/TargetSeparationPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/compiler/sscp/HostKernelNameExtractionPass.hpp"
#include "hipSYCL/compiler/sscp/AggregateArgumentExpansionPass.hpp"
#include "hipSYCL/compiler/sscp/StdBuiltinRemapperPass.hpp"
#include "hipSYCL/compiler/sscp/DynamicFunctionSupport.hpp"
#include "hipSYCL/compiler/sscp/StdAtomicRemapperPass.hpp"
#include "hipSYCL/compiler/CompilationState.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/utils/ProcessFunctionAnnotationsPass.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/common/hcf_container.hpp"

#include <cctype>
#include <cstddef>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Support/CommandLine.h>

#include <memory>
#include <string>
#include <fstream>
#include <random>
#include <chrono>


namespace hipsycl {
namespace compiler {

class Timer {
public:
  Timer(llvm::StringRef Name, bool PrintAtDestruction = false, llvm::StringRef Description = "")
  : Print{PrintAtDestruction}, Name{Name}, Description{Description} {
    Start = std::chrono::high_resolution_clock::now();;
    IsRunning = true;
  }

  double stop() {
    if(IsRunning) {
      Stop = std::chrono::high_resolution_clock::now();
      IsRunning = false;
    }

    auto Ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(Stop - Start).count();
    return static_cast<double>(Ticks) * 1.e-9;

    return Ticks;
  }

  double stopAndPrint() {
    double T = stop();
    HIPSYCL_DEBUG_INFO << "SSCP: Phase '" << Name << "' took " << T << " seconds\n"; 
    return T;
  }

  ~Timer() {
    if(Print)
      stopAndPrint();
    else
      stop();
  }
private:
  bool Print;
  bool IsRunning = false;
  std::string Name, Description;

  using TimePointT = 
    std::chrono::time_point<std::chrono::high_resolution_clock>;
  TimePointT Start;
  TimePointT Stop;
  
};

class ScopedPrintingTimer : private Timer {
public:
  ScopedPrintingTimer(llvm::StringRef Name, llvm::StringRef Description = "")
  : Timer{Name, true, Description} {}
};

static llvm::cl::opt<bool> SSCPEmitHcf{
    "acpp-sscp-emit-hcf", llvm::cl::init(false),
    llvm::cl::desc{"Emit HCF from AdaptiveCpp LLVM SSCP compilation flow"}};

static llvm::cl::opt<bool> PreoptimizeSSCPKernels{
    "acpp-sscp-preoptimize", llvm::cl::init(false),
    llvm::cl::desc{
        "Preoptimize SYCL kernels in LLVM IR instead of embedding unoptimized kernels and relying "
        "on optimization at runtime. This is mainly for AdaptiveCpp developers and NOT supported!"}};

static llvm::cl::opt<bool> ExportAllSymbols{
    "acpp-sscp-export-all", llvm::cl::init(false),
    llvm::cl::desc{
        "(experimental) export all functions for JIT-time linking"}};

static const char *SscpIsHostIdentifier = "__acpp_sscp_is_host";
static const char *SscpIsDeviceIdentifier = "__acpp_sscp_is_device";
static const char *SscpHcfObjectIdIdentifier = "__acpp_local_sscp_hcf_object_id";
static const char *SscpHcfObjectSizeIdentifier = "__acpp_local_sscp_hcf_object_size";
static const char *SscpHcfContentIdentifier = "__acpp_local_sscp_hcf_content";

template<class IntT>
IntT generateRandomNumber() {
  static std::mutex M;
  static std::random_device Dev;
  static std::mt19937 Rng{Dev()};
  static std::uniform_int_distribution<IntT> dist;

  std::lock_guard<std::mutex> lock {M};
  return dist(Rng);
}

enum class ParamType {
  Integer,
  FoatingPoint,
  Ptr,
  OtherByVal
};

struct KernelParam {
  std::size_t ByteSize;
  std::size_t ArgByteOffset;
  std::size_t OriginalArgIndex;
  ParamType Type;
  llvm::SmallVector<std::string> Annotations;
};

struct KernelInfo {
  std::string Name;
  std::vector<KernelParam> Parameters;

  KernelInfo() = default;
  KernelInfo(const std::string &KernelName, llvm::Module &M,
             const std::vector<OriginalParamInfo>& OriginalParamInfos) {

    this->Name = KernelName;
    if(auto* F = M.getFunction(KernelName)) {

      auto* FType = F->getFunctionType();
      assert(OriginalParamInfos.size() == FType->getNumParams());

      for(int i = 0; i < FType->getNumParams(); ++i) {
        ParamType PT;

        llvm::Type* ParamT = FType->getParamType(i);
        if(ParamT->isIntegerTy()) {
          PT = ParamType::Integer;
        } else if(ParamT->isFloatingPointTy()) {
          PT = ParamType::FoatingPoint;
        } else if(ParamT->isPointerTy()) {
          if(F->hasParamAttribute(i, llvm::Attribute::ByVal)) {
            PT = ParamType::OtherByVal;
          } else {
            PT = ParamType::Ptr;
          }
        } else {
          PT = ParamType::OtherByVal;
        }

        KernelParam KP;
        
        auto BitSize = M.getDataLayout().getTypeSizeInBits(ParamT);
        assert(BitSize % CHAR_BIT == 0);
        KP.ByteSize = BitSize / CHAR_BIT;
        KP.Type = PT;
        KP.ArgByteOffset = OriginalParamInfos[i].OffsetInOriginalParam;
        KP.OriginalArgIndex = OriginalParamInfos[i].OriginalParamIndex;
        KP.Annotations = OriginalParamInfos[i].Annotations;
        this->Parameters.push_back(KP);
      }
    }
  }
};

#ifdef _MSC_VER
void replaceInvalidMSABICharsInSymbolNames(llvm::Module &M) {
    auto UpdateName = [](auto &S) {
      std::string Name(S.getName());
      replaceInvalidMSABICharsInSymbolName(Name);
      S.setName(Name);
    };

    for(auto &F : M) {
      UpdateName(F);
    }
    for(auto &G : M.globals()){
      UpdateName(G);
    }
}
#endif

std::unique_ptr<llvm::Module> generateDeviceIR(llvm::Module &M,
                                               const std::vector<std::string>& DynamicFunctions,
                                               std::vector<KernelInfo> &KernelInfoOutput,
                                               std::vector<std::string> &ExportedSymbolsOutput,
                                               std::vector<std::string> &ImportedSymbolsOutput) {

  std::unique_ptr<llvm::Module> DeviceModule = llvm::CloneModule(M);
  DeviceModule->setModuleIdentifier("device." + DeviceModule->getModuleIdentifier());
  
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager DeviceMAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(DeviceMAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, DeviceMAM);

  // Strip module-level inline assembly. Module-level inline assembly is used
  // by some libstdc++ versions (>13?) in their headers. This causes problems
  // because we cannot infer whether global assembly code not contained in functions
  // is part of device or host code. Thus, such inline assembly can cause JIT
  // failures.
  // Because global inline assembly does not make sense in device code (there are
  // multiple JIT targets, each with their own inline assembly syntax), such code
  // cannot be relevant to device code and we can safely strip it from device code.
  DeviceModule->setModuleInlineAsm("");

  // Fix std:: math function calls to point to our builtins.
  // This is required such that e.g. std::sin() can be called in kernels.
  // This should be done prior to kernel outlining, such that the now defunct
  // std math functions can be thrown away during kernel outlining.
  StdBuiltinRemapperPass SBMP;
  SBMP.run(*DeviceModule, DeviceMAM);
  // Remap atomics
  StdAtomicRemapperPass SAMP;
  SAMP.run(*DeviceModule, DeviceMAM);

  // Fix attributes for generic IR representation
  llvm::SmallVector<llvm::Attribute::AttrKind, 16> AttrsToRemove;
  llvm::SmallVector<std::string, 16> StringAttrsToRemove;
  AttrsToRemove.push_back(llvm::Attribute::AttrKind::UWTable);
  AttrsToRemove.push_back(llvm::Attribute::AttrKind::StackProtectStrong);
  AttrsToRemove.push_back(llvm::Attribute::AttrKind::StackProtect);
  AttrsToRemove.push_back(llvm::Attribute::AttrKind::StackProtectReq);
  StringAttrsToRemove.push_back("frame-pointer");
  StringAttrsToRemove.push_back("min-legal-vector-width");
  StringAttrsToRemove.push_back("no-trapping-math");
  StringAttrsToRemove.push_back("stack-protector-buffer-size");
  StringAttrsToRemove.push_back("target-cpu");
  StringAttrsToRemove.push_back("target-features");
  StringAttrsToRemove.push_back("tune-cpu");

  llvm::SmallSet<llvm::Function *, 16> AcppNoInlineFunctions;
  utils::findFunctionsWithStringAnnotations(
      *DeviceModule, [&](llvm::Function *F, llvm::StringRef Annotation) {
        if (F) {
          if (Annotation.compare("acpp_no_s1_device_inline") == 0) {
            AcppNoInlineFunctions.insert(F);
          }
        }
      });
  for(const auto& FName : DynamicFunctions) {
    if(auto* F = DeviceModule->getFunction(FName))
      AcppNoInlineFunctions.insert(F);
  }

  for(auto& F : *DeviceModule) {
    for(auto& A : AttrsToRemove) {
      if(F.hasFnAttribute(A))
        F.removeFnAttr(A);
    }
    for(auto& A : StringAttrsToRemove) {
      if(F.hasFnAttribute(A))
        F.removeFnAttr(A);
    }

    // Need to enable inlining so that we can efficiently JIT even when
    // the user compiles with -O0. However, we need skip functions
    // that have the acpp_no_s1_inline annotation.
    bool IsAcppNoInline = AcppNoInlineFunctions.contains(&F);
    if(F.hasFnAttribute(llvm::Attribute::NoInline)) {
      if(!IsAcppNoInline)
        F.removeFnAttr(llvm::Attribute::NoInline);
    } else {
      if(IsAcppNoInline)
        F.addFnAttr(llvm::Attribute::NoInline);
    }
  }

#ifdef _MSC_VER
  replaceInvalidMSABICharsInSymbolNames(*DeviceModule);
#endif

  EntrypointPreparationPass EPP{ExportAllSymbols};
  EPP.run(*DeviceModule, DeviceMAM);
  
  ExportedSymbolsOutput = EPP.getNonKernelOutliningEntrypoints();

  KernelArgumentCanonicalizationPass KACPass{EPP.getKernelNames()};
  KACPass.run(*DeviceModule, DeviceMAM);

  // Still need to make sure that at least dummy values are there on
  // the device side to avoid undefined references.
  // SscpIsHostIdentifier can also be used in device code.
  S1IRConstantReplacer DeviceSideReplacer{
      {{SscpIsHostIdentifier, 0}, {SscpIsDeviceIdentifier, 1}},
      {{SscpHcfObjectIdIdentifier, 0 /* Dummy value */}, {SscpHcfObjectSizeIdentifier, 0}},
      {{SscpHcfContentIdentifier, std::string{}}}};
  DeviceSideReplacer.run(*DeviceModule, DeviceMAM);

  IRConstant::optimizeCodeAfterConstantModification(*DeviceModule, DeviceMAM);

  // getOutlinigEntrypoints() returns both kernels as well as non-kernel (i.e. SYCL_EXTERNAL)
  // entrypoints

  S2IRConstant::forEachS2IRConstant(*DeviceModule, [](S2IRConstant IRC){
    // This is important to avoid GlobalOpt during Kernel outlining from removing these
    // unitialized variables.
    IRC.getGlobalVariable()->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
  });

  KernelOutliningPass KP{EPP.getOutliningEntrypoints()};
  KP.run(*DeviceModule, DeviceMAM);

  // Scan for imported function definitions
  ImportedSymbolsOutput.clear();
  for(auto& F : *DeviceModule) {
    if(F.size() == 0) {
      // We currently use the heuristic that functions are imported
      // if they are not defined, not an intrinsic and don't start with
      // __ like our hipSYCL builtins. This is a hack, it would
      // be better if we could tell clang to annotate the declaration for us :(
      if(!F.isIntrinsic() && !llvmutils::starts_with(F.getName(), "__"))
        ImportedSymbolsOutput.push_back(F.getName().str());
    }
  }

  AggregateArgumentExpansionPass KernelArgExpansionPass{EPP.getKernelNames()};
  KernelArgExpansionPass.run(*DeviceModule, DeviceMAM);

  DeviceMAM.clear();
  if(!PreoptimizeSSCPKernels) {
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O0);
    MPM.run(*DeviceModule, DeviceMAM);
  } else {
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    MPM.run(*DeviceModule, DeviceMAM);
  }

  KernelInfoOutput.clear();
  for(auto Name : EPP.getKernelNames()) {
    auto* OriginalParamInfos = KernelArgExpansionPass.getInfosOnOriginalParams(Name);
    assert(OriginalParamInfos);

    KernelInfo KI{Name, *DeviceModule, *OriginalParamInfos};
    KernelInfoOutput.push_back(KI);
  }

  return DeviceModule;
}

std::string
generateHCF(llvm::Module &DeviceModule, std::size_t HcfObjectId,
            const std::vector<KernelInfo> &Kernels, const std::vector<std::string> &ExportedSymbols,
            const std::vector<std::string> &ImportedSymbols,
            const std::vector<std::string> &KernelCompileFlags,
            const std::vector<std::pair<std::string, std::string>> &KernelCompileOptions) {

  std::string ModuleContent;
  llvm::raw_string_ostream OutputStream{ModuleContent};
  llvm::WriteBitcodeToFile(DeviceModule, OutputStream);

  common::hcf_container HcfObject;
  HcfObject.root_node()->set("object-id", std::to_string(HcfObjectId));
  HcfObject.root_node()->set("generator", "hipSYCL SSCP");

  auto *DeviceImagesNodes = HcfObject.root_node()->add_subnode("images");
  auto* LLVMIRNode = DeviceImagesNodes->add_subnode("llvm-ir.global");
  LLVMIRNode->set("variant", "global-module");
  LLVMIRNode->set("format", "llvm-ir");
  HcfObject.attach_binary_content(LLVMIRNode, ModuleContent);

  for(const auto& ES : ExportedSymbols) {
    HIPSYCL_DEBUG_INFO << "HCF generation: Image exports symbol: " << ES << "\n";
  }
  for(const auto& IS : ImportedSymbols) {
    HIPSYCL_DEBUG_INFO << "HCF generation: Image imports symbol: " << IS << "\n";
  }
  
  LLVMIRNode->set_as_list("exported-symbols", ExportedSymbols);
  LLVMIRNode->set_as_list("imported-symbols", ImportedSymbols);

  auto* KernelsNode = HcfObject.root_node()->add_subnode("kernels");
  for(const auto& Kernel : Kernels) {
    auto* K = KernelsNode->add_subnode(Kernel.Name);
    K->set_as_list("image-providers", {std::string{"llvm-ir.global"}});
    
    auto* FlagsNode = K->add_subnode("compile-flags");
    for(const auto& F : KernelCompileFlags) {
      FlagsNode->set(F, "1");
    }
    auto *OptsNode = K->add_subnode("compile-options");
    for(const auto& O : KernelCompileOptions) {
      OptsNode->set(O.first, O.second);
    }

    auto* ParamsNode = K->add_subnode("parameters");

    for(std::size_t i = 0; i < Kernel.Parameters.size(); ++i) {
      const auto& ParamInfo = Kernel.Parameters[i];
      auto* P = ParamsNode->add_subnode(std::to_string(i));
      P->set("byte-offset", std::to_string(ParamInfo.ArgByteOffset));
      P->set("byte-size", std::to_string(ParamInfo.ByteSize));
      P->set("original-index", std::to_string(ParamInfo.OriginalArgIndex));
      ParamType Type = ParamInfo.Type;
      std::string TypeDescription;
      if(Type == ParamType::Integer) {
        TypeDescription = "integer";
      } else if(Type == ParamType::FoatingPoint) {
        TypeDescription = "floating-point";
      } else if(Type == ParamType::Ptr) {
        TypeDescription = "pointer";
      } else if(Type == ParamType::OtherByVal) {
        TypeDescription = "other-by-value";
      }
      P->set("type", TypeDescription);

      auto* AnnotationsNode = P->add_subnode("annotations");
      for(const auto& A : ParamInfo.Annotations) {
        AnnotationsNode->set(A, "1");
      }
    }
  }
  

  return HcfObject.serialize();
}

llvm::PreservedAnalyses TargetSeparationPass::run(llvm::Module &M,
                                                  llvm::ModuleAnalysisManager &MAM) {

  DynamicFunctionIdentifactionPass DFI;
  DFI.run(M, MAM);

  {
    ScopedPrintingTimer totalTimer{"TargetSeparationPass (total)"};
    // TODO If we know that the SSCP compilation flow is the only one using HCF,
    // we could just enumerate the objects instead of generating (hopefully)
    // unique random numbers.
    std::size_t HcfObjectId = generateRandomNumber<std::size_t>();
    std::string HcfString;


    // Only run SSCP kernel extraction in the host pass in case
    // there are also CUDA/HIP compilation flows going on
    if(!CompilationStateManager::getASTPassState().isDeviceCompilation()) {
      
      std::vector<KernelInfo> Kernels;
      std::vector<std::string> ExportedSymbols;
      std::vector<std::string> ImportedSymbols;
      
      
      Timer IRGenTimer{"generateDeviceIR", true};
      std::unique_ptr<llvm::Module> DeviceIR = generateDeviceIR(
          M, DFI.getDynamicFunctionNames(), Kernels, ExportedSymbols, ImportedSymbols);
      IRGenTimer.stopAndPrint();

      Timer HCFGenTimer{"generateHCF"};
      HcfString = generateHCF(*DeviceIR, HcfObjectId, Kernels, ExportedSymbols, ImportedSymbols,
                              CompilationFlags, CompilationOptions);
      HCFGenTimer.stopAndPrint();

      if(SSCPEmitHcf) {
        std::string Filename = M.getSourceFileName()+".hcf";
        std::ofstream OutputFile{Filename.c_str(), std::ios::trunc|std::ios::binary};
        OutputFile.write(HcfString.c_str(), HcfString.size());
        OutputFile.close();
      }
    }

    {
      ScopedPrintingTimer Timer {"HostKernelNameExtractionPass"};
      HostKernelNameExtractionPass KernelNamingPass;
      KernelNamingPass.run(M, MAM);
    }

    {
      ScopedPrintingTimer Timer {"Host-side dynamic function handling"};

      HostSideDynamicFunctionHandlerPass HDFH{DFI.getDynamicFunctionNames(),
                                              DFI.getDynamicFunctionDefinitionNames()};

      HDFH.run(M, MAM);

      // Remove argument_used hints, which are no longer needed once IR
      // has been generated. This is primarily needed for dynamic functions.
      // TODO: We should consider whether it might make more sense to move this to late-stage
      // JIT, at least for the device part.
      utils::ProcessFunctionAnnotationPass PFA({"argument_used"});
      PFA.run(M, MAM);
    }

    {
      ScopedPrintingTimer Timer {"S1 IR constant application"};
      S1IRConstantReplacer HostSideReplacer{
          {{SscpIsHostIdentifier, 1}, {SscpIsDeviceIdentifier, 0}},
          {{SscpHcfObjectIdIdentifier, HcfObjectId}, {SscpHcfObjectSizeIdentifier, HcfString.size()}},
          {{SscpHcfContentIdentifier, HcfString}}};

      HostSideReplacer.run(M, MAM);
    }

    {
      ScopedPrintingTimer Timer {"S1 IR constant branching optimization"};
      IRConstant::optimizeCodeAfterConstantModification(M, MAM);
    }
  }
  return llvm::PreservedAnalyses::none();
}

TargetSeparationPass::TargetSeparationPass(const std::string &KernelCompilationOptions) {
  llvm::StringRef Opts{KernelCompilationOptions};

  llvm::SmallVector<llvm::StringRef> Fragments;
  Opts.split(Fragments, ',', -1, false);

  for(auto& S : Fragments) {
    if(S.contains("=")) {
      llvm::SmallVector<llvm::StringRef> OptionFragments;
      S.split(OptionFragments, '=', 1, false);
      if(OptionFragments.size() == 2) {
        CompilationOptions.push_back(
            std::make_pair(std::string{OptionFragments[0]}, std::string{OptionFragments[1]}));
      }
    } else {
      CompilationFlags.push_back(std::string{S});
    }
  }
}
}
}
