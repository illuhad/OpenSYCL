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
#include "hipSYCL/common/config.hpp"

#include "hipSYCL/compiler/GlobalsPruningPass.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"

#ifdef HIPSYCL_WITH_STDPAR_COMPILER
#include "hipSYCL/compiler/stdpar/MallocToUSM.hpp"
#include "hipSYCL/compiler/stdpar/SyncElision.hpp"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

#ifdef HIPSYCL_WITH_ACCELERATED_CPU
#include "hipSYCL/compiler/cbs/LoopsParallelMarker.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#endif

#ifdef HIPSYCL_WITH_SSCP_COMPILER
#include "hipSYCL/compiler/sscp/TargetSeparationPass.hpp"
#endif

#ifdef HIPSYCL_WITH_REFLECTION_BUILTINS
#include "hipSYCL/compiler/reflection/IntrospectStructPass.hpp"
#include "hipSYCL/compiler/reflection/FunctionNameExtractionPass.hpp"
#endif

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

#if LLVM_VERSION_MAJOR < 16
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#endif

namespace hipsycl {
namespace compiler {

static llvm::cl::opt<bool> EnableLLVMSSCP{
    "acpp-sscp", llvm::cl::init(false),
    llvm::cl::desc{"Enable AdaptiveCpp LLVM SSCP compilation flow"}};

static llvm::cl::opt<std::string> LLVMSSCPKernelOpts{
    "acpp-sscp-kernel-opts", llvm::cl::init(""),
    llvm::cl::desc{
        "Specify compilation options to use when JIT-compiling AdaptiveCpp SSCP kernels"}};

static llvm::cl::opt<bool> EnableStdPar{
    "acpp-stdpar", llvm::cl::init(false),
    llvm::cl::desc{"Enable hipSYCL C++ standard parallelism support"}};

static llvm::cl::opt<bool> StdparNoMallocToUSM{
    "acpp-stdpar-no-malloc-to-usm", llvm::cl::init(false),
    llvm::cl::desc{"Disable hipSYCL C++ standard parallelism malloc-to-usm compiler-side support"}};

// Register and activate passes

#if LLVM_VERSION_MAJOR < 16
static void registerGlobalsPruningPass(const llvm::PassManagerBuilder &,
                                       llvm::legacy::PassManagerBase &PM) {
  PM.add(new GlobalsPruningPassLegacy{});
}

static llvm::RegisterStandardPasses
    RegisterGlobalsPruningPassOptLevel0(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                                        registerGlobalsPruningPass);

static llvm::RegisterStandardPasses
    RegisterGlobalsPruningPassOptimizerLast(llvm::PassManagerBuilder::EP_OptimizerLast,
                                            registerGlobalsPruningPass);

#ifdef HIPSYCL_WITH_ACCELERATED_CPU
static llvm::RegisterPass<SplitterAnnotationAnalysisLegacy>
    splitterAnnotationReg("splitter-annot-ana", "hipSYCL splitter annotation analysis pass",
                          true /* Only looks at CFG */, true /* Analysis Pass */);

static void registerLoopSplitAtBarrierPasses(const llvm::PassManagerBuilder &,
                                             llvm::legacy::PassManagerBase &PM) {
  registerCBSPipelineLegacy(PM);
}

static llvm::RegisterStandardPasses
    RegisterLoopSplitAtBarrierPassOptimizerFirst(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                                                 registerLoopSplitAtBarrierPasses);

static void registerMarkParallelPass(const llvm::PassManagerBuilder &,
                                     llvm::legacy::PassManagerBase &PM) {
  PM.add(new LoopsParallelMarkerPassLegacy());
}

// SROA adds loads / stores without adopting the llvm.access.group MD, need to re-add.
static llvm::RegisterStandardPasses
    RegisterMarkParallelBeforeVectorizer(llvm::PassManagerBuilder::EP_VectorizerStart,
                                         registerMarkParallelPass);
#endif // HIPSYCL_WITH_ACCELERATED_CPU
#endif // LLVM_VERSION_MAJOR < 16

}}

#if !defined(_WIN32)
#define HIPSYCL_RESOLVE_AND_QUOTE(V) #V
#define HIPSYCL_STRINGIFY(V) HIPSYCL_RESOLVE_AND_QUOTE(V)
#define HIPSYCL_PLUGIN_VERSION_STRING                                                              \
  "v" HIPSYCL_STRINGIFY(ACPP_VERSION_MAJOR) "." HIPSYCL_STRINGIFY(                              \
      ACPP_VERSION_MINOR) "." HIPSYCL_STRINGIFY(ACPP_VERSION_PATCH)

llvm::PassPluginLibraryInfo getAdaptiveCppPluginInfo(){
  using namespace hipsycl::compiler;
return {
    LLVM_PLUGIN_API_VERSION, "AdaptiveCpp Clang plugin", HIPSYCL_PLUGIN_VERSION_STRING,
        [](llvm::PassBuilder &PB) {
          // Note: for Clang < 12, this EP is not called for O0, but the new PM isn't
          // really used there anyways..
          PB.registerOptimizerLastEPCallback([](llvm::ModulePassManager &MPM, OptLevel) {
            MPM.addPass(hipsycl::compiler::GlobalsPruningPass{});
          });
#ifdef HIPSYCL_WITH_REFLECTION_BUILTINS
          PB.registerPipelineStartEPCallback(
                [&](llvm::ModulePassManager &MPM, OptLevel Level) {
                  MPM.addPass(IntrospectStructPass{});
                  MPM.addPass(FunctionNameExtractionPass{});
                });
#endif

#ifdef HIPSYCL_WITH_STDPAR_COMPILER
          if(EnableStdPar) {
            PB.registerPipelineStartEPCallback([&](llvm::ModulePassManager &MPM, OptLevel Level) {
              if(!StdparNoMallocToUSM) {
                MPM.addPass(MallocToUSMPass{});
              }
            });
          
            PB.registerOptimizerLastEPCallback([&](llvm::ModulePassManager &MPM, OptLevel Level) {
              MPM.addPass(SyncElisionInliningPass{});
              MPM.addPass(llvm::AlwaysInlinerPass{});
              MPM.addPass(SyncElisionPass{});
            });
          }
#endif

#ifdef HIPSYCL_WITH_SSCP_COMPILER
          if(EnableLLVMSSCP){
            PB.registerPipelineStartEPCallback(
                [&](llvm::ModulePassManager &MPM, OptLevel Level) {
                  MPM.addPass(TargetSeparationPass{LLVMSSCPKernelOpts});
                });
          }
#endif


#ifdef HIPSYCL_WITH_ACCELERATED_CPU
          PB.registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager &MAM) {
            if(!CompilationStateManager::getASTPassState().isDeviceCompilation())
              MAM.registerPass([] { return SplitterAnnotationAnalysis{}; });
          });

          PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM, OptLevel Opt) {

            if(!CompilationStateManager::getASTPassState().isDeviceCompilation())
              registerCBSPipeline(MPM, Opt, false);
          });
          // SROA adds loads / stores without adopting the llvm.access.group MD, need to re-add.
          // todo: check back with LLVM 13, might be fixed with https://reviews.llvm.org/D103254
          PB.registerVectorizerStartEPCallback([](llvm::FunctionPassManager &FPM, OptLevel) {
            if(!CompilationStateManager::getASTPassState().isDeviceCompilation())
              FPM.addPass(LoopsParallelMarkerPass{});
          });
#endif
        }
  };
}

#ifndef LLVM_ADAPTIVECPP_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getAdaptiveCppPluginInfo();
}
#endif
#endif // !_WIN32
