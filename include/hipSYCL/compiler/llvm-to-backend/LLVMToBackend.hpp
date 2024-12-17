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
#ifndef HIPSYCL_LLVM_TO_BACKEND_HPP
#define HIPSYCL_LLVM_TO_BACKEND_HPP

#ifndef _WIN32
#define HIPSYCL_BACKEND_API_EXPORT
#else
#define HIPSYCL_BACKEND_API_EXPORT __declspec(dllexport)
#endif

// Note: This file should not include any LLVM headers or include
// dependencies that rely on LLVM headers in order to not spill
// LLVM code into the hipSYCL runtime.
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <typeinfo>
#include <functional>
#include "AddressSpaceMap.hpp"
#include "hipSYCL/glue/llvm-sscp/s2_ir_constants.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace llvm {
class Module;
class Function;
}

namespace hipsycl {
namespace compiler {

struct PassHandler;

struct TranslationHints {
  std::optional<std::size_t> RequestedLocalMemSize;
  std::optional<std::size_t> SubgroupSize;
  std::optional<rt::range<3>> WorkGroupSize;
};

class LLVMToBackendTranslator {
public:
  LLVMToBackendTranslator(int S2IRConstantCurrentBackendId,
    const std::vector<std::string>& OutliningEntrypoints,
    const std::vector<std::string>& KernelNames);

  virtual ~LLVMToBackendTranslator() {}

  // Do not use inside llvm-to-backend infrastructure targets to avoid
  // requiring RTTI-enabled LLVM
  template<auto& ConstantName, class T>
  void setS2IRConstant(const T& value) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                  "Unsupported type for S2 IR constant");

#ifdef _MSC_VER // Windows - not MINGW!
    std::string name = typeid(__acpp_sscp_s2_ir_constant<ConstantName, T>).raw_name();
    name = name.substr(sizeof(".?AU?$")); // this seems to be prepended here, but not in the actual variable name.
    replaceInvalidCharsInSymbolName(name);
#else
    std::string name = typeid(__acpp_sscp_s2_ir_constant<ConstantName, T>).name();
#endif
    setS2IRConstant<T>(name, value);
  }

  template<class T>
  void setS2IRConstant(const std::string& name, T value) {
    setS2IRConstant(name, static_cast<const void*>(&value));
  }

  void setS2IRConstant(const std::string& name, const void* ValueBuffer);
  void specializeKernelArgument(const std::string &KernelName, int ParamIndex,
                                const void *ValueBuffer);
  void specializeFunctionCalls(const std::string &FuncName,
                             const std::vector<std::string> &ReplacementCalls,
                             bool OverrideOnlyUndefined=true);

  bool setBuildFlag(const std::string &Flag);
  bool setBuildOption(const std::string &Option, const std::string &Value);
  bool setBuildToolArguments(const std::string &ToolName, const std::vector<std::string> &Args);

  template<class T>
  bool setBuildOption(const std::string& Option, const T& Value) {
    return setBuildOption(Option, std::to_string(Value));
  }

  // Does partial transformation to backend-flavored LLVM IR
  bool partialTransformation(const std::string& LLVMIR, std::string& out);

  // Does full transformation to backend specific format
  bool fullTransformation(const std::string& LLVMIR, std::string& out);
  bool prepareIR(llvm::Module& M);
  bool translatePreparedIR(llvm::Module& FlavoredModule, std::string& out);


  const std::vector<std::string>& getErrorLog() const {
    return Errors;
  }

  // Returns IR that caused the error in case an error occurs
  const std::string& getFailedIR() const {
    return ErroringCode;
  }

  const std::vector<std::string>& getOutliningEntrypoints() const {
    return OutliningEntrypoints;
  }

  const std::vector<std::string>& getKernels () const {
    return Kernels;
  }

  std::string getErrorLogAsString() const {
    std::string Result;
    for(int i = 0; i < getErrorLog().size(); ++i) {
      Result += std::to_string(i);
      Result += ": ";
      Result += getErrorLog()[i] + '\n';
    }
    return Result;
  }

  int getBackendId() const {
    return S2IRConstantBackendId;
  }

  using SymbolListType = std::vector<std::string>;

  class ExternalSymbolResolver {
  public:
    using LLVMModuleId = unsigned long long;
    // SymbolToModuleMapper is responsible for return a list of identifiers of LLVM modules
    // that should be linked in order to resolve the provided symbol list.
    using SymbolsToModuleIdMapperType =
        std::function<std::vector<LLVMModuleId>(const SymbolListType &SymbolList)>;
    // BitcodeStringRetriever will return the IR bitcode string as well as the imported symbols,
    // given a unique LLVM module id.
    using BitcodeStringRetrieverType = std::function<std::string (LLVMModuleId, SymbolListType&)>;

    ExternalSymbolResolver() = default;
    ExternalSymbolResolver(const SymbolsToModuleIdMapperType &SymbolMapper,
                           const BitcodeStringRetrieverType &Retriever,
                           const SymbolListType &ImportedSymbols)
        : SymbolModuleMapper{SymbolMapper}, BitcodeRetriever{Retriever}, ImportedSymbols{
                                                                             ImportedSymbols} {}

    auto mapSymbolsToModuleIds(const SymbolListType& symbols) const {
      return SymbolModuleMapper(symbols);
    }

    auto retrieveBitcode(LLVMModuleId MID, SymbolListType& ImportedSymbolsOut) const {
      return BitcodeRetriever(MID, ImportedSymbolsOut);
    }

    // retrieve imported symbols for the primary bitcode file
    const SymbolListType& getImportedSymbols() const {
      return ImportedSymbols;
    }
  private:
    SymbolsToModuleIdMapperType SymbolModuleMapper;
    BitcodeStringRetrieverType BitcodeRetriever;
    SymbolListType ImportedSymbols;
  };

  void provideExternalSymbolResolver(ExternalSymbolResolver Resolver);

  // Enable dead argument elimination. If non-null, RetainedArgumentIndices will be filled
  // with the indices of the parameters that were not removed in ascending order.
  void enableDeadArgumentElminiation(const std::string &FunctionName,
                                     std::vector<int> *RetainedArgumentIndices = nullptr);

  const std::vector<std::pair<std::string, std::vector<int>*>>& getDeadArgumentEliminationConfig() const;
protected:
  virtual AddressSpaceMap getAddressSpaceMap() const = 0;
  virtual bool isKernelAfterFlavoring(llvm::Function& F) = 0;
  virtual bool applyBuildFlag(const std::string &Flag) { return false; }
  virtual bool applyBuildOption(const std::string &Option, const std::string &Value) { return false; }
  virtual bool applyBuildToolArguments(const std::string &ToolName,
                                       const std::vector<std::string> &Args) {
    return false;
  }

  // Link against bitcode contained in file or string. If ForcedTriple/ForcedDataLayout are non-empty,
  // sets triple and data layout in contained bitcode to the provided values.
  
  bool linkBitcodeFile(llvm::Module &M, const std::string &BitcodeFile,
                       const std::string &ForcedTriple = "",
                       const std::string &ForcedDataLayout = "",
                       bool LinkOnlyNeeded = true);
  bool linkBitcodeString(llvm::Module &M, const std::string &Bitcode,
                         const std::string &ForcedTriple = "",
                         const std::string &ForcedDataLayout = "",
                         bool LinkOnlyNeeded = true);
  // If backend needs to set IR constants, it should do so here.
  virtual bool prepareBackendFlavor(llvm::Module& M) = 0;
  // Transform LLVM IR as much as required to backend-specific flavor
  virtual bool toBackendFlavor(llvm::Module &M, PassHandler& PH) = 0;
  virtual bool translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) = 0;

  // By default, just runs regular O3 pipeline. Backends may override
  // if they want to do something more specific.
  virtual bool optimizeFlavoredIR(llvm::Module& M, PassHandler& PH);

  // Transfers kernel properties (e.g. kernel call conventions, additional metadata) from one kernel
  // "From" to another "To". This is useful e.g. for dead argument elimination, where a new
  // kernel entrypoint with different signature will be created post optimizations.
  // This assumes that To has been created with a matching function signature from From,
  // including function and parameter attributes.
  virtual void migrateKernelProperties(llvm::Function* From, llvm::Function* To) = 0;

  void registerError(const std::string& E) {
    Errors.push_back(E);
  }

  // These will be non-zero if work group sizes are known at jit time.
  // Backends should check these values for being != 0 before using them.
  int KnownGroupSizeX = 0;
  int KnownGroupSizeY = 0;
  int KnownGroupSizeZ = 0;

  // Will be >= 0 if set by option. Backends using this should therefore check >= 0.
  std::int64_t KnownLocalMemSize = -1;

  bool GlobalSizesFitInInt = false;
  bool IsFastMath = false;

private:

  void resolveExternalSymbols(llvm::Module& M);
  void setFailedIR(llvm::Module& M);
  void runKernelDeadArgumentElimination(llvm::Module &M, llvm::Function *F, PassHandler &PH,
                                        std::vector<int>& RetainedIndicesOut);

  int S2IRConstantBackendId;
  
  std::vector<std::string> OutliningEntrypoints;
  // function call specializations might result in additional outlining entrypoints
  // that we need to consider early on
  std::vector<std::string> FunctionCallSpecializationOutliningEntrypoints;
  std::vector<std::string> Kernels;

  std::vector<std::string> Errors;
  
  std::unordered_map<std::string, std::function<void(llvm::Module &)>> SpecializationApplicators;
  ExternalSymbolResolver SymbolResolver;
  bool HasExternalSymbolResolver = false;

  // In case an error occurs, the code will be stored here
  std::string ErroringCode;

  std::vector<std::pair<std::string, std::vector<int>*>> FunctionsForDeadArgumentElimination;

};

}
}

#endif
