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
#ifndef ACPP_S2_REFLECTION_HPP
#define ACPP_S2_REFLECTION_HPP

#include <llvm/IR/PassManager.h>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace hipsycl {
namespace compiler {

// Processes calls to 
// - __acpp_jit_reflect_<name> or __acpp_s2_reflect_<name> functions (different synonyms),
// replacing callsites with provided constants.
// - __acpp_jit_reflect_knows_<name> or __acpp_s2_reflect_knows_<>
class ProcessS2ReflectionPass : public llvm::PassInfoMixin<ProcessS2ReflectionPass> {
public:
  ProcessS2ReflectionPass(const std::unordered_map<std::string, uint64_t>& Fields);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  std::unordered_map<std::string, uint64_t> SupportedFields;
};

}
}

#endif

