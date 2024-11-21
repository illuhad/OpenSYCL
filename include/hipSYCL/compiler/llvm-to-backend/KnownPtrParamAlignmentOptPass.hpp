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
#ifndef HIPSYCL_SSCP_KNOWN_PTR_PARAM_ALIGNMENT_OPT_PASS_HPP
#define HIPSYCL_SSCP_KNOWN_PTR_PARAM_ALIGNMENT_OPT_PASS_HPP

#include <llvm/IR/PassManager.h>
#include <unordered_map>

namespace hipsycl {
namespace compiler {

class KnownPtrParamAlignmentOptPass : public llvm::PassInfoMixin<KnownPtrParamAlignmentOptPass> {
public:
  KnownPtrParamAlignmentOptPass(
      const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &KnownAlignments);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  std::unordered_map<std::string, std::vector<std::pair<int, int>>> KnownPtrParamAlignments;
};

}
}

#endif

