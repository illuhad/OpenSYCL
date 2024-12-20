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
#ifndef HIPSYCL_HOST_WRAPPER_PASS_HPP
#define HIPSYCL_HOST_WRAPPER_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

class HostKernelWrapperPass : public llvm::PassInfoMixin<HostKernelWrapperPass> {
  std::int64_t DynamicLocalMemSize;
  std::array<int, 3> KnownWgSize;

public:
  explicit HostKernelWrapperPass(std::int64_t DynamicLocalMemSize, int KnownGroupSizeX,
                                 int KnownGroupSizeY, int KnownGroupSizeZ)
      : DynamicLocalMemSize{DynamicLocalMemSize},
        KnownWgSize{KnownGroupSizeX, KnownGroupSizeY, KnownGroupSizeZ} {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace compiler
} // namespace hipsycl

#endif
