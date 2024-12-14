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
#ifndef HIPSYCL_LLVM_TO_HOST_FACTORY_HPP
#define HIPSYCL_LLVM_TO_HOST_FACTORY_HPP

#include <memory>
#include <vector>
#include <string>
#include "../LLVMToBackend.hpp"

namespace hipsycl {
namespace compiler {

HIPSYCL_BACKEND_API_EXPORT std::unique_ptr<LLVMToBackendTranslator>
createLLVMToHostTranslator(const std::vector<std::string> &KernelNames);

}
}

#endif