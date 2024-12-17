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
#ifndef HIPSYCL_COMPILATION_STATE_HPP
#define HIPSYCL_COMPILATION_STATE_HPP

#include <string>
#include <unordered_set>

namespace hipsycl {
namespace compiler {

class ASTPassState
{
  std::unordered_set<std::string> ImplicitlyMarkedHostDeviceFunctions;
  std::unordered_set<std::string> ExplicitDeviceFunctions;
  std::unordered_set<std::string> KernelFunctions;
  bool IsDeviceCompilation;
public:
  ASTPassState()
  : IsDeviceCompilation{false}
  {}

  bool isDeviceCompilation() const
  {
    return IsDeviceCompilation;
  }

  void setDeviceCompilation(bool IsDevice)
  {
    this->IsDeviceCompilation = IsDevice;
  }

  void addImplicitHostDeviceFunction(const std::string& Name)
  {
    ImplicitlyMarkedHostDeviceFunctions.insert(Name);
  }

  void addKernelFunction(const std::string& Name)
  {
    KernelFunctions.insert(Name);
  }

  void addExplicitDeviceFunction(const std::string& Name)
  {
    ExplicitDeviceFunctions.insert(Name);
  }

  bool isImplicitlyHostDevice(const std::string& FunctionName) const
  {
    return ImplicitlyMarkedHostDeviceFunctions.find(FunctionName) 
      != ImplicitlyMarkedHostDeviceFunctions.end();
  }

  bool isExplicitlyDevice(const std::string& FunctionName) const
  {
    return ExplicitDeviceFunctions.find(FunctionName)
      != ExplicitDeviceFunctions.end();
  }

  bool isKernel(const std::string& FunctionName) const
  {
    return KernelFunctions.find(FunctionName)
      != KernelFunctions.end();
  }

};

class CompilationStateManager
{
public:
  // is "okay" in header for now, as frontend & pass plugins are always
  // linked (together) statically in the end 
  static CompilationStateManager& get()
  {
    static CompilationStateManager m;
    return m;
  }

  void reset()
  {
    ASTState = ASTPassState();
  }

  static ASTPassState &getASTPassState() { return get().ASTState; }

private:
  CompilationStateManager() = default;
  ASTPassState ASTState;
};

} // namespace compiler
} // namespace hipsycl

#endif
