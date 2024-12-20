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
#include "hipSYCL/compiler/sscp/StdBuiltinRemapperPass.hpp"
#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <unordered_set>
#include <unordered_map>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)


namespace hipsycl {
namespace compiler {

// List of builtins taken from https://libc.llvm.org/math/
static constexpr std::array math_builtins = {
    "ceil",       "ceilf",       "ceill",       "copysign",  "copysignf",  "copysignl",
    "fabs",       "fabsf",       "fabsl",       "fdim",      "fdimf",      "fdiml",
    "floor",      "floorf",      "floorl",      "fmax",      "fmaxf",      "fmaxl",
    "fmin",       "fminf",       "fminl",       "fmod",      "fmodf",      "fmodl",
    "frexp",      "frexpf",      "frexpl",      "ilogb",     "ilogbf",     "ilogbl",
    "ldexp",      "ldexpf",      "ldexpl",      "llrint",    "llrintf",    "llrintl",
    "llround",    "llroundf",    "llroundl",    "logb",      "logbf",      "logbl",
    "lrint",      "lrintf",      "lrintl",      "lround",    "lroundf",    "lroundl",
    "modf",       "modff",       "modfl",       "nan",       "nanf",       "nanl",
    "nearbyint",  "nearbyintf",  "nearbyintl",  "nextafter", "nextafterf", "nextafterl",
    "nexttoward", "nexttowardf", "nexttowardl", "remainder", "remainderf", "remainderl",
    "remquo",     "remquof",     "remquol",     "rint",      "rintf",      "rintl",
    "round",      "roundf",      "roundl",      "scalbn",    "scalbnf",    "scalbnl",
    "trunc",      "truncf",      "truncl",      "acos",      "acosf",      "acosl",
    "acosh",      "acoshf",      "acoshl",      "asin",      "asinf",      "asinl",
    "asinh",      "asinhf",      "asinhl",      "atan",      "atanf",      "atanl",
    "atan2",      "atan2f",      "atan2l",      "atanh",     "atanhf",     "atanhl",
    "cbrt",       "cbrtf",       "cbrtl",       "cos",       "cosf",       "cosl",
    "cosh",       "coshf",       "coshl",       "erf",       "erff",       "erfl",
    "erfc",       "erfcf",       "erfcl",       "exp",       "expf",       "expl",
    "exp10",      "exp10f",      "exp10l",      "exp2",      "exp2f",      "exp2l",
    "expm1",      "expm1f",      "expm1l",      "fma",       "fmaf",       "fmal",
    "hypot",      "hypotf",      "hypotl",      "lgamma",    "lgammaf",    "lgammal",
    "log",        "logf",        "logl",        "log10",     "log10f",     "log10l",
    "log1p",      "log1pf",      "log1pl",      "log2",      "log2f",      "log2l",
    "pow",        "powf",        "powl",        "sin",       "sinf",       "sinl",
    "sincos",     "sincosf",     "sincosl",     "sinh",      "sinhf",      "sinhl",
    "sqrt",       "sqrtf",       "sqrtl",       "tan",       "tanf",       "tanl",
    "tanh",       "tanhf",       "tanhl",       "tgamma",    "tgammaf",    "tgammal"};

using builtin_mapping = std::array<const char*, 2>;
// We may want to complete this with soft-float functions defined here:
// https://gcc.gnu.org/onlinedocs/gccint/Soft-float-library-routines.html
static constexpr std::array explicitly_mapped_builtins = {
  // clang sometimes (e.g. -ffast-math) these builtins
  builtin_mapping{"__powisf2", "__acpp_sscp_pown_f32"},
  builtin_mapping{"__powidf2", "__acpp_sscp_pown_f64"},

#define ACPP_DECLARE_FINITE_BUILTIN_MAPPING(name) \
  builtin_mapping{STRINGIFY(__ ## name ## f_finite), STRINGIFY(__acpp_sscp_ ## name ## _f32)}, \
  builtin_mapping{STRINGIFY(__ ## name ## _finite), STRINGIFY(__acpp_sscp_ ## name ## _f64)}

  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(acos),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(acosh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(asin),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(asinh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(atan2),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(atanh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(cosh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(cos),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(exp10),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(exp2),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(exp),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(fmod),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(log10),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(log2),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(log),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(hypot),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(pow),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(remainder),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(sinh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(sin),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(sqrt),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(tan)
};

llvm::PreservedAnalyses StdBuiltinRemapperPass::run(llvm::Module &M,
                                                    llvm::ModuleAnalysisManager &MAM) {

  std::unordered_set<std::string> GenericBuiltinsToReplace;
  for(const char* name : math_builtins) {
    GenericBuiltinsToReplace.insert(std::string{name});
  }

  auto ParseBuiltinName = [&](const std::string &Name, std::string &BaseNameOut,
                              std::string &SuffixOut) -> bool {
    if(GenericBuiltinsToReplace.count(Name+"f") > 0) {
      SuffixOut = "f64";
      BaseNameOut = Name;
    } else {
      if(Name.find("f") == Name.size() - 1) {
        SuffixOut = "f32";
        BaseNameOut = Name.substr(0, Name.size() - 1);
      } else {
        return false;
      }
    }
    return true;
  };

  std::unordered_map<std::string, std::string> Replacements;

  for(const auto& B : GenericBuiltinsToReplace) {
    std::string Suffix;
    std::string BaseName;
    if(ParseBuiltinName(B, BaseName, Suffix)) {
      Replacements[B] = "__acpp_sscp_" + BaseName + "_" + Suffix;
    }
  }
   for(const auto& EM : explicitly_mapped_builtins) {
    Replacements[EM[0]] = EM[1];
  }

  for(const auto& B: Replacements) {
    // Find function to replace
    if(llvm::Function* F = M.getFunction(B.first)) {
      // See if we have replacement declaration
      llvm::Function* Replacement = M.getFunction(B.second);
      // If not, create declaration
      if(!Replacement) {
        Replacement = llvm::Function::Create(F->getFunctionType(), F->getLinkage(), B.second, M);
        Replacement->setLinkage(llvm::GlobalValue::ExternalLinkage);
      }
      if(F->getFunctionType() == Replacement->getFunctionType()) {
        HIPSYCL_DEBUG_INFO << "StdBuiltinRemapper: Remapping calls from " << B.first << " to "
                           << B.second << "\n";
        F->replaceAllUsesWith(Replacement);
      }
    }
  }

  return llvm::PreservedAnalyses::none();
}
}
}

