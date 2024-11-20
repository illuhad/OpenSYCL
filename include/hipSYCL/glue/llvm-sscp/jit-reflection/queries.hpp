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
#ifndef ACPP_GLUE_JIT_REFLECTION_QUERIES_HPP
#define ACPP_GLUE_JIT_REFLECTION_QUERIES_HPP


namespace hipsycl{
namespace sycl {
namespace AdaptiveCpp_jit {

enum class compiler_backend : int {
  spirv = 0,
  ptx = 1,
  amdgpu = 2,
  host = 3
};

namespace vendor_id {

inline constexpr int nvidia = 4318;
inline constexpr int amd = 1022;
inline constexpr int intel = 8086;

}

}
}
}



extern "C" bool __acpp_sscp_jit_reflect_knows_target_vendor_id();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_arch();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_has_independent_forward_progress();
extern "C" bool __acpp_sscp_jit_reflect_knows_runtime_backend();
extern "C" bool __acpp_sscp_jit_reflect_knows_compiler_backend();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_is_cpu();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_is_gpu();

extern "C" int __acpp_sscp_jit_reflect_target_vendor_id();
extern "C" int __acpp_sscp_jit_reflect_target_arch();
extern "C" bool __acpp_sscp_jit_reflect_target_is_cpu();
extern "C" bool __acpp_sscp_jit_reflect_target_is_gpu();
extern "C" bool __acpp_sscp_jit_reflect_target_has_independent_forward_progress();
extern "C" int __acpp_sscp_jit_reflect_runtime_backend();
extern "C" hipsycl::sycl::AdaptiveCpp_jit::compiler_backend __acpp_sscp_jit_reflect_compiler_backend();

namespace hipsycl {
namespace sycl {
namespace AdaptiveCpp_jit {


namespace reflection_query {

#define ACPP_DEFINE_REFLECT_QUERY(name)                                        \
  struct name {                                                                \
    __attribute__((always_inline)) static bool is_known() {                    \
      return __acpp_sscp_jit_reflect_knows_##name();                           \
    }                                                                          \
    __attribute__((always_inline)) static auto get() {                         \
      return __acpp_sscp_jit_reflect_##name();                                 \
    }                                                                          \
  };

ACPP_DEFINE_REFLECT_QUERY(target_vendor_id)
ACPP_DEFINE_REFLECT_QUERY(target_arch)
ACPP_DEFINE_REFLECT_QUERY(target_has_independent_forward_progress)
ACPP_DEFINE_REFLECT_QUERY(target_is_cpu)
ACPP_DEFINE_REFLECT_QUERY(target_is_gpu)
ACPP_DEFINE_REFLECT_QUERY(runtime_backend)
ACPP_DEFINE_REFLECT_QUERY(compiler_backend)

#undef ACPP_DEFINE_REFLECT_QUERY

}

template<class Query>
__attribute__((always_inline))
auto reflect() {
  return Query::get();
}

template<class Query>
__attribute__((always_inline))
bool knows() {
  return Query::is_known();
}

template<class F>
__attribute__((always_inline))
void compile_if(bool condition, F&& f) {
  if(condition) {
    f();
  }
}

template<class F, class G>
__attribute__((always_inline))
auto compile_if_else(bool condition, F&& if_branch, G&& else_branch) {
  if(condition) {
    return if_branch();
  } else{
    return else_branch();
  }
}

}
}
}
#endif
