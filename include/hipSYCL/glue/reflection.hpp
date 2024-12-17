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
#ifndef HIPSYCL_REFLECTION_HPP
#define HIPSYCL_REFLECTION_HPP

#include "hipSYCL/runtime/support/reflection.hpp"

template <class StructT>
void __acpp_introspect_flattened_struct(void *s, int **num_flattened_members,
                                        int **member_offsets,
                                        int **member_sizes, int **member_kinds);

namespace hipsycl::glue::reflection {

enum class type_kind : int { other = 0, pointer = 1, integer_type = 2, float_type = 3 };

class introspect_flattened_struct {
public:
  template<class StructT>
  introspect_flattened_struct(const StructT& s) 
  : _num_members{nullptr} {
    // Compiler needs this copy to figure out
    // the struct type in the presence of opaque pointers.
    // Currently it checks if the first operand of the call
    // to the builtin comes from an alloca instruction.
    StructT s_copy = s;
    __acpp_introspect_flattened_struct<StructT>(&s_copy, &_num_members,
                                                   &_member_offsets, &_member_sizes,
                                                   reinterpret_cast<int **>(&_member_kinds));
  }

  int get_num_members() const {
    if (!_num_members)
      return 0;
    return *_num_members;
  }

  int get_member_offset(int idx) const {
    return _member_offsets[idx];
  }

  int get_member_size(int idx) const {
    return _member_sizes[idx];
  }

  type_kind get_member_kind(int idx) const {
    return _member_kinds[idx];
  }
private:
  int* _num_members;
  int* _member_offsets;
  int* _member_sizes;
  type_kind* _member_kinds;
};

template<class Ret, typename... Args>
const char* resolve_function_name(Ret (*func)(Args...)) {
  return rt::support::symbol_information::get().resolve_symbol_name((void*)func);
}

}


// Will be defined by compiler
extern "C" void __acpp_reflection_init_registered_function_pointers();
extern "C" void __acpp_function_annotation_needs_function_ptr_argument_reflection();

// Compiler will generate calls to this function to register functions.
__attribute__((used))
inline void __acpp_reflection_associate_function_pointer(const void *func_ptr,
                                             const char *func_name) {
  hipsycl::rt::support::symbol_information::get()
      .register_function_symbol(func_ptr, func_name);
}

struct __acpp_reflection_tu_init_trigger {
  __acpp_reflection_tu_init_trigger() {
    __acpp_reflection_init_registered_function_pointers();
  }
};

static __acpp_reflection_tu_init_trigger __reflection_init;


namespace hipsycl::glue::reflection {

template<class Ret, typename... Args>
__attribute__((noinline))
void enable_function_symbol_reflection(Ret (*func)(Args...)) {
  __acpp_function_annotation_needs_function_ptr_argument_reflection();
}

}

#endif
