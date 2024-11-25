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
#ifndef HIPSYCL_ATOMIC_HIPLIKE_BUILTINS_HPP
#define HIPSYCL_ATOMIC_HIPLIKE_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/bit_cast.hpp"

#include <type_traits>


#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl {
namespace detail::hiplike_builtins {

inline constexpr int builtin_memory_order(memory_order o) noexcept {
  switch(o){
    case memory_order::relaxed:
      return __ATOMIC_RELAXED;
    case memory_order::acquire:
      return __ATOMIC_ACQUIRE;
    case memory_order::release:
      return __ATOMIC_RELEASE;
    case memory_order::acq_rel:
      return __ATOMIC_ACQ_REL;
    case memory_order::seq_cst:
      return __ATOMIC_SEQ_CST;
  }
  return __ATOMIC_RELAXED;
}

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA

// LLVM NVPTX backend does currently not properly support acquire/release
// atomics. We workaround this for two load/store instructions that we
// need for the algorithms library using inline assembly.
__attribute__((always_inline)) 
void __acpp_cuda_atomic_store_device_rel_i32(int32_t *ptr, int32_t x) {
  asm volatile("st.release.gpu.s32 [%0], %1;"
              :
              :"l"(ptr), "r"(x)
              : "memory");
}

__attribute__((always_inline)) 
int32_t __acpp_cuda_atomic_load_device_acq_i32(int32_t *ptr) {
  int32_t result;
  asm volatile("ld.acquire.gpu.u32 %0,[%1];"
                : "=r"(result)
                : "l"(ptr)
                : "memory");
  return result;  
}

#endif

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN void
__acpp_atomic_store(T *addr, T x, memory_order order,
                       memory_scope scope) noexcept {
  if constexpr(sizeof(T) == sizeof(int32_t)) {
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
    if(scope == memory_scope::device && order == memory_order::release){
      __acpp_cuda_atomic_store_device_rel_i32(reinterpret_cast<int32_t*>(addr),
                                              bit_cast<int32_t>(x));
      return;
    }
#endif
  }
  __atomic_store_n(addr, x, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN void
__acpp_atomic_store(float *addr, float x, memory_order order,
                       memory_scope scope) noexcept {
  __atomic_store_n(reinterpret_cast<unsigned int *>(addr),
                   bit_cast<unsigned int>(x), builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN void
__acpp_atomic_store(double *addr, double x, memory_order order,
                       memory_scope scope) noexcept {
  __atomic_store_n(reinterpret_cast<unsigned long long *>(addr),
                   bit_cast<unsigned long long>(x), builtin_memory_order(order));
}

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_load(T *addr, memory_order order,
                                                memory_scope scope) noexcept {
  if constexpr(sizeof(T) == sizeof(int32_t)) {
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
    if(scope == memory_scope::device && order == memory_order::acquire){
      return bit_cast<T>(__acpp_cuda_atomic_load_device_acq_i32(
          reinterpret_cast<int32_t*>(addr)));
    }
#endif
  }
  return __atomic_load_n(addr, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN float
__acpp_atomic_load(float *addr, memory_order order,
                      memory_scope scope) noexcept {
  return bit_cast<float>(__atomic_load_n(reinterpret_cast<unsigned int *>(addr),
                                         builtin_memory_order(order)));
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_load(double *addr, memory_order order,
                      memory_scope scope) noexcept {
  return bit_cast<double>(
      __atomic_load_n(reinterpret_cast<unsigned long long *>(addr),
                      builtin_memory_order(order)));
}

//********************* atomic exchange ***********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
  return atomicExch(addr, x);
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_bitcast_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {

  return bit_cast<T>(__acpp_atomic_exchange<S>(
      reinterpret_cast<InvokedT*>(addr), bit_cast<InvokedT>(x), order, scope));
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_staticcast_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {

  return static_cast<T>(__acpp_atomic_exchange<S>(
      reinterpret_cast<InvokedT*>(addr), static_cast<InvokedT>(x), order, scope));
}


// No direct support for long long, long and unsigned long, and double
template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN long long
__acpp_atomic_exchange(long long *addr, long long x, memory_order order,
                          memory_scope scope) noexcept {

  return __acpp_bitcast_atomic_exchange<S, unsigned long long>(
      addr, x, order, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN long
__acpp_atomic_exchange(long *addr, long x, memory_order order,
                          memory_scope scope) noexcept {
  if constexpr(sizeof(long) == 4) {
    return __acpp_staticcast_atomic_exchange<S, int>(addr, x, order, scope);
  } else {
    return __acpp_bitcast_atomic_exchange<S, unsigned long long>(addr, x, order,
                                                                    scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN unsigned long
__acpp_atomic_exchange(unsigned long *addr, unsigned long x,
                          memory_order order, memory_scope scope) noexcept {
  if constexpr (sizeof(long) == 4) {
    return __acpp_staticcast_atomic_exchange<S, unsigned int>(
        addr, x, order, scope);
  } else {
    return __acpp_staticcast_atomic_exchange<S, unsigned long long>(
        addr, x, order, scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_exchange(double *addr, double x, memory_order order,
                          memory_scope scope) noexcept {
  return __acpp_bitcast_atomic_exchange<S, unsigned long long>(
      addr, x, order, scope);
}

// ******************* atomic compare exchange strong ********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  T old = expected;
  expected = atomicCAS(addr, expected, desired);
  return old == expected;
}


template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_bitcast_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  InvokedT cast_expected = bit_cast<InvokedT>(expected);
  
  bool r = __acpp_atomic_compare_exchange_strong<S>(
      reinterpret_cast<InvokedT *>(addr), cast_expected,
      bit_cast<InvokedT>(desired), success, failure, scope);

  expected = bit_cast<T>(cast_expected);
  return r;
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_staticcast_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  InvokedT cast_expected = static_cast<InvokedT>(expected);
  
  bool r = __acpp_atomic_compare_exchange_strong<S>(
      reinterpret_cast<InvokedT *>(addr), cast_expected,
      static_cast<InvokedT>(desired), success, failure, scope);

  expected = static_cast<T>(cast_expected);
  return r;
}

// No direct support for long long, long and unsigned long, float, double

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    long long *addr, long long &expected, long long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __acpp_bitcast_compare_exchange_strong<S, unsigned long long>(
      addr, expected, desired, success, failure, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    long *addr, long &expected, long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  if constexpr (sizeof(long) == 4) {
    return __acpp_staticcast_compare_exchange_strong<S, int>(
        addr, expected, desired, success, failure, scope);
  } else {
    return __acpp_bitcast_compare_exchange_strong<S, unsigned long long>(
        addr, expected, desired, success, failure, scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    unsigned long *addr, unsigned long &expected, unsigned long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  if constexpr(sizeof(unsigned long) == 4) {
    return __acpp_staticcast_compare_exchange_strong<S, unsigned int>(
        addr, expected, desired, success, failure, scope);
  } else {
    return __acpp_staticcast_compare_exchange_strong<S, unsigned long long>(
        addr, expected, desired, success, failure, scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    float *addr, float &expected, float desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __acpp_bitcast_compare_exchange_strong<S, int>(
      addr, expected, desired, success, failure, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    double *addr, double &expected, double desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __acpp_bitcast_compare_exchange_strong<S, unsigned long long>(
      addr, expected, desired, success, failure, scope);
}

// ******************* atomic compare exchange weak ********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  return __acpp_atomic_compare_exchange_strong<S>(
      addr, expected, desired, success, failure, scope);
}

// *********************** Integral values only *****************************

// Defines overloads for integral atomics for types that are not natively
// supported using bitcasts: long long, long and unsigned long
#define HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(            \
    name, bitcast_name)                                                        \
  template <access::address_space S, class InvokedT, class T>                  \
  HIPSYCL_HIPLIKE_BUILTIN T bitcast_name(T *addr, T x, memory_order order,     \
                                         memory_scope scope) noexcept {        \
    return bit_cast<T>(name<S>(reinterpret_cast<InvokedT *>(addr),             \
                               bit_cast<InvokedT>(x), order, scope));          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN long long name(long long *addr, long long x,         \
                                         memory_order order,                   \
                                         memory_scope scope) noexcept {        \
    return bitcast_name<S, unsigned long long>(addr, x, order, scope);         \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN long name(long *addr, long x, memory_order order,    \
                                    memory_scope scope) noexcept {             \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, int>(addr, x, order, scope);                      \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN unsigned long name(                                  \
      unsigned long *addr, unsigned long x, memory_order order,                \
      memory_scope scope) noexcept {                                           \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, unsigned int>(addr, x, order, scope);             \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }

// ******************** atomic fetch and ************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_and(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
  return atomicAnd(addr, x);
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __acpp_atomic_fetch_and, __acpp_bitcast_atomic_fetch_and);

// ************************ atomic fetch or **********************


template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_or(T *addr, T x, memory_order order,
                                                    memory_scope scope) noexcept {
   return atomicOr(addr, x);
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __acpp_atomic_fetch_or, __acpp_bitcast_atomic_fetch_or);

// ************************ atomic fetch xor **********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_xor(T *addr, T x, memory_order order,
                                                     memory_scope scope) noexcept {
   return atomicXor(addr, x);
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __acpp_atomic_fetch_xor, __acpp_bitcast_atomic_fetch_xor);

// ***************** Floating point and integral values *************


// *********************** atomic add ****************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_add(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
  return atomicAdd(addr, x);
}

// atomicAdd supports float and double, so we only need to complete the integral
// overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __acpp_atomic_fetch_add, __acpp_bitcast_atomic_fetch_add);

// *********************** atomic sub ****************************

// Only supports int and unsigned
template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_sub(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
  return atomicSub(addr, x);
}

// Need CAS loop for 64bit types - the definition of this
// is required in order to be able to use the 
// HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS macro

#define HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(Type)                   \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN Type __acpp_atomic_fetch_sub(                     \
      Type *addr, Type x, memory_order order, memory_scope scope) noexcept {   \
    Type old = __acpp_atomic_load<S>(addr, order, scope);                   \
    while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, old - x,    \
                                                        order, order, scope))  \
      ;                                                                        \
    return old;                                                                \
  }

HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(long long)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(unsigned long long)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(float)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(double)

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN long
__acpp_atomic_fetch_sub(long *addr, long x, memory_order order,
                           memory_scope scope) noexcept {
  if constexpr (sizeof(long) == 4) {
    return static_cast<long>(__acpp_atomic_fetch_sub<S>(
        reinterpret_cast<int *>(addr), static_cast<int>(x), order, scope));
  } else {
    return static_cast<long>(__acpp_atomic_fetch_sub<S>(
        reinterpret_cast<long long *>(addr), static_cast<long long>(x), order,
        scope));
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN unsigned long
__acpp_atomic_fetch_sub(unsigned long *addr, unsigned long x,
                           memory_order order, memory_scope scope) noexcept {
  if constexpr (sizeof(unsigned long) == 4) {
    return static_cast<unsigned long>(__acpp_atomic_fetch_sub<S>(
        reinterpret_cast<unsigned *>(addr), static_cast<unsigned>(x), order,
        scope));
  } else {
    return static_cast<unsigned long>(
        __acpp_atomic_fetch_sub<S>(
            reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(x), order, scope));
  }
}

// *********************** atomic min ****************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_min(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
  return atomicMin(addr, x);
}

// Need completion of integral overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __acpp_atomic_fetch_min, __acpp_bitcast_atomic_fetch_min);

// Need CAS loop for emulation of floating point overloads
template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN float
__acpp_atomic_fetch_min(float *addr, float x, memory_order order,
                           memory_scope scope) noexcept {
  float old = __acpp_atomic_load<S>(addr, order, scope);
  do {
    if (old < x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_fetch_min(double *addr, double x, memory_order order,
                           memory_scope scope) noexcept {
  double old = __acpp_atomic_load<S>(addr, order, scope);
  do {
    if (old < x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

// *********************** atomic max ****************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_max(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
  return atomicMax(addr, x);
}

// Need completion of integral overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __acpp_atomic_fetch_max, __acpp_bitcast_atomic_fetch_max);

// Need CAS loop for emulation of floating point overloads
template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN float
__acpp_atomic_fetch_max(float *addr, float x, memory_order order,
                           memory_scope scope) noexcept {
  float old = __acpp_atomic_load<S>(addr, order, scope);
  do {
    if (old > x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_fetch_max(double *addr, double x, memory_order order,
                           memory_scope scope) noexcept {
  double old = __acpp_atomic_load<S>(addr, order, scope);
  do {
    if (old > x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}
}
}
}

#endif

#endif
