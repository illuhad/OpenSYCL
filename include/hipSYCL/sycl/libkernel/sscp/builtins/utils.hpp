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
#include "builtin_config.hpp"


#define  ACPP_CUDALIKE_SHMEM_ATTRIBUTE static __attribute__((loader_uninitialized))  __attribute__((address_space(3)))

HIPSYCL_SSCP_BUILTIN void* __acpp_sscp_host_get_internal_local_memory();

#ifndef HIPSYCL_SSCP_UTILS_BUILTINS_HPP
#define HIPSYCL_SSCP_UTILS_BUILTINS_HPP

#if __has_builtin(__builtin_bit_cast)

#define HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, in, out)                           \
  out = __builtin_bit_cast(Tout, in)

#else

#define HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, in, out)                           \
  {                                                                            \
    union {                                                                    \
      Tout union_out;                                                          \
      Tin union_in;                                                            \
    } u;                                                                       \
    u.union_in = in;                                                           \
    out = u.union_out;                                                         \
  }
#endif

template <class Tout, class Tin>
Tout bit_cast(Tin x) {
  Tout result;
  HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, x, result);
  return result;
}

__acpp_uint64 get_active_mask();

struct plus
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs+rhs;
    }
};

struct min
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs < rhs ? lhs : rhs;
    }
};

struct max
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs < rhs ? rhs : lhs;
    }
};

struct multiply
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs * rhs;
    }
};

struct bit_and
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs & rhs ;
    }
};

struct bit_or
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs | rhs;
    }
};

struct bit_xor
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs ^ rhs;
    }
};

struct logical_and
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs and rhs;
    }
};

struct logical_or
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs or rhs;
    }
};

template<__acpp_sscp_algorithm_op op>
struct get_op{
};

#define MAP_SSCP_ALGORITHM_OP(sscp_algo_op,impl) \
template<> \
struct get_op< sscp_algo_op >{ \
    using type =impl; \
}; \

MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::plus,plus)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::multiply,multiply)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::min,min)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::max,max)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::bit_and,bit_and)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::bit_or,bit_or)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::bit_xor,bit_xor)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::logical_and,logical_and)
MAP_SSCP_ALGORITHM_OP(__acpp_sscp_algorithm_op::logical_or,logical_or)


template<typename T>
struct integer_type{
    using type = T;
};

// __acpp_f16 is unsigned short currently so
// This will resolve to the same type
// template<>
// struct integer_type<__acpp_f16>{
//   using type = __acpp_int16;
// };

template<>
struct integer_type<__acpp_f32>{
  using type = __acpp_int32;
};

template<>
struct integer_type<__acpp_f64>{
  using type = __acpp_int64;
};

template<>
struct integer_type<__acpp_uint8>{
  using type = __acpp_int8;
};

template<>
struct integer_type<__acpp_uint16>{
  using type = __acpp_int16;
};

template<>
struct integer_type<__acpp_uint32>{
  using type = __acpp_int32;
};

template<>
struct integer_type<__acpp_uint64>{
  using type = __acpp_int64;
};



#endif
