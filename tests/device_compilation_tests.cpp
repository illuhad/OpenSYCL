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

// Test workaround for https://bugs.llvm.org/show_bug.cgi?id=50383
#include <atomic>
#include <complex>

#define BOOST_TEST_MODULE device compilation tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/included/unit_test.hpp>

#include <initializer_list>

// NOTE: Since these tests are concerned with device compilation, the first
// and arguably most important "test" is that this file compiles at all. We
// then also include a few unit tests to assert the correct behavior of the
// resulting programs.

#include <CL/sycl.hpp>
#include "common/reset.hpp"
BOOST_FIXTURE_TEST_SUITE(device_compilation_test_suite, reset_device_fixture)

int add_five(int v) {
  return v + 5;
}

void call_lambda(int& v) {
  ([&](){ v = add_five(v); })();
}

struct MyParent {
  MyParent(std::initializer_list<int> init) {
    for(auto&& e : init) {
      parent_value += e;
    }
    call_lambda(parent_value);
  }

  virtual ~MyParent() {
    *increment_in_dtor += parent_value + 7;
  };

  int parent_value = 0;
  int* increment_in_dtor;
};

struct Conversion {
  Conversion(float value) : value(value) {}

  operator int() {
    return static_cast<int>(value);
  }

  float value;
};

int convert(Conversion conv) {
  return add_five(conv);
}

struct MyStruct : MyParent {
  MyStruct(int& value) : MyParent::MyParent{value, 4, 5, 7}, value(value) {
    increment_in_dtor = &value;
  }

  ~MyStruct() {
    value = convert(static_cast<float>(value));
  }

  int& value;
};

template<typename T, typename U>
void template_fn(U& u) {
  T t(u);
}

float some_fn(int value) {
  template_fn<MyStruct>(value);
  return value;
}

// What we're testing here is different forms of function calls, including
// implicit and templated functions and constructors as well as destructors.
// The hipSYCL Clang plugin must recognize that each of these is being called
// by the user-provided kernel functor, and mark them as __device__ functions
// accordingly.
BOOST_AUTO_TEST_CASE(complex_device_call_graph) {
  cl::sycl::queue queue;
  cl::sycl::buffer<float, 1> buf(1);

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class some_kernel>([=]() {
        acc[0] = some_fn(2);
      });
  });

  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  BOOST_REQUIRE(acc[0] == 37);
}


template<int T, int* I, typename U>
class complex_kn;

template<typename T>
class templated_kn;

enum class my_enum { HELLO = 42, WORLD };

template<my_enum E>
class enum_kn;

template<template <typename, int, typename> class T>
class templated_template_kn;

BOOST_AUTO_TEST_CASE(kernel_name_mangling) {
  cl::sycl::queue queue;

  // Qualified / modified types
  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_kn<const unsigned int>>
        ([](){});
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_kn<complex_kn<32, nullptr, enum_kn<my_enum::HELLO>>>>
        ([](){});
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_template_kn<cl::sycl::buffer>>
        ([](){});
  });

  // No collision happens between the following two names
  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <templated_kn<class collision>>
        ([](){});
  });

  queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task
        <class templated_kn_collision>
        ([](){});
  });
}

template<class AccessorT>
struct KernelFunctor {

  KernelFunctor(AccessorT acc) : acc(acc) {};

  void operator() (cl::sycl::item<1> item) const {
    acc[0] = 300 + item.get_linear_id();
  }

  AccessorT acc;
};

// It's allowed to omit the name if the functor is globally visible
BOOST_AUTO_TEST_CASE(omit_kernel_name) {
  cl::sycl::queue queue;
  cl::sycl::buffer<size_t, 1> buf(1);

  {
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(cl::sycl::range<1>(1), KernelFunctor(acc));
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    BOOST_REQUIRE(acc[0] == 300);
  }

  {
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(cl::sycl::range<1>(1), cl::sycl::id<1>(1), KernelFunctor(acc));
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    BOOST_REQUIRE(acc[0] == 301);
  }
}

#ifndef __ACPP_ENABLE_LLVM_SSCP_TARGET__
BOOST_AUTO_TEST_CASE(hierarchical_invoke_shared_memory) {
  cl::sycl::queue queue;

  // The basic case, as outlined in the SYCL spec.
  {
    cl::sycl::buffer<size_t, 1> buf(4);
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for_work_group<class shmem_one>(
        cl::sycl::range<1>(2), cl::sycl::range<1>(2), [=](cl::sycl::group<1> group) {
          { // Do this in a block to make sure the Clang plugin handles this correctly.
            size_t shmem[2]; // Should be shared
            group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
              shmem[h_item.get_local().get_linear_id()] = h_item.get_global().get_linear_id();
            });
            group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
              if(h_item.get_local().get_linear_id() == 0) {
                auto offset = h_item.get_global().get_linear_id();
                acc[offset + 0] = shmem[0];
                acc[offset + 1] = shmem[1];
              }
            });
          }
        });
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    for(size_t i = 0; i < 4; ++i) {
      BOOST_REQUIRE(acc[i] == i);
    }
  }

  // In this case the functionality remains the same, but
  // is moved into a separate function. We expect it to still
  // behave as before.
  {
    auto operate_on_shmem = [](cl::sycl::group<1> group, auto acc) {
      size_t shmem[2]; // Should be shared
      group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
        shmem[h_item.get_local().get_linear_id()] = h_item.get_global().get_linear_id();
      });
      group.parallel_for_work_item([&](cl::sycl::h_item<1> h_item) {
        if(h_item.get_local().get_linear_id() == 0) {
          auto offset = h_item.get_global().get_linear_id();
          acc[offset + 0] = shmem[0];
          acc[offset + 1] = shmem[1];
        }
      });
    };

    cl::sycl::buffer<size_t, 1> buf(4);
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for_work_group<class shmem_two>(
        cl::sycl::range<1>(2), cl::sycl::range<1>(2), [=](cl::sycl::group<1> group) {
          operate_on_shmem(group, acc);
        });
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    for(size_t i = 0; i < 4; ++i) {
      BOOST_REQUIRE(acc[i] == i);
    }
  }
}
#endif

void forward_declared2();

template<class T>
void forward_declared1();

template<class T>
class forward_declared_test_kernel;

BOOST_AUTO_TEST_CASE(forward_declared_function) {
  cl::sycl::queue q;

  // Here, the clang plugin must build the correct call graph
  // and emit both forward_declared1() and forward_declared2()
  // to device code, even though the call expression below refers
  // to the forward declarations above.
  q.submit([&](cl::sycl::handler& cgh){
    cgh.single_task<forward_declared_test_kernel<int>>([=](){
      forward_declared1<int>();
    });
  });

  q.wait_and_throw();
}


template <class T>
void forward_declared1(){forward_declared2();}

void forward_declared2() {}

#ifndef HIPSYCL_DISABLE_UNNAMED_LAMBDA_TESTS
BOOST_AUTO_TEST_CASE(optional_lambda_naming) {
  cl::sycl::queue q;

  auto lambda =
      [&]() {
        q.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task([=]() {

          });
        });

        q.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task([=]() {

          });
        });

        q.submit([&](cl::sycl::handler &cgh) {
          cgh.parallel_for(cl::sycl::range<1>{1024}, [=](cl::sycl::id<1> idx) {

          });
        });

        q.submit([&](cl::sycl::handler &cgh) {
          cgh.parallel_for(cl::sycl::range<1>{1024}, [=](cl::sycl::id<1> idx) {

          });
        });
      };
  lambda();

  q.wait_and_throw();
}
#endif

template <class ...>
struct VariadicKernelTP {};

template <auto ...>
struct VariadicKernelNTTP {};

BOOST_AUTO_TEST_CASE(variadic_kernel_name) {
  cl::sycl::queue queue;
  cl::sycl::buffer<size_t, 1> buf(1);
  {
    queue.submit([&](cl::sycl::handler& cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<VariadicKernelTP<int, bool, char>>(cl::sycl::range<1>(1), [=](cl::sycl::item<1>) {});
    });
  }
  {
    queue.submit([&](cl::sycl::handler& cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<VariadicKernelNTTP<1, true, 'a'>>(cl::sycl::range<1>(1), [=](cl::sycl::item<1>) {});
    });
  }
}

void h(){}
void f(){ h(); }
void g(){}

BOOST_AUTO_TEST_CASE(generic_lambda_outlining) {
  cl::sycl::queue q;
  q.parallel_for(cl::sycl::range{1024}, [=](auto item){
    f();

    int x = 3;
    auto invoke = [&](auto value){
      g();
    };
    invoke(x);
  });
}

BOOST_AUTO_TEST_CASE(nd_range) {
  cl::sycl::queue q;
  cl::sycl::buffer<size_t, 1> buf(1);

  {
    q.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(cl::sycl::nd_range<1>{{1},{1}}, [=](cl::sycl::nd_item<1> item) {
        acc[0] = 300 + item.get_global_linear_id();
      });
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    BOOST_REQUIRE(acc[0] == 300);
  }

  {
    q.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.parallel_for(cl::sycl::nd_range<2>{{1,1},{1,1}}, [=](cl::sycl::nd_item<2> item) {
        acc[0] = 300 + item.get_global_linear_id();
      });
    });
    auto acc = buf.get_access<cl::sycl::access::mode::read>();
    BOOST_REQUIRE(acc[0] == 300);
  }

}
BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line

