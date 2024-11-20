// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <sycl/sycl.hpp>
#include "common.hpp"

SYCL_EXTERNAL void myfunction1(int* data, sycl::item<1> idx) {
  *data += 1;
}

SYCL_EXTERNAL void myfunction2(int* data, sycl::item<1> idx) {
  *data += 2;
}



__attribute__((noinline))
void execute_operations_with_definition(int* data, sycl::item<1> idx) {
  sycl::AdaptiveCpp_jit::arguments_are_used(data, idx);
}

void execute_operations_without_definition(int* data, sycl::item<1> idx);


int main() {
  sycl::queue q = get_queue();
  int*data = sycl::malloc_shared<int>(1, q);

  {
    *data = 0;
  
    sycl::AdaptiveCpp_jit::dynamic_function_config dyn_function_config;
    dyn_function_config.define(&execute_operations_without_definition, &myfunction1);
    q.parallel_for(sycl::range{1}, dyn_function_config.apply([=](sycl::item<1> idx){
      execute_operations_without_definition(data, idx);
    }));

    q.wait();
    // CHECK: 1
    std::cout << *data << std::endl;
  }

  {
    *data = 0;
  
    sycl::AdaptiveCpp_jit::dynamic_function_config dyn_function_config;
    dyn_function_config.define(&execute_operations_without_definition, &myfunction1);
    q.parallel_for(sycl::range{1}, dyn_function_config.apply([=](sycl::item<1> idx){
      execute_operations_without_definition(data, idx);
    }));

    q.wait();
    // CHECK: 1
    std::cout << *data << std::endl;
  }

  {
    *data = 0;
  
    sycl::AdaptiveCpp_jit::dynamic_function_config dyn_function_config;
    dyn_function_config.define_as_call_sequence(&execute_operations_without_definition, {&myfunction1, &myfunction2});
    q.parallel_for(sycl::range{1}, dyn_function_config.apply([=](sycl::item<1> idx){
      execute_operations_without_definition(data, idx);
    }));

    q.wait();
    // CHECK: 3
    std::cout << *data << std::endl;
  }



  sycl::free(data, q);
}

