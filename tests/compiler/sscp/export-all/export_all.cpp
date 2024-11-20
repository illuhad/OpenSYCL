// RUN: %acpp %s %S/second_tu.cpp -o %t --acpp-targets=generic --acpp-export-all
// RUN: %t | FileCheck %s
// RUN: %acpp %s %S/second_tu.cpp -o %t --acpp-targets=generic -O3 --acpp-export-all
// RUN: %t | FileCheck %s
// RUN: %acpp %s %S/second_tu.cpp -o %t --acpp-targets=generic -g --acpp-export-all
// RUN: %t | FileCheck %s

#include <iostream>
#include <sycl/sycl.hpp>
#include "../common.hpp"

// defined in second_tu.cpp
int increment(int x);

int main() {
  sycl::queue q = get_queue();
  int* data = sycl::malloc_shared<int>(1, q);
  q.single_task([=](){
    *data = increment(123);
  });
  q.wait();

  // CHECK: 124
  std::cout << *data << std::endl;
  sycl::free(data, q);
}
