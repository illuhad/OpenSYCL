// REQUIRES: sscp
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>

#include <sycl/sycl.hpp>
#include "common.hpp"

constexpr int table [] = {10, 20, 30, 40};

int main()
{
  sycl::queue q = get_queue();

  int* data = sycl::malloc_device<int>(1024, q);
  q.parallel_for(sycl::range{1024}, [=](auto idx){
    data[idx] = idx + table[idx % 4];
  }).wait();

  std::vector<int> result(1024);
  q.memcpy(result.data(), data, result.size()).wait();
  for(int i = 0; i < 8; ++i) {
    // CHECK: 10
    // CHECK: 21
    // CHECK: 32
    // CHECK: 43
    // CHECK: 14
    // CHECK: 25
    // CHECK: 36
    // CHECK: 47
    std::cout << result[i] << std::endl;
  }
}
