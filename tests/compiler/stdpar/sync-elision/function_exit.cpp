// REQUIRES: stdpar
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s

#include <cstdio>
#include "common.hpp"

__attribute__((noinline))
void test() {
  stdpar_call();
}

// Compiler will currently always inline the first level of calls
// to stdpar functions, so we need another layer of function calls
// to actually trigger a function exit in LLVM IR
__attribute__((noinline)) void test_wrapper() {
  test();
}

int main() {
  test_wrapper();
  // CHECK: 0
  printf("%d\n", get_num_enqueued_ops());
}
