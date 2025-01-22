// REQUIRES: stdpar
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s

#include <cstdio>
#include "common.hpp"


int main() {
  stdpar_call();
  stdpar_call();
  
  // CHECK: 2
  printf("%d\n", get_num_enqueued_ops());
  // printf call should have triggered synchronization
  // CHECK: 0
  printf("%d\n", get_num_enqueued_ops());
}
