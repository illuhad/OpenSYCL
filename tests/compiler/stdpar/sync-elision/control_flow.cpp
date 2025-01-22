// REQUIRES: stdpar
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s

#include <cstdio>
#include "common.hpp"

__attribute__((noinline))
void test(int n = 10 /* to avoid all branches being optimized out*/) {
  stdpar_call();
  for(int i=0; i < n; ++i)
    stdpar_call();
  if(n > 10) {
    // Is not executed, and thus should not trigger sync
    printf("Triggering synchronization\n");
  }
  // CHECK: 11
  printf("%d\n", get_num_enqueued_ops());
}

int main() {
  test();
  // printf() call to external function will trigger synchronization
  // CHECK: 0
  printf("%d\n", get_num_enqueued_ops());
}
