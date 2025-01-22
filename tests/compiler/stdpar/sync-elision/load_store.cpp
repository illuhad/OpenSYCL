// REQUIRES: stdpar
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g --acpp-stdpar --acpp-stdpar-unconditional-offload
// RUN: %t | FileCheck %s

#include <cstdio>
#include <vector>
#include "common.hpp"



int main() {
  std::vector<int> data(1024);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i;

  stdpar_call();
  stdpar_call();
  
  int pre_load_queue_size = get_num_enqueued_ops();
  if(data[pre_load_queue_size] < 0) {
    printf("(output to prevent dead code elimination: %d\n)", data[pre_load_queue_size]);
  }
  int post_load_queue_size = get_num_enqueued_ops();
  // CHECK: 2
  // CHECK: 0
  printf("%d\n", pre_load_queue_size);
  printf("%d\n", post_load_queue_size);

  stdpar_call();
  data[0] = 10;
  // CHECK: 0
  printf("%d\n", get_num_enqueued_ops());
  // CHECK: 0
  printf("%d\n", data[0]);
}
