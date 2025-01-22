// REQUIRES: sscp
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -fsanitize=undefined
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -fsanitize=address
// RUN: %t | FileCheck %s

#include <iostream>
#include "hipSYCL/glue/reflection.hpp"

void print_introspection_result(
    const hipsycl::glue::reflection::introspect_flattened_struct &s) {
  std::cout << s.get_num_members() << " members: ";
  for(int i = 0; i < s.get_num_members(); ++i) {
    std::cout << " " << s.get_member_offset(i) << " " << s.get_member_size(i)
              << " " << static_cast<int>(s.get_member_kind(i));
  }
  std::cout << "\n";
}

struct S {
  int x;
  double y;
  void* z;
};

struct S_wrapped {
  int x;
  S s;
};

int main() {
  int s1 = 3;
  hipsycl::glue::reflection::introspect_flattened_struct i1{s1};
  double s2 = 3.0;
  hipsycl::glue::reflection::introspect_flattened_struct i2{s2};
  void* s3 = nullptr;
  hipsycl::glue::reflection::introspect_flattened_struct i3{s3};
  S s4;
  hipsycl::glue::reflection::introspect_flattened_struct i4{s4};
  S_wrapped s5;
  hipsycl::glue::reflection::introspect_flattened_struct i5{s5};

  print_introspection_result(i1);
  print_introspection_result(i2);
  print_introspection_result(i3);
  print_introspection_result(i4);
  print_introspection_result(i5);
  // CHECK: 1 members:  0 4 2
  // CHECK: 1 members:  0 8 3
  // CHECK: 1 members:  0 8 1
  // CHECK: 3 members:  0 4 2 8 8 3 16 8 1
  // CHECK: 4 members:  0 4 2 8 4 2 16 8 3 24 8 1
}
