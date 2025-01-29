// REQUIRES: sscp
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t

#include <sycl/sycl.hpp>
#include "common.hpp"

// Tests whether the SSCP compiler can separate host and device code
// if circular globals are present, which can cause JIT failure
// if they remain in device code.
struct S {
  void* ptr;
};

S s_v3 {
  &s_v3.ptr
};

int main() {
  sycl::queue q = get_queue();
    
  int* data=sycl::malloc_device<int>(1024, q);
  q.parallel_for(1024, [=](auto idx){data[idx] = static_cast<int>(idx);});
    
  q.wait();
}
