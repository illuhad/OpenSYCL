
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <cmath>
#include <sycl/sycl.hpp>
#include "common.hpp"
#include "hipSYCL/runtime/hardware.hpp"

extern "C" bool __acpp_sscp_jit_reflect_knows_random_unknown_thing();

int main() {
  sycl::queue q = get_queue();
  int* data = sycl::malloc_shared<int>(7, q);

  q.single_task([data]{
    __acpp_if_target_device(
      data[0] = __acpp_sscp_jit_reflect_runtime_backend();
      data[1] = __acpp_sscp_jit_reflect_target_arch();
      data[2] = __acpp_sscp_jit_reflect_target_is_cpu();
      data[3] = static_cast<int>(__acpp_sscp_jit_reflect_compiler_backend());
      data[4] = __acpp_sscp_jit_reflect_target_vendor_id();

      data[5] = __acpp_sscp_jit_reflect_knows_runtime_backend();
      data[6] = __acpp_sscp_jit_reflect_knows_random_unknown_thing();
    );
  }).wait();

  auto dev = q.get_device().AdaptiveCpp_device_id();
  hipsycl::rt::runtime_keep_alive_token rt;
  hipsycl::rt::hardware_context *ctx = rt.get()
                                           ->backends()
                                           .get(dev.get_backend())
                                           ->get_hardware_manager()
                                           ->get_device(dev.get_id());

  // CHECK: 1
  std::cout << (data[0] == static_cast<int>(dev.get_backend())) << std::endl;
  // CHECK: 1
  std::cout << (data[1] ==
                static_cast<int>(ctx->get_property(
                    hipsycl::rt::device_uint_property::architecture)))
            << std::endl;
  // CHECK: 1
  std::cout << (data[2] == static_cast<int>(ctx->is_cpu())) << std::endl;

  // We don't have a mechanism yet to query compiler backends on the host, so
  // cannot test data[3] for now.

  // CHECK: 1
  std::cout << (data[4] ==
                static_cast<int>(ctx->get_property(
                    hipsycl::rt::device_uint_property::vendor_id))) << std::endl;

  // CHECK: 1
  std::cout << data[5] << std::endl;
  // CHECK: 0
  std::cout << data[6] << std::endl;

  q.single_task([=]() {
    __acpp_if_target_device(
      auto backend = sycl::AdaptiveCpp_jit::reflect<
          sycl::AdaptiveCpp_jit::reflection_query::runtime_backend>();
      data[0] = sycl::AdaptiveCpp_jit::compile_if_else(
          backend == static_cast<int>(sycl::backend::omp), 
          []() { return 1; },
          []() { return 0; });
      data[1] = sycl::AdaptiveCpp_jit::knows<
          sycl::AdaptiveCpp_jit::reflection_query::runtime_backend>();
    );
  }).wait();
  // CHECK: 1
  std::cout << (data[0] == (q.get_device().get_backend() ==
                            sycl::backend::omp))
            << std::endl;

  // CHECK: 1
  std::cout << data[1] << std::endl;

  sycl::free(data, q);
}
