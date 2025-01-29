// REQUIRES: cbs
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s

#include <iostream>

#include <CL/sycl.hpp>

int main()
{
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<char> host_buf(global_size);

  for(size_t i = 0; i < 2 * local_size; ++i)
    host_buf[i] = false;
  for(size_t i = 2 * local_size; i < 4 * local_size; ++i)
    host_buf[i] = true;

  host_buf[10] = true;
  host_buf[2 * local_size + 10] = false;

  {
    cl::sycl::buffer<char, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<bool, 1, mode::read_write, target::local>{1, cgh};

      cgh.parallel_for<class test_kernel>(
        cl::sycl::nd_range<1>{global_size, local_size},
        [=](cl::sycl::nd_item<1> item) noexcept
        {
          auto g = item.get_group();
          const auto id = item.get_global_id();
          const bool pred = acc[id];

          scratch[0] = false;
          cl::sycl::group_barrier(g);

          if (pred)
            scratch[0] = pred;

          cl::sycl::group_barrier(g);

          acc[id] = scratch[0];
        });
    });
  }
  // CHECK: 1
  // CHECK: 0
  // CHECK: 1
  // CHECK: 1
  std::cout << static_cast<bool>(host_buf[0]) << "\n";
  std::cout << static_cast<bool>(host_buf[local_size]) << "\n";
  std::cout << static_cast<bool>(host_buf[local_size * 2]) << "\n";
  std::cout << static_cast<bool>(host_buf[local_size * 3]) << "\n";
}
