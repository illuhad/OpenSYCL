// REQUIRES: cbs
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %t | FileCheck %s
// Only O3 generated some invalid accesses at some point, so only test this..

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

int main()
{
  constexpr size_t dot_wgsize = 4;
  constexpr size_t dot_num_groups = 32;
  constexpr size_t array_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  std::vector<int> host_buf2;
  std::vector<int> host_outbuf(dot_num_groups);
  for(size_t i = 0; i < array_size; ++i)
  {
    host_buf.push_back(static_cast<int>(i));
    host_buf2.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> d_a{host_buf.data(), host_buf.size()};
    cl::sycl::buffer<int, 1> d_b{host_buf2.data(), host_buf2.size()};
    cl::sycl::buffer<int, 1> d_sum{host_outbuf.data(), host_outbuf.size()};

    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      auto ka = d_a.template get_access<mode::read>(cgh);
      auto kb = d_b.template get_access<mode::read>(cgh);
      auto ksum = d_sum.template get_access<mode::discard_write>(cgh);

      auto wg_sum = cl::sycl::accessor<int, 1, mode::read_write, target::local>(cl::sycl::range<1>(dot_wgsize), cgh);

      size_t N = array_size;
      cgh.parallel_for<class dot_kernel>(
        cl::sycl::nd_range<1>{dot_num_groups * dot_wgsize, dot_wgsize},
        [=](cl::sycl::nd_item<1> item) noexcept {
          size_t i = item.get_global_id(0);
          size_t li = item.get_local_id(0);
          size_t global_size = item.get_global_range()[0];

          wg_sum[li] = 0.0;
          for(; i < N; i += global_size)
            wg_sum[li] += ka[i] * kb[i];

          size_t local_size = item.get_local_range()[0];
          for(int offset = local_size / 2; offset > 0; offset /= 2)
          {
            item.barrier(cl::sycl::access::fence_space::local_space);
            if(li < offset)
              wg_sum[li] += wg_sum[li + offset];
          }

          if(li == 0)
            ksum[item.get_group(0)] = wg_sum[0];
        });
    });
  }
  for(size_t i = 0; i < dot_num_groups; ++i)
  {
    // CHECK: 9218160
    // CHECK: 9333744
    // CHECK: 9450352
    // CHECK: 9567984
    // CHECK: 9686640
    // CHECK: 9806320
    // CHECK: 9927024
    // CHECK: 10048752
    // CHECK: 10171504
    // CHECK: 10295280
    // CHECK: 10420080
    // CHECK: 10545904
    // CHECK: 10672752
    // CHECK: 10800624
    // CHECK: 10929520
    // CHECK: 11059440
    // CHECK: 11190384
    // CHECK: 11322352
    // CHECK: 11455344
    // CHECK: 11589360
    // CHECK: 11724400
    // CHECK: 11860464
    // CHECK: 11997552
    // CHECK: 12135664
    // CHECK: 12274800
    // CHECK: 12414960
    // CHECK: 12556144
    // CHECK: 12698352
    // CHECK: 12841584
    // CHECK: 12985840
    // CHECK: 13131120
    // CHECK: 13277424
    std::cout << host_outbuf[i] << "\n";
  }
}
