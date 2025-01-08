// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: ACPP_VISIBILITY_MASK=omp; %t | FileCheck %s

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

int main()
{
  constexpr size_t local_size = 8;
  constexpr size_t global_size = 16;

  cl::sycl::range global_range{global_size, global_size, global_size};
  cl::sycl::range local_range{local_size, local_size, local_size};

  cl::sycl::range data_range{global_size + 2, global_size + 2, global_size + 2};

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  host_buf.reserve(data_range.size());
  for(size_t i = 0; i < data_range.size(); ++i)
  {
    host_buf.push_back(static_cast<int>(i));
  }

  std::vector<int> host_result(global_range.size());
  {
    cl::sycl::buffer<int, 3> buf{host_buf.data(), data_range};
    cl::sycl::buffer<int, 3> bufOut{host_result.data(), global_range};

    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read>(cgh);
      auto out = bufOut.get_access<mode::discard_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 3, mode::read_write, target::local>{local_range + cl::sycl::range{2, 2, 2}, cgh};

      cgh.parallel_for<class local_mem_stencil>(
        cl::sycl::nd_range<3>{global_range, local_range},
        [=](cl::sycl::nd_item<3> item) noexcept {
          const auto lid = item.get_local_id();
          const auto group_size = item.get_local_range();
          const auto offset = cl::sycl::id{1, 1, 1};

          scratch[lid + offset] = acc[item.get_global_id() + offset];
          auto loadBorder = [&](int dim) {
            if(lid.get(dim) == 0)
            {
              cl::sycl::id<3> innerOffset{};
              innerOffset[dim] = 1;
              scratch[lid + innerOffset] = acc[item.get_global_id() + innerOffset];
              innerOffset[dim] = 1 + group_size.get(0);
              scratch[lid + innerOffset] = acc[item.get_global_id() + innerOffset];
            }
          };
          loadBorder(0);
          loadBorder(1);
          loadBorder(2);

          item.barrier();

          auto accumulator = scratch[lid + offset];

          accumulator += scratch[lid + offset - cl::sycl::id{1, 0, 0}];
          accumulator += scratch[lid + offset - cl::sycl::id{0, 1, 0}];
          accumulator += scratch[lid + offset - cl::sycl::id{0, 0, 1}];

          accumulator += scratch[lid + offset + cl::sycl::id{1, 0, 0}];
          accumulator += scratch[lid + offset + cl::sycl::id{0, 1, 0}];
          accumulator += scratch[lid + offset + cl::sycl::id{0, 0, 1}];

          out[item.get_global_id()] = accumulator;
        });
    });
  }
  // CHECK: 2401
  // CHECK: 2457
  // CHECK: 3409
  // CHECK: 3465
  // CHECK: 20545
  // CHECK: 20601
  // CHECK: 21553
  // CHECK: 21609
  for(size_t i = 0; i < global_size / local_size; ++i)
  {
    for(size_t j = 0; j < global_size / local_size; ++j)
    {
      for(size_t k = 0; k < global_size / local_size; ++k)
      {
        std::cout << host_result[i * local_size * global_size * global_size + j * local_size * global_size + k * local_size] << std::endl;
      }
    }
  }
}
