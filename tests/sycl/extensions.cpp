/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause


#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/backend.hpp"
#include "hipSYCL/sycl/buffer.hpp"
#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/info/event.hpp"
#include "hipSYCL/sycl/libkernel/accessor.hpp"
#include "hipSYCL/sycl/libkernel/cuda/cuda_backend.hpp"
#include "hipSYCL/sycl/libkernel/hip/hip_backend.hpp"
#include "hipSYCL/sycl/property.hpp"
#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/queue.hpp"

#include "sycl_test_suite.hpp"
#include <boost/test/tools/old/interface.hpp>

BOOST_FIXTURE_TEST_SUITE(extension_tests, reset_device_fixture)

#ifdef ACPP_EXT_AUTO_PLACEHOLDER_REQUIRE
BOOST_AUTO_TEST_CASE(auto_placeholder_require_extension) {
  namespace s = cl::sycl;

  s::queue q;
  s::buffer<int, 1> buff{1};
  s::accessor<int, 1, s::access::mode::read_write, 
    s::access::target::global_buffer, s::access::placeholder::true_t> acc{buff};

  // This will call handler::require(acc) for each
  // subsequently launched command group
  auto automatic_requirement = s::vendor::hipsycl::automatic_require(q, acc);
  BOOST_CHECK(automatic_requirement.is_required());

  q.submit([&](s::handler &cgh) {
    cgh.single_task<class auto_require_kernel0>([=]() {
      acc[0] = 1;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 1);
  }

  q.submit([&] (s::handler& cgh) {
    cgh.single_task<class auto_require_kernel1>([=] (){
      acc[0] = 2;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 2);
  }

  automatic_requirement.release();
  BOOST_CHECK(!automatic_requirement.is_required());

  { 
    auto host_acc = buff.get_access<s::access::mode::read_write>(); 
    host_acc[0] = 3;
  }

  automatic_requirement.reacquire();
  BOOST_CHECK(automatic_requirement.is_required());

  q.submit([&] (s::handler& cgh) {
    cgh.single_task<class auto_require_kernel2>([=] (){
      acc[0] += 1;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 4);
  }
}
#endif
#if defined(ACPP_EXT_CUSTOM_PFWI_SYNCHRONIZATION) &&                           \
    !defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
BOOST_AUTO_TEST_CASE(custom_pfwi_synchronization_extension) {
  namespace sync = cl::sycl::vendor::hipsycl::synchronization;

  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](cl::sycl::handler& cgh) {

      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto scratch =
          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>{local_size,
                                                                    cgh};

      cgh.parallel_for_work_group<class pfwi_dispatch>(
        cl::sycl::range<1>{global_size / local_size},
        cl::sycl::range<1>{local_size},
        [=](cl::sycl::group<1> wg) {

          wg.parallel_for_work_item<sync::local_barrier>(
            [&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] = acc[item.get_global_id()];
          });

          // By default, a barrier is used
          wg.parallel_for_work_item(
            [&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] *= 2;
          });

          // Testing the behavior of mem_fence() or 
          // that there is no synchronization is difficult,
          // so let's just test that things compile for now.
          wg.parallel_for_work_item<sync::none>(
            [&](cl::sycl::h_item<1> item) {
            acc[item.get_global_id()] = scratch[item.get_local_id()[0]];
          });

          wg.parallel_for_work_item<sync::local_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });

          wg.parallel_for_work_item<sync::global_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });

          wg.parallel_for_work_item<sync::global_and_local_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });
        });
    });
  }

  for(size_t i = 0; i < global_size; ++i) {
    BOOST_TEST(host_buf[i] == 2*i);
  }
}
#endif

#if defined(ACPP_EXT_SCOPED_PARALLELISM_V2) &&                                 \
    !defined(                                                                  \
        ACPP_LIBKERNEL_CUDA_NVCXX) && /*nvc++ currently crashed with sp code*/ \
    !defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)

template<class KernelName, int N>
class enumerated_kernel_name;

template<class KernelName, int Dim>
void test_distribute_groups(){
  namespace s = cl::sycl;
  s::queue q;

  s::range<Dim> input_size;
  s::range<Dim> group_size;

  if constexpr(Dim == 1){
    input_size = s::range{1024};
    group_size = s::range{128};
  } else if constexpr(Dim == 2){
    input_size = s::range{512,512};
    group_size = s::range{16,16};
  } else {
    input_size = s::range{64,64,64};
    group_size = s::range{4,4,8};
  }

  s::buffer<int> output_buff{input_size.size()};

  q.submit([&](s::handler &cgh) {
    s::accessor acc{output_buff, cgh, s::no_init};
    cgh.parallel<enumerated_kernel_name<KernelName,0>>(
        s::range{input_size / group_size}, s::range{group_size},
        [=](auto grp) {
          s::distribute_groups(grp, [&](auto subgrp) {
            s::distribute_groups(subgrp, [&](auto subsubgrp) {
              s::distribute_items(subsubgrp, [&](s::s_item<Dim> idx) {
                int* ptr = acc.get_pointer();
                ptr[idx.get_global_linear_id()] =
                    idx.get_global_linear_id();
              });
            });
          });
        });
      });
  {
    s::host_accessor hacc{output_buff};
    int* result_ptr = hacc.get_pointer();
    for(std::size_t i = 0; i < input_size.size(); ++i){
      BOOST_CHECK(result_ptr[i] == static_cast<int>(i));
    }
  }

  q.submit([&](s::handler& cgh){
    s::accessor acc{output_buff, cgh, s::no_init};
    cgh.fill(acc, 0);
  });

  q.submit([&](s::handler &cgh) {
    s::accessor acc{output_buff, cgh, s::no_init};
    cgh.parallel<enumerated_kernel_name<KernelName, 1>>(
        s::range{input_size / group_size}, s::range{group_size}, [=](auto grp) {
          s::distribute_groups(grp, [&](auto subgrp) {
            s::distribute_items(subgrp, [&](s::s_item<Dim> idx) {
              acc[idx.get_global_linear_id()] = idx.get_global_linear_id();
            });
          });
        });
      });

  {
    s::host_accessor hacc{output_buff};
    int* result_ptr = hacc.get_pointer();
    for(std::size_t i = 0; i < input_size.size(); ++i){
      BOOST_CHECK(result_ptr[i] == static_cast<int>(i));
    }
  }
  
}

template<class Name, int D>
class nd_kernel_name;

BOOST_AUTO_TEST_CASE_TEMPLATE(scoped_parallelism_api, _dimensions,
                              test_dimensions::type) {
  constexpr int d = _dimensions::value;
  test_distribute_groups<nd_kernel_name<class ScopedParallelismDistrGroups, d>,
                         d>();
}

BOOST_AUTO_TEST_CASE(scoped_parallelism_reduction) {
  namespace s = cl::sycl;
  s::queue q;
  
  std::size_t input_size = 256;
  std::vector<int> input(input_size);
  for(int i = 0; i < input.size(); ++i)
    input[i] = i;
  
  s::buffer<int> buff{input.data(), s::range<1>{input_size}};
  
  constexpr size_t Group_size = 64;
  
  q.submit([&](s::handler& cgh){
    auto data_accessor = buff.get_access<s::access::mode::read_write>(cgh);
    cgh.parallel<class ScopedReductionKernel>(
        s::range<1>{input_size / Group_size}, s::range<1>{Group_size},
        [=](auto grp) {
          
          s::memory_environment(grp, 
            s::require_local_mem<int[Group_size]>(),
            s::require_private_mem<int>(),
            [&](auto &scratch, auto &load) {
            
            s::distribute_items(grp, [&](s::s_item<1> idx) {
              load(idx) = data_accessor[idx.get_global_id(0)];
            });
            s::distribute_items(grp, [&](s::s_item<1> idx) {
              scratch[idx.get_innermost_local_id(0)] = load(idx);
            });

            s::group_barrier(grp);

            for (int i = Group_size / 2; i > 0; i /= 2) {
              s::distribute_items_and_wait(grp, [&](s::s_item<1> idx) {
                size_t lid = idx.get_innermost_local_id(0);
                if (lid < i)
                  scratch[lid] += scratch[lid + i];
              });
            }

            s::single_item(grp, [&]() {
              data_accessor[grp.get_group_id(0) * Group_size] = scratch[0];
            });
          });
        });
  });
  
  auto host_acc = buff.get_access<s::access::mode::read>();
  
  for(int grp = 0; grp < input_size/Group_size; ++grp){
    int host_result = 0;
    for(int i = grp * Group_size; i < (grp+1) * Group_size; ++i)
      host_result += i;
    
    BOOST_TEST(host_result == host_acc[grp * Group_size]);
  }
} 

BOOST_AUTO_TEST_CASE(scoped_parallelism_memory_environment) {
  namespace s = cl::sycl;

  s::queue q;
  std::size_t input_size = 1024;
  s::buffer<int> buff{input_size};
  constexpr std::size_t Group_size = 256;

  q.submit([&](s::handler& cgh){
    s::accessor acc{buff, cgh, s::no_init};
    cgh.parallel<class ScopedReductionMemEnv>(
      s::range{input_size / Group_size},
      s::range{Group_size}, [=](auto grp){
      
      s::memory_environment(grp,
        s::require_local_mem<int[16][16]>(3),
        s::require_private_mem<int>(4),
        [&](auto& local, auto& private_mem){
        
        if(grp.get_group_id(0) == 0) {
          s::distribute_items(grp, [&](s::s_item<1> idx) {
            int* local_ptr = &local[0][0];
            acc[idx.get_global_linear_id()] =
                local_ptr[idx.get_innermost_local_linear_id()];
          });
        }
        if(grp.get_group_id(0) == 1) {
          s::distribute_items(grp, [&](s::s_item<1> idx) {
            acc[idx.get_global_linear_id()] =
                private_mem(idx);
          });
        }
      });
      s::local_memory_environment<int [Group_size]>(grp, 
        [&](auto& local){

        if(grp.get_group_id(0) == 2) {
          s::distribute_items(grp, [&](s::s_item<1> idx) {
            local[idx.get_innermost_local_linear_id()] =
                idx.get_innermost_local_linear_id();
            acc[idx.get_global_linear_id()] =
                local[idx.get_innermost_local_linear_id()];
          });
        }
      });
      const s::vec<int,8> init_val{0,1,2,3,4,5,6,7};
      s::memory_environment(grp, 
        s::require_private_mem<s::vec<int,8>>(init_val),
        [&](auto& priv_mem){

        if(grp.get_group_id(0) == 3) {
          s::distribute_items(grp, [&](s::s_item<1> idx) {
            int res = 0;
            auto v = priv_mem(idx) + init_val;
            for(int i = 0; i < init_val.size(); ++i) {
              res += v[i];
            }
            acc[idx.get_global_linear_id()] = res;
          });
        }
      });
    });
  });
  s::host_accessor hacc{buff};
  for(int grp = 0; grp < 4; ++grp) {
    for(int lid = 0; lid < Group_size; ++lid){
      const int gid = grp * Group_size + lid;
      if(grp == 0){
        BOOST_CHECK(hacc[gid] == 3);
      } else if(grp == 1) {
        BOOST_CHECK(hacc[gid] == 4);
      } else if(grp == 2) {
        BOOST_CHECK(hacc[gid] == lid);
      } else if(grp == 4) {
        const s::vec<int,8> expected_v{0,2,4,6,8,10,12,14};
        int expected = 0; 
        for (int i = 0; i < expected_v.size(); ++i)
          expected += expected_v[i];
        BOOST_CHECK(hacc[gid] == expected);
      }
    }
  }

}
BOOST_AUTO_TEST_CASE(scoped_parallelism_odd_group_size) {
  using namespace cl;
  sycl::queue q;
  const size_t test_size = 1000;
  sycl::buffer<int> buff{sycl::range{test_size}};

  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc {buff, cgh, sycl::no_init};
    cgh.parallel<class ScopedOddGroupSize>(sycl::range{10}, sycl::range{100}, 
      [=](auto grp){
      sycl::distribute_groups(grp, [&](auto subgroup){
        sycl::distribute_groups(subgroup, [&](auto subsubgroup){
          sycl::distribute_items(subsubgroup, [&](sycl::s_item<1> idx){
            acc[idx.get_global_linear_id()] = static_cast<int>(idx.get_global_linear_id());
          });
        });
      });
    });
  });
  {
    sycl::host_accessor hacc{buff};
    for (int i = 0; i < test_size; ++i)
      BOOST_CHECK(hacc[i] == i);
  }
}

#endif
#ifdef ACPP_EXT_ENQUEUE_CUSTOM_OPERATION

template<cl::sycl::backend B>
void test_interop(cl::sycl::queue& q) {
  using namespace cl;

  const std::size_t test_size = 1024;

  std::vector<int> initial_data(test_size, 14);
  std::vector<int> target_data(test_size);
  int* target_ptr = target_data.data();

  sycl::buffer<int, 1> buff{initial_data.data(), sycl::range<1>{test_size}};

  q.submit([&](sycl::handler &cgh) {
    auto acc = buff.get_access<sycl::access::mode::read>(cgh);

    cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle &h) {
      // All backends support obtaining native memory
      void *native_mem = h.get_native_mem<B>(acc);

      // OpenMP backend doesn't support extracting a native queue or device
      if constexpr(B == sycl::backend::cuda) {
        auto stream = h.get_native_queue<B>();
        // dev is not really used, just test that this function call works for now
        typename sycl::backend_traits<B>::template native_type<sycl::device> dev =
            h.get_native_device<B>();
        
        // Even though we can target multiple backends simultaneously,
        // the HIP headers cannot be included simultaneously with CUDA.
        // We can therefore only directly call either CUDA or HIP runtime functions.
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA
        cudaMemcpyAsync(target_ptr, native_mem, test_size * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
#endif
      }
      else if constexpr(B == sycl::backend::hip) {
      
        auto stream = h.get_native_queue<B>();
        // dev is not really used, just test that this function call works for now
        typename sycl::backend_traits<B>::template native_type<sycl::device> dev =
            h.get_native_device<B>();
        
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP
        hipMemcpyAsync(target_ptr, native_mem, test_size * sizeof(int),
                        hipMemcpyDeviceToHost, stream);
#endif
      }
    });
  });

  q.wait();

  constexpr bool has_hip_memcpy_test = (B == sycl::backend::hip) &&
                    ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP;
  constexpr bool has_cuda_memcpy_test = (B == sycl::backend::cuda) &&
                    ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA;
  if constexpr (has_hip_memcpy_test || has_cuda_memcpy_test) {
    for (std::size_t i = 0; i < test_size; ++i) {
      BOOST_TEST(initial_data[i] == target_data[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(custom_enqueue) {
  using namespace cl;

  sycl::queue q;
  sycl::backend b = q.get_device().get_backend();
  
  if(b == sycl::backend::cuda)
    test_interop<sycl::backend::cuda>(q);
  else if(b == sycl::backend::hip)
    test_interop<sycl::backend::hip>(q);
  else if(b == sycl::backend::level_zero)
    test_interop<sycl::backend::level_zero>(q);
  else if(b == sycl::backend::omp)
    test_interop<sycl::backend::omp>(q);
}
#endif
#ifdef ACPP_EXT_CG_PROPERTY_RETARGET
BOOST_AUTO_TEST_CASE(cg_property_retarget) {
  using namespace cl;

  auto all_devices = sycl::device::get_devices();

  std::vector<sycl::device> target_devices;
  for(const auto& dev : all_devices) {
    if (dev.AdaptiveCpp_has_compiled_kernels() && dev.is_gpu()) {
      target_devices.push_back(dev);
    }
  }
  sycl::device host_device{sycl::detail::get_host_device()};

  if(target_devices.size() > 0) {
    sycl::queue q{
        target_devices[0],
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::AdaptiveCpp_retargetable{}}};
    int* ptr = sycl::malloc_shared<int>(1, q);
    *ptr = 0;

    q.parallel_for<class retarget_gpu_kernel>(sycl::range{128}, 
      [=](sycl::id<1> idx){
      
      if(idx[0] == 0)
        ++ptr[0];
    });

    q.submit({sycl::property::command_group::AdaptiveCpp_retarget{host_device}},
      [&](sycl::handler& cgh){
        cgh.single_task<class retarget_host_kernel>([=](){
          ++ptr[0];
        });  
      });

    q.wait();

    BOOST_TEST(ptr[0] == 2);

    sycl::free(ptr, q);
  }
}
#endif


ACPP_KERNEL_TARGET
int get_total_group_size() {
  int group_size = 0;
  __acpp_if_target_device(
    group_size = __acpp_lsize_x * __acpp_lsize_y * __acpp_lsize_z;
  );
  return group_size;
}


#ifdef ACPP_EXT_CG_PROPERTY_PREFER_GROUP_SIZE
BOOST_AUTO_TEST_CASE(cg_property_preferred_group_size) {
  using namespace cl;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  int* gsize = sycl::malloc_shared<int>(3, q);

  auto group_size1d = sycl::range{100};
  auto group_size2d = sycl::range{9,9};
  auto group_size3d = sycl::range{5,5,5};

#if defined(__ACPP_ENABLE_CUDA_TARGET__) ||                                 \
    defined(__ACPP_ENABLE_HIP_TARGET__) ||                                  \
    defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
#define DEVICE_MODEL
#endif

  q.submit({sycl::property::command_group::AdaptiveCpp_prefer_group_size{
               group_size1d}},
           [&](sycl::handler &cgh) {
             cgh.parallel_for<class property_preferred_group_size1>(
                sycl::range{1000}, [=](sycl::id<1> idx) {
                  if (idx[0] == 0) {
#if defined(DEVICE_MODEL)
                    __acpp_if_target_device(
                      gsize[0] = get_total_group_size();
                    );
                    __acpp_if_target_host(
                      gsize[0] = 1;
                    );
#else
                    gsize[0] = 1;
#endif
                  }
                });
           });

  q.submit({sycl::property::command_group::AdaptiveCpp_prefer_group_size{
               group_size2d}},
           [&](sycl::handler &cgh) {
             cgh.parallel_for<class property_preferred_group_size2>(
                 sycl::range{30,30}, [=](sycl::id<2> idx) {
                   if (idx[0] == 0 && idx[1] == 0) {
#if defined(DEVICE_MODEL)
                    __acpp_if_target_device(
                      gsize[1] = get_total_group_size();
                    );
                    __acpp_if_target_host(
                      gsize[1] = 2;
                    );
#else
                     gsize[1] = 2;
#endif
                   }
                 });
           });


  q.submit({sycl::property::command_group::AdaptiveCpp_prefer_group_size{
               group_size3d}},
           [&](sycl::handler &cgh) {
             cgh.parallel_for<class property_preferred_group_size3>(
                 sycl::range{10,10,10}, [=](sycl::id<3> idx) {
                   if (idx[0] == 0 && idx[1] == 0) {
#if defined(DEVICE_MODEL)
                    __acpp_if_target_device(
                     gsize[2] = get_total_group_size();
                    );
                    __acpp_if_target_host(
                     gsize[2] = 3;
                    );
#else
                     gsize[2] = 3;
#endif
                   }
                 });
           });

  q.wait();

  if(q.get_device().get_backend() == sycl::backend::cuda || 
    q.get_device().get_backend() == sycl::backend::hip) {
    BOOST_TEST(gsize[0] == group_size1d.size());
    BOOST_TEST(gsize[1] == group_size2d.size());
    BOOST_TEST(gsize[2] == group_size3d.size());
  } else {
    /* Don't test this - it's meaningless for the extension,
       and might not be true if the SSCP JIT executes the kernel
       on another target apart from CUDA or HIP.

    BOOST_TEST(gsize[0] == 1);
    BOOST_TEST(gsize[1] == 2);
    BOOST_TEST(gsize[2] == 3);
    */
  }

  sycl::free(gsize, q);
}
#endif

#ifdef ACPP_EXT_CG_PROPERTY_PREFER_EXECUTION_LANE

BOOST_AUTO_TEST_CASE(cg_property_prefer_execution_lane) {

  cl::sycl::queue q;

  // Only compile testing for now
  for(std::size_t i = 0; i < 100; ++i) {
    q.submit(
        {cl::sycl::property::command_group::AdaptiveCpp_prefer_execution_lane{i}},
        [&](cl::sycl::handler &cgh) {
          cgh.single_task<class prefer_execution_lane_test>([=]() {});
        });
  }
  q.wait();
}

#endif

#ifdef ACPP_EXT_PREFETCH_HOST
BOOST_AUTO_TEST_CASE(prefetch_host) {
  using namespace cl;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};

  std::size_t test_size = 4096;
  int *shared_mem = sycl::malloc_shared<int>(test_size, q);

  for (std::size_t i = 0; i < test_size; ++i)
    shared_mem[i] = i;

  q.parallel_for<class usm_prefetch_host_test_kernel>(
      sycl::range<1>{test_size},
      [=](sycl::id<1> idx) { shared_mem[idx.get(0)] += 1; });
  q.prefetch_host(shared_mem, test_size * sizeof(int));
  q.wait_and_throw();

  for (std::size_t i = 0; i < test_size; ++i)
    BOOST_TEST(shared_mem[i] == i + 1);

  sycl::free(shared_mem, q);
}
#endif
#ifdef ACPP_EXT_BUFFER_USM_INTEROP
BOOST_AUTO_TEST_CASE(buffer_introspection) {
  using namespace cl;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}}};
  sycl::range size{1024};

  int* usm_ptr = nullptr;
  {
    sycl::buffer<int> buff{size};

    q.submit([&](sycl::handler& cgh){
      auto acc = buff.get_access<sycl::access::mode::discard_write>(cgh);
      // Force allocation of buffer on target device
      cgh.single_task([=](){});
    });

    q.wait();

    BOOST_TEST(buff.has_allocation(q.get_device()));
    usm_ptr = buff.get_pointer(q.get_device());
    BOOST_TEST(usm_ptr != nullptr);

    // Query information
    sycl::buffer_allocation::descriptor<int> alloc =
        buff.get_allocation(usm_ptr);
    BOOST_TEST(alloc.ptr == usm_ptr);
    BOOST_CHECK(alloc.dev == q.get_device());
    BOOST_TEST(alloc.is_owned == true);

    // This doesn't change anything as the allocation is already
    // owned because the buffer constructor was not provided a pointer.
    // Execute both variants to make sure both interfaces work.
    buff.own_allocation(usm_ptr);
    buff.own_allocation(q.get_device());
    alloc = buff.get_allocation(usm_ptr);
    BOOST_TEST(alloc.is_owned == true);

    // Disown allocation so that we can use it outside the
    // buffer scope
    buff.disown_allocation(usm_ptr);

    alloc = buff.get_allocation(usm_ptr);
    BOOST_TEST(alloc.is_owned == false);

    std::vector<int*> allocations;
    buff.for_each_allocation(
        [&](const sycl::buffer_allocation::descriptor<int> &a) {
          allocations.push_back(a.ptr);
        });

    BOOST_TEST(allocations.size() >= 1);
    bool found = false;
    for(std::size_t i = 0; i < allocations.size(); ++i) {
      if(allocations[i] == usm_ptr)
        found = true;
    }
    BOOST_TEST(found);
  }

  // Use extracted USM pointer directly
  std::vector<int> host_mem(size[0]);

  q.parallel_for(size, [usm_ptr](sycl::id<1> idx){
    usm_ptr[idx[0]] = idx[0];
  });
  q.memcpy(host_mem.data(), usm_ptr, sizeof(int)*size[0]);
  q.wait();

  for(std::size_t i = 0; i < host_mem.size(); ++i) {
    BOOST_CHECK(host_mem[i] == i);
  }

  sycl::free(usm_ptr, q);


}

BOOST_AUTO_TEST_CASE(buffers_over_usm_pointers) {
  using namespace cl;

  sycl::queue q;
  sycl::range size{1024};

  int* alloc1 = sycl::malloc_shared<int>(size.size(), q);
  int* alloc2 = sycl::malloc_shared<int>(size.size(), q);

  {
    sycl::buffer<int> b1{
        {sycl::buffer_allocation::empty_view(alloc1, q.get_device())}, size};

    BOOST_CHECK(b1.has_allocation(q.get_device()));
    BOOST_CHECK(b1.get_pointer(q.get_device()) == alloc1);
    b1.for_each_allocation([&](const auto& alloc){
      if(alloc.ptr == alloc1){
        BOOST_CHECK(!alloc.is_owned);
      }
    });

    q.submit([&](sycl::handler& cgh){
      sycl::accessor<int> acc{b1, cgh};

      cgh.parallel_for(size, [=](sycl::id<1> idx){
        acc[idx] = idx.get(0);
      });
    });
  }
  q.wait();
  for(int i = 0; i < size.get(0); ++i){
    BOOST_CHECK(alloc1[i] == i);
  }
  {
    sycl::buffer<int> b2{
        {sycl::buffer_allocation::view(alloc1, q.get_device())}, size};
    
    q.submit([&](sycl::handler& cgh){
      sycl::accessor<int> acc{b2, cgh};

      cgh.parallel_for(size, [=](sycl::id<1> idx){
        alloc2[idx.get(0)] = acc[idx];
      });
    });

    // Check that data state tracking works and migrating back to host
    sycl::host_accessor<int> hacc{b2};
    for(int i = 0; i < size.get(0); ++i){
      BOOST_CHECK(hacc[i] == i);
    }  
  }
  
  for(int i = 0; i < size.get(0); ++i){
    BOOST_CHECK(alloc2[i] == i);
  }

  sycl::free(alloc1, q);
  sycl::free(alloc2, q);
}

#endif
#ifdef ACPP_EXT_BUFFER_PAGE_SIZE

BOOST_AUTO_TEST_CASE(buffer_page_size) {
  using namespace cl;

  sycl::queue q;

  // Deliberately choose page_size so that size is not a mulitple of it
  // to test the more complicated case.
  const std::size_t size = 1000;
  const std::size_t page_size = 512;
  sycl::buffer<int, 2> buff{sycl::range{size, size},
                            sycl::property::buffer::AdaptiveCpp_page_size<2>{
                                sycl::range{page_size, page_size}}};

  // We have 4 pages
  for(std::size_t offset_x = 0; offset_x < size; offset_x += page_size) {
    for(std::size_t offset_y = 0; offset_y < size; offset_y += page_size) {
      auto event = q.submit([&](sycl::handler &cgh) {

        sycl::range range{std::min(page_size, size - offset_x),
                          std::min(page_size, size - offset_y)};
        sycl::id offset{offset_x, offset_y};

        for(int i = 0; i < 2; ++i){
          assert(offset[i]+range[i] <= size);
        }

        sycl::accessor<int, 2> acc{buff, cgh, range, offset};

        cgh.parallel_for(sycl::range{range}, [=](sycl::id<2> idx){
          acc[idx] = static_cast<int>(idx[0] + offset[0] + idx[1] + offset[1]);
        });
      });

      // All kernels should be independent, in that case we should
      // have a wait list of exactly one element: The one accessor
      // we have requested.
      // TODO This does not really guarantee that the kernels run in-
      // dependently as access conflicts are typically added to the requirements
      // of the accessor, not the kernel.
      BOOST_CHECK(event.get_wait_list().size() == 1);
    }
  }

  sycl::host_accessor<int, 2> hacc{buff};

  for(int i = 0; i < size; ++i) {
    for(int j = 0; j < size; ++j) {
      BOOST_REQUIRE(hacc[i][j] == i+j);
    }
  }
}

#endif
#ifdef ACPP_EXT_EXPLICIT_BUFFER_POLICIES
BOOST_AUTO_TEST_CASE(explicit_buffer_policies) {
  using namespace cl;
  sycl::queue q;
  sycl::range size{1024};

  {
    std::vector<int> input_vec(size.size());
    
    for(int i = 0; i < input_vec.size(); ++i)
      input_vec[i] = i;
    
    auto b1 = sycl::make_async_buffer(input_vec.data(), size);
    // Because of buffer semantics we should be able to modify the input
    // pointer again
    input_vec[20] = 0;

    q.submit([&](sycl::handler& cgh){
      sycl::accessor acc{b1, cgh};
      cgh.parallel_for(size, [=](sycl::id<1> idx){
        acc[idx.get(0)] += 1;
      });
    });

    sycl::host_accessor hacc{b1};
    for(int i = 0; i < size.size(); ++i) {
      BOOST_CHECK(hacc[i] == i+1);
    }

    // Submit another operation before buffer goes out of
    // scope to make sure operations work even if the buffer leaves
    // scope.
    q.submit([&](sycl::handler& cgh){
      sycl::accessor acc{b1, cgh};
      cgh.parallel_for(size, [=](sycl::id<1> idx){
        acc[idx.get(0)] -= 1;
      });
    });
  }

  {
    std::vector<int> input_vec(size.size());
    
    for(int i = 0; i < input_vec.size(); ++i)
      input_vec[i] = i;
    {
      auto b1 = sycl::make_sync_writeback_view(input_vec.data(), size);

      q.submit([&](sycl::handler& cgh){
        sycl::accessor acc{b1, cgh};
        cgh.parallel_for(size, [=](sycl::id<1> idx){
          acc[idx.get(0)] += 1;
        });
      });
    }
    for(int i = 0; i < input_vec.size(); ++i) {
      BOOST_CHECK(input_vec[i] == i+1);
    }
  }

  {
    std::vector<int> input_vec(size.size());
    
    for(int i = 0; i < input_vec.size(); ++i)
      input_vec[i] = i;
    {
      auto b1 = sycl::make_async_writeback_view(input_vec.data(), size, q);

      q.submit([&](sycl::handler& cgh){
        sycl::accessor acc{b1, cgh};
        cgh.parallel_for(size, [=](sycl::id<1> idx){
          acc[idx.get(0)] += 1;
        });
      });
    }

    q.wait();

    for(int i = 0; i < input_vec.size(); ++i) {
      BOOST_CHECK(input_vec[i] == i+1);
    }
  }

}
#endif
#ifdef ACPP_EXT_ACCESSOR_VARIANTS
#ifdef ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION

template <class T, int Dim, cl::sycl::access_mode M, cl::sycl::target Tgt,
          cl::sycl::accessor_variant V>
constexpr cl::sycl::accessor_variant
get_accessor_variant(cl::sycl::accessor<T, Dim, M, Tgt, V>) {
  return V;
}

BOOST_AUTO_TEST_CASE(accessor_variants) {
  using namespace cl;
  sycl::queue q;
  sycl::range size{1024};
  sycl::range subrange{512};
  sycl::id<1> offset{256};

  sycl::buffer<int> buff{size};
  sycl::accessor unranged_placeholder{buff, sycl::read_write, sycl::no_init};
  sycl::accessor ranged_placeholder{buff, subrange, offset, sycl::read_write,
                                    sycl::no_init};

  BOOST_CHECK(get_accessor_variant(unranged_placeholder) ==
              sycl::accessor_variant::unranged_placeholder);
  BOOST_CHECK(get_accessor_variant(ranged_placeholder) ==
              sycl::accessor_variant::ranged_placeholder);

  BOOST_CHECK(unranged_placeholder.get_offset() == sycl::id<1>{});
  BOOST_CHECK(ranged_placeholder.get_offset() == offset);
  BOOST_CHECK(unranged_placeholder.get_range() == size);
  BOOST_CHECK(ranged_placeholder.get_range() == subrange);
  BOOST_CHECK(unranged_placeholder.is_placeholder());
  BOOST_CHECK(ranged_placeholder.is_placeholder());


  BOOST_CHECK(sizeof(ranged_placeholder) > sizeof(unranged_placeholder));

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor unranged_acc{buff, cgh, sycl::read_write, sycl::no_init};
    sycl::accessor ranged_acc{
        buff, cgh, subrange, offset, sycl::read_write, sycl::no_init};

    BOOST_CHECK(get_accessor_variant(unranged_acc) ==
              sycl::accessor_variant::unranged);
    BOOST_CHECK(get_accessor_variant(ranged_acc) ==
              sycl::accessor_variant::ranged);

    BOOST_CHECK(sizeof(ranged_acc) > sizeof(unranged_acc));
    BOOST_CHECK(sizeof(ranged_placeholder) > sizeof(ranged_acc));

    BOOST_CHECK(unranged_acc.get_offset() == sycl::id<1>{});
    BOOST_CHECK(ranged_acc.get_offset() == offset);
    BOOST_CHECK(unranged_acc.get_range() == size);
    BOOST_CHECK(ranged_acc.get_range() == subrange);
    BOOST_CHECK(!unranged_acc.is_placeholder());
    BOOST_CHECK(!ranged_acc.is_placeholder());

    sycl::accessor raw_acc{buff, cgh, sycl::read_write_raw, sycl::no_init};

    BOOST_CHECK(sizeof(raw_acc) < sizeof(ranged_acc));

    auto kernel = [=](sycl::id<1> idx){
      raw_acc[idx] = idx.get(0);
    };

    cgh.parallel_for(size, kernel);
  });

  sycl::host_accessor hacc{buff};
  for(std::size_t i = 0; i < size[0]; ++i){
    BOOST_CHECK(hacc[i] == static_cast<int>(i));
  }
}

#endif
#endif
#if defined(ACPP_EXT_UPDATE_DEVICE) &&                                      \
    defined(ACPP_EXT_BUFFER_USM_INTEROP)
BOOST_AUTO_TEST_CASE(update_device) {
  using namespace cl;
  sycl::queue q;
  sycl::range size{1024};
  sycl::buffer<int> buff{size};
  {
    sycl::host_accessor hacc{buff};

    for(std::size_t i = 0; i < size[0]; ++i){
      hacc[i] = static_cast<int>(i);
    }
  }

  std::vector<int> target_buff(size[0]);

  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{buff, cgh};
    cgh.update(acc);
  }).wait();

  // We have to use a USM copy to get the data for testing
  // because we cannot know whether it was the accessor
  // or the udpate() that updated the data if we use
  // handler::copy()
  int* dev_ptr = buff.get_pointer(q.get_device());
  BOOST_CHECK(dev_ptr != nullptr);

  q.memcpy(target_buff.data(), dev_ptr, size[0] * sizeof(int)).wait();

  for(std::size_t i = 0; i < size[0]; ++i)
    BOOST_CHECK(target_buff[i] == static_cast<int>(i));
}
#endif
#ifdef ACPP_EXT_QUEUE_WAIT_LIST

BOOST_AUTO_TEST_CASE(queue_wait_list) {
  using namespace cl;
  sycl::queue out_of_order_q;
  sycl::queue in_order_q{
      sycl::property_list{sycl::property::queue::in_order{},
                          // Needed for accurate get_wait_list results
                          sycl::property::queue::AdaptiveCpp_retargetable{}}};

  auto test = [](sycl::queue& q){
    std::vector<sycl::event> evts;
    for(int i = 0; i < 10; ++i)
      evts.push_back(q.single_task([=](){}));
    auto wait_list = q.get_wait_list();
    
    q.single_task(wait_list, [=](){}).wait();
    for(sycl::event e : evts) {
      BOOST_CHECK(e.get_info<sycl::info::event::command_execution_status>() ==
                  sycl::info::event_command_status::complete);
    }
  };

  test(out_of_order_q);
  test(in_order_q);
}

#endif
#if defined(ACPP_EXT_MULTI_DEVICE_QUEUE) && defined(ACPP_TEST_MULTI_DEVICE_QUEUE)

BOOST_AUTO_TEST_CASE(multi_device_queue) {
  using namespace cl;
  sycl::queue q{sycl::system_selector_v};

  sycl::buffer<int> buff{sycl::range{1}};

  q.submit([&](sycl::handler& cgh){
    sycl::accessor<int> acc{buff, cgh, sycl::no_init};
    cgh.single_task([=](){
      acc[0] = 1;
    });
  });
  q.submit([&](sycl::handler& cgh){
    sycl::accessor<int> acc{buff, cgh};
    cgh.single_task([=](){
      acc[0] += 100;
    });
  });
  q.submit([&](sycl::handler& cgh){
    sycl::accessor<int> acc{buff, cgh};
    cgh.single_task([=](){
      acc[0]++;
    });
  });

  sycl::host_accessor hacc{buff};
  BOOST_CHECK(hacc[0] == 102);
}
#endif
#ifdef ACPP_EXT_COARSE_GRAINED_EVENTS
BOOST_AUTO_TEST_CASE(coarse_grained_events) {
  using namespace cl;
  sycl::queue q{sycl::property::queue::AdaptiveCpp_coarse_grained_events{}};
  
  auto e1 = q.single_task([=](){});
  std::vector<sycl::event> events;
  for(int i=0; i < 10; ++i) {
    auto e = q.submit(
        {sycl::property::command_group::AdaptiveCpp_prefer_execution_lane{
            static_cast<std::size_t>(
                i)}}, // Make sure we alternate across all lanes/streams
        [&](sycl::handler &cgh) {
          cgh.depends_on(e1); // Test depends_on synchronization
          cgh.single_task([=]() {});
        });
    events.push_back(e);
  }
  for(auto& e : events) {
    e.wait();
    BOOST_CHECK(e.get_info<sycl::info::event::command_execution_status>() ==
                sycl::info::event_command_status::complete);
  }
}
#endif

#ifdef ACPP_EXT_SPECIALIZED
BOOST_AUTO_TEST_CASE(sycl_specialized) {
  using namespace cl;
  sycl::queue q;

  uint64_t* data = sycl::malloc_shared<uint64_t>(1, q);

  //Ctor
  {
    *data = 1;
    sycl::specialized<uint64_t> s{10};
    q.submit([&](sycl::handler& cgh){
      cgh.single_task([=](){
        *data += s;
      });
    }).wait();

    BOOST_CHECK(*data == 11);
  }

  //Copy assignment operator (const T&)
  {
    *data = 1;
    sycl::specialized<uint64_t> s;
    q.submit([&](sycl::handler& cgh){
      s = 10;
      cgh.single_task([=](){
        *data += s;
      });
    }).wait();

    BOOST_CHECK(*data == 11);
  }

   //Copy assignment operator (sycl::specialized)
  {
    *data = 1;
    sycl::specialized<uint64_t> s_tmp{10};
    q.submit([&](sycl::handler& cgh){
      sycl::specialized<uint64_t> s = s_tmp;
      cgh.single_task([=](){
        *data += s;
      });
    }).wait();

    BOOST_CHECK(*data == 11);
  }

  sycl::free(data, q);
}
#endif
#ifdef SYCL_KHR_DEFAULT_CONTEXT
BOOST_AUTO_TEST_CASE(khr_default_context) {
  using namespace cl;
  sycl::queue q1;
  sycl::queue q2;

  BOOST_CHECK(q1.get_context() == q2.get_context());
  BOOST_CHECK(q1.get_device().get_platform().khr_get_default_context() ==
              q1.get_context());
  BOOST_CHECK(sycl::context{} != q1.get_context());
}
#endif

BOOST_AUTO_TEST_SUITE_END()
