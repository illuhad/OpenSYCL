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
#ifndef HIPSYCL_OPENMP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_OPENMP_KERNEL_LAUNCHER_HPP

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include <cassert>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"
#include "hipSYCL/sycl/libkernel/detail/data_layout.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

#include "../generic/host/collective_execution_engine.hpp"
#include "../generic/host/iterate_range.hpp"

namespace hipsycl {
namespace glue {
namespace omp_dispatch {

inline int get_my_thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline int get_max_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int get_num_threads() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

template <class Function>
void parallel_invocation(Function kernel) noexcept {
#ifndef _OPENMP
  HIPSYCL_DEBUG_WARNING
      << "omp_kernel_launcher: Kernel launcher was built without OpenMP "
         "support, the kernel will execute sequentially!"
      << std::endl;
#else
#pragma omp parallel
#endif
  {
    kernel();
  }
}

#ifdef __ACPP_USE_ACCELERATED_CPU__
extern "C" size_t __acpp_cbs_local_id_x;
extern "C" size_t __acpp_cbs_local_id_y;
extern "C" size_t __acpp_cbs_local_id_z;

template <int Dim, class Function>
[[clang::annotate("acpp_cbs_kernel_dimension", Dim)]]
HIPSYCL_LOOP_SPLIT_ND_KERNEL __attribute__((noinline))
inline void iterate_nd_range_omp(Function f, const sycl::id<Dim> &&group_id, const sycl::range<Dim> num_groups,
  HIPSYCL_LOOP_SPLIT_ND_KERNEL_LOCAL_SIZE_ARG const sycl::range<Dim> local_size, const sycl::id<Dim> offset,
  size_t num_local_mem_bytes, void* group_shared_memory_ptr,
  std::function<void()> &barrier_impl) noexcept {
  if constexpr (Dim == 1) {
    sycl::id<Dim> local_id{__acpp_cbs_local_id_x};
    sycl::nd_item<Dim> this_item{&offset,    group_id,   local_id,
      local_size, num_groups, &barrier_impl, group_shared_memory_ptr};
    f(this_item);
  } else if constexpr (Dim == 2) {
    sycl::id<Dim> local_id{__acpp_cbs_local_id_x, __acpp_cbs_local_id_y};
    sycl::nd_item<Dim> this_item{&offset, group_id,
      local_id, local_size, num_groups,
      &barrier_impl, group_shared_memory_ptr};
    f(this_item);
  } else if constexpr (Dim == 3) {
    sycl::id<Dim> local_id{__acpp_cbs_local_id_x, __acpp_cbs_local_id_y, __acpp_cbs_local_id_z};
    sycl::nd_item<Dim> this_item{&offset,    group_id,
      local_id,   local_size,
      num_groups, &barrier_impl, group_shared_memory_ptr};
    f(this_item);
  }
}
#endif

template<class Function>
inline
void single_task_kernel(Function f) noexcept
{
  f();
}

template <int Dim, class Function>
inline void parallel_for_kernel(Function f,
                                const sycl::range<Dim> execution_range) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");

  parallel_invocation([=](){
    host::iterate_range_omp_for(execution_range, [&](sycl::id<Dim> idx) {
      auto this_item =
        sycl::detail::make_item<Dim>(idx, execution_range);

      f(this_item);
    });
  });
}

template <int Dim, class Function>
inline void parallel_for_kernel_offset(Function f,
                                       const sycl::range<Dim> execution_range,
                                       const sycl::id<Dim> offset) noexcept {
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");


  parallel_invocation([=](){
    host::iterate_range_omp_for(offset, execution_range, [&](sycl::id<Dim> idx) {
      auto this_item =
        sycl::detail::make_item<Dim>(idx, execution_range, offset);

      f(this_item);
    });
  });
}

template <int Dim, class Function>
inline void parallel_for_ndrange_kernel(
    Function f, const sycl::range<Dim> num_groups,
    const sycl::range<Dim> local_size, const sycl::id<Dim> offset,
    size_t num_local_mem_bytes) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1 - 3 are supported.");

  parallel_invocation([=](){
    if(num_groups.size() == 0 || local_size.size() == 0)
      return;

    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);

    // 128 kiB as local memory for group algorithms
    std::aligned_storage_t<128*1024, sizeof(double) * 16> group_shared_memory_ptr{};
#ifdef __ACPP_USE_ACCELERATED_CPU__
    std::function<void()> barrier_impl = [] () noexcept {
      assert(false && "splitting seems to have failed");
      std::terminate();
    };

    host::iterate_range_omp_for(num_groups, [&](sycl::id<Dim> &&group_id) {
      iterate_nd_range_omp(f, std::move(group_id), num_groups, local_size, offset,
        num_local_mem_bytes, &group_shared_memory_ptr, barrier_impl);
    });
#elif defined(HIPSYCL_HAS_FIBERS)
    host::static_range_decomposition<Dim> group_decomposition{
        num_groups, get_num_threads()};

    host::collective_execution_engine<Dim> engine{num_groups, local_size,
                                                  offset, group_decomposition,
                                                  get_my_thread_id()};

    std::function<void()> barrier_impl = [&]() { engine.barrier(); };

    engine.run_kernel([&](sycl::id<Dim> local_id, sycl::id<Dim> group_id) {

      auto linear_group_id =
          sycl::detail::linear_id<Dim>::get(group_id, num_groups);

      sycl::nd_item<Dim> this_item{&offset,
                                    group_id,
                                    local_id,
                                    local_size,
                                    num_groups,
                                    &barrier_impl,
                                    &group_shared_memory_ptr};

      f(this_item);
    });
#endif

    sycl::detail::host_local_memory::release();
  });
}

template <int Dim, class Function>
inline void parallel_for_workgroup(Function f,
                                   const sycl::range<Dim> num_groups,
                                   const sycl::range<Dim> local_size,
                                   size_t num_local_mem_bytes) noexcept
{
  static_assert(Dim > 0 && Dim <= 3, "Only dimensions 1,2,3 are supported");  

  parallel_invocation([=](){
    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);

    host::iterate_range_omp_for(num_groups, [&, f](sycl::id<Dim> group_id) {
      sycl::group<Dim> this_group{group_id, local_size, num_groups};

      f(this_group);
    });

    sycl::detail::host_local_memory::release();
  });
}

template <class HierarchicalDecomposition,
          class Function, int dimensions>
inline void parallel_region(Function f,
                            const sycl::range<dimensions> num_groups,
                            const sycl::range<dimensions> group_size,
                            std::size_t num_local_mem_bytes)
{
  static_assert(dimensions > 0 && dimensions <= 3,
                "Only dimensions 1,2,3 are supported");

  parallel_invocation([=]() {
    sycl::detail::host_local_memory::request_from_threadprivate_pool(
        num_local_mem_bytes);

    host::iterate_range_omp_for(num_groups, [&](sycl::id<dimensions> group_id) {
      using group_properties =
          sycl::detail::sp_property_descriptor<dimensions, 0,
                                               HierarchicalDecomposition>;

      sycl::detail::sp_group<
          sycl::detail::host_sp_property_descriptor<group_properties>>
          this_group{sycl::group<dimensions>{group_id, group_size, num_groups}};

      f(this_group);
    });

    sycl::detail::host_local_memory::release();
  });
}

template<int Dim, int MaxGuaranteedWorkgroupSize>
constexpr auto determine_hierarchical_decomposition() {
  using namespace sycl::detail;
  using fallback_decomposition =
      nested_range<unknown_static_range, nested_range<static_range<1>>>;

  if constexpr(Dim == 1) {
    if constexpr(MaxGuaranteedWorkgroupSize % 16 == 0) {
      return nested_range<
        unknown_static_range,
        nested_range<
          static_range<16>
        >
      >{};
    } else {
      return fallback_decomposition{};
    }
  } else if constexpr(Dim == 2){
    if constexpr(MaxGuaranteedWorkgroupSize % 4 == 0) {
      return nested_range<
          unknown_static_range,
          nested_range<
            static_range<4,4>
          >
        >{};
    } else {
      return fallback_decomposition{};
    }
  } else {
    if constexpr(MaxGuaranteedWorkgroupSize % 2 == 0) {
      return nested_range<
          unknown_static_range,
          nested_range<
            static_range<2,2,2>
          >
        >{};
    } else {
      return fallback_decomposition{};
    }
  }
}

}

class omp_kernel_launcher : public rt::backend_kernel_launcher
{
public:

  omp_kernel_launcher() {}
  virtual ~omp_kernel_launcher(){}

  virtual void set_params(void*) override {}

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k) {

    this->_type = type;
#if !defined(HIPSYCL_HAS_FIBERS) && !defined(__ACPP_USE_ACCELERATED_CPU__)
    if (type == rt::kernel_type::ndrange_parallel_for) {
      this->_invoker = [](rt::dag_node* node) {};

      throw sycl::exception{sycl::make_error_code(sycl::errc::feature_not_supported),
        "nd_range kernels on CPU are only supported if either compiler support (requires using Clang)\n"
        "or fibers are enabled, as otherwise they cannot be efficiently implemented. It is recommended:\n"
        " * to verify that you really need the features of nd_range parallel for.\n"
        "   If you do not need local memory, use basic parallel for instead.\n"
        " * users targeting SYCL 1.2.1 may use hierarchical parallel for, which\n"
        "   can express the same algorithms, but may have functionality caveats in hipSYCL\n"
        "   and/or other SYCL implementations.\n"
        " * if you use hipSYCL exclusively, you are encouraged to use scoped parallelism:\n"
        "   https://github.com/illuhad/hipSYCL/blob/develop/doc/scoped-parallelism.md\n"
        " * if you can use Clang, enable the compiler support\n"
        "   CMake: -DHIPSYCL_USE_ACCELERATED_CPU=ON, syclcc: --hipsycl-use-accelerated-cpu\n"
        " * if you absolutely need nd_range parallel for and cannot use Clang, enable fiber support in hipSYCL."
      };
    }
#endif

    this->_invoker = [=] (rt::dag_node* node) mutable {

      static_cast<rt::kernel_operation *>(node->get_operation())
          ->initialize_embedded_pointers(k);

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      auto get_grid_range = [&]() {
        for (int i = 0; i < Dim; ++i){
          if (global_range[i] % local_range[i] != 0) {
            rt::register_error(__acpp_here(),
                               rt::error_info{"omp_dispatch: global range is "
                                              "not divisible by local range"});
          }
        }

        return global_range / local_range;
      };

      if constexpr(type == rt::kernel_type::single_task){

        omp_dispatch::single_task_kernel(k);

      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

        if(!is_with_offset) {
          omp_dispatch::parallel_for_kernel(k, global_range);
        } else {
          omp_dispatch::parallel_for_kernel_offset(k, global_range, offset);
        }

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

        omp_dispatch::parallel_for_ndrange_kernel(
            k, get_grid_range(), local_range, offset, dynamic_local_memory);

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

        omp_dispatch::parallel_for_workgroup(k, get_grid_range(), local_range,
                                             dynamic_local_memory);
      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

        auto local_range_is_divisible_by = [&](int x) -> bool {
          for(int i = 0; i < Dim; ++i) {
            if(local_range[i] % x != 0)
              return false;
          }
          return true;
        };

        if(local_range_is_divisible_by(64)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<
                       Dim, 64>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory);
        } else if(local_range_is_divisible_by(32)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<
                       Dim, 32>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory);
        } else if(local_range_is_divisible_by(16)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<
                       Dim, 16>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory);
        } else if(local_range_is_divisible_by(8)) {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<Dim,
                                                                          8>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory);
        } else {
          using decomposition_type =
              decltype(omp_dispatch::determine_hierarchical_decomposition<Dim,
                                                                          1>());

          omp_dispatch::parallel_region<decomposition_type>(
              k, get_grid_range(), local_range, dynamic_local_memory);
        }
      } else if constexpr (type == rt::kernel_type::custom) {
        sycl::interop_handle handle{
            rt::device_id{rt::backend_descriptor{rt::hardware_platform::cpu,
                                                 rt::api_platform::omp},
                          0},
            static_cast<void*>(nullptr)};

        k(handle);
      }
      else {
        assert(false && "Unsupported kernel type");
      }

    };
  }

  virtual int get_backend_score(rt::backend_id b) const final override {
    return (b == rt::backend_id::omp) ? 2 : -1;
  }

  virtual void invoke(rt::dag_node *node,
                      const rt::kernel_configuration &) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:

  std::function<void (rt::dag_node*)> _invoker;
  rt::kernel_type _type;
};

}
}

#endif
