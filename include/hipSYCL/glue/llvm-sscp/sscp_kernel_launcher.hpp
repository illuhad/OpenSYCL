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
#ifndef HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/glue/generic/code_object.hpp"
#include "hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"
#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "ir_constants.hpp"
#include "../kernel_launcher_data.hpp"

#include <array>
#include <atomic>
#include <string_view>


template <typename KernelType>
// hipsycl_sscp_kernel causes kernel entries to be emitted to the HCF
[[clang::annotate("hipsycl_sscp_kernel")]]
// hipsycl_sscp_outlining creates an entrypoint for outlining of device code
[[clang::annotate("hipsycl_sscp_outlining")]]
void __acpp_sscp_kernel(const KernelType& kernel) {
  if(__acpp_sscp_is_device) {
    // The copy here creates an alloca that can help inferring the argument
    // type in case of opaque pointers.
    KernelType k = kernel;
    k();
  }
}


// hipSYCL SSCP LLVM magic will add definition, but clang warns - suppress until
// we find a better solution to implement things
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal" 
// The SSCP compiler will use this invocation to connect the kernel function
// to a global variable where the kernel name can be stored.
// First argument has to be a function pointer to the kernel,
// second one a pointer to a global variable.
// No indirection is allowed! If I say, the argument has to be a global variable,
// I mean it. Directly. No passing through other functions first.
template <class Kernel>
[[clang::annotate("acpp_sscp_extract_kernel_name")]]
void __acpp_sscp_extract_kernel_name(void (*Func)(const Kernel&),
                                        const char *target);
#pragma clang diagnostic pop

namespace hipsycl {
namespace glue {

namespace sscp {

static std::string get_local_hcf_object() {
  return std::string{
      reinterpret_cast<const char *>(__acpp_local_sscp_hcf_content),
      __acpp_local_sscp_hcf_object_size};
}

// TODO: Maybe this can be unified with the ACPP_STATIC_HCF_REGISTRATION
// macro. We cannot use this macro directly because it expects
// the object id to be constexpr, which it is not for the SSCP case.
struct static_hcf_registration {
  static_hcf_registration(const std::string& hcf_data) {
    this->_hcf_object = rt::hcf_cache::get().register_hcf_object(
        common::hcf_container{hcf_data});
  }

  ~static_hcf_registration() {
    rt::hcf_cache::get().unregister_hcf_object(_hcf_object);
  }
private:
  rt::hcf_object_id _hcf_object;
};
static static_hcf_registration
    __acpp_register_sscp_hcf_object{get_local_hcf_object()};


// This class effectively caches queries into the HCF cache: For each
// kernel lambda type, a separate object is instantiated which queries the HCF cache.
// This allows us to not query the cache for every single submission, instead just
// reusing the pointer here.
struct kernel_info_retriever {
  template <class Kernel>
  static const rt::hcf_kernel_info *get(const Kernel &, rt::hcf_object_id obj,
                         std::string_view kernel_name) {
    static kernel_info_retriever retriever;

    const rt::hcf_kernel_info *ptr =
        retriever._cached_info_ptr.load(std::memory_order_relaxed);
    if(!ptr) {
      ptr = rt::hcf_cache::get().get_kernel_info(obj, kernel_name);
      retriever._cached_info_ptr.store(ptr, std::memory_order_relaxed);
    }
    return ptr;
  }

private:
  std::atomic<const rt::hcf_kernel_info*> _cached_info_ptr = nullptr;
};

}

// Do not change this namespace name - compiler may look for
// this name to identify structs passed in as kernel arguments.
namespace __sscp_dispatch {

template <int Dimensions, bool WithOffset>
bool item_is_in_range(const sycl::item<Dimensions, WithOffset> &item,
                      const sycl::range<Dimensions> &execution_range,
                      const sycl::id<Dimensions>& offset = {}) {

  for(int i = 0; i < Dimensions; ++i) {
    if constexpr(WithOffset) {
      if(item.get_id(i) >= offset.get(i) + execution_range.get(i))
        return false;
    } else {
      if(item.get_id(i) >= execution_range.get(i))
        return false;
    }
  }
  return true;
}

template<class UserKernel>
class single_task {
public:
  single_task(const UserKernel& k)
  : _k{k} {}

  [[clang::annotate("hipsycl_kernel_dimension", 0)]]
  void operator()() const {
    _k();
  }
private:
  UserKernel _k;
};

template<class UserKernel, int Dimensions>
class basic_parallel_for {
public:
  basic_parallel_for(const UserKernel &k,
                     sycl::range<Dimensions> execution_range)
      : _k{k}, _range{execution_range} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    auto this_item = sycl::detail::make_item<Dimensions>(
      sycl::detail::get_global_id<Dimensions>(), _range
    );
    if(item_is_in_range(this_item, _range))
      _k(this_item);
  }
private:
  UserKernel _k;
  sycl::range<Dimensions> _range;
};

template<class UserKernel, int Dimensions>
class basic_parallel_for_offset {
public:
  basic_parallel_for_offset(const UserKernel &k, sycl::id<Dimensions> offset,
                            sycl::range<Dimensions> execution_range)
      : _k{k}, _range{execution_range}, _offset{offset} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    auto this_item = sycl::detail::make_item<Dimensions>(
        sycl::detail::get_global_id<Dimensions>() + _offset, _range, _offset);
    
    if(item_is_in_range(this_item, _range, _offset))
      _k(this_item);
  }
private:
  UserKernel _k;
  sycl::range<Dimensions> _range;
  sycl::id<Dimensions> _offset;
};


template<class UserKernel, int Dimensions>
class ndrange_parallel_for {
public:
  ndrange_parallel_for(const UserKernel& k)
  : _k{k} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    const sycl::id<Dimensions> zero_offset{};
    sycl::nd_item<Dimensions> this_item{
        &zero_offset, sycl::detail::get_group_id<Dimensions>(),
        sycl::detail::get_local_id<Dimensions>(),
        sycl::detail::get_local_size<Dimensions>(),
        sycl::detail::get_grid_size<Dimensions>()};

    _k(this_item);
  };
private:
  UserKernel _k;
};

template<class UserKernel, int Dimensions>
class ndrange_parallel_for_offset {
public:
  ndrange_parallel_for_offset(const UserKernel& k, sycl::id<Dimensions> offset)
  : _k{k}, _offset{offset} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    sycl::nd_item<Dimensions> this_item{
        &_offset, sycl::detail::get_group_id<Dimensions>(),
        sycl::detail::get_local_id<Dimensions>(),
        sycl::detail::get_local_size<Dimensions>(),
        sycl::detail::get_grid_size<Dimensions>()};

    _k(this_item);
  };
private:
  UserKernel _k;
  const sycl::id<Dimensions> _offset;
};

}

// NOTE: This class no longer follows the backend_kernel_launcher concept,
// in an effort to reduce latencies and indirections. It is likely that this
// new concept will continue to change as the other launchers are also aligned
// with it.
class sscp_kernel_launcher
{
public:
  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel>
  static void create(kernel_launcher_data &data, sycl::id<Dim> offset,
                     sycl::range<Dim> global_range,
                     sycl::range<Dim> local_range,
                     std::size_t dynamic_local_memory, Kernel k) {

    data.type = type;
    data.sscp_hcf_object_id = __acpp_local_sscp_hcf_object_id;
    data.sscp_invoker = &invoke;

    if constexpr(type == rt::kernel_type::single_task){

      configure_launch_with_global_range(data, __sscp_dispatch::single_task{k},
                                         sycl::range{1}, sycl::range{1},
                                         dynamic_local_memory);

    } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

      if(offset == sycl::id<Dim>{}) {
        configure_launch_with_global_range(data,
            __sscp_dispatch::basic_parallel_for{k, global_range}, global_range,
            local_range, dynamic_local_memory);
      } else {
        configure_launch_with_global_range(data,
            __sscp_dispatch::basic_parallel_for_offset{k, offset, global_range},
            global_range, local_range, dynamic_local_memory);
      }

    } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

      if(offset == sycl::id<Dim>{}) {
        configure_launch_with_global_range(data,
            __sscp_dispatch::ndrange_parallel_for<Kernel, Dim>{k}, global_range,
            local_range, dynamic_local_memory);
      } else {
        configure_launch_with_global_range(data,
            __sscp_dispatch::ndrange_parallel_for_offset<Kernel, Dim>{k, offset},
            global_range, local_range, dynamic_local_memory);
      }

    } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

    } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {
      
    } else if constexpr (type == rt::kernel_type::custom) {
      // handled at invoke time
      data.custom_op = [k](rt::kernel_operation * kernel_op, sycl::interop_handle& ih) mutable {
          kernel_op->initialize_embedded_pointers(
              static_cast<void*>(&k),
              sizeof(Kernel));
          k(ih);
       };
    }
    else {
      assert(false && "Unsupported kernel type");
    }
  }

  static rt::result
  invoke(const kernel_launcher_data& launch_config, rt::dag_node *node,
         const rt::kernel_configuration &kernel_config,
         const rt::backend_kernel_launch_capabilities &launch_capabilities,
         void *backend_params = nullptr) {
    assert(node);

    if(launch_config.type == rt::kernel_type::custom) {
      assert(backend_params);
      sycl::interop_handle handle{node->get_assigned_device(),
                                  backend_params};
      auto *kernel_op =
          static_cast<rt::kernel_operation *>(node->get_operation());
      launch_config.custom_op(kernel_op, handle);

      return rt::make_success();
    } else {
      auto *kernel_op =
          static_cast<rt::kernel_operation *>(node->get_operation());

      kernel_op->initialize_embedded_pointers(launch_config.kernel_args.data(),
                                              launch_config.kernel_args.size());

      auto sscp_invoker = launch_capabilities.get_sscp_invoker();
      if(!sscp_invoker) {
        return rt::make_error(
          __acpp_here(),
          rt::error_info{"Attempted to prepare to launch SSCP kernel, but the backend "
                         "did not configure the kernel launcher for SSCP."});
      }
      auto *invoker = sscp_invoker.value();

      auto selected_group_size = launch_config.group_size;
      if (launch_config.group_size.size() == 0)
        selected_group_size = invoker->select_group_size(
            launch_config.global_size, launch_config.group_size);

      rt::range<3> num_groups;
      for(int i = 0; i < 3; ++i) {
        num_groups[i] = (launch_config.global_size[i] + selected_group_size[i] - 1) /
                        selected_group_size[i];
      }

      std::array<const void*, 1> args{launch_config.kernel_args.data()};
      std::size_t arg_size = launch_config.kernel_args.size();

      return invoker->submit_kernel(
          *kernel_op, launch_config.sscp_hcf_object_id, num_groups,
          selected_group_size, launch_config.local_mem_size,
          const_cast<void **>(args.data()), &arg_size, args.size(),
          launch_config.sscp_kernel_id, launch_config.kernel_info,
          kernel_config);
    }
  }

private:
  template <int Dim>
  static rt::range<3> flip_range(const sycl::range<Dim> &r) {
    rt::range<3> rt_range{1,1,1};

    for (int i = 0; i < Dim; ++i) {
      rt_range[i] = r[Dim - i - 1];
    }

    return rt_range;
  }

  template <class Kernel, int Dim>
  static void configure_launch_with_global_range(kernel_launcher_data& out,
                                          const Kernel &k,
                                          const sycl::range<Dim> &global_range,
                                          const sycl::range<Dim> &group_size,
                                          unsigned local_mem_size) {

    out.global_size = flip_range(global_range);
    out.group_size = flip_range(group_size);
    out.local_mem_size = local_mem_size;
    out.kernel_args.resize(sizeof(Kernel));
    std::memcpy(out.kernel_args.data(), &k, sizeof(Kernel));

    out.sscp_kernel_id = generate_kernel(k);
    out.kernel_info = sscp::kernel_info_retriever::get(
        k, out.sscp_hcf_object_id, out.sscp_kernel_id);
  }


  // Generate SSCP kernel and return name of the generated kernel
  template<class Kernel>
  static const char* generate_kernel(const Kernel& k) {
    if (__acpp_sscp_is_device) {
      __acpp_sscp_kernel(k);
    }

    // Compiler will change the number of elements to the kernel name length
    static char __acpp_sscp_kernel_name [] = "kernel-name-extraction-failed";

    __acpp_sscp_extract_kernel_name<Kernel>(
        &__acpp_sscp_kernel<Kernel>,
        &__acpp_sscp_kernel_name[0]);
    return &__acpp_sscp_kernel_name[0];
  }
  
  kernel_launcher_data* _data;
};

}
}

#endif
