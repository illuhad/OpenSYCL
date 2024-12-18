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
#ifndef HIPSYCL_QUEUE_HPP
#define HIPSYCL_QUEUE_HPP

#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_executor.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/sycl/backend.hpp"
#include "types.hpp"
#include "exception.hpp"

#include "property.hpp"
#include "libkernel/backend.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "context.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "info/info.hpp"
#include "detail/function_set.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <atomic>

namespace hipsycl {
namespace sycl {

namespace detail {

template<typename, int, access::mode, access::target>
class automatic_placeholder_requirement_impl;

using queue_submission_hooks =
  function_set<sycl::handler&>;
using queue_submission_hooks_ptr = 
  shared_ptr_class<queue_submission_hooks>;

}

namespace property::command_group {

template<int Dim>
struct AdaptiveCpp_prefer_group_size : public detail::cg_property{
  AdaptiveCpp_prefer_group_size(range<Dim> r)
  : size{r} {}

  const range<Dim> size;
};

struct AdaptiveCpp_retarget : public detail::cg_property{
  AdaptiveCpp_retarget(const device& d)
  : dev{d} {}

  const sycl::device dev;
};

struct AdaptiveCpp_prefer_execution_lane : public detail::cg_property{
  AdaptiveCpp_prefer_execution_lane(std::size_t lane_id)
  : lane{lane_id} {}

  const std::size_t lane;
};

struct AdaptiveCpp_coarse_grained_events : public detail::cg_property {};

// backwards compatibility
template<int Dim>
using hipSYCL_prefer_group_size = AdaptiveCpp_prefer_group_size<Dim>;
using hipSYCL_retarget = AdaptiveCpp_retarget;
using hipSYCL_prefer_execution_lane = AdaptiveCpp_prefer_execution_lane;
using hipSYCL_coarse_grained_events = AdaptiveCpp_coarse_grained_events;

}


namespace property::queue {

class in_order : public detail::queue_property
{};

class enable_profiling : public detail::property
{};

class AdaptiveCpp_coarse_grained_events : public detail::queue_property
{};

struct AdaptiveCpp_priority : public detail::queue_property {
  AdaptiveCpp_priority(int queue_execution_priority)
  : priority{queue_execution_priority} {}

  int priority;
};

struct AdaptiveCpp_retargetable : public detail::queue_property {};

// backwards compatibility
using hipSYCL_coarse_grained_events = AdaptiveCpp_coarse_grained_events;
using hipSYCL_priority = AdaptiveCpp_priority;

}


class queue : public detail::property_carrying_object
{
  struct queue_impl {
    queue_impl(const context &c, const async_handler &h)
        : ctx{c}, handler{h}, allocation_cache{
                                  algorithms::util::allocation_type::device} {}

    rt::runtime_keep_alive_token requires_runtime;  
    detail::queue_submission_hooks_ptr hooks;

    rt::execution_hints default_hints;
    context ctx;
    async_handler handler;
    bool is_in_order = false;
    bool is_retargetable = false;
    // if this is true, in-order queues will be emulated by storing the most recent
    // event from a submission, and adding that as dependency for the next event.
    bool needs_in_order_emulation = true;
    

    // Note: This must not be a weak_ptr, since in the case of instant submissions,
    // the lifetime of nodes is not guaranteed to exceed task runtime.
    rt::dag_node_ptr previous_submission = nullptr;
    std::mutex lock;
    std::size_t node_group_id = -1;
    std::shared_ptr<rt::backend_executor> dedicated_inorder_executor = nullptr;
  
    // These fields are exclusively hauled around for SYCL 2020 reductions
    // due to the incredible ingenuity of this API...
    algorithms::util::allocation_cache allocation_cache;
    std::weak_ptr<rt::dag_node> most_recent_reduction_kernel;

    // Prevents kernel cache from becoming invalid while we have a queue
    std::shared_ptr<rt::kernel_cache> kernel_cache;
    // For non-emulated in-order queues only
    std::atomic<bool> has_non_instant_operations = false;
  };

  template<typename, int, access::mode, access::target>
  friend class detail::automatic_placeholder_requirement_impl;

public:
  explicit queue(const property_list &propList = {})
      : queue{default_selector_v,
              [](exception_list e) { glue::default_async_handler(e); },
              propList} {
    assert(_impl->default_hints.has_hint<rt::hints::bind_to_device>());
  }

  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue{default_selector_v, asyncHandler, propList} {
    assert(_impl->default_hints.has_hint<rt::hints::bind_to_device>());
  }

  template <
      class DeviceSelector,
      std::enable_if_t<detail::is_device_selector_v<DeviceSelector>, int> = 0>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue{detail::select_devices(deviceSelector), propList} {}

  template <
      class DeviceSelector,
      std::enable_if_t<detail::is_device_selector_v<DeviceSelector>, int> = 0>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue{detail::select_devices(deviceSelector), asyncHandler, propList} {}

  explicit queue(const device &syclDevice, const property_list &propList = {})
      : queue{get_default_context(syclDevice), std::vector<device>{syclDevice},
              propList} {}

  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue{get_default_context(syclDevice), std::vector<device>{syclDevice},
              asyncHandler, propList} {}

  template <
      class DeviceSelector,
      std::enable_if_t<detail::is_device_selector_v<DeviceSelector>, int> = 0>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue(syclContext, detail::select_devices(deviceSelector), propList) {}

  template <
      class DeviceSelector,
      std::enable_if_t<detail::is_device_selector_v<DeviceSelector>, int> = 0>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(syclContext, detail::select_devices(deviceSelector), asyncHandler,
              propList) {}

  explicit queue(const context &syclContext,
                 const device &syclDevice,
                 const property_list &propList = {})
      : queue{syclContext, std::vector<device>{syclDevice}, propList} {}

  explicit queue(const context &syclContext, const device &syclDevice,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue{syclContext, std::vector<device>{syclDevice}, asyncHandler,
              propList} {}

  // hipSYCL-specific constructors for multiple devices

  explicit queue(const std::vector<device> &devices,
                 const async_handler &handler,
                 const property_list &propList = {})
      : queue{get_default_context(devices), devices, handler, propList} {}

  explicit queue(const std::vector<device>& devices, const property_list& propList = {})
    : queue{get_default_context(devices), devices, propList} {}

  explicit queue(const context &syclContext, const std::vector<device> &devices,
                 const property_list &propList = {})
      : queue{syclContext, devices, async_handler{syclContext._impl->handler},
              propList} {}

  explicit queue(const context &syclContext,
                 const std::vector<device> &devices,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : detail::property_carrying_object{propList} {
    
    _impl = std::make_shared<queue_impl>(syclContext, asyncHandler);

    if(devices.empty()) {
      throw exception{make_error_code(errc::invalid),
                      "queue: No devices in device list"};
    }

    for(const auto& dev : devices)
      if (!is_device_in_context(dev, syclContext))
        throw exception{make_error_code(errc::invalid),
                        "queue: Device is not in context"};

    if(devices.size() == 1){
      _impl->default_hints.set_hint(rt::hints::bind_to_device{
          detail::extract_rt_device(devices[0])});
    }
    else if(devices.size() > 1) {
      std::vector<rt::device_id> rt_devs;
      for(const auto& d : devices) {
        rt_devs.push_back(detail::extract_rt_device(d));
      }
      _impl->default_hints.set_hint(
          rt::hints::bind_to_device_group{rt_devs});
    }
    // Otherwise we are in completely unrestricted scheduling land - don't
    // add any hints regarding target device.

    this->init();
  }

  ~queue() {
    this->throw_asynchronous();
  }


  context get_context() const {
    return _impl->ctx;
  }

  device get_device() const {
    if (_impl->default_hints.has_hint<rt::hints::bind_to_device>()) {
      rt::device_id id =
          _impl->default_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();
      return device{id};
    } else {
      throw exception{make_error_code(errc::feature_not_supported),
                      "queue::get_device() is unsupported for "
                      "multi-device queues"};
    }
  }

  std::vector<device> get_devices() const {
    if(_impl->default_hints.has_hint<rt::hints::bind_to_device>()) {

      rt::device_id id =
          _impl->default_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();
      return std::vector<device>{device{id}};

    } else if(_impl->default_hints.has_hint<rt::hints::bind_to_device_group>()) {

      std::vector<device> devs;
      for (const auto &d :
           _impl->default_hints.get_hint<rt::hints::bind_to_device_group>()
               ->get_devices()) {
        devs.push_back(device{d});
      }

      return devs;
    }
    return std::vector<device>{};
  }

  bool is_host() const {
    auto devs = get_devices();
    if(devs.empty())
      return false;
    
    for(const auto& d : devs) {
      if(!d.is_host())
        return false;
    }
    return true;
  }

  bool is_in_order() const {
    return _impl->is_in_order;
  }

  void wait() {
    if(_impl->is_in_order) {
      if(_impl->needs_in_order_emulation) {
        rt::dag_node_ptr most_recent_event = nullptr;
        {
          std::lock_guard<std::mutex> lock{_impl->lock};

          most_recent_event = _impl->previous_submission;
        }
        if(most_recent_event) {
          // Flush DAG for non-submitted events. Note that this does not affect
          // instant nodes, as they immediately assume the submitted state.
          if(!most_recent_event->is_submitted())
            _impl->requires_runtime.get()->dag().flush_sync();
          
          most_recent_event->wait();
        }
      } else {
        rt::inorder_executor* exec = AdaptiveCpp_inorder_executor();
        assert(exec);
        // Need to ensure everything is submitted before waiting on the stream
        // in case we have non-instant operations
        if(_impl->has_non_instant_operations.load(std::memory_order_relaxed))
          _impl->requires_runtime.get()->dag().flush_sync();
        
        auto err = exec->wait();
        if(!err.is_success()) {
          // We might want to throw a synchronous error here?
          rt::register_error(err);
        }
      }
    } else {
      _impl->requires_runtime.get()->dag().flush_sync();
      _impl->requires_runtime.get()->dag().wait(_impl->node_group_id);
    }
  }

  void wait_and_throw() {
    this->wait();
    this->throw_asynchronous();
  }

  void throw_asynchronous() {
    glue::throw_asynchronous_errors(_impl->handler);
  }

  template <typename Param>
  typename Param::return_type get_info() const;


  template <typename T>
  event submit(const property_list& prop_list, T cgf) {
    std::lock_guard<std::mutex> lock{_impl->lock};

    rt::execution_hints hints = _impl->default_hints;
    
    if(prop_list.has_property<property::command_group::AdaptiveCpp_retarget>()) {
      if(!_impl->is_retargetable)
        throw exception{make_error_code(errc::invalid),
                        "queue: Attempted to use AdaptiveCpp_retarget "
                        "extension with a queue that is not constructed with "
                        "AdaptiveCpp_retargetable property"};

      rt::device_id dev = detail::extract_rt_device(
          prop_list.get_property<property::command_group::AdaptiveCpp_retarget>()
              .dev);

      if(!detail::extract_context_devices(_impl->ctx).contains_device(dev)) {
        HIPSYCL_DEBUG_WARNING
            << "queue: Warning: Retargeting operation for a device that is not "
               "part of the queue's context. This can cause terrible problems if the "
               "operation uses USM allocations that were allocated using the "
               "queue's context."
            << std::endl;
      }

      hints.set_hint(rt::hints::bind_to_device{dev});
    }
    if (prop_list.has_property<
            property::command_group::AdaptiveCpp_prefer_execution_lane>()) {

      std::size_t lane_id =
          prop_list
              .get_property<
                  property::command_group::AdaptiveCpp_prefer_execution_lane>()
              .lane;

      hints.set_hint(rt::hints::prefer_execution_lane{lane_id});
    }
    if (prop_list.has_property<
            property::command_group::AdaptiveCpp_coarse_grained_events>()) {
      hints.set_hint(rt::hints::coarse_grained_synchronization{});
    }
    // Should always have node_group hint from default hints
    assert(hints.has_hint<rt::hints::node_group>());

    handler cgh{get_context(),
                _impl->handler,
                hints,
                _impl->requires_runtime.get(),
                &(_impl->allocation_cache),
                &(_impl->most_recent_reduction_kernel)};

    apply_preferred_group_size<1>(prop_list, cgh);
    apply_preferred_group_size<2>(prop_list, cgh);
    apply_preferred_group_size<3>(prop_list, cgh);

    this->get_hooks()->run_all(cgh);

    rt::dag_node_ptr node = execute_submission(cgf, cgh);
    
    return event{node, _impl->handler};
  }


  template <typename T>
  event submit(T cgf) {
    return submit(property_list{}, cgf);
  }

  template <typename T>
  event submit(T cgf, queue &secondaryQueue,
               const property_list &prop_list = {}) {
    try {

      size_t num_errors_begin =
          rt::application::errors().num_errors();

      event evt = submit(prop_list, cgf);
      // Flush so that we see any errors during submission
      _impl->requires_runtime.get()->dag().flush_sync();

      size_t num_errors_end =
          rt::application::errors().num_errors();

      bool submission_failed = false;
      // TODO This approach fails if an async handler has consumed
      // the errors in the meantime
      if(num_errors_end != num_errors_begin) {
        // Need to check if there was a kernel error..
        rt::application::errors().for_each_error(
            [&](const rt::result &err) {
              if (!err.is_success()) {
                if (err.info().get_error_type() ==
                    rt::error_type::kernel_error) {
                  submission_failed = true;
                }
              }
            });
      }

      if(!submission_failed) {
        return evt;
      } else {
        return secondaryQueue.submit(prop_list, cgf);
      }
    }
    catch(exception&) {
      return secondaryQueue.submit(prop_list, cgf);
    }
  }

  friend bool operator==(const queue& lhs, const queue& rhs)
  { return lhs._impl == rhs._impl; }

  friend bool operator!=(const queue& lhs, const queue& rhs)
  { return !(lhs == rhs); }

  std::vector<event> get_wait_list() {
    if(is_in_order()) {
      if(_impl->needs_in_order_emulation) {
        std::lock_guard<std::mutex> lock{_impl->lock};

        if(auto prev = _impl->previous_submission){
          if(!prev->is_known_complete()) {
            return std::vector<event>{event{prev, _impl->handler}};
          }
        }
        // If we don't have a previous event or it's complete,
        // just return empty vector
        return std::vector<event>{};
      } else {
        // If we don't have in-order emulation, we need to create a new
        // event. We use enqueue_custom_operation to effectively obtain
        // an asynchronous barrier.
        return std::vector{AdaptiveCpp_enqueue_custom_operation([](auto&){})};
      }
      
    } else {
      // for non-in-order queues we need to ask the runtime for
      // all nodes of this node group
      _impl->requires_runtime.get()->dag().flush_sync();
      auto nodes =
          _impl->requires_runtime.get()->dag().get_group(_impl->node_group_id);
      std::vector<event> evts;
      for(auto node : nodes){
        if(!node->is_known_complete())
          evts.push_back(event{node, _impl->handler});
      }

      return evts;
    }
  }

  // ---- Queue shortcuts ------

  template <typename KernelName = __acpp_unnamed_kernel, typename KernelType>
  event single_task(const KernelType &KernelFunc) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.single_task<KernelName>(KernelFunc);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel, typename KernelType>
  event single_task(event dependency, const KernelType &KernelFunc) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.single_task<KernelName>(KernelFunc);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel, typename KernelType>
  event single_task(const std::vector<event> &dependencies,
                    const KernelType &KernelFunc) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.single_task<KernelName>(KernelFunc);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel, 
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(range<Dims> NumWorkItems, 
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(NumWorkItems, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel>
  event parallel_for(range<1> NumWorkItems,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(NumWorkItems, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(range<Dims> NumWorkItems, event dependency,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<KernelName>(NumWorkItems, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel>
  event parallel_for(range<1> NumWorkItems, event dependency,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<KernelName>(NumWorkItems, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(range<Dims> NumWorkItems,
                     const std::vector<event> &dependencies,
                     const ReductionsAndKernel& ... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<KernelName>(NumWorkItems, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel>
  event parallel_for(range<1> NumWorkItems,
                     const std::vector<event> &dependencies,
                     const ReductionsAndKernel& ... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<KernelName>(NumWorkItems, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     const ReductionsAndKernel& ... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(NumWorkItems, WorkItemOffset,
                                   redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     event dependency,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<KernelName>(NumWorkItems, WorkItemOffset,
                                   redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                     const std::vector<event> &dependencies,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<KernelName>(NumWorkItems, WorkItemOffset,
                                   redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(ExecutionRange, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange, event dependency,
                     const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<KernelName>(ExecutionRange, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int Dims>
  event parallel_for(nd_range<Dims> ExecutionRange,
                     const std::vector<event> &dependencies,
                     const ReductionsAndKernel& ... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<KernelName>(ExecutionRange, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  event parallel(range<dimensions> numWorkGroups,
                range<dimensions> workGroupSize,
                const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.parallel<KernelName>(numWorkGroups, workGroupSize, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  event parallel(range<dimensions> numWorkGroups,
                range<dimensions> workGroupSize, event dependency,
                const ReductionsAndKernel& ... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.parallel<KernelName>(numWorkGroups, workGroupSize, redu_kernel...);
    });
  }

  template <typename KernelName = __acpp_unnamed_kernel,
            typename... ReductionsAndKernel, int dimensions>
  event parallel(range<dimensions> numWorkGroups,
                range<dimensions> workGroupSize,
                const std::vector<event> &dependencies,
                const ReductionsAndKernel &... redu_kernel) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel<KernelName>(numWorkGroups, workGroupSize, redu_kernel...);
    });
  }

  event memcpy(void *dest, const void *src, std::size_t num_bytes) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.memcpy(dest, src, num_bytes);
    });
  }

  event memcpy(void *dest, const void *src, std::size_t num_bytes,
               event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.memcpy(dest, src, num_bytes);
    });
  }

  event memcpy(void *dest, const void *src, std::size_t num_bytes,
               const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.memcpy(dest, src, num_bytes);
    });
  }
  
  template <typename T>
  event copy(const T* src, T* dest, std::size_t count) {
    return this->memcpy(static_cast<void*>(dest), 
      static_cast<const void*>(src), count * sizeof(T));
  }
  
  template <typename T>
  event copy(const T* src, T* dest, std::size_t count, 
             event dependency) {
    return this->memcpy(static_cast<void*>(dest), 
      static_cast<const void*>(src), count * sizeof(T), dependency);
  }
  
  template <typename T>
  event copy(const T* src, T* dest, std::size_t count, 
             const std::vector<event>& dependencies) {
    return this->memcpy(static_cast<void*>(dest), 
      static_cast<const void*>(src), count * sizeof(T), dependencies);
  }

  event memset(void *ptr, int value, std::size_t num_bytes) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.memset(ptr, value, num_bytes);
    });
  }

  event memset(void *ptr, int value, std::size_t num_bytes, event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.memset(ptr, value, num_bytes);
    });
  }

  event memset(void *ptr, int value, std::size_t num_bytes,
               const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.memset(ptr, value, num_bytes);
    });
  }

  template <class T>
  event fill(void *ptr, const T &pattern, std::size_t count) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.fill(ptr, pattern, count);
    });
  }

  template <class T>
  event fill(void *ptr, const T &pattern, std::size_t count, event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.fill(ptr, pattern, count);
    });
  }

  template <class T>
  event fill(void *ptr, const T &pattern, std::size_t count,
             const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.fill(ptr, pattern, count);
    });
  }

  event prefetch(const void *ptr, std::size_t num_bytes) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.prefetch(ptr, num_bytes);
    });
  }

  event prefetch(const void *ptr, std::size_t num_bytes, event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.prefetch(ptr, num_bytes);
    });
  }

  event prefetch(const void *ptr, std::size_t num_bytes,
                 const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.prefetch(ptr, num_bytes);
    });
  }

  event prefetch_host(const void *ptr, std::size_t num_bytes) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.prefetch_host(ptr, num_bytes);
    });
  }

  event prefetch_host(const void *ptr, std::size_t num_bytes, event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.prefetch_host(ptr, num_bytes);
    });
  }

  event prefetch_host(const void *ptr, std::size_t num_bytes,
                      const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.prefetch_host(ptr, num_bytes);
    });
  }

  event mem_advise(const void *addr, std::size_t num_bytes, int advice) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.mem_advise(addr, num_bytes, advice);
    });
  }

  event mem_advise(const void *addr, std::size_t num_bytes, int advice,
                   event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.mem_advise(addr, num_bytes, advice);
    });
  }

  event mem_advise(const void *addr, std::size_t num_bytes, int advice,
                   const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.mem_advise(addr, num_bytes, advice);
    });
  }

  template<class InteropFunction>
  event AdaptiveCpp_enqueue_custom_operation(InteropFunction op) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.AdaptiveCpp_enqueue_custom_operation(op);
    });
  }

  template <class InteropFunction>
  event AdaptiveCpp_enqueue_custom_operation(InteropFunction op, event dependency) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependency);
      cgh.AdaptiveCpp_enqueue_custom_operation(op);
    });
  }

  template <class InteropFunction>
  event
  AdaptiveCpp_enqueue_custom_operation(InteropFunction op,
                                   const std::vector<event> &dependencies) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.depends_on(dependencies);
      cgh.AdaptiveCpp_enqueue_custom_operation(op);
    });
  }

  template<class InteropFunction>
  [[deprecated("Use AdaptiveCpp_enqueue_custom_operation()")]]
  event hipSYCL_enqueue_custom_operation(InteropFunction op) {
    return AdaptiveCpp_enqueue_custom_operation(op);
  }

  template <class InteropFunction>
  [[deprecated("Use AdaptiveCpp_enqueue_custom_operation()")]]
  event hipSYCL_enqueue_custom_operation(InteropFunction op, event dependency) {
    return AdaptiveCpp_enqueue_custom_operation(op, dependency);
  }

  template <class InteropFunction>
  [[deprecated("Use AdaptiveCpp_enqueue_custom_operation()")]]
  event
  hipSYCL_enqueue_custom_operation(InteropFunction op,
                                   const std::vector<event> &dependencies) {
    return AdaptiveCpp_enqueue_custom_operation(op, dependencies);
  }

  /// Placeholder accessor shortcuts
  
  // Explicit copy functions
  
  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event copy(accessor<T, dim, mode, tgt, isPlaceholder> src,
             shared_ptr_class<T> dest) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(src);
      cgh.copy(src, dest);
    });           
  }
  
  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event copy(shared_ptr_class<T> src,
             accessor<T, dim, mode, tgt, isPlaceholder> dest) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(dest);
      cgh.copy(src, dest);
    });           
  }

  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event copy(accessor<T, dim, mode, tgt, isPlaceholder> src,
             T *dest) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(src);
      cgh.copy(src, dest);
    });     
  }

  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event copy(const T *src,
             accessor<T, dim, mode, tgt, isPlaceholder> dest) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(dest);
      cgh.copy(src, dest);
    });             
  }

  template <typename T, int dim, access_mode srcMode, access_mode dstMode,
            target srcTgt, target destTgt,
            accessor_variant isPlaceholderSrc, accessor_variant isPlaceholderDst>
  event copy(accessor<T, dim, srcMode, srcTgt, isPlaceholderSrc> src,
             accessor<T, dim, dstMode, destTgt, isPlaceholderDst> dest) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(src);
      cgh.require(dest);
      cgh.copy(src, dest);
    });  
  }

  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event update_host(accessor<T, dim, mode, tgt, isPlaceholder> acc) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(acc);
      cgh.update_host(acc);
    });  
  }
  
  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event update(accessor<T, dim, mode, tgt, isPlaceholder> acc) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(acc);
      cgh.update(acc);
    });  
  }

  template <typename T, int dim, access_mode mode, target tgt,
            accessor_variant isPlaceholder>
  event fill(accessor<T, dim, mode, tgt, isPlaceholder> dest, const T &src) {
    return this->submit([&](sycl::handler &cgh) {
      cgh.require(dest);
      cgh.fill(dest, src);
    });  
  }

  std::size_t AdaptiveCpp_hash_code() const {
    return _impl->node_group_id;
  }

  rt::inorder_executor* AdaptiveCpp_inorder_executor() const {
    if(!_impl->dedicated_inorder_executor)
      return nullptr;
    return static_cast<rt::inorder_executor *>(
        _impl->dedicated_inorder_executor.get());
  }


  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

  [[deprecated("Use AdaptiveCpp_inorder_executor()")]]
  auto hipSYCL_inorder_executor() const {
    return AdaptiveCpp_inorder_executor();
  }
private:
  static context get_default_context(const device& dev) {
    return context{detail::default_context_tag_t{}, dev.get_platform()};
  }

  static context get_default_context(const std::vector<device> &devices) {
    if(devices.empty())
      return context{detail::default_context_tag_t{}};
    if(devices.size() == 1){
      return context{detail::default_context_tag_t{}, devices[0].get_platform()};
    } else {
      return context{detail::default_context_tag_t{}, devices};
    }
  }

  template<int Dim>
  void apply_preferred_group_size(const property_list& prop_list, handler& cgh) {
    if(prop_list.has_property<property::command_group::AdaptiveCpp_prefer_group_size<Dim>>()){
      sycl::range<Dim> preferred_group_size =
          prop_list
              .get_property<
                  property::command_group::AdaptiveCpp_prefer_group_size<Dim>>()
              .size;
      cgh.set_preferred_group_size(preferred_group_size);
    }
  }

  template <class Cgf>
  rt::dag_node_ptr execute_submission(Cgf cgf, handler &cgh) {
    if (is_in_order() && _impl->needs_in_order_emulation) {
      auto previous = _impl->previous_submission;
      if(previous)
        cgh.depends_on(event{previous, _impl->handler});
    }
    
    cgf(cgh);

    rt::dag_node_ptr node = this->extract_dag_node(cgh);
    if (is_in_order()) {
      if(_impl->needs_in_order_emulation) {
        _impl->previous_submission = node;
      } else if(cgh.contains_non_instant_nodes()) {
        _impl->has_non_instant_operations.store(true, std::memory_order_relaxed);
        // If we have instant submission enabled, non-emulated in-order queue
        // but non-instant tasks, we need to flush the dag, otherwise future instant
        // tasks might not wait on the tasks that have been cached in the dag
        // builder.
#if ACPP_ALLOW_INSTANT_SUBMISSION
        _impl->requires_runtime.get()->dag().flush_sync();
#endif
      }
    }

    return node;
  }
      
  bool is_device_in_context(const device &dev, const context &ctx) const {    
    std::vector<device> devices = ctx.get_devices();
    for (const auto context_dev : devices) {
      if (context_dev == dev)
        return true;
    }
    return false;
  }

  rt::dag_node_ptr extract_dag_node(sycl::handler& cgh) {

    const rt::node_list_t &dag_nodes = cgh.get_cg_nodes();

    if(dag_nodes.empty()) {
      HIPSYCL_DEBUG_ERROR
          << "queue: Command queue evaluation did not result in the creation "
             "of events. Are there operations inside the command group?"
          << std::endl;
      return nullptr;
    }
    if(dag_nodes.size() > 1) {
      HIPSYCL_DEBUG_ERROR
          << "queue: Multiple events returned from command group evaluation; "
             "multiple operations in a single command group is not SYCL "
             "conformant. Returning event to the last operation"
          << std::endl;
    }
    return dag_nodes.back();
  }


  void init() {
    assert(_impl);

    static std::atomic<std::size_t> node_group_id;
    _impl->node_group_id = ++node_group_id;
    
    HIPSYCL_DEBUG_INFO << "queue: Constructed queue with node group id "
                       << _impl->node_group_id << std::endl;

    _impl->default_hints.set_hint(rt::hints::node_group{_impl->node_group_id});

    if (this->has_property<property::queue::enable_profiling>()) {
      _impl->default_hints.set_hint(
          rt::hints::request_instrumentation_submission_timestamp{});
      _impl->default_hints.set_hint(
              rt::hints::request_instrumentation_start_timestamp{});
      _impl->default_hints.set_hint(
              rt::hints::request_instrumentation_finish_timestamp{});
    }
    if(this->has_property<property::queue::AdaptiveCpp_coarse_grained_events>()){
      _impl->default_hints.set_hint(
          rt::hints::coarse_grained_synchronization{});
    }

    if(this->has_property<property::queue::AdaptiveCpp_retargetable>()) {
      _impl->is_retargetable = true;
    }

    _impl->is_in_order = this->has_property<property::queue::in_order>();

    if(_impl->is_in_order && get_devices().size() == 1) {
      int priority = 0;
      if(this->has_property<property::queue::AdaptiveCpp_priority>()) {
        priority = this->get_property<property::queue::AdaptiveCpp_priority>().priority;
      }

      rt::device_id rt_dev = detail::extract_rt_device(this->get_device());
      // Dedicated executor may not be supported by all backends,
      // so this might return nullptr.
      _impl->dedicated_inorder_executor =
          _impl->requires_runtime.get()
              ->backends()
              .get(rt_dev.get_backend())
              ->create_inorder_executor(rt_dev, priority);
      
      if(_impl->dedicated_inorder_executor) {
        _impl->default_hints.set_hint(
            rt::hints::prefer_executor{_impl->dedicated_inorder_executor});
      }
    }

    _impl->hooks = detail::queue_submission_hooks_ptr{
          new detail::queue_submission_hooks{}};

    if (_impl->is_in_order && _impl->dedicated_inorder_executor &&
        !_impl->is_retargetable &&
        _impl->default_hints.has_hint<rt::hints::bind_to_device>())
      _impl->needs_in_order_emulation = false;
    
    _impl->kernel_cache = rt::kernel_cache::get();
  }


  detail::queue_submission_hooks_ptr get_hooks() const
  {
    return _impl->hooks;
  }

  std::shared_ptr<queue_impl> _impl;
};

HIPSYCL_SPECIALIZE_GET_INFO(queue, context)
{
  return get_context();
}

HIPSYCL_SPECIALIZE_GET_INFO(queue, device)
{
  return get_device();
}

HIPSYCL_SPECIALIZE_GET_INFO(queue, reference_count)
{
  return 1;
}

HIPSYCL_SPECIALIZE_GET_INFO(queue, AdaptiveCpp_node_group)
{
  return _impl->node_group_id;
}

namespace detail{


template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
class automatic_placeholder_requirement_impl
{
public:
  automatic_placeholder_requirement_impl(sycl::queue &q, 
      sycl::accessor<dataT, dimensions, accessMode, accessTarget,
                access::placeholder::true_t>* acc)
    : _acc{acc}, _is_required{false}, _hooks{q.get_hooks()}
  {
    acquire();
  }

  void reacquire()
  {
    if(!_is_required)
      acquire();
  }

  void release()
  {
    if(_is_required)
      _hooks->remove(_hook_id);
    _is_required = false;
  }

  ~automatic_placeholder_requirement_impl()
  {
    if(_is_required)
      release();
  }

  bool is_required() const { return _is_required; }
  
private:
  void acquire()
  {
    auto acc = _acc;
    _hook_id = _hooks->add([acc] (sycl::handler& cgh) mutable{
      cgh.require(*acc);
    });

    _is_required = true;
  }

  bool _is_required;

  sycl::accessor<dataT, dimensions, accessMode, accessTarget,
                                  access::placeholder::true_t>* _acc;

  std::size_t _hook_id;
  detail::queue_submission_hooks_ptr _hooks;
};

}

namespace vendor {
namespace hipsycl {

template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
class automatic_placeholder_requirement
{
public:
  using impl_type = detail::automatic_placeholder_requirement_impl<
    dataT,dimensions,accessMode,accessTarget>;

  automatic_placeholder_requirement(queue &q, 
      accessor<dataT, dimensions, accessMode, accessTarget,
                access::placeholder::true_t>& acc)
  {
    _impl = std::make_unique<impl_type>(q, &acc);
  }

  automatic_placeholder_requirement(std::unique_ptr<impl_type> impl)
  : _impl{std::move(impl)}
  {}

  void reacquire()
  {
    _impl->reacquire();
  }

  void release()
  {
    _impl->release();
  }

  bool is_required() const
  {
    return _impl->is_required();
  }

private:
  std::unique_ptr<impl_type> _impl;
};

template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
inline auto automatic_require(queue &q, 
    accessor<dataT, dimensions, accessMode, accessTarget,access::placeholder::true_t>& acc)
{
  using requirement_type = automatic_placeholder_requirement<
    dataT, dimensions, accessMode, accessTarget>;

  using impl_type = typename requirement_type::impl_type;

  return requirement_type{std::make_unique<impl_type>(q, &acc)};
}

} // hipsycl
} // vendor



}// namespace sycl
}// namespace hipsycl

namespace std {
template <>
struct hash<hipsycl::sycl::queue>
{
  std::size_t operator()(const hipsycl::sycl::queue& q) const
  {
    return q.AdaptiveCpp_hash_code();
  }
};

}

#endif
