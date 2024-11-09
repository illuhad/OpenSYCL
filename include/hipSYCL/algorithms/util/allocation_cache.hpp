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
#ifndef HIPSYCL_ALGORITHM_UTIL_ALLOCATION_CACHE_HPP
#define HIPSYCL_ALGORITHM_UTIL_ALLOCATION_CACHE_HPP

#include <vector>
#include <mutex>

#include "hipSYCL/common/small_vector.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/sycl/device.hpp"

namespace hipsycl::rt {
  class runtime;
}

namespace hipsycl::algorithms::util {


struct allocation {
  void* ptr;
  std::size_t size;
  rt::device_id dev;
};

class allocation_group;

enum class allocation_type {
  device, shared, host
};

class allocation_cache {
  friend class allocation_group;
public:
  allocation_cache(allocation_type alloc_type)
  : _alloc_type{alloc_type} {}

  ~allocation_cache() {
    purge();
  }

  void purge() {
    std::lock_guard<std::mutex> lock{_mutex};
    
    for(auto& allocation : _allocations) {
      auto* allocator = _rt.get()->backends()
          .get(allocation.dev.get_backend())
          ->get_allocator(allocation.dev);
      rt::deallocate(allocator, allocation.ptr);
    }
    _allocations.clear();
  }
private:
  
  allocation find_or_alloc(std::size_t min_size, std::size_t min_alignment,
                           rt::device_id dev) {
    allocation result;
    if(!find_allocation(min_size, min_alignment, dev, result)){  
      result.dev = dev;
      result.size = min_size;

      auto allocator = _rt.get()->backends()
                       .get(dev.get_backend())
                       ->get_allocator(dev);

      if(_alloc_type == allocation_type::device)
        result.ptr = rt::allocate_device(allocator, min_alignment, min_size);
      else if(_alloc_type == allocation_type::shared)
        result.ptr = rt::allocate_shared(allocator, min_size);
      else
        result.ptr =
            rt::allocate_host(allocator, min_alignment, min_size);
    }
    return result;
  }

  bool find_allocation(std::size_t min_size, std::size_t min_alignment,
                       rt::device_id dev, allocation &out) {
    std::lock_guard<std::mutex> lock{_mutex};

    bool found = false;
    std::size_t found_index = 0;
    for (std::size_t i = 0; i < _allocations.size(); ++i) {
      const auto& allocation = _allocations[i];
      if (allocation.dev == dev) {
        if (allocation.size >= min_size &&
            reinterpret_cast<std::size_t>(allocation.ptr) % min_alignment ==
                0) {
          // If we already have found a candidate: We want the smallest
          // allocation that has the required size so that larger allocations
          // remain available for larger requests.
          if (!found || allocation.size < out.size) {
            out = allocation;
            found_index = i;
          }
          found = true;
        }
      }
    }
    if(found) {
      // The allocation is no longer available for other requests,
      // so remove for now.
      _allocations.erase(_allocations.begin() + found_index);
    }
    return found;
  }

  void return_allocation(const allocation& alloc) {
    std::lock_guard<std::mutex> lock{_mutex};
    _allocations.push_back(alloc);
  }

  rt::runtime_keep_alive_token _rt;
  common::auto_small_vector<allocation> _allocations;
  std::mutex _mutex;
  allocation_type _alloc_type;
};

/// allocation_group represents allocation requests that belong together
/// semantically and share lifetime - for example, all scratch allocations
/// required by one reduction invocation.
///
/// allocation_group is not thread-safe.
///
/// When the object releases its managed allocations, they are returned to the
/// parent cache from where they might be reused to serve other allocation
/// requests.
/// The user should therefore ensure that the lifetime of this object is either
/// at least as long as the operations using its allocations,
/// or all operations are ordered such that there is no hazard of race conditions, e.g.
/// if only in-order queues are involved and one allocation_cache exists per in-order queue.
class allocation_group {
public:
  allocation_group(allocation_cache *parent_cache, rt::device_id dev)
      : _parent{parent_cache}, _dev{dev} {}

  allocation_group(allocation_cache *parent_cache, const sycl::device &dev)
      : _parent{parent_cache},
        _dev{dev.AdaptiveCpp_device_id()} {}

  allocation_group() = default;
  allocation_group(const allocation_group&) = delete;
  allocation_group& operator=(const allocation_group&) = delete;

  ~allocation_group() {
    release();
  }

  void release() {
    for(const auto& allocation : _managed_allocations) {
      _parent->return_allocation(allocation);
    }
    _managed_allocations.clear();
  }

  template<class T>
  T* obtain(std::size_t count) {
    allocation alloc =
        _parent->find_or_alloc(count * sizeof(T), alignof(T), _dev);
    _managed_allocations.push_back(alloc);
    return static_cast<T*>(alloc.ptr);
  }

  rt::device_id get_device() const {
    return _dev;
  }
private:
  allocation_cache* _parent;
  rt::device_id _dev;
  common::auto_small_vector<allocation> _managed_allocations;
};


}

#endif
