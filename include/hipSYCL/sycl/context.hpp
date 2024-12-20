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
#ifndef HIPSYCL_CONTEXT_HPP
#define HIPSYCL_CONTEXT_HPP

#include <cassert>
#include <memory>
#include <vector>

#include "types.hpp"
#include "platform.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "info/info.hpp"
#include "exception.hpp"

#include "hipSYCL/runtime/device_list.hpp"
#include "hipSYCL/glue/error.hpp"

namespace hipsycl {
namespace sycl {

class context;

namespace detail {
const rt::unique_device_list& extract_context_devices(const context&);

struct default_context_tag_t{};
}

class context
{
public:
  friend class queue;

  friend const rt::unique_device_list &
  detail::extract_context_devices(const context &);

  explicit context(async_handler handler = [](exception_list e) {
    glue::default_async_handler(e);
  }) : context{detail::select_devices(default_selector_v), handler} {}

  explicit context(
      const device &dev, async_handler handler = [](exception_list e) {
        glue::default_async_handler(e);
      }) {
    this->init(handler, dev);
  }

  explicit context(
      const platform &plt, async_handler handler = [](exception_list e) {
        glue::default_async_handler(e);
      }) {
    this->init(handler);
    std::vector<device> devices = plt.get_devices();
    for (const auto &dev : devices) {
      _impl->devices.add(dev._device_id);
    }
    // Always need to add the host device
    _impl->devices.add(detail::get_host_device());
  }

  explicit context(
      const std::vector<device> &deviceList,
      async_handler handler = [](exception_list e) {
        glue::default_async_handler(e);
      }) {
    
    if(deviceList.empty())
      throw exception{make_error_code(errc::platform),
                      "context: Cannot construct context for empty device list"};

    this->init(handler);
    for(const device& dev : deviceList) {
      _impl->devices.add(dev._device_id);
    }
    // Always need to add the host device
    _impl->devices.add(detail::get_host_device());
  }

  explicit context(
      detail::default_context_tag_t,
      async_handler handler =
          [](exception_list e) { glue::default_async_handler(e); })
      : context{handler} {
    _impl->is_default_context = true;
  }

  explicit context(
      detail::default_context_tag_t, const device &dev,
      async_handler handler =
          [](exception_list e) { glue::default_async_handler(e); })
      : context{dev, handler} {
    _impl->is_default_context = true;
  }

  explicit context(
      detail::default_context_tag_t, const platform &plt,
      async_handler handler =
          [](exception_list e) { glue::default_async_handler(e); })
      : context{plt, handler} {
    _impl->is_default_context = true;
  }

  explicit context(
      detail::default_context_tag_t, const std::vector<device> &deviceList,
      async_handler handler =
          [](exception_list e) { glue::default_async_handler(e); })
      : context{deviceList, handler} {
    _impl->is_default_context = true;
  }

  bool is_host() const {
    bool has_non_host_devices = false;
    _impl->devices.for_each_device([&](rt::device_id d) {
      if (!d.is_host())
        has_non_host_devices = true;
    });
    return !has_non_host_devices;
  }

  platform get_platform() const {
    bool found_device_platform = false;
    rt::platform_id last_platform;

    this->_impl->devices.for_each_backend([&](rt::backend_id b) {
      rt::backend* backend = this->_impl->requires_runtime.get()->backends().get(b);

      for (std::size_t platform_index = 0;
           platform_index < backend->get_hardware_manager()->get_num_platforms();
           ++platform_index) {
        if (b != detail::get_host_device().get_backend()) {
          if (found_device_platform) {
            // We already have a device backend
            HIPSYCL_DEBUG_WARNING
                << "context: get_platform() was called but this context spans "
                  "multiple backends/platforms. Only returning last platform"
                << std::endl;
          }
          
          last_platform = rt::platform_id{b, platform_index};
          found_device_platform = true;
        }
      }
    });

    if (!found_device_platform) {
      last_platform = rt::platform_id{detail::get_host_device().get_backend(), 0}; 
    }

    return platform{last_platform};
  }

  vector_class<device> get_devices() const {
    std::vector<device> devs;
    _impl->devices.for_each_device([&](rt::device_id d) {
      devs.push_back(d);
    });
    return devs;
  }


  template <typename Param>
  typename Param::return_type get_info() const {
    throw exception{make_error_code(errc::runtime),
                    "context::get_info() is unimplemented"};
  }

  std::size_t AdaptiveCpp_hash_code() const {
    if(_impl && _impl->is_default_context) {
      std::size_t hash = 0;
      _impl->devices.for_each_device([&](rt::device_id dev){
        // xor ensures that device order does not matter
        hash ^= dev.hash_code();
      });
      return hash;
    }
    return std::hash<void*>{}(_impl.get());
  }

  friend bool operator ==(const context& lhs, const context& rhs) {

    if (lhs._impl && rhs._impl && lhs._impl->is_default_context &&
        rhs._impl->is_default_context) {
      return lhs._impl->devices == rhs._impl->devices;
    }

    return lhs._impl == rhs._impl;
  }

  friend bool operator!=(const context& lhs, const context &rhs)
  { return !(lhs == rhs); }

  rt::runtime* AdaptiveCpp_runtime() const {
    return _impl->requires_runtime.get();
  }

  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

  [[deprecated("Use AdaptiveCpp_runtime()")]]
  auto hipSYCL_runtime() const {
    return AdaptiveCpp_runtime();
  }
private:
  void init(async_handler handler) {
    _impl = std::make_shared<context_impl>();
    _impl->handler = handler;
  }

  void init(async_handler handler, const device &d) {
    init(handler);
    _impl->devices.add(d._device_id);
    if(!d.is_host()) {
      // Always need to add the host device
      _impl->devices.add(detail::get_host_device());
    }
  }
  
  struct context_impl {
    rt::runtime_keep_alive_token requires_runtime;
    rt::unique_device_list devices;

    context_impl() : devices{requires_runtime.get()} {}

    async_handler handler;
    bool is_default_context = false;
  };

  std::shared_ptr<context_impl> _impl;
};


HIPSYCL_SPECIALIZE_GET_INFO(context, reference_count)
{ return _impl.use_count(); }

HIPSYCL_SPECIALIZE_GET_INFO(context, platform)
{ return get_platform(); }

HIPSYCL_SPECIALIZE_GET_INFO(context, devices)
{ return get_devices(); }

namespace detail {

inline const rt::unique_device_list &extract_context_devices(const context &ctx) {
  return ctx._impl->devices;
}

}

inline context platform::khr_get_default_context() const {
  return context{detail::default_context_tag_t{}, *this};
}

inline exception::exception(context ctx, std::error_code ec, const std::string& what_arg)
  : _context{std::make_shared<context>(ctx)}, error_code{ec},
    _msg{what_arg} {}

inline exception::exception(context ctx, std::error_code ec, const char* what_arg)
    : _context{std::make_shared<context>(ctx)}, error_code{ec},
      _msg{what_arg} {}

inline exception::exception(context ctx, std::error_code ec)
  : _context{std::make_shared<context>(ctx)}, error_code{ec} {}

inline exception::exception(context ctx, int ev, const std::error_category& ecat,
                     const std::string& what_arg)
  : _context{std::make_shared<context>(ctx)}, error_code{ev, ecat},
    _msg{what_arg} {}

inline exception::exception(context ctx, int ev,
                            const std::error_category &ecat,
                            const char *what_arg)
  : _context{std::make_shared<context>(ctx)},
    error_code{ev, ecat}, _msg{what_arg} {}

inline exception::exception(context ctx, int ev,
                            const std::error_category &ecat)
  : _context{std::make_shared<context>(ctx)},
    error_code{ev, ecat} {}

inline context exception::get_context() const {
  if (!has_context())
    throw exception{make_error_code(errc::invalid)};

  return *_context;
}

} // namespace sycl
} // namespace hipsycl

namespace std {

template <>
struct hash<hipsycl::sycl::context>
{
  std::size_t operator()(const hipsycl::sycl::context& c) const
  {
    return c.AdaptiveCpp_hash_code();
  }
};

}

#endif
