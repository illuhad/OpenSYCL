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
#ifndef HIPSYCL_PLATFORM_HPP
#define HIPSYCL_PLATFORM_HPP

#include <vector>

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/device_id.hpp"

#include "types.hpp"
#include "device_selector.hpp"
#include "info/info.hpp"
#include "version.hpp"

namespace hipsycl {
namespace sycl {

class device_selector;

class platform {

public:
  platform() : _platform{detail::get_host_device().get_backend(), 0} {}
  
  platform(rt::backend_id backend)
      : _platform{backend, 0} {}

  template<class DeviceSelector>
  explicit platform(const DeviceSelector &deviceSelector) {
    auto dev = detail::select_devices(deviceSelector)[0];
    this->_platform = rt::platform_id{dev._device_id};
  }


  std::vector<device>
  get_devices(info::device_type type = info::device_type::all) const {
    std::vector<device> result;
    rt::backend *b =
        _requires_runtime.get()->backends().get(_platform.get_backend());

    int num_devices = b->get_hardware_manager()->get_num_devices();
    for (int dev = 0; dev < num_devices; ++dev) {
      bool is_cpu = b->get_hardware_manager()->get_device(dev)->is_cpu();
      bool is_gpu = b->get_hardware_manager()->get_device(dev)->is_gpu();

      bool include_device = false;
      if (type == info::device_type::all ||
          (type == info::device_type::accelerator && is_gpu) ||
          (type == info::device_type::gpu && is_gpu) ||
          (type == info::device_type::host && is_cpu) ||
          (type == info::device_type::cpu && is_cpu)) {
        include_device = true;
      }

      if (include_device)
        result.push_back(device{rt::device_id{b->get_backend_descriptor(), dev}});
    }
  
    return result;
  }


  template <typename Param>
  typename Param::return_type get_info() const;


  /// \todo Think of a better solution
  bool has_extension(const string_class &extension) const {
    return false;
  }


  bool is_host() const {
    return _requires_runtime.get()->backends().get(_platform.get_backend())
               ->get_backend_descriptor()
               .hw_platform == rt::hardware_platform::cpu;
  }

  /// Returns true if all devices in this platform have the
  /// specified aspect
  bool has(aspect asp) const {
    auto devs = get_devices();
    for(const device& d : devs) {
      if(!d.has(asp))
        return false;
    }
    return true;
  }

  static std::vector<platform> get_platforms() {
    std::vector<platform> result;
    rt::runtime_keep_alive_token requires_runtime;

    requires_runtime.get()->backends().for_each_backend([&](rt::backend *b) {
      result.push_back(platform{b->get_unique_backend_id()});
    });

    return result;
  }

  friend bool operator==(const platform &lhs, const platform &rhs) {
    return lhs._platform == rhs._platform;
  }

  friend bool operator!=(const platform &lhs, const platform &rhs) {
    return !(lhs == rhs);
  }

  std::size_t AdaptiveCpp_hash_code() const {
    return std::hash<rt::platform_id>{}(_platform);
  }


  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

  
  context khr_get_default_context() const;
private:
  rt::platform_id _platform;
  rt::runtime_keep_alive_token _requires_runtime;
};


HIPSYCL_SPECIALIZE_GET_INFO(device, platform)
{ return this->get_platform(); }

HIPSYCL_SPECIALIZE_GET_INFO(platform, profile)
{ return "FULL_PROFILE"; }

HIPSYCL_SPECIALIZE_GET_INFO(platform, version)
{
  return detail::version_string();
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, name)
{
  rt::backend_id b = _platform.get_backend();
  return _requires_runtime.get()->backends().get(b)->get_name();
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, vendor)
{
  return "The AdaptiveCpp project";
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, extensions)
{
  return vector_class<string_class>{};
}

inline platform device::get_platform() const  {
  return platform{_device_id.get_backend()};
}

}// namespace sycl
}// namespace hipsycl

namespace std {

template <>
struct hash<hipsycl::sycl::platform>
{
  std::size_t operator()(const hipsycl::sycl::platform& p) const
  {
    return p.AdaptiveCpp_hash_code();
  }
};

}

#endif
