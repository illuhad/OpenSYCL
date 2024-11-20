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
#ifndef HIPSYCL_DEVICE_ID_HPP
#define HIPSYCL_DEVICE_ID_HPP

#include <functional>
#include <cassert>
#include <ostream>
#include <cstdint>

namespace hipsycl {
namespace rt {

enum class hardware_platform
{
  rocm,
  cuda,
  level_zero,
  ocl,
  cpu
};

enum class api_platform {
  cuda,
  hip,
  level_zero,
  ocl,
  omp
};

enum class backend_id {
  cuda,
  hip,
  level_zero,
  ocl,
  omp
};

struct backend_descriptor
{
  backend_id id;
  hardware_platform hw_platform;
  api_platform sw_platform;

  backend_descriptor() = default;
  backend_descriptor(hardware_platform hw_plat, api_platform sw_plat)
      : hw_platform{hw_plat}, sw_platform{sw_plat} {

    if (hw_plat == hardware_platform::cpu &&
        sw_plat == api_platform::omp)
      id = backend_id::omp;
    else if (sw_plat == api_platform::hip)
      id = backend_id::hip;
    else if (hw_plat == hardware_platform::cuda &&
             sw_plat == api_platform::cuda)
      id = backend_id::cuda;
    else if(hw_plat == hardware_platform::level_zero &&
            sw_plat == api_platform::level_zero)
      id = backend_id::level_zero;
    else if (hw_plat == hardware_platform::ocl && sw_plat == api_platform::ocl)
      id = backend_id::ocl;
    else
      assert(false && "Invalid combination of hardware/software platform for "
                      "backend descriptor.");
  }

  friend bool operator==(const backend_descriptor &a,
                         const backend_descriptor &b) {
    return a.id == b.id;
  }
};

class device_id
{
public:
  device_id() = default;
  device_id(backend_descriptor b, int id);
  
  bool is_host() const;
  backend_id get_backend() const;
  backend_descriptor get_full_backend_descriptor() const;
  
  int get_id() const;

  void dump(std::ostream& ostr) const;

  friend bool operator==(const device_id& a, const device_id& b)
  {
    return a._backend == b._backend && 
           a._device_id == b._device_id;
  }

  friend bool operator!=(const device_id& a, const device_id& b)
  {
    return !(a == b);
  }

  uint64_t hash_code() const {
    uint32_t backend = static_cast<uint32_t>(_backend.id);
    uint32_t id = _device_id;
    return (static_cast<uint64_t>(backend) << 32) | id;
  }
private:
  backend_descriptor _backend;
  int _device_id;
};

class platform_id {
public:
  platform_id(backend_id b, int platform)
      : _backend{b}, _platform_id{platform} {}

  platform_id() = default;

  platform_id(device_id dev) : _backend{dev.get_backend()}, _platform_id{0} {}

  backend_id get_backend() const { return _backend; }

  int get_platform() const { return _platform_id; }

  friend bool operator==(const platform_id &a, const platform_id &b) {
    return a._backend == b._backend && a._platform_id == b._platform_id;
  }

  friend bool operator!=(const platform_id &a, const platform_id &b) {
    return !(a == b);
  }
private:
  backend_id _backend;
  int _platform_id;
};

}
}


namespace std {

template <>
struct hash<hipsycl::rt::device_id>
{
  std::size_t operator()(const hipsycl::rt::device_id& k) const
  {
    return hash<int>()(static_cast<int>(k.get_backend())) ^ 
          (hash<int>()(k.get_id()) << 8);
  }
};

template <>
struct hash<hipsycl::rt::platform_id>
{
  std::size_t operator()(const hipsycl::rt::platform_id& p) const
  {
    return hash<int>()(static_cast<int>(p.get_backend())) ^ 
          (hash<int>()(p.get_platform()) << 8);
  }
};

}

#endif
