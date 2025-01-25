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
#include "hipSYCL/runtime/ur/ur_hardware_manager.hpp"

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/settings.hpp"
#include "ur_utils.hpp"

#include <cstddef>

using namespace hipsycl::rt;

ur_hardware_manager::ur_hardware_manager() {
  {
    const auto err = urLoaderInit(0, nullptr);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(
          __acpp_here(),
          ur_error_info("ur_hardware_manager: Could not initialize UR loader",
                        err));
      return;
    }
  }

  uint32_t num_adapters = 0;
  {
    const auto err = urAdapterGet(0, nullptr, &num_adapters);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(__acpp_here(),
                    ur_error_info("Could not get number of adapters", err));
      return;
    }
  }

  HIPSYCL_DEBUG_INFO << "ur_hardware_manager: Found " << num_adapters
                     << " adapters" << std::endl;
  _adapters = std::vector<ur_adapter_handle_t>(num_adapters);

  {
    const auto err =
        urAdapterGet(num_adapters, _adapters.data(), &num_adapters);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(__acpp_here(),
                    ur_error_info("Could not list adapters", err));
      return;
    }
  }

  uint32_t num_platforms = 0;
  {
    const auto err = urPlatformGet(_adapters.data(), num_adapters, 1, nullptr,
                                   &num_platforms);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(__acpp_here(),
                    ur_error_info("Could not get number of platforms", err));
      return;
    }
  }

  HIPSYCL_DEBUG_INFO << "ur_hardware_manager: Found " << num_platforms
                     << " platforms" << std::endl;
  _platforms = std::vector<ur_platform_handle_t>(num_platforms);

  {
    const auto err = urPlatformGet(_adapters.data(), num_adapters,
                                   num_platforms, _platforms.data(), nullptr);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(__acpp_here(),
                    ur_error_info("Could not list platforms", err));
      return;
    }
  }

  // for each platform, try to create a context with all devices
  for (std::size_t platform_idx = 0; platform_idx < _platforms.size();
       ++platform_idx) {

    const auto platform = _platforms[platform_idx];

    uint32_t num_devices = 0;
    {
      const auto err =
          urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
      if (err != UR_RESULT_SUCCESS) {
        print_warning(
            __acpp_here(),
            ur_error_info("Could not get number of devices of platform", err));
        continue;
      }
    }

    auto _devices = std::vector<ur_device_handle_t>(num_devices);

    {
      const auto err = urDeviceGet(platform, UR_DEVICE_TYPE_ALL, num_devices,
                                   _devices.data(), nullptr);

      if (err != UR_RESULT_SUCCESS) {
        print_warning(__acpp_here(),
                      ur_error_info("Could not list devices of platform", err));
        continue;
      }
    }

    _devices.resize(num_devices);

    // create a context for all devices
    ur_context_handle_t platform_ctx;

    {
      const auto err =
          urContextCreate(num_devices, _devices.data(), nullptr, &platform_ctx);
      if (err != UR_RESULT_SUCCESS) {
        print_warning(
            __acpp_here(),
            ur_error_info(
                "ur_hardware_manager: Could not create context for platform",
                err));
        continue; // TODO: fallback to single device contexts
      }
    }

    // for each device in the platform, create a hardware context
    const auto ctx_mgr = std::make_shared<ur_context_manager>(platform_ctx);
    for (const auto device : _devices) {
      _contexts.emplace_back(device, ctx_mgr, platform_idx);
    }
  }
}

ur_hardware_manager::~ur_hardware_manager() {
  for (const auto adapter : _adapters) {
    urAdapterRelease(adapter);
  }
}

std::size_t ur_hardware_manager::get_num_devices() const {
  return _contexts.size();
}
std::size_t ur_hardware_manager::get_num_platforms() const {
  return _platforms.size();
}
result ur_hardware_manager::device_handle_to_device_id(ur_device_handle_t d,
                                                       device_id &out) const {
  for (std::size_t i = 0; i < _contexts.size(); ++i) {
    if (_contexts[i].get_ur_device() == d) {
      out = get_device_id(i);
      return make_success();
    }
  }
  return make_error(__acpp_here(),
                    error_info{"ze_hardware_manager: Could not convert "
                               "ze_device_handle_t to hipSYCL device id"});
}

hardware_context *ur_hardware_manager::get_device(const std::size_t index) {
  assert(index < _contexts.size());
  return &_contexts[index];
}

device_id ur_hardware_manager::get_device_id(const std::size_t index) const {
  assert(index < _contexts.size());
  return device_id{backend_descriptor{hardware_platform::unified_runtime,
                                      api_platform::unified_runtime},
                   static_cast<int>(index)};
}

ur_context_handle_t ur_hardware_context::get_ur_context() const {
  return _context->get();
}
ur_device_handle_t ur_hardware_context::get_ur_device() const {
  return _device;
}
ur_allocator *ur_hardware_context::get_allocator() { return &_allocator; }

ur_hardware_context::ur_hardware_context(
    const ur_device_handle_t device,
    const std::shared_ptr<ur_context_manager> &ctx,
    const std::size_t platform_idx)
    : _allocator{ur_allocator{device, ctx->get(), platform_idx}}, _context(ctx),
      _platform_idx(platform_idx), _device(device) {}

ur_hardware_context::~ur_hardware_context() = default;

template <typename T>
static T get_device_property(const ur_device_handle_t device,
                             const ur_device_info_t prop) {
  T out;

  const ur_result_t err =
      urDeviceGetInfo(device, prop, sizeof(out), &out, nullptr);
  if (err != UR_RESULT_SUCCESS) {
    print_warning(__acpp_here(),
                  ur_error_info("Could not query device property", err));
    out = T{};
  }

  return out;
}

static std::string get_device_property_string(const ur_device_handle_t device,
                                              const ur_device_info_t prop) {

  size_t out_size = 0;

  {
    const ur_result_t err =
        urDeviceGetInfo(device, prop, 0, nullptr, &out_size);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(__acpp_here(),
                    ur_error_info("Could not query device property", err));
      return "";
    }
  }

  auto out = std::vector<char>(out_size);
  {
    const ur_result_t err =
        urDeviceGetInfo(device, prop, out_size, out.data(), nullptr);
    if (err != UR_RESULT_SUCCESS) {
      print_warning(__acpp_here(),
                    ur_error_info("Could not query device property", err));
      return "";
    }
  }

  return std::string{out.data()};
}

bool ur_hardware_context::is_cpu() const {
  const auto type =
      get_device_property<ur_device_type_t>(_device, UR_DEVICE_INFO_TYPE);
  return type == UR_DEVICE_TYPE_CPU;
}

bool ur_hardware_context::is_gpu() const {
  const auto type =
      get_device_property<ur_device_type_t>(_device, UR_DEVICE_INFO_TYPE);
  return type == UR_DEVICE_TYPE_GPU;
}
std::size_t ur_hardware_context::get_max_kernel_concurrency() const {
  return 1; // TODO
}
std::size_t ur_hardware_context::get_max_memcpy_concurrency() const {
  return 1; // TODO
}

std::string ur_hardware_context::get_device_name() const {
  return get_device_property_string(_device, UR_DEVICE_INFO_NAME);
}

std::string ur_hardware_context::get_vendor_name() const {
  return get_device_property_string(_device, UR_DEVICE_INFO_VENDOR);
}

std::string ur_hardware_context::get_device_arch() const {
  return "unknown"; // TODO
}
bool ur_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {

  case device_support_aspect::images:
    return get_device_property<ur_bool_t>(_device,
                                          UR_DEVICE_INFO_IMAGE_SUPPORTED);
  case device_support_aspect::error_correction:
    return get_device_property<ur_bool_t>(
        _device, UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT);
  case device_support_aspect::host_unified_memory:
    return get_device_property<ur_bool_t>(_device,
                                          UR_DEVICE_INFO_HOST_UNIFIED_MEMORY);
  case device_support_aspect::little_endian:
    return get_device_property<ur_bool_t>(_device,
                                          UR_DEVICE_INFO_ENDIAN_LITTLE);
  case device_support_aspect::global_mem_cache:
    return get_device_property<uint64_t>(
               _device, UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE) > 0;
  case device_support_aspect::usm_device_allocations:
    return get_device_property<ur_device_usm_access_capability_flags_t>(
               _device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT) &
           UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;
  case device_support_aspect::usm_host_allocations:
    return get_device_property<ur_device_usm_access_capability_flags_t>(
               _device, UR_DEVICE_INFO_USM_HOST_SUPPORT) &
           UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;
  case device_support_aspect::usm_shared_allocations:
    // or maybe UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT
    return get_device_property<ur_device_usm_access_capability_flags_t>(
               _device, UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT) &
           UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;

  case device_support_aspect::usm_system_allocations:
    return get_device_property<ur_device_usm_access_capability_flags_t>(
               _device, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT) &
           UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS;

  case device_support_aspect::global_mem_cache_read_only:
  case device_support_aspect::global_mem_cache_read_write:
  case device_support_aspect::emulated_local_memory:
  case device_support_aspect::sub_group_independent_forward_progress:
  case device_support_aspect::usm_atomic_host_allocations:
  case device_support_aspect::usm_atomic_shared_allocations:
  case device_support_aspect::execution_timestamps:
  case device_support_aspect::sscp_kernels:
  case device_support_aspect::work_item_independent_forward_progress:
    return false; // TODO
  }

  assert(false && "unreachable");
  return false;
}
std::size_t ur_hardware_context::get_property(device_uint_property prop) const {
  throw std::runtime_error("ur_hardware_context: get_property not implemented");
}

std::vector<std::size_t>
ur_hardware_context::get_property(device_uint_list_property prop) const {
  throw std::runtime_error("ur_hardware_context: get_property not implemented");
}

std::string ur_hardware_context::get_driver_version() const {
  return get_device_property<char *>(_device, UR_DEVICE_INFO_DRIVER_VERSION);
}
std::string ur_hardware_context::get_profile() const {
  return get_device_property<char *>(_device, UR_DEVICE_INFO_PROFILE);
}
std::size_t ur_hardware_context::get_platform_index() const {
  return _platform_idx;
}

ur_context_manager::ur_context_manager(const ur_context_handle_t ctx)
    : _context(ctx) {}

ur_context_manager::~ur_context_manager() {
  const auto err = urContextRelease(_context);
  if (err != UR_RESULT_SUCCESS) {
    print_warning(
        __acpp_here(),
        ur_error_info("ur_context_manager: Could not release context", err));
  }
}
