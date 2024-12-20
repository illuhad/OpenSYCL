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
#include "hipSYCL/runtime/cuda/cuda_hardware_manager.hpp"
#include "hipSYCL/runtime/cuda/cuda_event_pool.hpp"
#include "hipSYCL/runtime/cuda/cuda_allocator.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/settings.hpp"


#include <cuda_runtime_api.h>
#include <exception>
#include <cstdlib>
#include <limits>

namespace hipsycl {
namespace rt {

cuda_hardware_manager::cuda_hardware_manager(hardware_platform hw_platform)
    : _hw_platform(hw_platform) {

  if (has_device_visibility_mask(
          application::get_settings().get<setting::visibility_mask>(),
          backend_id::cuda)) {
    print_warning(
        __acpp_here(),
        error_info{
            "cuda_hardware_manager: CUDA backend does not support device "
            "visibility masks. Use CUDA_VISIBILE_DEVICES instead."});
  }

  int num_devices = 0;

  auto err = cudaGetDeviceCount(&num_devices);
  if (err != cudaSuccess) {
    num_devices = 0;

    if(err != cudaErrorNoDevice) {
      print_warning(
          __acpp_here(),
          error_info{"cuda_hardware_manager: Could not obtain number of devices",
                    error_code{"CUDA", err}});
    }
  }
  
  for (int dev = 0; dev < num_devices; ++dev) {
    _devices.emplace_back(dev);
  }

}


std::size_t cuda_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context *cuda_hardware_manager::get_device(std::size_t index) {
  if (index >= _devices.size()){
    register_error(__acpp_here(),
                   error_info{"cuda_hardware_manager: Attempt to access invalid "
                              "device detected."});
    return nullptr;
  }

  return &(_devices[index]);
}

device_id cuda_hardware_manager::get_device_id(std::size_t index) const {
  if (index >= _devices.size()){
    register_error(__acpp_here(),
                   error_info{"cuda_hardware_manager: Attempt to access invalid "
                              "device detected."});
  }

  return device_id{backend_descriptor{_hw_platform, api_platform::cuda},
                   static_cast<int>(index)};
}

std::size_t cuda_hardware_manager::get_num_platforms() const {
  return 1;
}

std::size_t cuda_hardware_context::get_platform_index() const {
  return 0;
}

cuda_hardware_context::cuda_hardware_context(int dev) 
  : _dev{dev} {
  _properties = std::make_unique<cudaDeviceProp>();
  auto err = cudaGetDeviceProperties(_properties.get(), dev);

  if (err != cudaSuccess) {
    register_error(
        __acpp_here(),
        error_info{"cuda_hardware_manager: Could not query device properties ",
                   error_code{"CUDA", err}});
  }

  _allocator = std::make_unique<cuda_allocator>(
      backend_descriptor{hardware_platform::cuda, api_platform::cuda}, _dev);
  _event_pool = std::make_unique<cuda_event_pool>(_dev);
}

cuda_allocator* cuda_hardware_context::get_allocator() const {
  return _allocator.get();
}

cuda_event_pool* cuda_hardware_context::get_event_pool() const {
  return _event_pool.get();
}

bool cuda_hardware_context::is_cpu() const {
  return !is_gpu();
}

bool cuda_hardware_context::is_gpu() const {
  return true;
}

std::size_t cuda_hardware_context::get_max_kernel_concurrency() const {
  return _properties->concurrentKernels + 1;
}

std::size_t cuda_hardware_context::get_max_memcpy_concurrency() const {
  // TODO: Modern CUDA as asyncEngineCount
  return get_max_kernel_concurrency();
}

std::string cuda_hardware_context::get_device_name() const {
  return _properties->name;
}

std::string cuda_hardware_context::get_vendor_name() const {
  return "NVIDIA";
}

std::string cuda_hardware_context::get_device_arch() const {
  return "sm_" + std::to_string(_properties->major) +
         std::to_string(_properties->minor);
}

bool cuda_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return false;
    break;
  case device_support_aspect::host_unified_memory:
    return false;
    break;
  case device_support_aspect::error_correction:
    return false; // TODO
    break;
  case device_support_aspect::global_mem_cache:
    return true;
    break;
  case device_support_aspect::global_mem_cache_read_only:
    return false;
    break;
  case device_support_aspect::global_mem_cache_read_write:
    // NVIDIA GPUs have read/write cache at least since Fermi architecture
    return true;
    break;
  case device_support_aspect::images:
    return false;
    break;
  case device_support_aspect::little_endian:
    return true;
    break;
  case device_support_aspect::sub_group_independent_forward_progress:
    return true;
    break;
  case device_support_aspect::usm_device_allocations:
    return true;
    break;
  case device_support_aspect::usm_host_allocations:
    return true;
    break;
  case device_support_aspect::usm_atomic_host_allocations:
    // TODO actually query this
    return false;
    break;
  case device_support_aspect::usm_shared_allocations:
    return true;
    break;
  case device_support_aspect::usm_atomic_shared_allocations:
    // TODO actually query this
    return false;
    break;
  case device_support_aspect::usm_system_allocations:
    return false;
    break;
  case device_support_aspect::execution_timestamps:
    return true;
    break;
  case device_support_aspect::sscp_kernels:
#ifdef HIPSYCL_WITH_SSCP_COMPILER
    return true;
#else
    return false;
#endif
    break;
  case device_support_aspect::work_item_independent_forward_progress:
    return _properties->major >= 7;
    break;
  }
  assert(false && "Unknown device aspect");
  std::terminate();
}

std::size_t
cuda_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return _properties->multiProcessorCount;
    break;
  case device_uint_property::max_global_size0:
    return static_cast<std::size_t>(_properties->maxThreadsDim[0]) *
                                    _properties->maxGridSize[0];
    break;
  case device_uint_property::max_global_size1:
    return static_cast<std::size_t>(_properties->maxThreadsDim[1]) *
                                    _properties->maxGridSize[1];
    break;
  case device_uint_property::max_global_size2:
    return static_cast<std::size_t>(_properties->maxThreadsDim[2]) *
                                    _properties->maxGridSize[2];
    break;
  case device_uint_property::max_group_size0:
    return _properties->maxThreadsDim[0];
    break;
  case device_uint_property::max_group_size1:
    return _properties->maxThreadsDim[1];
    break;
  case device_uint_property::max_group_size2:
    return _properties->maxThreadsDim[2];
    break;
  case device_uint_property::max_group_size:
    return _properties->maxThreadsPerBlock;
    break;
  case device_uint_property::max_num_sub_groups:
    return _properties->maxThreadsPerBlock / _properties->warpSize;
    break;
  case device_uint_property::needs_dimension_flip:
    return true;
    break;
  case device_uint_property::preferred_vector_width_char:
    return 4;
    break;
  case device_uint_property::preferred_vector_width_double:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_float:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_half:
    return 2;
    break;
  case device_uint_property::preferred_vector_width_int:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_long:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_short:
    return 2;
    break;
  case device_uint_property::native_vector_width_char:
    return 4;
    break;
  case device_uint_property::native_vector_width_double:
    return 1;
    break;
  case device_uint_property::native_vector_width_float:
    return 1;
    break;
  case device_uint_property::native_vector_width_half:
    return 2;
    break;
  case device_uint_property::native_vector_width_int:
    return 1;
    break;
  case device_uint_property::native_vector_width_long:
    return 1;
    break;
  case device_uint_property::native_vector_width_short:
    return 2;
    break;
  case device_uint_property::max_clock_speed:
    return _properties->clockRate / 1000;
    break;
  case device_uint_property::max_malloc_size:
    return _properties->totalGlobalMem;
    break;
  case device_uint_property::address_bits:
    return 64;
    break;
  case device_uint_property::max_read_image_args:
    return 0;
    break;
  case device_uint_property::max_write_image_args:
    return 0;
    break;
  case device_uint_property::image2d_max_width:
    return 0;
    break;
  case device_uint_property::image2d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_width:
    return 0;
    break;
  case device_uint_property::image3d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_depth:
    return 0;
    break;
  case device_uint_property::image_max_buffer_size:
    return 0;
    break;
  case device_uint_property::image_max_array_size:
    return 0;
    break;
  case device_uint_property::max_samplers:
    return 0;
    break;
  case device_uint_property::max_parameter_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::mem_base_addr_align:
    return 8; // TODO
    break;
  case device_uint_property::global_mem_cache_line_size:
    return 128; //TODO
    break;
  case device_uint_property::global_mem_cache_size:
    return _properties->l2CacheSize; // TODO
    break;
  case device_uint_property::global_mem_size:
    return _properties->totalGlobalMem;
    break;
  case device_uint_property::max_constant_buffer_size:
    return _properties->totalConstMem;
    break;
  case device_uint_property::max_constant_args:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::local_mem_size:
    return _properties->sharedMemPerBlock;
    break;
  case device_uint_property::printf_buffer_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::partition_max_sub_devices:
    return 0;
    break;
  case device_uint_property::vendor_id:
    return 4318;
    break;
  case device_uint_property::architecture:
    return _properties->major * 10 + _properties->minor;
    break;
  case device_uint_property::backend_id:
    return static_cast<int>(backend_id::cuda);
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::vector<std::size_t>
cuda_hardware_context::get_property(device_uint_list_property prop) const {
  switch (prop) {
  case device_uint_list_property::sub_group_sizes:
    return std::vector<std::size_t>{
        static_cast<std::size_t>(_properties->warpSize)};
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::string cuda_hardware_context::get_driver_version() const {
  int driver_version = 0;

  auto err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    register_error(
        __acpp_here(),
        error_info{"cuda_hardware_manager: Querying driver version failed",
                   error_code{"CUDA", err}});
  }
  
  return std::to_string(driver_version);
}

std::string cuda_hardware_context::get_profile() const {
  return "FULL_PROFILE";
}

cuda_hardware_context::~cuda_hardware_context(){}

unsigned cuda_hardware_context::get_compute_capability() const {
  return _properties->major * 10 + _properties->minor;
}

}
}
