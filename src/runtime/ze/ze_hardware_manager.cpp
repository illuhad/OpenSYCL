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
#include <cassert>
#include <cstdint>
#include <string>
#include <limits>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {


ze_context_manager::ze_context_manager(ze_driver_handle_t driver)
 : _driver{driver} {

  ze_context_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  desc.pNext = nullptr;
  desc.flags = 0;
  HIPSYCL_DEBUG_INFO << "ze_context_manager: Spawning context..." << std::endl;

  ze_context_handle_t handle;
  ze_result_t err = zeContextCreate(driver, &desc, &handle);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ze_context_manager: Could not create context",
                              error_code{"ze", static_cast<int>(err)}});
  }

  auto deleter = [](ze_context_handle_t *ptr) {
    if (ptr) {
      HIPSYCL_DEBUG_INFO << "ze_context_manager: Destroying context..."
                         << std::endl;
      ze_result_t err = zeContextDestroy(*ptr);
      
      if (err != ZE_RESULT_SUCCESS) {
        register_error(
            __acpp_here(),
            error_info{"ze_context_manager: Could not destroy context",
                       error_code{"ze", static_cast<int>(err)}});
      }

      delete ptr;
    }
  };

  _handle = std::shared_ptr<ze_context_handle_t>{
      new ze_context_handle_t{handle}, deleter};
}

ze_context_manager::~ze_context_manager() {}

ze_context_handle_t ze_context_manager::get() const {
  if(!_handle)
    return nullptr;
  
  return *(_handle.get());
}

ze_driver_handle_t ze_context_manager::get_driver() const {
  return _driver;
}

ze_event_pool_manager::ze_event_pool_manager(
    ze_context_handle_t ctx, const std::vector<ze_device_handle_t> &devices,
    std::size_t pool_size)
    : _devices{devices}, _ctx{ctx}, _pool_size{pool_size}, _num_used_events{0} {

  this->spawn_pool();
}

void ze_event_pool_manager::spawn_pool(){

  HIPSYCL_DEBUG_INFO << "ze_event_pool_manager: Spawning event pool..."
                     << std::endl;

  ze_event_pool_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  desc.pNext = nullptr;
  desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  desc.count = _pool_size;

  ze_event_pool_handle_t pool;

  // Nullptr means that event pool is visible to all devices
  ze_device_handle_t* devs = _devices.empty() ? nullptr : _devices.data();
  uint32_t num_devices = _devices.size();

  ze_result_t err = zeEventPoolCreate(_ctx, &desc, num_devices, devs, &pool);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
                  error_info{"ze_event_pool_manager: Could not construct event pool",
                             error_code{"ze", static_cast<int>(err)}});
  } else {
    // Assign new pool
    auto deleter = [](ze_event_pool_handle_t* ptr) {
        if(ptr) {
          ze_result_t err = zeEventPoolDestroy(*ptr);
    
          if(err != ZE_RESULT_SUCCESS) {
            register_error(__acpp_here(),
                          error_info{"ze_event_pool_manager: Could not destroy event pool",
                                    error_code{"ze", static_cast<int>(err)}});
          }

          delete ptr;
        }
      };

    _pool = std::shared_ptr<ze_event_pool_handle_t>{
      new ze_event_pool_handle_t{pool}, deleter 
    };

    // Reset number of used events
    _num_used_events = 0;
  }
}

ze_event_pool_manager::~ze_event_pool_manager() {}

std::shared_ptr<ze_event_pool_handle_t> ze_event_pool_manager::get_pool() const {
  return _pool;
}

ze_context_handle_t ze_event_pool_manager::get_ze_context() const {
  return _ctx;
}

std::shared_ptr<ze_event_pool_handle_t> 
ze_event_pool_manager::allocate_event(uint32_t& event_ordinal) {
  if(_num_used_events+1 >= _pool_size) {
    // If pool is full, spawn a new pool
    spawn_pool();
    event_ordinal = 0;

  } else {
    uint32_t ordinal = _num_used_events;
    ++_num_used_events;

    event_ordinal = ordinal;
  }
  return _pool;
}
  

ze_hardware_context::ze_hardware_context(ze_driver_handle_t driver,
                                         ze_device_handle_t device,
                                         ze_context_handle_t ctx)
    : _driver{driver}, _device{device}, _ctx{ctx} {

  ze_result_t err = zeDeviceGetProperties(_device, &_props);

  if(err != ZE_RESULT_SUCCESS) {
    print_error(__acpp_here(),
                  error_info{"ze_hardware_context: Could not query device properties",
                             error_code{"ze", static_cast<int>(err)}});
  }

  err = zeDeviceGetComputeProperties(_device, &_compute_props);

  if(err != ZE_RESULT_SUCCESS) {
    print_error(__acpp_here(),
                  error_info{"ze_hardware_context: Could not query device compute properties",
                             error_code{"ze", static_cast<int>(err)}});
  }

  uint32_t num_memory_properties = 0;
  err = zeDeviceGetMemoryProperties(_device, &num_memory_properties, nullptr);

  if(err != ZE_RESULT_SUCCESS) {
    print_error(__acpp_here(),
                  error_info{"ze_hardware_context: Could not query number of memory properties",
                             error_code{"ze", static_cast<int>(err)}});
  }
  if(num_memory_properties > 0) {
    _memory_props.resize(num_memory_properties);

    err = zeDeviceGetMemoryProperties(_device, &num_memory_properties, _memory_props.data());

    if(err != ZE_RESULT_SUCCESS) {
      print_error(__acpp_here(),
                  error_info{"ze_hardware_context: Could not query memory properties",
                             error_code{"ze", static_cast<int>(err)}});
    }
  }
}

bool ze_hardware_context::is_cpu() const {
  return _props.type == ZE_DEVICE_TYPE_CPU;
}

bool ze_hardware_context::is_gpu() const {
  return _props.type == ZE_DEVICE_TYPE_GPU;
}

std::size_t ze_hardware_context::get_max_kernel_concurrency() const {
  // TODO
  return 1;
}
  
std::size_t ze_hardware_context::get_max_memcpy_concurrency() const {
  // TODO
  return 1;
}

std::string ze_hardware_context::get_device_name() const {
  return std::string{_props.name};
}

std::string ze_hardware_context::get_vendor_name() const {
  return std::string{"pci:"}+std::to_string(_props.vendorId);
}

std::string ze_hardware_context::get_device_arch() const {
  return "spirv";
}

bool ze_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return false;
    break;
  case device_support_aspect::host_unified_memory:
    return _props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;
    break;
  case device_support_aspect::error_correction:
    return _props.flags & ZE_DEVICE_PROPERTY_FLAG_ECC;
    break;
  case device_support_aspect::global_mem_cache:
    return true;
    break;
  case device_support_aspect::global_mem_cache_read_only:
    return false;
    break;
  case device_support_aspect::global_mem_cache_read_write:
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
    return false;
    break;
  case device_support_aspect::sscp_kernels:
#ifdef HIPSYCL_WITH_SSCP_COMPILER
    return true;
#else
    return false;
#endif
    break;
  case device_support_aspect::work_item_independent_forward_progress:
    return false;
    break;
  }
  assert(false && "Unknown device aspect");
  std::terminate();
}

std::size_t ze_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return _props.numSlices * _props.numSubslicesPerSlice * _props.numEUsPerSubslice;
    break;
  case device_uint_property::max_global_size0:
    return _compute_props.maxGroupSizeX * _compute_props.maxGroupCountX;
    break;
  case device_uint_property::max_global_size1:
    return _compute_props.maxGroupSizeY * _compute_props.maxGroupCountY;
    break;
  case device_uint_property::max_global_size2:
    return _compute_props.maxGroupSizeZ * _compute_props.maxGroupCountZ;
    break;
  case device_uint_property::max_group_size0:
    return _compute_props.maxGroupSizeX;
    break;
  case device_uint_property::max_group_size1:
    return _compute_props.maxGroupSizeY;
    break;
  case device_uint_property::max_group_size2:
    return _compute_props.maxGroupSizeZ;
    break;
  case device_uint_property::max_group_size:
    return _compute_props.maxTotalGroupSize;
    break;
  case device_uint_property::max_num_sub_groups:
    {
      std::size_t min_subgroup_size = std::numeric_limits<std::size_t>::max();
      for(int i = 0; i < _compute_props.numSubGroupSizes; ++i){
        if(_compute_props.subGroupSizes[i] < min_subgroup_size)
          min_subgroup_size = _compute_props.subGroupSizes[i];
      }
      return _compute_props.maxTotalGroupSize / min_subgroup_size;
    }
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
    return _props.coreClockRate / 1000;
    break;
  case device_uint_property::max_malloc_size:
    return _props.maxMemAllocSize;
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
    return 128; // TODO
    break;
  case device_uint_property::global_mem_size:
    return _props.maxMemAllocSize; // TODO Is this correct?
    break;
  case device_uint_property::max_constant_buffer_size:
    return 0; // TODO
    break;
  case device_uint_property::max_constant_args:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::local_mem_size:
    return _compute_props.maxSharedLocalMemory;
    break;
  case device_uint_property::printf_buffer_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::partition_max_sub_devices:
    return 0;
    break;
  case device_uint_property::vendor_id:
    return _props.vendorId;
    break;
  case device_uint_property::architecture:
    // TODO
    return 0;
    break;
  case device_uint_property::backend_id:
    return static_cast<int>(backend_id::level_zero);
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::vector<std::size_t>
ze_hardware_context::get_property(device_uint_list_property prop) const {
  switch(prop) {
  case device_uint_list_property::sub_group_sizes:
    {
      std::vector<std::size_t> result(_compute_props.numSubGroupSizes);
      for(int i = 0; i < _compute_props.numSubGroupSizes; ++i)
        result[i] = _compute_props.subGroupSizes[i];
      return result;
    }
    break;
  }

  assert(false && "Invalid device property");
  std::terminate();
}
  
std::string ze_hardware_context::get_driver_version() const {
  ze_driver_properties_t props;
  ze_result_t err = zeDriverGetProperties(_driver, &props);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
                  error_info{"ze_hardware_context: Could not query driver properties",
                             error_code{"ze", static_cast<int>(err)}});
    return "<unknown>";
  } else {
    return std::to_string(props.driverVersion);
  }
}

std::string ze_hardware_context::get_profile() const {
  return "FULL_PROFILE";
}

ze_hardware_context::~ze_hardware_context(){}

uint32_t ze_hardware_context::get_ze_global_memory_ordinal() const {
  uint32_t result = 0;

  std::size_t max_found_mem = 0;
  for(std::size_t i = 0; i < _memory_props.size(); ++i) {
    if(_memory_props[i].totalSize > max_found_mem) {
      max_found_mem = _memory_props[i].totalSize;
      result = i;
    }
  }

  return result;
}

ze_hardware_manager::ze_hardware_manager() {

  if (has_device_visibility_mask(
          application::get_settings().get<setting::visibility_mask>(),
          backend_id::level_zero)) {
    print_warning(
        __acpp_here(),
        error_info{
            "ze_hardware_manager: Level Zero backend does not support device "
            "visibility masks. Use ZE_AFFINITY_MASK instead."});
  }

  uint32_t num_drivers = 0;
  ze_result_t err = zeDriverGet(&num_drivers, nullptr);

  if (err != ZE_RESULT_SUCCESS) {
    print_warning(__acpp_here(),
                  error_info{"ze_hardware_manager: Could not get number of drivers, "
                             "assuming no drivers available.",
                             error_code{"ze", static_cast<int>(err)}});
  }
  
  if(num_drivers > 0) {
    std::vector<ze_driver_handle_t> drivers(num_drivers);
    err = zeDriverGet(&num_drivers, drivers.data());

    if (err != ZE_RESULT_SUCCESS) {
      print_error(__acpp_here(),
                    error_info{"ze_hardware_manager: Could not obtain driver handles",
                              error_code{"ze", static_cast<int>(err)}});
      num_drivers = 0;
    }

    for(int i = 0; i < num_drivers; ++i) {
      
      _drivers.push_back(drivers[i]);
      _contexts.push_back(ze_context_manager{drivers[i]});

      uint32_t num_devices = 0;
      err = zeDeviceGet(drivers[i], &num_devices, nullptr);

      if (err != ZE_RESULT_SUCCESS) {
        print_error(__acpp_here(),
                    error_info{"ze_hardware_manager: Could not obtain number of devices",
                              error_code{"ze", static_cast<int>(err)}});
        num_devices = 0;
      }

      
      std::vector<ze_device_handle_t> devices;
      if(num_devices > 0) {
        devices.resize(num_devices);
        
        err = zeDeviceGet(drivers[i], &num_devices, devices.data());

        for(int dev = 0; dev < num_devices; ++dev) {
          _devices.push_back(ze_hardware_context{drivers[i], devices[dev],
                                                 _contexts.back().get()});
        }

      }

      _event_pools.push_back(
          ze_event_pool_manager{_contexts.back().get(), devices});
    }
  }
}


std::size_t ze_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context *ze_hardware_manager::get_device(std::size_t index) {
  assert(index < _devices.size());
  return &(_devices[index]);
}

device_id ze_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{backend_descriptor{
    hardware_platform::level_zero, api_platform::level_zero}, 
    static_cast<int>(index)
  };
}

ze_context_handle_t ze_hardware_manager::get_ze_context(std::size_t device_index) const {
  assert(device_index < _devices.size());

  return _devices[device_index].get_ze_context();
}

result ze_hardware_manager::device_handle_to_device_id(ze_device_handle_t d, device_id &out) const {

  for(std::size_t i = 0; i < _devices.size(); ++i) {
    if(_devices[i].get_ze_device() == d) {
      out = get_device_id(i);
      return make_success();
    }
  }
  return make_error(__acpp_here(),
                    error_info{"ze_hardware_manager: Could not convert "
                               "ze_device_handle_t to hipSYCL device id"});
}

ze_event_pool_manager* ze_hardware_manager::get_event_pool_manager(std::size_t device_index) {
  assert(device_index < _devices.size());

  ze_context_handle_t ctx = _devices[device_index].get_ze_context();

  for(auto& pool : _event_pools) {
    if(pool.get_ze_context() == ctx) {
      return &pool;
    }
  }
  return nullptr;
}

}
}
