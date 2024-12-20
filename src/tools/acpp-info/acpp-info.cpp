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
#include <iostream>

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/executor.hpp"

using namespace hipsycl;

void list_backends(rt::runtime* rt) {
  std::vector<rt::backend*> backends;
  rt->backends().for_each_backend([&](rt::backend* b){
    backends.push_back(b);
  });

  for(std::size_t i = 0; i < backends.size(); ++i) {
    std::cout << "Loaded backend " << i << ": " << backends[i]->get_name()
              << std::endl;
    int num_devices = backends[i]->get_hardware_manager()->get_num_devices();
    if(num_devices == 0) {
      std::cout << "  (no devices found)" << std::endl;
    }

    for (int dev = 0;
         dev < num_devices; ++dev) {
      std::cout << "  Found device: "
                << backends[i]
                       ->get_hardware_manager()
                       ->get_device(dev)
                       ->get_device_name()
                << std::endl;
    }
  }
}

template<class T>
void print_info(const std::string& info_name, const T& val, int indentation = 0) {
  for(int i = 0; i < indentation; ++i)
    std::cout << " ";
  std::cout << info_name << ": " << val << std::endl;
}

template <class T>
void print_info(const std::string &info_name, const std::vector<T> &val,
                int indentation = 0) {
  for(int i = 0; i < indentation; ++i)
    std::cout << " ";
  std::cout << info_name << ": ";
  for(int i = 0; i < val.size(); ++i) {
    std::cout << val[i] << " ";
  }
  std::cout << std::endl;
}

void list_device_details(rt::device_id dev, rt::backend *b,
                         rt::hardware_context *hw) {

  std::cout << "Device " << dev.get_id() << ":" <<std::endl;
  std::cout << " General device information:" << std::endl;
  print_info("Name", hw->get_device_name(), 2);
  print_info("Backend", b->get_name(), 2);
  print_info("Platform",
             "Backend " +
                 std::to_string(static_cast<int>(b->get_unique_backend_id())) +
                 " / Platform " + std::to_string(hw->get_platform_index()),
             2);
  print_info("Vendor", hw->get_vendor_name(), 2);
  print_info("Arch", hw->get_device_arch(), 2);
  print_info("Driver version", hw->get_driver_version(), 2);
  print_info("Is CPU", hw->is_cpu(), 2);
  print_info("Is GPU", hw->is_gpu(), 2);

  std::cout << " Default executor information:" << std::endl;
  print_info("Is in-order queue", b->get_executor(dev)->is_inorder_queue(), 2);
  print_info("Is out-of-order queue", b->get_executor(dev)->is_outoforder_queue(), 2);
  print_info("Is task graph", b->get_executor(dev)->is_taskgraph(), 2);
  
  std::cout << " Device support queries:" << std::endl;
#define PRINT_DEVICE_SUPPORT_ASPECT(name) \
  print_info(#name, hw->has(rt::device_support_aspect::name), 2);
  
  PRINT_DEVICE_SUPPORT_ASPECT(images)
  PRINT_DEVICE_SUPPORT_ASPECT(error_correction)
  PRINT_DEVICE_SUPPORT_ASPECT(host_unified_memory)
  PRINT_DEVICE_SUPPORT_ASPECT(little_endian)
  PRINT_DEVICE_SUPPORT_ASPECT(global_mem_cache)
  PRINT_DEVICE_SUPPORT_ASPECT(global_mem_cache_read_only)
  PRINT_DEVICE_SUPPORT_ASPECT(global_mem_cache_read_write)
  PRINT_DEVICE_SUPPORT_ASPECT(emulated_local_memory)
  PRINT_DEVICE_SUPPORT_ASPECT(sub_group_independent_forward_progress)
  PRINT_DEVICE_SUPPORT_ASPECT(usm_device_allocations)
  PRINT_DEVICE_SUPPORT_ASPECT(usm_host_allocations)
  PRINT_DEVICE_SUPPORT_ASPECT(usm_atomic_host_allocations)
  PRINT_DEVICE_SUPPORT_ASPECT(usm_shared_allocations)
  PRINT_DEVICE_SUPPORT_ASPECT(usm_atomic_shared_allocations)
  PRINT_DEVICE_SUPPORT_ASPECT(usm_system_allocations)
  PRINT_DEVICE_SUPPORT_ASPECT(execution_timestamps)
  PRINT_DEVICE_SUPPORT_ASPECT(sscp_kernels)
  std::cout << " Device properties:" << std::endl;

#define PRINT_DEVICE_UINT_PROPERTY(name)                                       \
  print_info(#name, hw->get_property(rt::device_uint_property::name), 2);
#define PRINT_DEVICE_UINT_LIST_PROPERTY(name)                                   \
  print_info(#name, hw->get_property(rt::device_uint_list_property::name), 2);

  PRINT_DEVICE_UINT_PROPERTY(max_compute_units);
  PRINT_DEVICE_UINT_PROPERTY(max_global_size0);
  PRINT_DEVICE_UINT_PROPERTY(max_global_size1);
  PRINT_DEVICE_UINT_PROPERTY(max_global_size2);
  PRINT_DEVICE_UINT_PROPERTY(max_group_size);
  PRINT_DEVICE_UINT_PROPERTY(max_num_sub_groups);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_char);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_double);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_float);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_half);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_int);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_long);
  PRINT_DEVICE_UINT_PROPERTY(preferred_vector_width_short);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_char);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_double);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_float);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_half);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_int);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_long);
  PRINT_DEVICE_UINT_PROPERTY(native_vector_width_short);
  PRINT_DEVICE_UINT_PROPERTY(max_clock_speed);
  PRINT_DEVICE_UINT_PROPERTY(max_malloc_size);
  PRINT_DEVICE_UINT_PROPERTY(address_bits);
  PRINT_DEVICE_UINT_PROPERTY(max_read_image_args);
  PRINT_DEVICE_UINT_PROPERTY(max_write_image_args);
  PRINT_DEVICE_UINT_PROPERTY(image2d_max_width);
  PRINT_DEVICE_UINT_PROPERTY(image2d_max_height);
  PRINT_DEVICE_UINT_PROPERTY(image3d_max_width);
  PRINT_DEVICE_UINT_PROPERTY(image3d_max_height);
  PRINT_DEVICE_UINT_PROPERTY(image3d_max_depth);
  PRINT_DEVICE_UINT_PROPERTY(image_max_buffer_size);
  PRINT_DEVICE_UINT_PROPERTY(image_max_array_size);
  PRINT_DEVICE_UINT_PROPERTY(max_samplers);
  PRINT_DEVICE_UINT_PROPERTY(max_parameter_size);
  PRINT_DEVICE_UINT_PROPERTY(mem_base_addr_align);
  PRINT_DEVICE_UINT_PROPERTY(global_mem_cache_line_size);
  PRINT_DEVICE_UINT_PROPERTY(global_mem_cache_size);
  PRINT_DEVICE_UINT_PROPERTY(global_mem_size);
  PRINT_DEVICE_UINT_PROPERTY(max_constant_buffer_size);
  PRINT_DEVICE_UINT_PROPERTY(max_constant_args);
  PRINT_DEVICE_UINT_PROPERTY(local_mem_size);
  PRINT_DEVICE_UINT_PROPERTY(printf_buffer_size);
  PRINT_DEVICE_UINT_PROPERTY(partition_max_sub_devices);
  PRINT_DEVICE_UINT_PROPERTY(vendor_id);
  PRINT_DEVICE_UINT_LIST_PROPERTY(sub_group_sizes);
}

void list_devices(rt::runtime* rt) {
  rt->backends().for_each_backend([&](rt::backend* b){
    std::size_t num_devices = b->get_hardware_manager()->get_num_devices();

    std::cout << "***************** Devices for backend " << b->get_name()
              << " *****************" << std::endl;
    if(num_devices == 0)
      std::cout << "  (no devices)\n\n" << std::endl;

    for(std::size_t i = 0; i < num_devices; ++i) {
      rt::hardware_context* hw = b->get_hardware_manager()->get_device(i);
      list_device_details(b->get_hardware_manager()->get_device_id(i), b, hw);

      std::cout << "\n" << std::endl;
    }
  });
}

void print_help(const char* exe_name)
{
    std::cout << "Usage: " << exe_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "\t-h, --help              Show this message.\n";
    std::cout << "\t-l, --list-devices      Only list backends and devices, without detailed information.\n";
}

int main(int argc, char *argv[]) {
  bool print_device_details = true;
  for (int arg = 1; arg < argc; arg++) {
    const std::string current_arg{argv[arg]};
    if (current_arg == "-h" || current_arg == "--help") {
      print_help(argv[0]);
      return 0;
    }
    else if (current_arg == "-l" || current_arg == "--list-devices") {
      print_device_details = false;
    }
    else {
      std::cerr << "Unknown option: " << argv[arg] << std::endl;
      print_help(argv[0]);
      return 1;
    }
  }

  rt::runtime_keep_alive_token rt_token;
  rt::runtime* rt = rt_token.get();

  std::cout << "=================Backend information==================="
            << std::endl;
  list_backends(rt);
  if (print_device_details) {
    std::cout << std::endl;
    std::cout << "=================Device information==================="
              << std::endl;
    list_devices(rt);
  }
}
