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
#ifndef HIPSYCL_OCL_HARDWARE_MANAGER_HPP
#define HIPSYCL_OCL_HARDWARE_MANAGER_HPP

#include <vector>
#include <memory>

#include <CL/opencl.hpp>

#include "../hardware.hpp"
#include "ocl_allocator.hpp"
#include "ocl_usm.hpp"

namespace hipsycl {
namespace rt {

class ocl_allocator;
class ocl_hardware_manager;

class ocl_hardware_context : public hardware_context
{
public:
  ocl_hardware_context() = default;
  ocl_hardware_context(const cl::Device &dev, const cl::Context &ctx,
                       int dev_id, int platform_id);
  ocl_hardware_context(const ocl_hardware_context&) = default;

  virtual bool is_cpu() const override;
  virtual bool is_gpu() const override;

  /// \return The maximum number of kernels that can be executed concurrently
  virtual std::size_t get_max_kernel_concurrency() const override;
  /// \return The maximum number of memory transfers that can be executed
  /// concurrently
  virtual std::size_t get_max_memcpy_concurrency() const override;

  virtual std::string get_device_name() const override;
  virtual std::string get_vendor_name() const override;
  virtual std::string get_device_arch() const override;

  virtual bool has(device_support_aspect aspect) const override;
  virtual std::size_t get_property(device_uint_property prop) const override;
  virtual std::vector<std::size_t>
  get_property(device_uint_list_property prop) const override;
  
  virtual std::string get_driver_version() const override;
  virtual std::string get_profile() const override;

  virtual ~ocl_hardware_context();

  virtual std::size_t get_platform_index() const override;

  ocl_allocator* get_allocator();
  ocl_usm* get_usm_provider();

  int get_platform_id() const;
  int get_device_id() const;
  cl::Device get_cl_device() const;
  cl::Context get_cl_context() const;

  bool has_intel_extension_profile() const;

  void init_allocator(ocl_hardware_manager* mgr);
private:
  int _dev_id;
  int _platform_id;
  cl::Context _ctx;
  cl::Device _dev;
  std::shared_ptr<ocl_usm> _usm_provider;
  ocl_allocator _alloc;
  bool _has_intel_extension_profile;
};

class ocl_hardware_manager : public backend_hardware_manager
{
public:
  ocl_hardware_manager();

  virtual std::size_t get_num_devices() const override;
  virtual hardware_context *get_device(std::size_t index) override;
  virtual device_id get_device_id(std::size_t index) const override;
  virtual std::size_t get_num_platforms() const override;

  virtual ~ocl_hardware_manager() {}
  
  cl::Platform get_platform(int platform_id);
  cl::Context get_context(device_id dev);
private:
  std::vector<ocl_hardware_context> _devices;
  std::vector<cl::Platform> _platforms;
  
  hardware_platform _hw_platform;
};

}
}

#endif
