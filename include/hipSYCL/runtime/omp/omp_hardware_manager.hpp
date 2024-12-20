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
#ifndef HIPSYCL_OMP_HARDWARE_MANAGER_HPP
#define HIPSYCL_OMP_HARDWARE_MANAGER_HPP

#include "../hardware.hpp"

namespace hipsycl {
namespace rt {

class omp_hardware_context : public hardware_context
{
public:
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

  virtual std::size_t get_platform_index() const override;

  virtual ~omp_hardware_context() {}
};

class omp_hardware_manager : public backend_hardware_manager
{
public:
  virtual std::size_t get_num_devices() const override;
  virtual hardware_context *get_device(std::size_t index) override;
  virtual device_id get_device_id(std::size_t index) const override;
  virtual std::size_t get_num_platforms() const override;

  virtual ~omp_hardware_manager(){}
private:
  omp_hardware_context _device;
};

} // namespace rt
} // namespace hipsycl

#endif
