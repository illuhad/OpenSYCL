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
#ifndef HIPSYCL_UR_HARDWARE_MANAGER_HPP
#define HIPSYCL_UR_HARDWARE_MANAGER_HPP

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/ur/ur_allocator.hpp"

#include <memory>
#include <ur_api.h>
#include <vector>

namespace hipsycl::rt {

/// A wrapper around an ur_context_handle_t that manages the lifetime of the
/// context and can be shared among multiple ur_hardware_context instances.
class ur_context_manager {
public:
  explicit ur_context_manager(ur_context_handle_t ctx);
  ~ur_context_manager();

  // No copying or moving

  ur_context_manager(const ur_context_manager &) = delete;
  ur_context_manager(ur_context_manager &&) = delete;
  ur_context_manager &operator=(const ur_context_manager &) = delete;
  ur_context_manager &operator=(ur_context_manager &&) = delete;

  [[nodiscard]] ur_context_handle_t get() const { return _context; }

private:
  ur_context_handle_t _context;
};

/// Represents a Unified Runtime device handle.
class ur_hardware_context final : public hardware_context {
public:
  ur_hardware_context(ur_device_handle_t device,
                      const std::shared_ptr<ur_context_manager> &ctx,
                      size_t platform_idx);
  ~ur_hardware_context() override;

  [[nodiscard]] bool is_cpu() const override;
  [[nodiscard]] bool is_gpu() const override;

  [[nodiscard]] std::size_t get_max_kernel_concurrency() const override;
  [[nodiscard]] std::size_t get_max_memcpy_concurrency() const override;

  [[nodiscard]] std::string get_device_name() const override;
  [[nodiscard]] std::string get_vendor_name() const override;
  [[nodiscard]] std::string get_device_arch() const override;

  [[nodiscard]] bool has(device_support_aspect aspect) const override;
  [[nodiscard]] std::size_t
  get_property(device_uint_property prop) const override;
  [[nodiscard]] std::vector<std::size_t>
  get_property(device_uint_list_property prop) const override;

  [[nodiscard]] std::string get_driver_version() const override;
  [[nodiscard]] std::string get_profile() const override;
  [[nodiscard]] std::size_t get_platform_index() const override;

  [[nodiscard]] ur_context_handle_t get_ur_context() const;
  [[nodiscard]] ur_device_handle_t get_ur_device() const;

  [[nodiscard]] ur_allocator *get_allocator();

private:
  ur_allocator _allocator;
  std::shared_ptr<ur_context_manager> _context;

  size_t _platform_idx;
  ur_device_handle_t _device;
};

class ur_hardware_manager final : public backend_hardware_manager {
public:
  ur_hardware_manager();
  ~ur_hardware_manager() override;

  [[nodiscard]] std::size_t get_num_devices() const override;
  [[nodiscard]] hardware_context *get_device(std::size_t index) override;
  [[nodiscard]] device_id get_device_id(std::size_t index) const override;
  [[nodiscard]] std::size_t get_num_platforms() const override;

  result device_handle_to_device_id(ur_device_handle_t d, device_id &out) const;

private:
  std::vector<ur_adapter_handle_t> _adapters{};
  std::vector<ur_platform_handle_t> _platforms{};

  std::vector<ur_hardware_context> _contexts{};
};

} // namespace hipsycl::rt

#endif
