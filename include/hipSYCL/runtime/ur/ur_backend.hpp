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
#ifndef HIPSYCL_UR_BACKEND_HPP
#define HIPSYCL_UR_BACKEND_HPP

#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include "hipSYCL/runtime/ur/ur_hardware_manager.hpp"

#include <memory>

namespace hipsycl::rt {

class ur_backend final : public backend {
public:
  ur_backend();
  ~ur_backend() override;

  [[nodiscard]] api_platform get_api_platform() const override;
  [[nodiscard]] hardware_platform get_hardware_platform() const override;
  [[nodiscard]] backend_id get_unique_backend_id() const override;

  [[nodiscard]] backend_executor *get_executor(device_id dev) const override;
  [[nodiscard]] ur_hardware_manager *get_hardware_manager() const override;
  [[nodiscard]] ur_allocator *get_allocator(device_id dev) const override;

  [[nodiscard]] std::string get_name() const override;

  std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) override;

private:
  std::unique_ptr<ur_hardware_manager> _hw_manager;
  std::unique_ptr<lazily_constructed_executor<multi_queue_executor>> _executor;
};

} // namespace hipsycl::rt

#endif
