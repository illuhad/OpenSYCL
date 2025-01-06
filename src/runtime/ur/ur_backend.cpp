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
#include "hipSYCL/runtime/ur/ur_backend.hpp"

#include "hipSYCL/runtime/ur/ur_queue.hpp"
#include "ur_utils.hpp"

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::ur_backend();
}

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() { return "ur"; }

using namespace hipsycl::rt;

namespace {
std::unique_ptr<multi_queue_executor>
create_multi_queue_executor(ur_backend *backend, ur_hardware_manager *hw_mgr) {

  auto queue_factory = [hw_mgr](const device_id dev) {
    return std::make_unique<ur_queue>(hw_mgr,
                                      static_cast<std::size_t>(dev.get_id()));
  };

  return std::make_unique<multi_queue_executor>(*backend, queue_factory);
}

bool is_device_id_valid(const device_id dev) {
  return dev.get_backend() == backend_id::unified_runtime;
}

} // namespace

ur_backend::ur_backend() {
  _hw_manager = std::make_unique<ur_hardware_manager>();
  _executor =
      std::make_unique<lazily_constructed_executor<multi_queue_executor>>(
          [this] {
            return create_multi_queue_executor(this, _hw_manager.get());
          });
}
ur_backend::~ur_backend() = default;

std::string ur_backend::get_name() const { return "Unified Runtime"; }

api_platform ur_backend::get_api_platform() const {
  return api_platform::unified_runtime;
}

backend_id ur_backend::get_unique_backend_id() const {
  return backend_id::unified_runtime;
}

hardware_platform ur_backend::get_hardware_platform() const {
  return hardware_platform::unified_runtime;
}

ur_hardware_manager *ur_backend::get_hardware_manager() const {
  return _hw_manager.get();
}

backend_executor *ur_backend::get_executor(const device_id dev) const {
  if (!is_device_id_valid(dev)) {
    register_error(
        __acpp_here(),
        error_info{"passed device_id does not belong to this backend"});
    return nullptr;
  }

  return _executor->get();
}

std::unique_ptr<backend_executor>
ur_backend::create_inorder_executor(const device_id dev, int priority) {
  if (!is_device_id_valid(dev)) {
    register_error(
        __acpp_here(),
        error_info{"passed device_id does not belong to this backend"});
    return nullptr;
  }

  std::unique_ptr<inorder_queue> q =
      std::make_unique<ur_queue>(_hw_manager.get(), dev.get_id());

  return std::make_unique<inorder_executor>(std::move(q));
}

ur_allocator *ur_backend::get_allocator(const device_id dev) const {
  if (!is_device_id_valid(dev)) {
    register_error(
        __acpp_here(),
        error_info{"passed device_id does not belong to this backend"});
    return nullptr;
  }

  const auto dev_ctx = dynamic_cast<ur_hardware_context *>(
      _hw_manager->get_device(dev.get_id()));
  return dev_ctx->get_allocator();
}
