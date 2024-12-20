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
#include <level_zero/ze_api.h>


#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include "hipSYCL/runtime/ze/ze_backend.hpp"
#include "hipSYCL/runtime/ze/ze_allocator.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/runtime/backend_loader.hpp"
#include "hipSYCL/runtime/device_id.hpp"

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::ze_backend();
}

static const char *backend_name = "ze";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}

namespace hipsycl {
namespace rt {


namespace {

std::unique_ptr<multi_queue_executor>
create_multi_queue_executor(ze_backend *b, ze_hardware_manager *mgr) {
  return std::make_unique<multi_queue_executor>(*b, [b, mgr](device_id dev) {
    return std::make_unique<ze_queue>(mgr,
                                      static_cast<std::size_t>(dev.get_id()));
  });
}
}


ze_backend::ze_backend() {
  ze_result_t err = zeInit(0);

  if (err != ZE_RESULT_SUCCESS) {
    print_warning(__acpp_here(),
                  error_info{"ze_backend: Call to zeInit() failed",
                            error_code{"ze", static_cast<int>(err)}});
  }

  _hardware_manager = std::make_unique<ze_hardware_manager>();
  for(std::size_t i = 0; i < _hardware_manager->get_num_devices(); ++i) {
    _allocators.push_back(ze_allocator{i,
        static_cast<ze_hardware_context *>(_hardware_manager->get_device(i)),
        _hardware_manager.get()});
  }

  _executor =
      std::make_unique<lazily_constructed_executor<multi_queue_executor>>(
          [this, hw_mgr = _hardware_manager.get()]() {
            return create_multi_queue_executor(this, hw_mgr);
          });
}

api_platform ze_backend::get_api_platform() const {
  return api_platform::level_zero;
}

hardware_platform ze_backend::get_hardware_platform() const {
  return hardware_platform::level_zero;
}

backend_id ze_backend::get_unique_backend_id() const {
  return backend_id::level_zero;
}
  
backend_hardware_manager* ze_backend::get_hardware_manager() const {
  return _hardware_manager.get();
}

backend_executor* ze_backend::get_executor(device_id dev) const {
  return _executor->get();
}

backend_allocator *ze_backend::get_allocator(device_id dev) const {
  assert(dev.get_id() < _allocators.size());

  return &(_allocators[dev.get_id()]);
}

std::string ze_backend::get_name() const {
  return "Level Zero";
}

std::unique_ptr<backend_executor>
ze_backend::create_inorder_executor(device_id dev, int priority){
  std::unique_ptr<inorder_queue> q =
      std::make_unique<ze_queue>(_hardware_manager.get(), dev.get_id());

  return std::make_unique<inorder_executor>(std::move(q));
}

}
}
