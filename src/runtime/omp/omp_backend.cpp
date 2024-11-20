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
#include "hipSYCL/runtime/backend_loader.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/omp/omp_backend.hpp"
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include <memory>


HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::omp_backend();
}

static const char *backend_name = "omp";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}



namespace hipsycl {
namespace rt {

namespace {

std::unique_ptr<inorder_queue> make_omp_queue(omp_backend* be, device_id dev) {
  return std::make_unique<omp_queue>(be, dev.get_id());
}

std::unique_ptr<multi_queue_executor>
create_multi_queue_executor(omp_backend *b) {
  return std::make_unique<multi_queue_executor>(*b, [b](device_id dev) {
    return make_omp_queue(b, dev);
  });
}

}

omp_backend::omp_backend()
    : _allocator{device_id{
          backend_descriptor{omp_backend::get_hardware_platform(), omp_backend::get_api_platform()}, 0}},
      _hw{},
      _executor([this](){
        return create_multi_queue_executor(this);
      }) {}

api_platform omp_backend::get_api_platform() const {
  return api_platform::omp;
}

hardware_platform omp_backend::get_hardware_platform() const {
  return hardware_platform::cpu;
}

backend_id omp_backend::get_unique_backend_id() const {
  return backend_id::omp;
}
  
backend_hardware_manager* omp_backend::get_hardware_manager() const {
  return &_hw;
}

backend_executor* omp_backend::get_executor(device_id dev) const {
  if(dev.get_backend() != this->get_unique_backend_id()) {
    register_error(__acpp_here(),
                   error_info{"omp_backend: Device id from other backend requested",
                              error_type::invalid_parameter_error});
    return nullptr;
  }

  return _executor.get();
}

backend_allocator* omp_backend::get_allocator(device_id dev) const {
  if(dev.get_backend() != this->get_unique_backend_id()) {
    register_error(__acpp_here(),
                   error_info{"omp_backend: Device id from other backend requested",
                              error_type::invalid_parameter_error});
    return nullptr;
  }
  return &_allocator;
}

std::string omp_backend::get_name() const {
  return "OpenMP";
}

std::unique_ptr<backend_executor>
omp_backend::create_inorder_executor(device_id dev, int priority){
  return nullptr;
}

}
}
