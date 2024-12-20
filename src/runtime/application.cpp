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
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/async_errors.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/hw_model/hw_model.hpp"
#include "hipSYCL/runtime/settings.hpp"
#include <memory>
#include <mutex>
#include <atomic>

namespace hipsycl {
namespace rt {

runtime_keep_alive_token::runtime_keep_alive_token()
: _rt{application::get_runtime_pointer()} {
  assert(_rt);
}

runtime* runtime_keep_alive_token::get() const {
  return _rt.get();
}

std::shared_ptr<runtime> application::get_runtime_pointer() {
  static std::mutex mutex;
  static std::weak_ptr<runtime> rt;

  std::lock_guard<std::mutex> lock{mutex};

  std::shared_ptr<runtime> rt_ptr = rt.lock();
  if(!rt_ptr) {
    rt_ptr = std::make_shared<runtime>();
    rt = rt_ptr;
  }
  assert(rt_ptr);
  return rt_ptr;
}

settings &application::get_settings() {
  static settings s;
  return s;
}

async_error_list& application::errors() {
  static async_error_list errors;
  return errors;
}

runtime_event_handlers& application::event_handler_layer() {
  static runtime_event_handlers h;
  return h;
}


}
}

