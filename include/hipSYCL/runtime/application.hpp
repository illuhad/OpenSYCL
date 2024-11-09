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
#ifndef HIPSYCL_APPLICATION_HPP
#define HIPSYCL_APPLICATION_HPP

#include <memory>

#include "backend.hpp"
#include "device_id.hpp"
#include "settings.hpp"
#include "runtime_event_handlers.hpp"

namespace hipsycl {
namespace rt {

class dag_manager;
class runtime;
class async_error_list;


class application
{
public:
  static settings& get_settings();
  // Should only be invoked from the SYCL interface, not
  // from the runtime or kernel launchers.
  static std::shared_ptr<runtime> get_runtime_pointer();
  static async_error_list& errors();
  static runtime_event_handlers& event_handler_layer();

  application() = delete;
};

class persistent_runtime {
public:
  persistent_runtime() : _rt{nullptr} {
    if(application::get_settings().get<setting::persistent_runtime>()) {
      _rt = application::get_runtime_pointer();
    }
  }
private:
  std::shared_ptr<runtime> _rt;
};

class runtime_keep_alive_token {
public:
  runtime_keep_alive_token();

  runtime* get() const;
private:
  std::shared_ptr<runtime> _rt;
};

}
}


#endif
