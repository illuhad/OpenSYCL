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
#ifndef HIPSYCL_DEBUG_HPP
#define HIPSYCL_DEBUG_HPP

#define HIPSYCL_DEBUG_LEVEL_NONE 0
#define HIPSYCL_DEBUG_LEVEL_ERROR 1
#define HIPSYCL_DEBUG_LEVEL_WARNING 2
#define HIPSYCL_DEBUG_LEVEL_INFO 3
#define HIPSYCL_DEBUG_LEVEL_VERBOSE 4

#ifndef HIPSYCL_DEBUG_LEVEL
#define HIPSYCL_DEBUG_LEVEL HIPSYCL_DEBUG_LEVEL_WARNING
#endif

#include <iostream>
#include <cstdlib>
#include <string>

#ifndef HIPSYCL_COMPILER_COMPONENT
#include "hipSYCL/runtime/application.hpp"
#endif

namespace hipsycl {
namespace common {

class output_stream {
public:
  static output_stream &get() {
    static output_stream ostr;
    return ostr;
  }

  std::ostream &get_stream() const { return _output_stream; }
  int get_debug_level() const { return _debug_level; }

private:

  output_stream()
  : _debug_level {HIPSYCL_DEBUG_LEVEL}, _output_stream{std::cerr} {
#if !defined(HIPSYCL_COMPILER_COMPONENT) && !defined(HIPSYCL_TOOL_COMPONENT)
    _debug_level =
        rt::application::get_settings().get<rt::setting::debug_level>();
#else
    
    auto process_env = [this](const char* e) {
      if (std::string{e}.find_first_not_of("0123456789") ==
          std::string::npos) {
        _debug_level = std::stoi(std::string{e});
      }
    };
    
    if (const char *env = std::getenv("ACPP_DEBUG_LEVEL")) {
      process_env(env);
    } else if (const char *env = std::getenv("HIPSYCL_DEBUG_LEVEL")){
      process_env(env);
    }
#endif
  }

  int _debug_level;
  std::ostream& _output_stream;
};

}
}
#ifndef HIPSYCL_COMPILER_COMPONENT
#define HIPSYCL_DEBUG_STREAM(level, prefix)                                    \
  if (level > ::hipsycl::common::output_stream::get().get_debug_level())       \
    ;                                                                          \
  else ::hipsycl::common::output_stream::get().get_stream() << prefix
#else
#define HIPSYCL_DEBUG_STREAM(level, prefix)                                    \
  if (level > ::hipsycl::common::output_stream::get().get_debug_level())       \
    ;                                                                          \
  else llvm::outs() << prefix
#endif

#ifdef HIPSYCL_DEBUG_NOCOLOR
#define HIPSYCL_DEBUG_PREFIX_ERROR   "[AdaptiveCpp Error] "
#define HIPSYCL_DEBUG_PREFIX_WARNING "[AdaptiveCpp Warning] "
#define HIPSYCL_DEBUG_PREFIX_INFO    "[AdaptiveCpp Info] "
#else
#define HIPSYCL_DEBUG_PREFIX_ERROR   "\033[1;31m[AdaptiveCpp Error] \033[0m"
#define HIPSYCL_DEBUG_PREFIX_WARNING "\033[;35m[AdaptiveCpp Warning] \033[0m"
#define HIPSYCL_DEBUG_PREFIX_INFO    "\033[;32m[AdaptiveCpp Info] \033[0m"
#endif

#define HIPSYCL_DEBUG_ERROR \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_ERROR, \
                      HIPSYCL_DEBUG_PREFIX_ERROR)


#define HIPSYCL_DEBUG_WARNING \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_WARNING, \
                      HIPSYCL_DEBUG_PREFIX_WARNING)


#define HIPSYCL_DEBUG_INFO \
  HIPSYCL_DEBUG_STREAM(HIPSYCL_DEBUG_LEVEL_INFO, \
                      HIPSYCL_DEBUG_PREFIX_INFO)

#define HIPSYCL_DEBUG_EXECUTE(level, content) \
  if(level <= ::hipsycl::common::output_stream::get().get_debug_level()) \
  {\
    content;\
  }

#define HIPSYCL_DEBUG_EXECUTE_VERBOSE(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_VERBOSE, content)
#define HIPSYCL_DEBUG_EXECUTE_INFO(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_INFO, content)
#define HIPSYCL_DEBUG_EXECUTE_WARNING(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_WARNING, content)
#define HIPSYCL_DEBUG_EXECUTE_ERROR(content) HIPSYCL_DEBUG_EXECUTE(HIPSYCL_DEBUG_LEVEL_ERROR, content)
#endif
