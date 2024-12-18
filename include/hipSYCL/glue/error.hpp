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
#ifndef HIPSYCL_GLUE_ERROR_HPP
#define HIPSYCL_GLUE_ERROR_HPP

#include <exception>
#include <vector>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/small_vector.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/async_errors.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/types.hpp"

namespace hipsycl {
namespace glue {

inline void print_async_errors(sycl::exception_list error_list) {
  if (error_list.size() > 0) {
    std::ostream& output_stream = common::output_stream::get().get_stream();
    output_stream << "============== AdaptiveCpp error report ============== "
                  << std::endl;

    output_stream
        << "AdaptiveCpp has caught the following unhandled asynchronous errors: "
        << std::endl << std::endl;

    int idx = 0;
    for(std::exception_ptr err : error_list) {
      
      try{
        if(err) {
          std::rethrow_exception(err);
        }
      }
      catch(sycl::exception &e) {
        output_stream << "   " <<  idx << ". " << e.what() << std::endl;
      }
      catch(std::exception &e) {
        output_stream << "   " <<  idx << ". " << e.what() << std::endl;
      }
      catch(...) {
        output_stream << "   " << idx << ". <unknown exception>" << std::endl;
      }

      ++idx;
    }
  }
}

inline void default_async_handler(sycl::exception_list error_list) {
  if (error_list.size() > 0) {
    print_async_errors(error_list);
    std::ostream &output_stream = common::output_stream::get().get_stream();
    output_stream << "The application will now be terminated." << std::endl;
    std::terminate();
  }
}

inline std::exception_ptr throw_result(const rt::result& r){
  if(!r.is_success()) {
    rt::error_type etype = r.info().get_error_type();

    using sycl::exception, sycl::make_error_code, sycl::errc;

    try {
      switch (etype) {
        // TODO: error_type::unimplemented has no equivalent errc
      case rt::error_type::unimplemented:
      case rt::error_type::runtime_error:
        throw exception{make_error_code(errc::runtime), r.what()};
        break;
      case rt::error_type::kernel_error:
        throw exception{make_error_code(errc::kernel), r.what()};
        break;
      case rt::error_type::accessor_error:
        throw exception{make_error_code(errc::accessor), r.what()};
        break;
      case rt::error_type::nd_range_error:
        throw exception{make_error_code(errc::nd_range), r.what()};
        break;
      case rt::error_type::event_error:
        throw exception{make_error_code(errc::event), r.what()};
        break;
      case rt::error_type::invalid_parameter_error:
        throw exception{make_error_code(errc::invalid), r.what()};
        break;
      case rt::error_type::device_error:
        /* TODO: error_type::device_error has no equivalent errc, however
           this isn't used anywhere, maybe remove in rt::error_type? */
        throw exception{make_error_code(errc::runtime), r.what()};
        break;
        /* TODO: Neither error_type::compile_program_error nor
           error_type::link_program_error have an equivalent errc,
           however they are both not used anywhere, maybe remove in
           rt::error_type? */
      case rt::error_type::compile_program_error:
      case rt::error_type::link_program_error:
        throw exception{make_error_code(errc::build), r.what()};
        break;
      case rt::error_type::invalid_object_error:
        throw exception{make_error_code(errc::invalid), r.what()};
        break;
      case rt::error_type::memory_allocation_error:
        throw exception{make_error_code(errc::memory_allocation), r.what()};
        break;
      case rt::error_type::platform_error:
        throw exception{make_error_code(errc::platform), r.what()};
        break;
      case rt::error_type::profiling_error:
        throw exception{make_error_code(errc::profiling), r.what()};
        break;
      case rt::error_type::feature_not_supported:
        throw exception{make_error_code(errc::feature_not_supported), r.what()};
        break;
      default:
        HIPSYCL_DEBUG_WARNING
            << "throw_result(): Encountered unknown exception type"
            << std::endl;
        throw exception{make_error_code(errc::runtime),
                        "Unknown error type encountered: " + r.what()};
      }
    } catch (...) {
      return std::current_exception();
    }
  }
  return std::exception_ptr{};
}

template<class Handler>
void throw_asynchronous_errors(Handler h){
  sycl::exception_list exceptions;

  common::auto_small_vector<rt::result> async_errors;
  rt::application::errors().pop_each_error(
      [&](const rt::result &err) {
        async_errors.push_back(err);
      });

  for(const auto& err : async_errors) {
    exceptions.push_back(throw_result(err));
  }

  if(!exceptions.empty())
    h(exceptions);
}

}
}

#endif
