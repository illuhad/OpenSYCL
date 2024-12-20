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
#ifndef HIPSYCL_COMMON_APP_DB_HPP
#define HIPSYCL_COMMON_APP_DB_HPP

#include <unordered_map>
#include <atomic>
#include <vector>
#include <string>
#include <ostream>

#include "msgpack/msgpack.hpp"

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "export.hpp"

namespace hipsycl::common::db {

struct kernel_arg_value_statistics {
  uint64_t value = 0; // The kernel argument value
  uint64_t count = 0; // How many times we have seen this value
  uint64_t last_used = 0; // The number of the kernel invocation where this value
                      // was last used

  ACPP_COMMON_EXPORT void dump(std::ostream& ostr, int indentation_level=0) const;

  template<class T>
  void pack(T &pack) {
    pack(value);
    pack(count);
    pack(last_used);
  }
};

struct kernel_arg_entry {
  static constexpr int max_tracked_values = 8;

  std::array<kernel_arg_value_statistics, max_tracked_values> common_values = {};
  std::array<bool, max_tracked_values> was_specialized = {};

  template<class T>
  void pack(T &pack) {
    pack(common_values);
    pack(was_specialized);
  }

  ACPP_COMMON_EXPORT void dump(std::ostream& ostr, int indentation_level=0) const;
};

struct kernel_entry {

  template<class T>
  void pack(T &pack) {
    pack(kernel_args);
    pack(num_registered_invocations);
    pack(retained_argument_indices);
    pack(first_iads_invocation_run);
  }

  ACPP_COMMON_EXPORT void dump(std::ostream& ostr, int indentation_level=0) const;

  std::vector<kernel_arg_entry> kernel_args;
  std::size_t num_registered_invocations = 0;
  std::vector<int> retained_argument_indices;

  // It seems there is a bug in msgpack serializing
  // std::numeric_limits<size_t>::max(). So we use 1 << 63
  // to denote an unset/invalid value.
  static constexpr uint64_t no_usage = 1ull << 63;
  uint64_t first_iads_invocation_run = no_usage;
};

struct binary_entry {
  std::string jit_cache_filename;

  template<class T>
  void pack(T &pack) {
    pack(jit_cache_filename);
  }

  ACPP_COMMON_EXPORT void dump(std::ostream& ostr, int indentation_level=0) const;
};

struct appdb_data {
  std::size_t content_version = 0;

  std::unordered_map<rt::kernel_configuration::id_type, kernel_entry,
                     rt::kernel_id_hash>
      kernels;
  std::unordered_map<rt::kernel_configuration::id_type, binary_entry,
                     rt::kernel_id_hash>
      binaries;

  template<class T>
  void pack(T &pack) {
    pack(kernels);
    pack(binaries);
    pack(content_version);
  }

  ACPP_COMMON_EXPORT void dump(std::ostream& ostr, int indentation_level=0) const;
};


ACPP_COMMON_EXPORT class appdb  {
public:
  // DO NOT FORGET TO INCREMENT THIS WHEN ADDING/REMOVING
  // FIELDS OR OTHERWISE CHANGING THE DATA LAYOUT!
  static const uint64_t format_version = 4;

  appdb(const std::string& db_path);
  ~appdb();

  template<class F>
  auto read_access(F&& handler) const{
    read_lock lock {_lock};
    return handler(_data);
  }

  template<class F>
  auto read_write_access(F&& handler) {
    write_lock lock {_lock};
    _was_modified = true;
    return handler(_data);
  }

private:
  
  struct write_lock {
  public:
    write_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = 0;
      while (!_op_counter.compare_exchange_strong(
          expected, -1, std::memory_order_release, std::memory_order_relaxed)) {
        expected = 0;
      }
    }

    ~write_lock() {
      _op_counter.store(0, std::memory_order_release);
    }
  private:
    std::atomic<int>& _op_counter;
  };

  struct read_lock {
  public:
    read_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = std::max(0, _op_counter.load(std::memory_order_acquire));
      while (!_op_counter.compare_exchange_strong(
          expected, expected+1, std::memory_order_release,
          std::memory_order_relaxed)) {
        if(expected < 0)
          expected = 0;
      }
    }

    ~read_lock() {
      _op_counter.fetch_sub(1, std::memory_order_acq_rel);
    }
  private:
   std::atomic<int>& _op_counter;
  };

  mutable std::atomic<int> _lock;
  bool _was_modified;

  std::string _db_path;

  appdb_data _data;
};


}

#endif
