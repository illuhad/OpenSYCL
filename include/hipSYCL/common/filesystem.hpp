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
#ifndef HIPSYCL_COMMON_FILESYSTEM_HPP
#define HIPSYCL_COMMON_FILESYSTEM_HPP

#include <string>
#include <vector>
#include <memory>
#include <atomic>

#include "appdb.hpp"


namespace hipsycl {
namespace common {

namespace filesystem {

std::string get_install_directory();

std::string join_path(const std::string& base, const std::string& extra);

bool exists(const std::string& path);

std::string absolute(const std::string& path);

std::string
join_path(const std::string &base,
          const std::vector<std::string> &additional_components);

std::vector<std::string> list_regular_files(const std::string &directory,
                                            std::error_code &EC);
std::vector<std::string> list_regular_files(const std::string &directory,
                                            const std::string &extension,
                                            std::error_code &EC);

/// Writes data atomically to filename
bool atomic_write(const std::string& filename, const std::string& data);

/// Removes a file, returns true if successful.
bool remove(const std::string &filename);

// Replaces runtime specialized variables, such as $HIPSYCL_PATH in the path.
std::string replace_known_variables(const std::string path);

class persistent_storage {
public:
  static persistent_storage& get() {
    static persistent_storage t;
    return t;
  }

  const std::string& get_base_dir() const {
    return _base_dir;
  }

  std::string generate_app_dir(const std::string& app_path) const;
  std::string generate_appdb_path(const std::string& app_path) const;

  const std::string& get_this_app_dir() const {
    return _this_app_dir;
  }

  const std::string& get_jit_cache_dir() const {
    return _jit_cache_dir;
  }

  db::appdb& get_this_app_db() {
    return *_this_app_db;
  }

  // Generates just the expected name of the file, without directories.
  std::string generate_app_db_filename() const;
private:

  persistent_storage();

  std::string _base_dir;
  std::string _this_app_dir;
  std::string _jit_cache_dir;

  std::unique_ptr<db::appdb> _this_app_db;
};

}

}
}

#endif
