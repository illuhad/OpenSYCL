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
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/common/config.hpp"
#include "hipSYCL/common/stable_running_hash.hpp"
#include "hipSYCL/common/debug.hpp"

#include "hipSYCL/runtime/settings.hpp"

#include <fstream>
#include <memory>
#include <random>
#include <cassert>

#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <limits.h>
#else
#include <windows.h> 
#endif

#include HIPSYCL_CXX_FILESYSTEM_HEADER
namespace fs = HIPSYCL_CXX_FILESYSTEM_NAMESPACE;


namespace hipsycl {
namespace common {
namespace filesystem {

namespace {


template<class T>
inline T random_number() {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen{rd()};
  thread_local std::uniform_int_distribution<T> distribution{0};

  return distribution(gen);
}

}

std::string get_install_directory() {

  std::vector<fs::path> paths;
#ifndef _WIN32
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&get_install_directory), &info)) {
    auto lib_path = fs::path{info.dli_fname}.parent_path();
    if(lib_path.has_parent_path())
      paths.emplace_back(lib_path.parent_path());
  }
  
#else
  if(HMODULE handle = GetModuleHandleA(HIPSYCL_COMMON_LIBRARY_NAME))
  {
    std::vector<char> path_buffer(MAX_PATH);
    if(GetModuleFileNameA(handle, path_buffer.data(), path_buffer.size()))
    {
      auto lib_path = fs::path{path_buffer.data()}.parent_path();
      if(lib_path.has_parent_path())
        paths.emplace_back(lib_path.parent_path());
    }
  }
  
#endif
  if(paths.empty() || !fs::is_directory(paths.back()))
    return fs::path{HIPSYCL_INSTALL_PREFIX}.string();
  return paths.back().string();
}

std::string join_path(const std::string& base, const std::string& extra) {
  return (fs::path(base) / extra).string();
}

std::string
join_path(const std::string &base,
          const std::vector<std::string> &additional_components) {
  std::string current = base;
  for(const auto& extra : additional_components) {
    current = join_path(current, extra);
  }
  return current;
}

std::vector<std::string> list_regular_files(const std::string& directory, std::error_code &EC) {
  fs::path p{directory};
  std::vector<std::string> result;
  for(const fs::directory_entry& entry : fs::directory_iterator(p, EC)) {
    if(fs::is_regular_file(entry.status())) {
      result.push_back(entry.path().string());
    }
  }
  return result;
}

std::vector<std::string> list_regular_files(const std::string& directory,
                                            const std::string& extension,
                                            std::error_code &EC) { 
  
  auto all_files = list_regular_files(directory, EC);
  std::vector<std::string> result;
  for(const auto& f : all_files) {
    if(fs::path(f).extension().string() == extension)
      result.push_back(f);
  }
  return result;
}

bool exists(const std::string& path) {
  return fs::exists(path);
}

std::string absolute(const std::string& path) {
  return fs::absolute(path).string();
}

bool atomic_write(const std::string &filename, const std::string &data) {
  fs::path p{filename};

  std::string temp_file = std::to_string(random_number<std::size_t>())+".tmp";
  fs::path tmp_path = p.parent_path() / temp_file;

  std::ofstream ostr{tmp_path, std::ios::binary|std::ios::out|std::ios::trunc};
  
  if(!ostr.is_open())
    return false;
  
  ostr.write(data.data(), data.size());
  ostr.close();

  fs::rename(tmp_path, p);

  return true;
}

bool remove(const std::string &filename) {
  try {
    return fs::remove(filename);
  } catch (const fs::filesystem_error &err) {}
  return false;
}

std::string replace_known_variables(std::string path) {
  if(auto pos = path.find("$ACPP_PATH");pos != std::string::npos){
    const auto install_dir = get_install_directory();
    path.replace(pos, std::string_view("$ACPP_PATH").size(), install_dir);
  }
  return path;
}

persistent_storage &persistent_storage::get() {
  static persistent_storage t;
  return t;
}

persistent_storage::persistent_storage() {
#ifndef _WIN32

  auto get_home = [](std::string &home_out, std::string &subdirectory) -> bool {
    if(const char* home = std::getenv("XDG_DATA_HOME")) {
      home_out = home;
      subdirectory = "acpp";
      return true;
    }

    if (const char* home = std::getenv("HOME")) {
      home_out = home;
      subdirectory = ".acpp";
      return true;
    }
        
    const char* home = getpwuid(getuid())->pw_dir;
    if(home) {
      home_out = home;
      subdirectory = ".acpp";
      return true;
    }

    return false;
  };

  if(!rt::try_get_environment_variable("appdb_dir", _base_dir)) {
    std::string home, subdirectory;
    if (get_home(home, subdirectory)) {
      _base_dir = (fs::path{home} / subdirectory).string();
    } else {
      _base_dir = (fs::current_path() / ".acpp").string();
    }
  }

  auto get_app_path = []() -> std::string{

    char result[PATH_MAX];
    ssize_t num_read = readlink("/proc/self/exe", result, PATH_MAX);
    if(num_read >= 0 && num_read <= PATH_MAX - 1) {
      result[num_read] = '\0';
      return std::string{result};
    }
    return std::string{};
  };

  std::string app_path = get_app_path();
  _this_app_dir = generate_app_dir(app_path);
  
#else
  _base_dir = (fs::current_path() / ".acpp").string();
  _this_app_dir = (fs::path{_base_dir} / "apps" / "global").string();
#endif

   _jit_cache_dir = (fs::path{_this_app_dir} / "jit-cache").string();

  fs::create_directories(_base_dir);
  fs::create_directories(_this_app_dir);
  fs::create_directories(_jit_cache_dir);

#ifndef _WIN32
  _this_app_db = std::make_unique<db::appdb>(generate_appdb_path(app_path));
#else
  _this_app_db = std::make_unique<db::appdb>(generate_appdb_path(""));
#endif
}

std::string persistent_storage::generate_app_dir(const std::string& app_path) const {
  std::string app_subdirectory = "global";
  if(!app_path.empty()) {
    std::string app_filename = fs::path{app_path}.filename().string();

    stable_running_hash h;
    h(app_path.data(), app_path.size());
    app_subdirectory = app_filename + "-" + std::to_string(h.get_current_hash());
  }

  return (fs::path{_base_dir} / "apps" / app_subdirectory).string();
}

std::string persistent_storage::generate_app_db_filename() const {
  auto version = db::appdb::format_version;
  return "app.v"+std::to_string(version)+".db";
}

std::string persistent_storage::generate_appdb_path(const std::string& app_path) const {
  return join_path(generate_app_dir(app_path), generate_app_db_filename());
}

}
}
}

