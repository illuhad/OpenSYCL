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
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/dylib_loader.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/config.hpp"
#include "hipSYCL/runtime/device_id.hpp"

#include <cassert>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h> 
#endif

#include HIPSYCL_CXX_FILESYSTEM_HEADER
namespace fs = HIPSYCL_CXX_FILESYSTEM_NAMESPACE;

namespace {

using namespace hipsycl::rt::detail;
bool load_plugin(const std::string &filename, void *&handle_out,
                 std::string &backend_name_out) {
  if(void *handle = load_library(filename, "backend_loader")) {
    if(void* symbol = get_symbol_from_library(handle, "hipsycl_backend_plugin_get_name", "backend_loader"))
    {
      auto get_name =
          reinterpret_cast<decltype(&hipsycl_backend_plugin_get_name)>(symbol);

      handle_out = handle;
      backend_name_out = get_name();

      return true;
    } else {
      close_library(handle, "backend_loader");
      return false;
    }
  } else {
    return false;
  } 
}

hipsycl::rt::backend *create_backend(void *plugin_handle) {
  assert(plugin_handle);

  if(void *symbol = get_symbol_from_library(plugin_handle, "hipsycl_backend_plugin_create", "backend_loader"))
  {
    auto create_backend_func =
        reinterpret_cast<decltype(&hipsycl_backend_plugin_create)>(symbol);

    return create_backend_func();
  }
  return nullptr;
}

std::vector<fs::path> get_plugin_search_paths()
{
  std::vector<fs::path> paths;
#ifndef _WIN32
  #define HIPSYCL_BACKEND_LIB_FOLDER "lib"
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&get_plugin_search_paths), &info)) {
    paths.emplace_back(fs::path{info.dli_fname}.parent_path() / "hipSYCL");
  }
#else
  #define HIPSYCL_BACKEND_LIB_FOLDER "bin"

  if(HMODULE handle = GetModuleHandleA(HIPSYCL_RT_LIBRARY_NAME))
  {
    std::vector<char> path_buffer(MAX_PATH);
    while(GetModuleFileNameA(handle, path_buffer.data(), path_buffer.size()) == path_buffer.size())
    {
      path_buffer.resize(path_buffer.size() * 2);
      if(path_buffer.size() >= 1024*1024) // 1MB paths? sure. I think it's time to give up...
        break;
    }
    paths.emplace_back(fs::path{path_buffer.data()}.parent_path() / "hipSYCL");
  }
#endif

  if(auto install_dir = hipsycl::common::filesystem::get_install_directory(); !install_dir.empty())
    paths.emplace_back(fs::path(install_dir) / HIPSYCL_BACKEND_LIB_FOLDER / "hipSYCL");

  const auto install_prefixed_path = fs::path{HIPSYCL_INSTALL_PREFIX} / HIPSYCL_BACKEND_LIB_FOLDER / "hipSYCL";

  if(paths.empty()
      || !fs::is_directory(paths.back())
      || (fs::is_directory(install_prefixed_path)
          && !fs::equivalent(install_prefixed_path, paths.back())))
    paths.emplace_back(std::move(install_prefixed_path));
  return paths;
}

bool is_plugin_active(const std::string& name)
{
  auto backends_active = hipsycl::rt::application::get_settings().get<hipsycl::rt::setting::visibility_mask>();
  if(backends_active.empty())
    return true;
  if(name == "omp") // we always need a cpu backend
    return true;

  hipsycl::rt::backend_id id;
  if(name == "cuda") {
    id = hipsycl::rt::backend_id::cuda;
  } else if(name == "hip") {
    id = hipsycl::rt::backend_id::hip;
  } else if(name == "ze") {
    id = hipsycl::rt::backend_id::level_zero;
  } else if(name == "ocl") {
    id = hipsycl::rt::backend_id::ocl;
  }
  return backends_active.find(id) != backends_active.cend();
}

}

namespace hipsycl {
namespace rt {

void backend_loader::query_backends() {
  std::vector<fs::path> backend_lib_paths = get_plugin_search_paths();

#ifdef __APPLE__
  std::string shared_lib_extension = ".dylib";
#elif defined(_WIN32)
  std::string shared_lib_extension = ".dll";
#else
  std::string shared_lib_extension = ".so";
#endif

  for(const fs::path& backend_lib_path : backend_lib_paths) {
    if(!fs::is_directory(backend_lib_path)) {
      HIPSYCL_DEBUG_INFO << "backend_loader: Backend lib search path candidate does not exists: "
                        << backend_lib_path << std::endl;
      continue;
    }

    HIPSYCL_DEBUG_INFO << "backend_loader: Searching path for backend libs: '"
                      << backend_lib_path << "'" << std::endl;

    for (const fs::directory_entry &entry :
        fs::directory_iterator(backend_lib_path)) {

      if(fs::is_regular_file(entry.status())){
        auto p = entry.path();
        if (p.extension().string() == shared_lib_extension) {
          std::string backend_name;
          void *handle;
          if (load_plugin(p.string(), handle, backend_name)) {
            if(!has_backend(backend_name) && is_plugin_active(backend_name)){
              HIPSYCL_DEBUG_INFO << "backend_loader: Successfully opened plugin: " << p
                                << " for backend '" << backend_name << "'"
                                << std::endl;
              _handles.emplace_back(std::make_pair(backend_name, handle));
            } else {
              close_library(handle, "backend_loader");
            }
          }
        }
      }
    }
  }
}

backend_loader::~backend_loader() {
  for (auto &handle : _handles) {
    assert(handle.second);
    close_library(handle.second, "backend_loader");
  }
}

std::size_t backend_loader::get_num_backends() const { return _handles.size(); }

std::string backend_loader::get_backend_name(std::size_t index) const {
  assert(index < _handles.size());
  return _handles[index].first;
}

bool backend_loader::has_backend(const std::string &name) const {
  for (const auto &h : _handles) {
    if (h.first == name)
      return true;
  }

  return false;
}

backend *backend_loader::create(std::size_t index) const {
  assert(index < _handles.size());
  
  return create_backend(_handles[index].second);
}

backend *backend_loader::create(const std::string &name) const {
  
  for (std::size_t i = 0; i < _handles.size(); ++i) {
    if (_handles[i].first == name)
      return create(i);
  }

  return nullptr;
}

}
} // namespace hipsycl
