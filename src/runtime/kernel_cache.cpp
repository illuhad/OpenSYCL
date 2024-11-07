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
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>

namespace hipsycl {
namespace rt {

namespace {

template<class F>
void for_each_device_image(const common::hcf_container& hcf, F&& handler) {
  if(hcf.root_node()->has_subnode("images")) {
    auto* img_node = hcf.root_node()->get_subnode("images");
    auto image_names = img_node->get_subnodes();
    for(const auto& image_name : image_names) {
      handler(img_node->get_subnode(image_name));
    }
  }
}

template<class F>
void for_each_exported_symbol_list(const common::hcf_container& hcf, F&& handler) {
  for_each_device_image(hcf, [&](const common::hcf_container::node* image_node){
    auto exported_symbols = image_node->get_as_list("exported-symbols");
    handler(image_node, exported_symbols);
  });
}

}

hcf_kernel_info::hcf_kernel_info(
    hcf_object_id id, const common::hcf_container::node *kernel_node)
    : _id{id} {

  if(!kernel_node->has_subnode("image-providers"))
    return;
  _image_providers = kernel_node->get_as_list("image-providers");

  // investigate parameters
  auto *parameters_node = kernel_node->get_subnode("parameters");

  if (!parameters_node)
    return;

  std::size_t num_subnodes = parameters_node->get_subnodes().size();

  for (int i = 0; i < num_subnodes; ++i) {
    const auto *param_info_node =
        parameters_node->get_subnode(std::to_string(i));

    if (!param_info_node)
      return;

    auto *byte_size = param_info_node->get_value("byte-size");
    auto *byte_offset = param_info_node->get_value("byte-offset");
    auto *original_index = param_info_node->get_value("original-index");
    auto *type = param_info_node->get_value("type");

    if (!byte_size)
      return;
    if (!byte_offset)
      return;
    if (!type)
      return;

    std::size_t arg_size = std::stoll(*byte_size);
    std::size_t arg_offset = std::stoll(*byte_offset);
    std::size_t arg_original_index = std::stoll(*original_index);
    if(*type == "pointer") {
      _arg_types.push_back(pointer);
    } else {
      _arg_types.push_back(other);
    }
    _arg_offsets.push_back(arg_offset);
    _arg_sizes.push_back(arg_size);
    _original_arg_indices.push_back(arg_original_index);
    // Let's accept annotation nodes not being provided
    _string_annotations.push_back({});
    _known_annotations.push_back({});
    if(auto* annotation_node = param_info_node->get_subnode("annotations")){
      for(const auto& entry : annotation_node->key_value_pairs) {
        // Ignore entries that are not set to "1" for now
        if(entry.second == "1") {
          if(entry.first == "specialized") {
            _known_annotations.back().push_back(annotation_type::specialized);
          } else if(entry.first == "fcall_specialized_config") {
            _known_annotations.back().push_back(
                annotation_type::fcall_specialized_config);
          } else if(entry.first == "restrict") {
            _known_annotations.back().push_back(annotation_type::restrict);
          } else {
            _string_annotations.back().push_back(entry.first);
          }
        }
      }
    }
  }

  if(const auto* flags_node = kernel_node->get_subnode("compile-flags")) {
    for(const auto& flag : flags_node->key_value_pairs) {
      auto f = to_build_flag(flag.first);
      if(f.has_value())
        _compilation_flags.push_back(f.value());
    }
  }
  if(const auto* options_node = kernel_node->get_subnode("compile-options")) {
    for(const auto& option : options_node->key_value_pairs) {
      auto o = to_build_option(option.first);
      if(o.has_value())
        _compilation_options.push_back(
            std::make_pair(o.value(), option.second));
    }
  }

  _parsing_successful = true;
}

std::size_t hcf_kernel_info::get_num_parameters() const {
  return _arg_sizes.size();
}

bool hcf_kernel_info::is_valid() const {
  return _parsing_successful;
}


std::size_t hcf_kernel_info::get_argument_offset(std::size_t i) const {
  return _arg_offsets[i];
}

std::size_t hcf_kernel_info::get_argument_size(std::size_t i) const {
  return _arg_sizes[i];
}

std::size_t hcf_kernel_info::get_original_argument_index(std::size_t i) const {
  return _original_arg_indices[i];
}

hcf_kernel_info::argument_type hcf_kernel_info::get_argument_type(std::size_t i) const {
  return _arg_types[i];
}

const std::vector<std::string> &
hcf_kernel_info::get_string_annotations(std::size_t i) const {
  return _string_annotations[i];
}

const std::vector<hcf_kernel_info::annotation_type> &
hcf_kernel_info::get_known_annotations(std::size_t i) const {
  return _known_annotations[i];
}

const std::vector<std::string> &
hcf_kernel_info::get_images_containing_kernel() const {
  return _image_providers;
}

hcf_object_id hcf_kernel_info::get_hcf_object_id() const {
  return _id;
}

const std::vector<kernel_build_flag> &
hcf_kernel_info::get_compilation_flags() const {
  return _compilation_flags;
}

const std::vector<std::pair<kernel_build_option, std::string>> &
hcf_kernel_info::get_compilation_options() const {
  return _compilation_options;
}

const std::string& hcf_image_info::get_format() const {
  return _format;
}

const std::string& hcf_image_info::get_variant() const {
  return _variant;
}

hcf_image_info::hcf_image_info(const common::hcf_container *hcf,
                               const common::hcf_container::node *image_node) {
  assert(hcf);
  assert(image_node);
  if(!image_node->has_key("format"))
    return;
  if(!image_node->has_key("variant"))
    return;
  
  _format = *image_node->get_value("format");
  _variant = *image_node->get_value("variant");

  // Currently we need to obtain contained kernels list
  // by walking through all kernels, and matching against their image-providers.
  auto* kernels = hcf->root_node()->get_subnode("kernels");
  if(!kernels)
    return;
  
  std::string image_name = image_node->node_id;
  for(const auto& kernel : kernels->get_subnodes()) {
    std::vector<std::string> image_providers =
        kernels->get_subnode(kernel)->get_as_list("image-providers");
    for(const std::string& provider : image_providers) {
      if(provider == image_name)
        _contained_kernels.push_back(kernel);
    }
  }

  _parsing_successful = true;

}

const std::vector<std::string> &hcf_image_info::get_contained_kernels() const {
  return _contained_kernels;
}

bool hcf_image_info::is_valid() const {
  return _parsing_successful;
}

hcf_cache& hcf_cache::get() {
  static hcf_cache c;
  return c;
}

hcf_object_id hcf_cache::register_hcf_object(const common::hcf_container &obj) {

  std::lock_guard<std::mutex> lock{_mutex};

  if (!obj.root_node()->has_key("object-id")) {
    HIPSYCL_DEBUG_ERROR
        << "hcf_cache: Invalid hcf object (missing object id)" << std::endl;
  }
  const std::string *data = obj.root_node()->get_value("object-id");

  assert(data);
  hcf_object_id id = std::stoull(*data);
  HIPSYCL_DEBUG_INFO << "hcf_cache: Registering HCF object " << id << "..." << std::endl;

  if (_hcf_objects.count(id) > 0) {
    HIPSYCL_DEBUG_ERROR
        << "hcf_cache: Detected hcf object id collision " << id
        << ", this should not happen. Some kernels might be unavailable."
        << std::endl;
  } else {
    common::hcf_container* stored_obj = new common::hcf_container{obj};
    _hcf_objects[id] = std::unique_ptr<common::hcf_container>{stored_obj};
    // Check if the HCF exports some symbols
    for_each_exported_symbol_list(
        // Don't use obj here, since we have copied it into the cache, and need
        // to ensure that the pointers to image nodes are stable
        *stored_obj,
        [&](const common::hcf_container::node *image_node,
            const std::vector<std::string> &exported_symbols) {

          for (const auto &symbol : exported_symbols) {

            _exported_symbol_providers[symbol].push_back(
                device_image_id{id, image_node});

            HIPSYCL_DEBUG_INFO << "hcf_cache: Symbol " << symbol
                               << " is registered as exported by object " << id
                               << " and image " << image_node->node_id
                               << " @" << image_node << std::endl;
          }
        });
    // See if stored object has kernel nodes that we can parse
    if(auto* kernels_node = stored_obj->root_node()->get_subnode("kernels")) {
      for(const auto& kernel_name : kernels_node->get_subnodes()) {
        std::unique_ptr<hcf_kernel_info> kernel_info{
            new hcf_kernel_info{id, kernels_node->get_subnode(kernel_name)}};
        if(kernel_info->is_valid()) {
          HIPSYCL_DEBUG_INFO << "hcf_cache: Registering kernel info for kernel "
                             << kernel_name << " from HCF object " << id
                             << std::endl;
          HIPSYCL_DEBUG_INFO << "  kernel_info: hcf object id = "
                             << kernel_info->get_hcf_object_id() << std::endl;
          for(int i = 0; i < kernel_info->get_num_parameters(); ++i) {
            HIPSYCL_DEBUG_INFO
                << "  kernel_info: parameter " << i
                << ": offset = " << kernel_info->get_argument_offset(i)
                << " size = " << kernel_info->get_argument_size(i)
                << " original index = "
                << kernel_info->get_original_argument_index(i) << std::endl;
          }
          _hcf_kernel_info[generate_info_id(id, kernel_name)] =
              std::move(kernel_info);
        }
      }
    }
    // Same for image nodes
    if(auto* images_node = stored_obj->root_node()->get_subnode("images")) {
      for(const auto& image_name : images_node->get_subnodes()) {
        std::unique_ptr<hcf_image_info> image_info{new hcf_image_info{
            stored_obj, images_node->get_subnode(image_name)}};
        
        if(image_info->is_valid()) {
          HIPSYCL_DEBUG_INFO << "hcf_cache: Registering image info for image "
                             << image_name << " from HCF object " << id
                             << std::endl;
          _hcf_image_info[generate_info_id(id, image_name)] =
              std::move(image_info);
        }
      }
    }
  }

  std::string hcf_dump_dir =
      application::get_settings().get<setting::hcf_dump_directory>();
  if(!hcf_dump_dir.empty()) {
    std::string out_filename = hcf_dump_dir;

    if(out_filename.back() != '/' && out_filename.back() != '\\')
      out_filename += '/';
    
    out_filename += "hipsycl_object_"+std::to_string(id)+".hcf";

    std::ofstream out_file(out_filename.c_str(), std::ios::binary);
    if(!out_file.is_open()) {
      HIPSYCL_DEBUG_ERROR << "Could not open file " << out_filename
                          << " for writing." << std::endl;

    } else {
      std::string hcf_data = obj.serialize();
      out_file.write(hcf_data.c_str(), hcf_data.size());
    }
  }

  return id;
}

void hcf_cache::unregister_hcf_object(hcf_object_id id) {
  std::lock_guard<std::mutex> lock{_mutex};

  auto it = _hcf_objects.find(id);
  if(it != _hcf_objects.end()) {
    // First remove the HCF object as a symbol provider for runtime linking and
    // symbol resolution. This ensures that it gets no longer selected
    // for symbol resolution.

    // 1. Go through each symbol list (all device images) exported by this
    // HCF file
    for_each_exported_symbol_list(
        *(it->second), [&](const common::hcf_container::node* image_node,
                 const std::vector<std::string> &exported_symbols) {
          // 2. Iterate over all symbols exported in this HCF
          for (const auto &symbol : exported_symbols) {
            // 3. Remove all references to this HCF in the symbol providers map  
            auto& symbol_providers = _exported_symbol_providers[symbol];
            symbol_providers.erase(
                std::remove_if(symbol_providers.begin(), symbol_providers.end(),
                               [&](const device_image_id &img) {
                                 return img.hcf_id == id;
                               }),
                symbol_providers.end());
          }
        });
    // Then we can remove the HCF itself.
    // Note: We don't necessarily need to remove the HCF kernel info, since
    // just maintaining this data won't have any side effects as long as 
    // the HCF object is no longer selected for execution.
    _hcf_objects.erase(id);
  }
}

const common::hcf_container* hcf_cache::get_hcf(hcf_object_id obj) const {
  std::lock_guard<std::mutex> lock{_mutex};

  auto it = _hcf_objects.find(obj);
  if(it == _hcf_objects.end())
    return nullptr;
  return it->second.get();
}

const hcf_kernel_info *
hcf_cache::get_kernel_info(hcf_object_id obj,
                           std::string_view kernel_name) const {
  std::lock_guard<std::mutex> lock{_mutex};
  auto it = _hcf_kernel_info.find(generate_info_id(obj, kernel_name));
  if(it == _hcf_kernel_info.end())
    return nullptr;
  return it->second.get();
}

const hcf_kernel_info *
hcf_cache::get_kernel_info(hcf_object_id obj,
                           const std::string &kernel_name) const {
  return get_kernel_info(obj, std::string_view{kernel_name});
}

const hcf_image_info *
hcf_cache::get_image_info(hcf_object_id obj,
                          const std::string &image_name) const {
  std::lock_guard<std::mutex> lock{_mutex};
  auto it = _hcf_image_info.find(generate_info_id(obj, image_name));
  if(it == _hcf_image_info.end())
    return nullptr;
  return it->second.get();
}




std::shared_ptr<kernel_cache> kernel_cache::get() {
  // required since kernel_cache has a private default constructor
  struct make_shared_enabler : public kernel_cache {};
  static std::shared_ptr<kernel_cache> c = std::make_shared<make_shared_enabler>();
  return c;
}

void kernel_cache::unload() {
  std::lock_guard<std::mutex> lock{_mutex};

  _code_objects.clear();
}

const code_object* kernel_cache::get_code_object(code_object_id id) const {
  std::lock_guard<std::mutex> lock{_mutex};
  return get_code_object_impl(id);
}

const code_object* kernel_cache::get_code_object_impl(code_object_id id) const {
  auto it = _code_objects.find(id);
  if(it == _code_objects.end())
    return nullptr;
  return it->second.get();
}

std::string kernel_cache::get_persistent_cache_file(code_object_id id_of_binary) {
  using namespace common::filesystem;
  std::string cache_dir = persistent_storage::get().get_jit_cache_dir();
  return join_path(cache_dir, kernel_configuration::to_string(id_of_binary)+".jit");
}

bool kernel_cache::persistent_cache_lookup(code_object_id id_of_binary,
                                           std::string &out) const {
  std::string filename;

  bool filename_lookup_succeeded =
      common::filesystem::persistent_storage::get()
          .get_this_app_db()
          .read_access([&](const common::db::appdb_data &appdb) {
            auto binary = appdb.binaries.find(id_of_binary);
            if (binary == appdb.binaries.end())
              return false;

            filename = binary->second.jit_cache_filename;
            return true;
          });

  if(!filename_lookup_succeeded)
    return false;

  std::ifstream file{filename, std::ios::in | std::ios::binary | std::ios::ate};
  
  if(!file.is_open())
    return false;

  HIPSYCL_DEBUG_INFO << "kernel_cache: Persistent cache hit for id "
                     << kernel_configuration::to_string(id_of_binary)
                     << " in file " << filename << std::endl;

  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  out.resize(file_size);
  file.read(out.data(), file_size);
  
  return true;
}

void kernel_cache::persistent_cache_store(code_object_id id_of_binary,
                                          const std::string &data) const {
  if(application::get_settings().get<setting::no_jit_cache_population>())
    return;

  std::string filename = get_persistent_cache_file(id_of_binary);

  HIPSYCL_DEBUG_INFO << "kernel_cache: Storing compiled binary with id "
                     << kernel_configuration::to_string(id_of_binary)
                     << " in persistent cache file " << filename << std::endl;
  

  if(!common::filesystem::atomic_write(filename, data)) {
    HIPSYCL_DEBUG_ERROR
        << "Could not store JIT result in persistent kernel cache in file "
        << filename << std::endl;
  }

  common::filesystem::persistent_storage::get()
      .get_this_app_db()
      .read_write_access([&](common::db::appdb_data &appdb) {
        appdb.binaries[id_of_binary].jit_cache_filename = filename;
      });
}

} // rt
} // hipsycl
