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
#ifndef HIPSYCL_KERNEL_CONFIGURATION_HPP
#define HIPSYCL_KERNEL_CONFIGURATION_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <array>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>
#include <functional>
#include <cassert>
#include <optional>
#include <unordered_map>
#include <string_view>

#include "hipSYCL/common/stable_running_hash.hpp"
#include "hipSYCL/glue/llvm-sscp/fcall_specialization.hpp"


namespace hipsycl {
namespace rt {

enum class kernel_base_config_parameter : int {
  backend_id = 0,
  compilation_flow = 1,
  hcf_object_id = 2,
  target_arch = 3,
  runtime_device = 4,
  runtime_context = 5,
  single_kernel = 6
};

enum class kernel_build_option : int {
  known_group_size_x,
  known_group_size_y,
  known_group_size_z,
  known_local_mem_size,

  ptx_version,
  ptx_target_device,

  amdgpu_target_device,
  amdgpu_rocm_device_libs_path,
  amdgpu_rocm_path,

  spirv_dynamic_local_mem_allocation_size
};

enum class kernel_build_flag : int {
  global_sizes_fit_in_int,
  fast_math,

  ptx_ftz,
  ptx_approx_div,
  ptx_approx_sqrt,

  spirv_enable_intel_llvm_spirv_options
};



std::string to_string(kernel_build_flag f);

std::string to_string(kernel_build_option o);

std::optional<kernel_build_option>
to_build_option(const std::string& s);

std::optional<kernel_build_flag>
to_build_flag(const std::string& s);


class kernel_configuration {
public:
  struct int_or_string{
    std::optional<uint64_t> int_value;
    std::optional<std::string> string_value;
  };

  using id_type = std::array<uint64_t, 2>;

  void set_specialized_kernel_argument(int param_index, uint64_t buffer_value) {
    for(int i = 0; i < _specialized_kernel_args.size(); ++i) {
      if(_specialized_kernel_args[i].first == param_index) {
        _specialized_kernel_args[i] = std::make_pair(param_index, buffer_value);
        return;
      }
    }
    _specialized_kernel_args.push_back(
        std::make_pair(param_index, buffer_value));
  }

  void set_function_call_specialization_config(
      int param_index, glue::sscp::fcall_config_kernel_property_t config) {
    _function_call_specializations.push_back(config);
  }

  void set_build_option(kernel_build_option option, const std::string& value) {
    int_or_string ios;
    ios.string_value = value;
    _build_options.push_back(std::make_pair(option, ios));
  }

  template<class T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
  void set_build_option(kernel_build_option option, T int_value) {
    int_or_string ios;
    ios.int_value = static_cast<uint64_t>(int_value);
    _build_options.push_back(std::make_pair(option, ios));
  }

  template<class T, std::enable_if_t<!std::is_unsigned_v<T>, int> = 0>
  void set_build_option(kernel_build_option option, const T& value) {
    set_build_option(option, std::to_string(value));
  }

  void set_build_flag(kernel_build_flag flag) {
    _build_flags.push_back(flag);
  }

  void set_known_alignment(int param_index, int alignment) {
    for(auto& entry : _known_alignments) {
      if(entry.first == param_index) {
        entry.second = alignment;
        return;
      }
    }
    _known_alignments.push_back(std::make_pair(param_index, alignment));
  }

  template <class ValueT>
  void append_base_configuration(kernel_base_config_parameter key,
                                 const ValueT &value) {
    add_entry_to_hash(_base_configuration_result, data_ptr(key), data_size(key),
                      data_ptr(value), data_size(value));
  }

  template<class KeyT, class ValueT>
  static void extend_hash(id_type& hash, const KeyT& key, const ValueT& value) {
    add_entry_to_hash(hash, data_ptr(key), data_size(key),
                      data_ptr(value), data_size(value));
  }

  static std::string to_string(const id_type& id) {
    return std::to_string(id[0])+"."+std::to_string(id[1]);
  }

  id_type generate_id() const {
    id_type result = _base_configuration_result;

    for(const auto& entry : _build_options) {
      uint64_t numeric_option_id = static_cast<uint64_t>(entry.first) | (1ull << 32);
      if(entry.second.int_value.has_value()) {
        auto numeric_value = entry.second.int_value.value();
        add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        &numeric_value, sizeof(numeric_value));
      } else {
        const std::string& string_value = entry.second.string_value.value();
        add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        string_value.data(), string_value.size());
      }
    }

    for(const auto& entry : _build_flags) {
      uint64_t numeric_flag_id = static_cast<uint64_t>(entry) | (1ull << 33);
      add_entry_to_hash(result, &numeric_flag_id, sizeof(numeric_flag_id),
                        "", 0);
    }

    for(const auto& entry : _specialized_kernel_args) {
      uint64_t numeric_option_id = static_cast<uint64_t>(entry.first) | (1ull << 34);
      add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        &entry.second, sizeof(entry.second));
    }

    for(int i = 0; i < _function_call_specializations.size(); ++i) {
      uint64_t numeric_option_id = static_cast<uint64_t>(i) | (1ull << 35);
      uint64_t config_id = _function_call_specializations[i].value->unique_hash;
      add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        &config_id, sizeof(config_id));
    }

    for(const auto& entry : _known_alignments) {
      uint64_t numeric_option_id = static_cast<uint64_t>(entry.first) | (1ull << 37);
      uint64_t config_id = entry.second;
      add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        &config_id, sizeof(config_id));
    }

    return result;
  }

  const auto& build_options() const {
    return _build_options;
  }

  const auto& build_flags() const {
    return _build_flags;
  }

  const auto& specialized_arguments() const {
    return _specialized_kernel_args;
  }

  const auto& function_call_specialization_config() const {
    return _function_call_specializations;
  }

  const auto& known_alignments() const {
    return _known_alignments;
  }

private:
  static const void* data_ptr(const char* data) {
    return data_ptr(std::string{data});
  }

  static const void* data_ptr(const std::string& data) {
    return data.data();
  }

  static const void* data_ptr(std::string_view data) {
    return data.data();
  }

  template<class T>
  static const void* data_ptr(const std::vector<T>& data) {
    return data.data();
  }

  template<class T>
  static const void* data_ptr(const T& data) {
    return &data;
  }

  static std::size_t data_size(const char* data) {
    return data_size(std::string{data});
  }

  static std::size_t data_size(const std::string& data) {
    return data.size();
  }

  static std::size_t data_size(std::string_view data) {
    return data.size();
  }

  template<class T>
  static std::size_t data_size(const std::vector<T>& data) {
    return data.size();
  }

  template<class T>
  static std::size_t data_size(const T& data) {
    return sizeof(T);
  }

  static void add_entry_to_hash(id_type &hash, const void *key_data,
                         std::size_t key_size, const void *data,
                         std::size_t data_size) {
    common::stable_running_hash h;
    h(key_data, key_size);
    h(data, data_size);
    auto entry_hash = h.get_current_hash();
    hash[entry_hash % hash.size()] ^= entry_hash;
  }

  std::vector<kernel_build_flag> _build_flags;
  std::vector<std::pair<kernel_build_option, int_or_string>> _build_options;
  std::vector<std::pair<int, uint64_t>> _specialized_kernel_args;
  std::vector<glue::sscp::fcall_config_kernel_property_t>
      _function_call_specializations;
  std::vector<std::pair<int, int>> _known_alignments;

  id_type _base_configuration_result = {};
};

struct kernel_id_hash{
  std::size_t operator() (const kernel_configuration::id_type &id) const {
    std::size_t hash = 0;
    for(std::size_t i = 0; i < id.size(); ++i)
      hash ^= std::hash<kernel_configuration::id_type::value_type>{}(id[i]);
    return hash;
  }
};

}
}

#endif
