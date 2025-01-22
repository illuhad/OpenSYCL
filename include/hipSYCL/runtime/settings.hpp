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
#ifndef HIPSYCL_RT_SETTINGS_HPP
#define HIPSYCL_RT_SETTINGS_HPP

#include "hipSYCL/runtime/device_id.hpp"

#include <ios>
#include <optional>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
namespace hipsycl {
namespace rt {

enum class scheduler_type { direct, unbound };
enum class default_selector_behavior { strict, multigpu, system };

struct device_visibility_condition{
  int device_index_equality = -1;
  int platform_index_equality = -1;
  std::string device_name_match;
  std::string platform_name_match;
};

using visibility_mask_t =
    std::unordered_map<rt::backend_id,
                       std::vector<device_visibility_condition>>;

bool device_matches(const visibility_mask_t &mask, backend_id backend,
                    int global_device_index, int platform_device_index,
                    int platform_index, const std::string &dev_name,
                    const std::string &platform_name);

bool device_matches(const visibility_mask_t::mapped_type &visibility_conditions,
                    int global_device_index, int platform_device_index,
                    int platform_index, const std::string &dev_name,
                    const std::string &platform_name);
bool has_device_visibility_mask(const visibility_mask_t& mask, backend_id backend);

std::istream &operator>>(std::istream &istr, scheduler_type &out);
std::istream &operator>>(std::istream &istr, visibility_mask_t &out);
std::istream &operator>>(std::istream &istr, default_selector_behavior& out);

template <class T>
bool try_get_environment_variable(const std::string& name, T& out) {
  std::string env_name = name;

  std::transform(env_name.begin(), env_name.end(), env_name.begin(), ::toupper);

  std::string env;
  if (const char *env_value =
          std::getenv(("ACPP_"+env_name).c_str())) {
    env_name = "ACPP_"+env_name;
    env = std::string{env_value};
  } else if (const char *env_value =
          std::getenv(("HIPSYCL_"+env_name).c_str())) {
    env_name = "HIPSYCL_"+env_name;
    env = std::string{env_value};
  }
  
  if (!env.empty()) {
    
    T val;
    std::stringstream sstr{std::string{env}};
    sstr >> val;

    if (sstr.fail() || sstr.bad()) {
      std::cerr << "AdaptiveCpp settings parsing: Could not parse value of environment "
                    "variable: "
                << env_name << std::endl;
      return false;
    }
    out = val;
    return true;
  }
  return false;
}

enum class setting {
  debug_level,
  scheduler_type,
  visibility_mask,
  dag_req_optimization_depth,
  mqe_lane_statistics_max_size,
  mqe_lane_statistics_decay_time_sec,
  default_selector_behavior,
  hcf_dump_directory,
  persistent_runtime,
  max_cached_nodes,
  sscp_failed_ir_dump_directory,
  gc_trigger_batch_size,
  ocl_no_shared_context,
  ocl_show_all_devices,
  no_jit_cache_population,
  adaptivity_level,
  jitopt_iads_relative_threshold,
  jitopt_iads_relative_eviction_threshold,
  jitopt_iads_relative_threshold_min_data,
  enable_allocation_tracking
};

template <setting S> struct setting_trait {};

#define HIPSYCL_RT_MAKE_SETTING_TRAIT(S, string_identifier, setting_type)      \
  template <> struct setting_trait<S> {                                        \
    using type = setting_type;                                                 \
    static constexpr const char *str = string_identifier;                      \
  };


HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::debug_level, "debug_level", int)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::scheduler_type, "rt_scheduler", scheduler_type)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::visibility_mask, "visibility_mask", visibility_mask_t)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::dag_req_optimization_depth,
                              "rt_dag_req_optimization_depth", std::size_t);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::mqe_lane_statistics_max_size,
                              "rt_mqe_lane_statistics_max_size", std::size_t);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::mqe_lane_statistics_decay_time_sec,
                              "rt_mqe_lane_statistics_decay_time_sec", double);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::default_selector_behavior,
                              "default_selector_behavior", default_selector_behavior);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::hcf_dump_directory,
                              "hcf_dump_directory", std::string);
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::persistent_runtime, "persistent_runtime", bool)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::max_cached_nodes, "rt_max_cached_nodes", std::size_t)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::sscp_failed_ir_dump_directory,
                              "sscp_failed_ir_dump_directory", std::string)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::gc_trigger_batch_size, "rt_gc_trigger_batch_size", std::size_t)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::ocl_no_shared_context, "rt_ocl_no_shared_context", bool)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::ocl_show_all_devices, "rt_ocl_show_all_devices", bool)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::no_jit_cache_population, "rt_no_jit_cache_population", bool)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::adaptivity_level, "adaptivity_level", int)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::jitopt_iads_relative_threshold, "jitopt_iads_relative_threshold", double)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::jitopt_iads_relative_eviction_threshold, "jitopt_iads_relative_eviction_threshold", double)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::jitopt_iads_relative_threshold_min_data,
                              "jitopt_iads_relative_threshold_min_data",
                              std::size_t)
HIPSYCL_RT_MAKE_SETTING_TRAIT(setting::enable_allocation_tracking, "allocation_tracking", bool)

class settings
{
public:

  template <setting S> typename setting_trait<S>::type get() const {
    if constexpr(S == setting::debug_level){
      return _debug_level;
    } else if constexpr (S == setting::scheduler_type) {
      return _scheduler_type;
    } else if constexpr (S == setting::visibility_mask) {
      return _visibility_mask;
    } else if constexpr (S == setting::dag_req_optimization_depth) {
      return _dag_requirement_optimization_depth;
    } else if constexpr (S == setting::mqe_lane_statistics_max_size) {
      return _mqe_lane_statistics_max_size;
    } else if constexpr (S == setting::mqe_lane_statistics_decay_time_sec) {
      return _mqe_lane_statistics_decay_time_sec;
    } else if constexpr (S == setting::default_selector_behavior) {
      return _default_selector_behavior;
    } else if constexpr (S == setting::hcf_dump_directory) {
      return _hcf_dump_directory;
    } else if constexpr (S == setting::persistent_runtime) {
      return _persistent_runtime;
    } else if constexpr (S == setting::max_cached_nodes) {
      return _max_cached_nodes;
    } else if constexpr(S == setting::sscp_failed_ir_dump_directory) {
      return _sscp_failed_ir_dump_directory;
    } else if constexpr(S == setting::gc_trigger_batch_size) {
      return _gc_trigger_batch_size;
    } else if constexpr(S == setting::ocl_no_shared_context) {
      return _ocl_no_shared_context;
    } else if constexpr(S == setting::ocl_show_all_devices) {
      return _ocl_show_all_devices;
    } else if constexpr(S == setting::no_jit_cache_population) {
      return _no_jit_cache_population;
    } else if constexpr(S == setting::adaptivity_level) {
      return _adaptivity_level;
    } else if constexpr(S == setting::jitopt_iads_relative_threshold) {
      return _jitopt_iads_relative_threshold;
    } else if constexpr(S == setting::jitopt_iads_relative_threshold_min_data) {
      return _jitopt_iads_relative_threshold_min_data;
    } else if constexpr(S == setting::jitopt_iads_relative_eviction_threshold) {
      return _jitopt_iads_relative_eviction_threshold;
    } else if constexpr(S == setting::enable_allocation_tracking) {
      return _enable_allocation_tracking;
    }
    return typename setting_trait<S>::type{};
  }

  template <setting S> void set(typename setting_trait<S>::type value) {
    if constexpr(S == setting::debug_level){
      _debug_level = value;
    } else if constexpr (S == setting::scheduler_type) {
      _scheduler_type = value;
    } else if constexpr (S == setting::visibility_mask) {
      _visibility_mask = value;
    } else if constexpr (S == setting::dag_req_optimization_depth) {
      _dag_requirement_optimization_depth = value;
    } else if constexpr (S == setting::mqe_lane_statistics_max_size) {
      _mqe_lane_statistics_max_size = value;
    } else if constexpr (S == setting::mqe_lane_statistics_decay_time_sec) {
      _mqe_lane_statistics_decay_time_sec = value;
    } else if constexpr (S == setting::default_selector_behavior) {
      _default_selector_behavior = value;
    } else if constexpr (S == setting::hcf_dump_directory) {
      _hcf_dump_directory = value;
    } else if constexpr (S == setting::persistent_runtime) {
      _persistent_runtime = value;
    } else if constexpr (S == setting::max_cached_nodes) {
      _max_cached_nodes = value;
    } else if constexpr(S == setting::sscp_failed_ir_dump_directory) {
      _sscp_failed_ir_dump_directory = value;
    } else if constexpr(S == setting::gc_trigger_batch_size) {
      _gc_trigger_batch_size = value;
    } else if constexpr(S == setting::ocl_no_shared_context) {
      _ocl_no_shared_context = value;
    } else if constexpr(S == setting::ocl_show_all_devices) {
      _ocl_show_all_devices = value;
    } else if constexpr(S == setting::no_jit_cache_population) {
        _no_jit_cache_population = value;
    } else if constexpr(S == setting::adaptivity_level) {
      _adaptivity_level = value;
    }
  }

  settings() {
    int default_debug_level = 2;
#ifdef HIPSYCL_DEBUG_LEVEL
    default_debug_level = HIPSYCL_DEBUG_LEVEL;
#endif
    _debug_level = get_environment_variable_or_default<setting::debug_level>(
        default_debug_level);
    _scheduler_type =
        get_environment_variable_or_default<setting::scheduler_type>(
            scheduler_type::unbound);
    _visibility_mask =
        get_environment_variable_or_default<setting::visibility_mask>(
            visibility_mask_t{});
    _dag_requirement_optimization_depth = get_environment_variable_or_default<
        setting::dag_req_optimization_depth>(10);
    _mqe_lane_statistics_max_size = get_environment_variable_or_default<
        setting::mqe_lane_statistics_max_size>(100);
    _mqe_lane_statistics_decay_time_sec = get_environment_variable_or_default<
        setting::mqe_lane_statistics_decay_time_sec>(10.0);
    _default_selector_behavior =
        get_environment_variable_or_default<setting::default_selector_behavior>(
            default_selector_behavior::strict);
    _hcf_dump_directory =
        get_environment_variable_or_default<setting::hcf_dump_directory>(
            std::string{});
    _persistent_runtime =
        get_environment_variable_or_default<setting::persistent_runtime>(false);
    _max_cached_nodes =
        get_environment_variable_or_default<setting::max_cached_nodes>(100);
    _sscp_failed_ir_dump_directory = get_environment_variable_or_default<
        setting::sscp_failed_ir_dump_directory>(std::string{});
    _gc_trigger_batch_size =
        get_environment_variable_or_default<setting::gc_trigger_batch_size>(128);
    _ocl_no_shared_context =
        get_environment_variable_or_default<setting::ocl_no_shared_context>(false);
    _ocl_show_all_devices =
        get_environment_variable_or_default<setting::ocl_show_all_devices>(false);
    _no_jit_cache_population =
        get_environment_variable_or_default<setting::no_jit_cache_population>(false);
    _adaptivity_level =
        get_environment_variable_or_default<setting::adaptivity_level>(1);
    
    _jitopt_iads_relative_threshold =
        get_environment_variable_or_default<setting::jitopt_iads_relative_threshold>(0.8);
    _jitopt_iads_relative_eviction_threshold =
        get_environment_variable_or_default<setting::jitopt_iads_relative_eviction_threshold>(0.1);
    _jitopt_iads_relative_threshold_min_data =
        get_environment_variable_or_default<setting::jitopt_iads_relative_threshold_min_data>(1024);
    _enable_allocation_tracking =
        get_environment_variable_or_default<setting::enable_allocation_tracking>(false);
  }

private:
  template <setting S, class T>
  T get_environment_variable_or_default(const T &default_value) {
    T out;
    if(try_get_environment_variable(setting_trait<S>::str, out)) {
      return out;
    }
    return default_value;
  }

  int _debug_level;
  scheduler_type _scheduler_type;
  std::size_t _dag_requirement_optimization_depth;
  std::size_t _mqe_lane_statistics_max_size;
  double _mqe_lane_statistics_decay_time_sec;
  default_selector_behavior _default_selector_behavior;
  std::string _hcf_dump_directory;
  bool _persistent_runtime;
  std::size_t _max_cached_nodes;
  std::string _sscp_failed_ir_dump_directory;
  std::size_t _gc_trigger_batch_size;
  visibility_mask_t _visibility_mask;
  bool _ocl_no_shared_context;
  bool _ocl_show_all_devices;
  bool _no_jit_cache_population;
  int _adaptivity_level;
  double _jitopt_iads_relative_threshold;
  double _jitopt_iads_relative_eviction_threshold;
  std::size_t _jitopt_iads_relative_threshold_min_data;
  bool _enable_allocation_tracking;
};

}
}

#endif
