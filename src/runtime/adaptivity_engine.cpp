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
#include "hipSYCL/runtime/adaptivity_engine.hpp"

#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/glue/llvm-sscp/fcall_specialization.hpp"
#include "hipSYCL/runtime/allocation_tracker.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/runtime/runtime_event_handlers.hpp"
#include <cstdint>
#include <limits>


namespace hipsycl {
namespace rt {

namespace {

template<class F>
void access_appdb(common::db::appdb& db, bool needs_write_access, F&& handler) {
  if(needs_write_access) {
    db.read_write_access(handler);
  } else {
    db.read_access(handler);
  }
}

bool has_annotation(const hcf_kernel_info *info, int param_index,
                    hcf_kernel_info::annotation_type annotation) {
  for(auto a : info->get_known_annotations(param_index)) {
    if(a == annotation)
      return true;
  }
  return false;
}

// Estimates whether kernel arguments might be invariant. This also updates
// the statistics in the appdb, so if this function returns true,
// a specialization should be carried out by the calling code in order
// to ensure consistency of the appdb with what is actually happening.
bool is_likely_invariant_argument(common::db::kernel_entry &kernel_entry,
                                  int param_index, std::size_t application_run,
                                  uint64_t current_value) {
  auto& args = kernel_entry.kernel_args;

  const double relative_specialization_threshold =
      application::get_settings().get<setting::jitopt_iads_relative_threshold>();
  const double relative_eviction_threshold =
      application::get_settings().get<setting::jitopt_iads_relative_eviction_threshold>();
  const int relative_trigger_min_size =
      application::get_settings().get<setting::jitopt_iads_relative_threshold_min_data>();

  // In case we find an empty slot, this stores its index.
  int empty_slot = -1;

  for(int i = 0; i < common::db::kernel_arg_entry::max_tracked_values; ++i) {
    // How many times the current kernel parameter was set to
    // args[param_index].common_values[i]

    auto& arg_statistics = args[param_index].common_values[i];
    uint64_t& arg_value_count = arg_statistics.count;
    // Is the argument the same as an argument from a previous submission that we
    // are tracking as commonly used?
    if(arg_value_count > 0 && arg_statistics.value == current_value) {
      // Yep, we've hit it again, increase counter
      ++arg_value_count;
      arg_statistics.last_used = kernel_entry.num_registered_invocations;

      bool& is_already_specialized = args[param_index].was_specialized[i];
      // If we already have specialized in the past, continue to specialize.
      // This prevents performance regressions if the first the value is specialized,
      // then not used for a long while and we are now seeing it again.
      if(is_already_specialized)
        return true;

      double fraction_of_all_invocations = static_cast<double>(arg_value_count) /
               kernel_entry.num_registered_invocations;

      bool can_use_fraction_of_all_invocations =
          (application_run > kernel_entry.first_iads_invocation_run) ||
          (arg_value_count > relative_trigger_min_size);

      if (can_use_fraction_of_all_invocations &&
          (fraction_of_all_invocations > relative_specialization_threshold)) {
        is_already_specialized = true;
        return true;
      } else
        return false;
    } else if(arg_value_count == 0) {
      // Remember that we have hit an unused slot in case we don't find any
      // matches with values that are know to be commonly occuring
      empty_slot = i; 
    }
  }

  auto create_new_entry = [&](int slot_index) {
    common::db::kernel_arg_value_statistics new_arg_entry;
    new_arg_entry.value = current_value;
    new_arg_entry.count = 1;
    new_arg_entry.last_used = kernel_entry.num_registered_invocations;
    args[param_index].common_values[slot_index] = new_arg_entry;
    args[param_index].was_specialized[slot_index] = false;
  };

  // If we arrive here, we are dealing with a value that we have
  // not encountered before.
  if(empty_slot >= 0) {
    // If we have an empty slot, store the current argument in case
    // it gets used a lot by future kernel invocations.
    create_new_entry(empty_slot);
  } else {
    // Try to find an old entry that we can evict.
    int eviction_candidate_slot = -1;
    uint64_t eviction_candidate_last_used_time = std::numeric_limits<uint64_t>::max();

    for(int i = 0; i < common::db::kernel_arg_entry::max_tracked_values; ++i) {
      auto& arg_statistics = args[param_index].common_values[i];
      auto& was_specialized = args[param_index].was_specialized[i];

      if(arg_statistics.last_used < eviction_candidate_last_used_time) {

        double fraction_of_all_invocations =
            static_cast<double>(arg_statistics.count) /
            kernel_entry.num_registered_invocations;

        if (!was_specialized ||
            (fraction_of_all_invocations < relative_eviction_threshold)) {
          auto age = kernel_entry.num_registered_invocations - arg_statistics.last_used;
          if (age > relative_trigger_min_size) {
            
            // Update least-recently-used so that we can find potential entries to evict
            eviction_candidate_slot = i;
            eviction_candidate_last_used_time = arg_statistics.last_used;
          }
        }
      }
    }
    
    if (eviction_candidate_slot >= 0) {
      create_new_entry(eviction_candidate_slot);
    }
  }

  return false;
}

int determine_ptr_alignment(uint64_t ptrval) {
  if(ptrval == 0)
    return 0;
  
#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER) && \
    !defined(__NVCOMPILER)
  // gcc supports __builtin_ctz, but versions prior to 10
  // do not support __has_builtin
  #define ACPP_HAS_BUILTIN_CTZ
#else
  #if __has_builtin(__builtin_ctzll)
    #define ACPP_HAS_BUILTIN_CTZ
  #endif
#endif

#ifdef ACPP_HAS_BUILTIN_CTZ
  uint64_t alignment = 1ull << __builtin_ctzll(ptrval);
  return alignment >= 32 ? 32 : 0;
#else
  return 0;
#endif
}

}

kernel_adaptivity_engine::kernel_adaptivity_engine(
    hcf_object_id hcf_object, std::string_view backend_kernel_name,
    const hcf_kernel_info *kernel_info,
    const glue::jit::cxx_argument_mapper &arg_mapper,
    const range<3> &num_groups, const range<3> &block_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args, std::size_t local_mem_size)
    : _hcf{hcf_object}, _kernel_name{backend_kernel_name},
      _kernel_info{kernel_info}, _arg_mapper{arg_mapper},
      _num_groups{num_groups}, _block_size{block_size}, _args{args},
      _arg_sizes{arg_sizes}, _num_args{num_args},
      _local_mem_size(local_mem_size) {

  _adaptivity_level = application::get_settings().get<setting::adaptivity_level>();
}

kernel_configuration::id_type
kernel_adaptivity_engine::finalize_binary_configuration(
    kernel_configuration &config) {
    
  // At any adaptivity level need to handle function call specializations.
  for (int i = 0; i < _kernel_info->get_num_parameters(); ++i) {
    auto &annotations = _kernel_info->get_known_annotations(i);
    std::size_t arg_size = _kernel_info->get_argument_size(i);
    for (auto annotation : annotations) {
      if (annotation ==
              hcf_kernel_info::annotation_type::fcall_specialized_config &&
          arg_size == sizeof(glue::sscp::fcall_config_kernel_property_t)) {
        glue::sscp::fcall_config_kernel_property_t value;
        std::memcpy(&value, _arg_mapper.get_mapped_args()[i], arg_size);
        config.set_function_call_specialization_config(i, value);
      }
    }
  }

  if(_adaptivity_level > 0) {
    // Enter single-kernel code model
    config.append_base_configuration(
        kernel_base_config_parameter::single_kernel, _kernel_name);

    if(_adaptivity_level >=1 ) {
      // Hard-code group sizes into the JIT binary
      config.set_build_option(kernel_build_option::known_group_size_x,
                              _block_size[0]);
      config.set_build_option(kernel_build_option::known_group_size_y,
                              _block_size[1]);
      config.set_build_option(kernel_build_option::known_group_size_z,
                              _block_size[2]);
    }

    // Try to optimize size_t -> i32 for queries if those fit in int
    if(_adaptivity_level >=2 ) {
      auto global_size = _num_groups * _block_size;
      auto int_max = std::numeric_limits<int>::max();
      if (global_size[0] * global_size[1] * global_size[2] < int_max)
        config.set_build_flag(kernel_build_flag::global_sizes_fit_in_int);
    }
    // Hard-code local memory size into the JIT binary
    config.set_build_option(kernel_build_option::known_local_mem_size,
                            _local_mem_size);
    // Handle kernel parameter optimization hints
    if(_adaptivity_level >= 3) {
      for(int i = 0; i < _kernel_info->get_num_parameters(); ++i) {
        std::size_t arg_size = _kernel_info->get_argument_size(i);
        if (has_annotation(_kernel_info, i,
                          hcf_kernel_info::annotation_type::specialized) &&
            arg_size <= sizeof(uint64_t)) {
          uint64_t buffer_value = 0;
          std::memcpy(&buffer_value, _arg_mapper.get_mapped_args()[i], arg_size);
          config.set_specialized_kernel_argument(i, buffer_value);
        }

        if (_kernel_info->get_argument_type(i) ==
            hcf_kernel_info::argument_type::pointer) {
          if (has_annotation(_kernel_info, i,
                            hcf_kernel_info::annotation_type::noalias)) {
            config.set_kernel_param_flag(i, kernel_param_flag::noalias);
          }
        }
      }
    }

    // Handle auto alignment specialization
    if(_adaptivity_level >= 4) {
      for(int i = 0; i < _kernel_info->get_num_parameters(); ++i) {
        std::size_t arg_size = _kernel_info->get_argument_size(i);
        if (_kernel_info->get_argument_type(i) == hcf_kernel_info::argument_type::pointer) {
          uint64_t buffer = 0;
          std::memcpy(&buffer, _arg_mapper.get_mapped_args()[i],
                      _kernel_info->get_argument_size(i));

          int alignment = determine_ptr_alignment(buffer);
          if(alignment > 0) {
            HIPSYCL_DEBUG_INFO
                << "adaptivity_engine: Inferred pointer alignment of "
                << alignment << " for kernel argument " << i << std::endl;
            config.set_known_alignment(i, alignment);
          }
        }
      }
    }

    if(application::get_settings().get<setting::enable_allocation_tracking>() &&
      _adaptivity_level >= 5) {
      // Detect whether pointer arguments qualify for NoAlias/restrict semantics.
      // This is achieved by determining the base of the allocations for all pointer
      // kernel arguments, and checking whether there are other pointer arguments
      // from the same allocation.
      constexpr int max_allocations = 32;
      uint64_t allocation_base_addresses [max_allocations] = {};
      bool allocations_exceeded = false;
      for(int alloc_index = 0, i = 0; i < _kernel_info->get_num_parameters(); ++i) {
        if(_kernel_info->get_argument_type(i) == hcf_kernel_info::argument_type::pointer) {
          auto arg_size = _kernel_info->get_argument_size(i);
          if(arg_size == sizeof(void*)) {
            void* ptr_arg;
            std::memcpy(&ptr_arg, _arg_mapper.get_mapped_args()[i], arg_size);
            if (ptr_arg) {
              allocation_info ainfo;
              uint64_t allocation_base;
              if(allocation_tracker::query_allocation(ptr_arg, ainfo, allocation_base)) {
                allocation_base_addresses[alloc_index] = allocation_base;
              }
            }
          }
          ++alloc_index;
          if (alloc_index >= max_allocations) {
            allocations_exceeded = true;
            break;
          }
        }
      }
      if (!allocations_exceeded) {
        for (int alloc_index = 0, i = 0; i < _kernel_info->get_num_parameters();
            ++i) {
          if (_kernel_info->get_argument_type(i) ==
              hcf_kernel_info::argument_type::pointer) {
            if (allocation_base_addresses[alloc_index] != 0) {
              bool argument_might_alias = false;
              for (int k = 0; k < max_allocations; ++k) {
                if (k != alloc_index) {
                  if (allocation_base_addresses[alloc_index] ==
                      allocation_base_addresses[k]) {
                    argument_might_alias = true;
                    break;
                  }
                }
              }
              if (!argument_might_alias) {
                HIPSYCL_DEBUG_INFO << "adaptivity_engine: Inferred noalias "
                                      "pointer semantics for kernel argument "
                                  << i << std::endl;
                config.set_kernel_param_flag(i, kernel_param_flag::noalias);
              }
            }
            ++alloc_index;
          }
        }
      }
    }
  }
  
  if(_adaptivity_level >= 6) {

    auto base_id = config.generate_id();
    
    // Automatic application of specialization constants by detecting
    // invariant kernel arguments
    auto& appdb = common::filesystem::persistent_storage::get().get_this_app_db();
    appdb.read_write_access([&](common::db::appdb_data& data){
      
      auto& kernel_entry = data.kernels[base_id];
      if (kernel_entry.first_iads_invocation_run ==
          common::db::kernel_entry::no_usage) {
        kernel_entry.first_iads_invocation_run = data.content_version;
      }
      ++kernel_entry.num_registered_invocations;

      std::size_t num_kernel_args = _kernel_info->get_num_parameters();
      if(kernel_entry.kernel_args.size() != num_kernel_args)
        kernel_entry.kernel_args.resize(num_kernel_args);

      auto process_kernel_arg = [&](int i) {
        uint64_t arg_value = 0;
        std::memcpy(&arg_value, _arg_mapper.get_mapped_args()[i],
                    _kernel_info->get_argument_size(i));
        if (_kernel_info->get_argument_type(i) !=
                hcf_kernel_info::argument_type::pointer &&
            is_likely_invariant_argument(kernel_entry, i, data.content_version,
                                         arg_value) &&
            !has_annotation(_kernel_info, i,
                            hcf_kernel_info::annotation_type::specialized)) {
          HIPSYCL_DEBUG_INFO << "adaptivity_engine: Kernel argument " << i
                             << " is invariant or common, specializing."
                             << std::endl;
          config.set_specialized_kernel_argument(i, arg_value);
        } else {
          HIPSYCL_DEBUG_INFO << "adaptivity_engine: Not specializing kernel argument " << i
                             << std::endl;
        }
      };

      if(!kernel_entry.retained_argument_indices.empty()) {
        for(auto arg_index : kernel_entry.retained_argument_indices) {
          process_kernel_arg(arg_index);
        }
      } else {
        for(int i = 0; i < num_kernel_args; ++i) {
          process_kernel_arg(i);
        }
      }
    });
  }

  return config.generate_id();
}

std::string kernel_adaptivity_engine::select_image_and_kernels(
    std::vector<std::string> *kernel_names_out) {
  if(_adaptivity_level > 0) {
    *kernel_names_out = std::vector{std::string{_kernel_name}};

    std::vector<std::string> all_kernels_in_image;
    return  glue::jit::select_image(_kernel_info, &all_kernels_in_image);
  } else {
    return glue::jit::select_image(_kernel_info, kernel_names_out);
  }
}

}
}