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
#ifndef HIPSYCL_GLUE_JIT_HPP
#define HIPSYCL_GLUE_JIT_HPP

#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/small_map.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "jit-reflection/reflection_map.hpp"
#include <cstddef>
#include <vector>
#include <atomic>
#include <fstream>
#include <string>

namespace hipsycl {
namespace glue {
namespace jit {


// Map arguments passed to a kernel function on the C++ level to
// arguments of the kernel function. This is necessary because
// C++ arguments (especially structs) may have been decomposed
// into their elements in the kernel function prototype.
class cxx_argument_mapper {
public:
  cxx_argument_mapper() = default;
  cxx_argument_mapper(const rt::hcf_kernel_info &kernel_info, void **args,
                      const std::size_t *arg_sizes, std::size_t num_args) {
    construct_mapping(kernel_info, args, arg_sizes, num_args);
  }

  void construct_mapping(const rt::hcf_kernel_info &kernel_info, void **args,
                         const std::size_t *arg_sizes, std::size_t num_args) {

    std::size_t num_params = kernel_info.get_num_parameters();

    _mapped_data.clear();
    _mapped_sizes.clear();
    _mapped_data.reserve(num_params);
    _mapped_sizes.reserve(num_params);

    for (int i = 0; i < num_params; ++i) {
      std::size_t arg_size = kernel_info.get_argument_size(i);
      std::size_t arg_offset = kernel_info.get_argument_offset(i);
      std::size_t arg_original_index =
          kernel_info.get_original_argument_index(i);

      assert(arg_original_index < num_args);

      void *data_ptr = add_offset(args[arg_original_index], arg_offset);

      if (!data_ptr)
        return;

      _mapped_data.push_back(data_ptr);
      _mapped_sizes.push_back(arg_size);
    }

    _mapping_result = true;
  }

  bool mapping_available() const {
    return _mapping_result;
  }

  void** get_mapped_args() {
    return _mapped_data.data();
  }

  void* const* get_mapped_args() const {
    return _mapped_data.data();
  }

  const std::size_t* get_mapped_arg_sizes() const {
    return _mapped_sizes.data();
  }

  std::size_t get_mapped_num_args() const {
    return _mapped_data.size();
  }

  void apply_dead_argument_elimination_mask(
      const std::vector<int> &retained_argument_indices) {
    assert(retained_argument_indices.size() <= _mapped_data.size());

    for(int i = 0; i < retained_argument_indices.size(); ++i) {
      assert(retained_argument_indices[i] < _mapped_data.size());
      assert(retained_argument_indices[i] >= i);
      _mapped_data[i] = _mapped_data[retained_argument_indices[i]];
      _mapped_sizes[i] = _mapped_sizes[retained_argument_indices[i]];
    }
    _mapped_data.erase(_mapped_data.begin() + retained_argument_indices.size(),
                       _mapped_data.end());
    _mapped_sizes.erase(_mapped_sizes.begin() +
                            retained_argument_indices.size(),
                        _mapped_sizes.end());
  }

private:
  void *add_offset(void *ptr, std::size_t offset_bytes) const {
    return static_cast<void *>(static_cast<char *>(ptr) + offset_bytes);
  }

  bool _mapping_result = false;
  std::vector<void*> _mapped_data;
  std::vector<std::size_t> _mapped_sizes; 
};

class default_llvm_image_selector {
public:
  std::string operator()(const rt::hcf_kernel_info* kernel_info) const {
    return "llvm-ir.global";
  }
};

template <class ImageSelector = default_llvm_image_selector>
std::string select_image(const rt::hcf_kernel_info* kernel_info,
                         std::vector<std::string>* all_kernels_in_image_out,
                         const ImageSelector &sel = ImageSelector{}) {
  std::string image_name = sel(kernel_info);

  const rt::hcf_image_info *selected_image_info =
      rt::hcf_cache::get().get_image_info(kernel_info->get_hcf_object_id(),
                                          image_name);

  if (!selected_image_info)
    return {};

  if(all_kernels_in_image_out) {
    *all_kernels_in_image_out = selected_image_info->get_contained_kernels();
  }
  return image_name;
}

using symbol_list_t = compiler::LLVMToBackendTranslator::SymbolListType;

class runtime_linker {
  
public:
  using resolver = compiler::LLVMToBackendTranslator::ExternalSymbolResolver;
  using llvm_module_id = resolver::LLVMModuleId;


  runtime_linker(compiler::LLVMToBackendTranslator *translator,
                            const symbol_list_t &imported_symbol_names) {

    auto symbol_mapper = [this](const symbol_list_t& sl){ return this->map_smybols(sl); };

    auto bitcode_retriever = [this](llvm_module_id id,
                                    symbol_list_t &imported_symbols) {
      return this->retrieve_bitcode(id, imported_symbols);
    };

    translator->provideExternalSymbolResolver(
        resolver{symbol_mapper, bitcode_retriever, imported_symbol_names});
  }


private:

  std::vector<llvm_module_id> map_smybols(const symbol_list_t& sym_list) {
    std::vector<llvm_module_id> ir_modules_to_link;

    auto candidate_selector = [&, this](const std::string &symbol_name,
            const rt::hcf_cache::symbol_resolver_list &images) {
      for (const auto &img : images) {
        // Always attempt to link with global LLVM IR for now
        if (img.image_node->node_id == "llvm-ir.global") {
          _image_node_to_hcf_map[img.image_node] = img.hcf_id;
          ir_modules_to_link.push_back(
              reinterpret_cast<llvm_module_id>(img.image_node));
        } else {

          HIPSYCL_DEBUG_INFO << "jit::setup_linking: Discarding image "
                            << img.image_node->node_id << " @"
                            << img.image_node << " from HCF " << img.hcf_id
                            << "\n";
        }
      }
    };

    rt::hcf_cache::get().symbol_lookup(
        sym_list, candidate_selector);

    return ir_modules_to_link;
  }

  std::string retrieve_bitcode(llvm_module_id id, symbol_list_t& imported_symbols) const {

    const auto* hcf_image_node = reinterpret_cast<common::hcf_container::node*>(id);

    assert(_image_node_to_hcf_map.contains(hcf_image_node));

    auto v = _image_node_to_hcf_map.find(hcf_image_node);
    
    if(v == _image_node_to_hcf_map.end())
      return {};
    
    rt::hcf_object_id hcf_id = v->second;
    imported_symbols = hcf_image_node->get_as_list("imported-symbols");

    std::string bitcode;
    rt::hcf_cache::get().get_hcf(hcf_id)->get_binary_attachment(hcf_image_node, bitcode);

    return bitcode;
  }

  // This is used to map images to the owning HCF object ids.
  common::small_map<const common::hcf_container::node *, rt::hcf_object_id>
        _image_node_to_hcf_map;

};

inline rt::result compile(compiler::LLVMToBackendTranslator *translator,
                          const std::string &source,
                          const rt::kernel_configuration &config,
                          const symbol_list_t& imported_symbol_names,
                          const reflection_map& refl_map,
                          std::string &output) {

  assert(translator);
  runtime_linker configure_linker {translator, imported_symbol_names};

  // Apply configuration
  if(translator->getKernels().size() == 1) {
    // Currently we only can specialize kernel arguments for the 
    // single-kernel code object model
    for(const auto& entry : config.specialized_arguments()) {
      translator->specializeKernelArgument(translator->getKernels().front(),
                                          entry.first, &entry.second);
    }

    for(const auto& entry : config.known_alignments()) {
      translator->setKnownPtrParamAlignment(translator->getKernels().front(),
                                            entry.first, entry.second);
    }
  }
  for(const auto& entry : config.function_call_specialization_config()) {
    auto& config = entry.value->function_call_map;
    for(const auto& call_specialization : config) {
      translator->specializeFunctionCalls(call_specialization.first,
                                          call_specialization.second, false);
    }
  }

  for(const auto& option : config.build_options()) {
    std::string option_name = rt::to_string(option.first);
    std::string option_value =
        option.second.int_value.has_value()
            ? std::to_string(option.second.int_value.value())
            : option.second.string_value.value();
    
    translator->setBuildOption(option_name, option_value);
  }

  for(const auto& flag : config.build_flags()) {
    translator->setBuildFlag(rt::to_string(flag));
  }

  // Set up JIT-time reflection for the code we compile
  for(const auto& KV : refl_map) {
    translator->setReflectionField(KV.first, KV.second);
  }

  // Transform code
  if(!translator->fullTransformation(source, output)) {
    // In case of failure, if a dump directory for IR is set,
    // dump the IR
    auto failure_dump_directory =
        rt::application::get_settings()
            .get<rt::setting::sscp_failed_ir_dump_directory>();
            
    if(!failure_dump_directory.empty()) {
      static std::atomic<std::size_t> failure_index = 0;
      std::string filename = common::filesystem::join_path(
          failure_dump_directory,
          "failed_ir_" + std::to_string(failure_index) + ".bc");
      
      std::ofstream out{filename.c_str(), std::ios::trunc|std::ios::binary};
      if(out.is_open()) {
        const std::string& failed_ir = translator->getFailedIR();
        out.write(failed_ir.c_str(), failed_ir.size());
      }

      ++failure_index;
    }
    
    return rt::make_error(__acpp_here(),
                      rt::error_info{"jit::compile: Encountered errors:\n" +
                                 translator->getErrorLogAsString()});
  }

  return rt::make_success();
}


inline rt::result compile(compiler::LLVMToBackendTranslator* translator,
                          const common::hcf_container* hcf,
                          const std::string& image_name,
                          const rt::kernel_configuration &config,
                          const reflection_map& refl_map,
                          std::string &output) {
  assert(hcf);
  assert(hcf->root_node());


  auto images_node = hcf->root_node()->get_subnode("images");
  if(!images_node) {
    return rt::make_error(
        __acpp_here(),
        rt::error_info{
            "jit::compile: Invalid HCF, no node named 'images' was found"});
  }

  auto target_image_node = images_node->get_subnode(image_name);
  if(!target_image_node) {
    return rt::make_error(__acpp_here(),
                          rt::error_info{"jit::compile: Requested image " +
                                         image_name +
                                         " was not defined in HCF"});
  }

  if(!target_image_node->has_binary_data_attached()) {
    return rt::make_error(
        __acpp_here(),
        rt::error_info{"jit::compile: Image " + image_name +
                       " was defined in HCF without data"});
  }
  std::string source;
  if(!hcf->get_binary_attachment(target_image_node, source)) {
    return rt::make_error(
        __acpp_here(),
        rt::error_info{
            "jit::compile: Could not extract binary data for HCF image " +
            image_name});
  }

  symbol_list_t imported_symbol_names =
      target_image_node->get_as_list("imported-symbols");

  return compile(translator, source, config, imported_symbol_names, refl_map,
                 output);
}

inline rt::result compile(compiler::LLVMToBackendTranslator* translator,
                          rt::hcf_object_id hcf_object,
                          const std::string& image_name,
                          const rt::kernel_configuration &config,
                          const reflection_map& refl_map,
                          std::string &output) {
  const common::hcf_container* hcf = rt::hcf_cache::get().get_hcf(hcf_object);
  if(!hcf) {
    return rt::make_error(
        __acpp_here(),
        rt::error_info{"jit::compile: Could not obtain HCF object"});
  }

  return compile(translator, hcf, image_name, config,
                 refl_map, output);
}

namespace dead_argument_elimination {
// Compiles with dead-argument-elimination for the kernels, and saves
// the retained argument mask in the appdb. This only works for single-kernel
// compilations!
inline rt::result compile_kernel(compiler::LLVMToBackendTranslator *translator,
                                 rt::hcf_object_id hcf_object,
                                 const std::string &image_name,
                                 const rt::kernel_configuration &config,
                                 rt::kernel_configuration::id_type binary_id,
                                 const reflection_map &refl_map,
                                 std::string &output) {

  assert(translator->getKernels().size() == 1);

  rt::result err = rt::make_success();

  common::filesystem::persistent_storage::get()
      .get_this_app_db()
      .read_write_access([&](common::db::appdb_data &appdb) {
        std::vector<int> *retained_args =
            &(appdb.kernels[binary_id].retained_argument_indices);

        translator->enableDeadArgumentElminiation(translator->getKernels()[0],
                                                  retained_args);
        err = compile(translator, hcf_object, image_name, config, refl_map,
                      output);
      });

  return err;
}

inline std::vector<int>
retrieve_retained_arguments_mask(rt::kernel_configuration::id_type binary_id) {
  return common::filesystem::persistent_storage::get().get_this_app_db().read_access(
      [&](const common::db::appdb_data &appdb) {
        auto it = appdb.kernels.find(binary_id);
        if(it != appdb.kernels.end()) {
          return it->second.retained_argument_indices;
        } else {
          return std::vector<int>{};
        }
      });
}
}

}
}
}

#endif
