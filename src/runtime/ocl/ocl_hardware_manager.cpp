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
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/ocl/ocl_allocator.hpp"
#include "hipSYCL/runtime/ocl/ocl_hardware_manager.hpp"
#include "hipSYCL/runtime/settings.hpp"

#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <cstddef>
#include <array>
#include <optional>


namespace hipsycl {
namespace rt {

namespace {

// These commonly encountered OpenCL platforms will be silently hidden due to
// known incompatibilities. The hardware they support is eithers supported via other backends,
// or not supported at all (e.g. FPGA). All other platforms will be hidden with a warning if
// they fail feature support queries.
constexpr std::array incompatible_ocl_platforms = {
    "NVIDIA CUDA", "Intel(R) FPGA Emulation Platform for OpenCL(TM)",
    "AMD Accelerated Parallel Processing"};

template<int Query, class ResultT>
ResultT info_query(const cl::Device& dev) {
  ResultT r{};
  cl_int err = dev.getInfo(Query, &r);
  if(err != CL_SUCCESS) {
    register_error(
          __acpp_here(),
          error_info{"ocl_hardware_context: Could not obtain device info",
                    error_code{"CL", err}});
  }
  return r;
}

template<int Query, class ResultT>
ResultT platform_info_query(const cl::Device& dev) {
  cl_platform_id platform = info_query<CL_DEVICE_PLATFORM, cl_platform_id>(dev);
  cl::Platform p{platform};

  ResultT r{};
  cl_int err = p.getInfo(Query, &r);
  if(err != CL_SUCCESS) {
    register_error(
          __acpp_here(),
          error_info{"ocl_hardware_context: Could not obtain platform info",
                    error_code{"CL", err}});
  }
  return r;
}

bool parse_ocl_version_string(const std::string& s, int& major_version_out) {
  const std::string identifier = "OpenCL ";
  const auto pos = s.find(identifier);
  if(pos != 0)
    return false;
  std::string version_substring = s.substr(identifier.length());
  auto dot_pos = version_substring.find(".");
  if(dot_pos != std::string::npos) {
    major_version_out = std::stoi(version_substring.substr(0, dot_pos));
    return true;
  }

  return false;
}

bool should_include_platform(const std::string& platform_name, const cl::Platform& p) {
  const bool show_all_devices = application::get_settings().get<setting::ocl_show_all_devices>();
  if(show_all_devices)
    return true;

  for(const auto& incompatible_platform : incompatible_ocl_platforms) {
    if(platform_name == incompatible_platform) {
      HIPSYCL_DEBUG_INFO << "ocl_hardware_manager: Hiding platform '"
                         << platform_name
                         << "' because it has known incompatibilities. Set "
                            "ACPP_RT_OCL_SHOW_ALL_DEVICES=1 to override."
                         << std::endl;
      return false;
    }
  }

  std::string ocl_version;
  cl_int err = p.getInfo(CL_PLATFORM_VERSION, &ocl_version);
  if (err != CL_SUCCESS) {
    print_warning(__acpp_here(),
                  error_info{"ocl_hardware_manager: Could not retrieve OpenCL "
                             "version for platform " +
                                 platform_name,
                             error_code{"CL", err}});
    return false;
  } else {
    int ocl_version_major = 0;
    if (!parse_ocl_version_string(ocl_version, ocl_version_major)) {
      HIPSYCL_DEBUG_WARNING
          << "ocl_hardware_manager: Could not parse OpenCL version string "
              "of platform '"
          << platform_name
          << "'; hiding platform. Set ACPP_RT_OCL_SHOW_ALL_DEVICES=1 "
              "to override. (OpenCL version string was: '"
          << ocl_version << "')" << std::endl;
      return false;
    } else {
      if (ocl_version_major < 3) {
        HIPSYCL_DEBUG_WARNING
            << "ocl_hardware_manager: Platform '"
            << platform_name
            << "' does not support OpenCL 3.0; hiding platform. Set "
               "ACPP_RT_OCL_SHOW_ALL_DEVICES=1 "
               "to override. (OpenCL version string was: '"
            << ocl_version << "')" << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool should_include_device(const std::string& dev_name, const cl::Device& dev) {
  const bool show_all_devices = application::get_settings().get<setting::ocl_show_all_devices>();
  if(show_all_devices)
    return true;

  if(info_query<CL_DEVICE_IL_VERSION, std::string>(dev).find("SPIR-V") ==
           std::string::npos) {
    HIPSYCL_DEBUG_WARNING
            << "ocl_hardware_manager: OpenCL device '"
            << dev_name
            << "' does not support SPIR-V; hiding device. Set "
               "ACPP_RT_OCL_SHOW_ALL_DEVICES=1 "
               "to override." << std::endl;
    return false;
  }

  cl_device_svm_capabilities cap =
      info_query<CL_DEVICE_SVM_CAPABILITIES, cl_device_svm_capabilities>(dev);

  bool has_usm_extension = info_query<CL_DEVICE_EXTENSIONS, std::string>(dev).find("cl_intel_unified_shared_memory") != std::string::npos;
  bool has_system_svm = cap & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;

  if(!has_usm_extension && !has_system_svm) {
    HIPSYCL_DEBUG_WARNING << "ocl_hardware_manager: OpenCL device '" << dev_name
                          << "' does not support USM extensions or system SVM. "
                             "Allocations are not possible; hiding device. Set "
                             "ACPP_RT_OCL_SHOW_ALL_DEVICES=1 "
                             "to override."
                          << std::endl;
    return false;
  }

  return true;
}

}

ocl_hardware_context::ocl_hardware_context(const cl::Device &dev,
                                           const cl::Context &ctx, int dev_id,
                                           int platform_id)
    : _dev_id{dev_id}, _platform_id{platform_id}, _ctx{ctx}, _dev{dev},
      _alloc{}, _has_intel_extension_profile{false} {
  std::string platform_name =
      platform_info_query<CL_PLATFORM_NAME, std::string>(_dev);

  if(platform_name == "Intel(R) OpenCL Graphics" || platform_name == "Intel(R) OpenCL")
    _has_intel_extension_profile = true;
}

bool ocl_hardware_context::is_cpu() const {
  return info_query<CL_DEVICE_TYPE, cl_device_type>(_dev) & CL_DEVICE_TYPE_CPU;
}

bool ocl_hardware_context::is_gpu() const {
  return info_query<CL_DEVICE_TYPE, cl_device_type>(_dev) & CL_DEVICE_TYPE_GPU;
}

std::size_t ocl_hardware_context::get_max_kernel_concurrency() const {
  return 1; // TODO
}

std::size_t ocl_hardware_context::get_max_memcpy_concurrency() const {
  return 1; // TODO
}

std::string ocl_hardware_context::get_device_name() const {
  return info_query<CL_DEVICE_NAME, std::string>(_dev);
}

std::string ocl_hardware_context::get_vendor_name() const {
  return info_query<CL_DEVICE_NAME, std::string>(_dev);
}

std::string ocl_hardware_context::get_device_arch() const {
  return "OpenCL " + info_query<CL_DEVICE_OPENCL_C_VERSION, std::string>(_dev);
}

bool ocl_hardware_context::has_intel_extension_profile() const {
  return _has_intel_extension_profile;
}

bool ocl_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return info_query<CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type>(
               _dev) == CL_GLOBAL;
    break;
  case device_support_aspect::host_unified_memory:
    return false;// deprecated in OpenCL 2.0, don't bother with it
    break;
  case device_support_aspect::error_correction:
    return info_query<CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool>(_dev);
    break;
  case device_support_aspect::global_mem_cache:
    return info_query<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                      cl_device_mem_cache_type>(_dev) != CL_NONE;
    break;
  case device_support_aspect::global_mem_cache_read_only:
    return info_query<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                      cl_device_mem_cache_type>(_dev) != CL_READ_ONLY_CACHE;
    break;
  case device_support_aspect::global_mem_cache_read_write:
    return info_query<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                      cl_device_mem_cache_type>(_dev) != CL_READ_WRITE_CACHE;
    break;
  case device_support_aspect::images:
    return false;
    break;
  case device_support_aspect::little_endian:
    return info_query<CL_DEVICE_ENDIAN_LITTLE, cl_bool>(_dev);
    break;
  case device_support_aspect::sub_group_independent_forward_progress:
    return info_query<CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
                      cl_bool>(_dev);
    break;
  case device_support_aspect::usm_device_allocations:
    return this->_usm_provider->has_usm_device_allocations();
    break;
  case device_support_aspect::usm_host_allocations:
    return this->_usm_provider->has_usm_host_allocations();
    break;
  case device_support_aspect::usm_atomic_host_allocations:
    return this->_usm_provider->has_usm_atomic_host_allocations();
    break;
  case device_support_aspect::usm_shared_allocations:
    return this->_usm_provider->has_usm_shared_allocations();
    break;
  case device_support_aspect::usm_atomic_shared_allocations:
    return this->_usm_provider->has_usm_atomic_shared_allocations();
    break;
  case device_support_aspect::usm_system_allocations:
    return this->_usm_provider->has_usm_system_allocations();
    break;
  case device_support_aspect::execution_timestamps:
    return false;
    break;
  case device_support_aspect::sscp_kernels:
#ifdef HIPSYCL_WITH_SSCP_COMPILER
    return info_query<CL_DEVICE_IL_VERSION, std::string>(_dev).find("SPIR-V") !=
           std::string::npos;
#else
    return false;
#endif
    break;
  case device_support_aspect::work_item_independent_forward_progress:
    return false;
    break;
  }
  assert(false && "Unknown device aspect");
  std::terminate();
}

std::size_t ocl_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint>(_dev));
    break;
  case device_uint_property::max_global_size0:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_ITEM_SIZES, std::vector<std::size_t>>(
            _dev)[0]);
    break;
  case device_uint_property::max_global_size1:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_ITEM_SIZES, std::vector<std::size_t>>(
            _dev)[1]);
    break;
  case device_uint_property::max_global_size2:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_ITEM_SIZES, std::vector<std::size_t>>(
            _dev)[2]);
    break;
  case device_uint_property::max_group_size0:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_GROUP_SIZE, std::size_t>(_dev));
    break;
  case device_uint_property::max_group_size1:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_GROUP_SIZE, std::size_t>(_dev));
    break;
  case device_uint_property::max_group_size2:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_GROUP_SIZE, std::size_t>(_dev));
    break;
  case device_uint_property::max_group_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_WORK_GROUP_SIZE, std::size_t>(_dev));
    break;
  case device_uint_property::max_num_sub_groups:
    return std::max(static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_NUM_SUB_GROUPS, cl_uint>(_dev)), std::size_t{1});
    break;
  case device_uint_property::needs_dimension_flip:
    return true;
    break;
  case device_uint_property::preferred_vector_width_char:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint>(_dev));
    break;
  case device_uint_property::preferred_vector_width_double:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint>(_dev));
    break;
  case device_uint_property::preferred_vector_width_float:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint>(_dev));
    break;
  case device_uint_property::preferred_vector_width_half:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, cl_uint>(_dev));
    break;
  case device_uint_property::preferred_vector_width_int:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint>(_dev));
    break;
  case device_uint_property::preferred_vector_width_long:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint>(_dev));
    break;
  case device_uint_property::preferred_vector_width_short:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_char:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_double:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_float:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_half:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_int:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_long:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint>(_dev));
    break;
  case device_uint_property::native_vector_width_short:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, cl_uint>(_dev));
    break;
  case device_uint_property::max_clock_speed:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint>(_dev));
    break;
  case device_uint_property::max_malloc_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong>(_dev));
    break;
  case device_uint_property::address_bits:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_ADDRESS_BITS, cl_uint>(_dev));
    break;
  case device_uint_property::max_read_image_args:
    return 0;
    break;
  case device_uint_property::max_write_image_args:
    return 0;
    break;
  case device_uint_property::image2d_max_width:
    return 0;
    break;
  case device_uint_property::image2d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_width:
    return 0;
    break;
  case device_uint_property::image3d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_depth:
    return 0;
    break;
  case device_uint_property::image_max_buffer_size:
    return 0;
    break;
  case device_uint_property::image_max_array_size:
    return 0;
    break;
  case device_uint_property::max_samplers:
    return 0;
    break;
  case device_uint_property::max_parameter_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_PARAMETER_SIZE, std::size_t>(_dev));
    break;
  case device_uint_property::mem_base_addr_align:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint>(_dev));
    break;
  case device_uint_property::global_mem_cache_line_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint>(_dev));
    break;
  case device_uint_property::global_mem_cache_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong>(_dev));
    break;
  case device_uint_property::global_mem_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong>(_dev));
    break;
  case device_uint_property::max_constant_buffer_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong>(_dev));
    break;
  case device_uint_property::max_constant_args:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint>(_dev));
    break;
  case device_uint_property::local_mem_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong>(_dev));
    break;
  case device_uint_property::printf_buffer_size:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_PRINTF_BUFFER_SIZE, std::size_t>(_dev));
    break;
  case device_uint_property::partition_max_sub_devices:
    return 0;
    break;
  case device_uint_property::vendor_id:
    return static_cast<std::size_t>(
        info_query<CL_DEVICE_VENDOR_ID, cl_uint>(_dev));
    break;
  case device_uint_property::architecture:
    // TODO
    return 0;
    break;
  case device_uint_property::backend_id:
    return static_cast<int>(backend_id::ocl);
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::vector<std::size_t>
ocl_hardware_context::get_property(device_uint_list_property prop) const {
  switch (prop) {
  case device_uint_list_property::sub_group_sizes:
    // TODO - there does not seem to be a direct query for this.
    // So we return all power of two values between 1 and
    // the max workgroup size. The latter corresponds to the case when there is
    // only a single subgroup for the whole work group.
    // Unfortunately there does not seem to be a good way to get a min bound.
    //
    // This is a heuristic that returns all possible subgroup sizes, but may
    // also return values that are not actually possible as subgroups.
    std::size_t max_num_sub_groups =
        get_property(device_uint_property::max_num_sub_groups);
    std::size_t max_group_size =
        get_property(device_uint_property::max_group_size);
    
    auto min_bound = 1;
    auto max_bound = std::max(max_num_sub_groups, max_group_size);

    std::vector<std::size_t> result;

    for(std::size_t i = 1; i < max_bound; i *= 2) {
      if(i >= min_bound)
        result.push_back(i);
    }

    return result;
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::string ocl_hardware_context::get_driver_version() const {
  return info_query<CL_DRIVER_VERSION, std::string>(_dev);
}

std::string ocl_hardware_context::get_profile() const {
  return info_query<CL_DEVICE_PROFILE, std::string>(_dev);
}

ocl_hardware_context::~ocl_hardware_context() {}

ocl_allocator * ocl_hardware_context::get_allocator() {
  return &_alloc;
}

ocl_usm* ocl_hardware_context::get_usm_provider() {
  return _usm_provider.get();
}

int ocl_hardware_context::get_platform_id() const {
  return _platform_id;
}

int ocl_hardware_context::get_device_id() const {
  return _dev_id;
}

cl::Device ocl_hardware_context::get_cl_device() const {
  return _dev;
}

cl::Context ocl_hardware_context::get_cl_context() const {
  return _ctx;
}

void ocl_hardware_context::init_allocator(ocl_hardware_manager *mgr) {
  _usm_provider = ocl_usm::from_intel_extension(mgr, _dev_id);
  if(!_usm_provider->is_available()) {
    // Try SVM fine-grained system as an alternative
    _usm_provider = ocl_usm::from_fine_grained_system_svm(mgr, _dev_id);
    if(_usm_provider->is_available()) {
      HIPSYCL_DEBUG_WARNING << "OpenCL device " << get_device_name()
                            << " does not support Intel USM extensions; "
                               "falling back to fine-grained system SVM. USM "
                               "pointer info queries have limited support."
                            << std::endl;
    }
  }
  if(!_usm_provider->is_available()) {
    HIPSYCL_DEBUG_WARNING << "OpenCL device " << get_device_name()
                          << " does not have a valid USM provider. Memory "
                             "allocations are not possible on that device."
                          << std::endl;
  }
  device_id dev{
      backend_descriptor{hardware_platform::ocl, api_platform::ocl},
      _dev_id};
  _alloc = ocl_allocator{dev, _usm_provider.get()};
}

std::size_t ocl_hardware_context::get_platform_index() const {
  return static_cast<std::size_t>(_platform_id);
}

std::size_t ocl_hardware_manager::get_num_platforms() const {
  return _platforms.size();
}

ocl_hardware_manager::ocl_hardware_manager()
: _hw_platform{hardware_platform::ocl} {
  const auto visibility_mask =
      application::get_settings().get<setting::visibility_mask>();
  const bool no_shared_contexts =
      application::get_settings().get<setting::ocl_no_shared_context>();

  std::vector<cl::Platform> platforms;
  cl_int err = cl::Platform::get(&platforms);
  if(err != CL_SUCCESS) {
    print_warning(
          __acpp_here(),
          error_info{"ocl_hardware_manager: Could not obtain platform list",
                    error_code{"CL", err}});
    platforms.clear();
  }

  int global_device_index = 0;
  for(const auto& p : platforms) {
    
    std::string platform_name;
    err = p.getInfo(CL_PLATFORM_NAME, &platform_name);
    if(err != CL_SUCCESS) {
      print_warning(
          __acpp_here(),
          error_info{"ocl_hardware_manager: Could not retrieve platform name",
                    error_code{"CL", err}});
    }

    if(should_include_platform(platform_name, p)) {
      HIPSYCL_DEBUG_INFO << "ocl_hardware_manager: Discovered OpenCL platform "
                         << platform_name << std::endl;

      _platforms.push_back(p);
      int platform_id = _platforms.size() - 1;

      std::vector<cl::Device> devs;
      // CL param validation layer does not like CL_DEVICE_TYPE_ALL here
      err = p.getDevices(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU |
                             CL_DEVICE_TYPE_ACCELERATOR,
                         &devs);
      if (err != CL_SUCCESS) {
        print_warning(
            __acpp_here(),
            error_info{
                "ocl_hardware_manager: Could not list devices of platform",
                error_code{"CL", err}});
      } else {
        cl_platform_id pid = p.cl::detail::Wrapper<cl_platform_id>::get();

        std::optional<cl::Context> platform_ctx;

        if (!no_shared_contexts) {
          // First attempt to create shared context
          cl_context_properties ctx_props[] = {CL_CONTEXT_PLATFORM,
                                               (cl_context_properties)pid, 0};
          cl::Context ctx{devs, ctx_props, nullptr, nullptr, &err};
          if (err == CL_SUCCESS)
            platform_ctx = ctx;
          else {
            print_warning(
                __acpp_here(),
                error_info{"ocl_hardware_manager: Shared context construction "
                           "failed. Will attempt to fall back to individual "
                           "context per device, but this may prevent data "
                           "transfers between devices from working.",
                           error_code{"CL", err}});
          }
        } else {
          print_warning(
              __acpp_here(),
              error_info{
                  "ocl_hardware_manager: Not constructing shared context "
                  "across devices. Note that this may prevent data "
                  "transfers between devices from working."});
        }

        int platform_device_index = 0;
        for (const auto &dev : devs) {
          std::string dev_name = info_query<CL_DEVICE_NAME, std::string>(dev);
          if(should_include_device(dev_name, dev)) {

            std::optional<cl::Context> chosen_context = platform_ctx;

            if (!chosen_context.has_value()) {
              // If we don't have a shared context yet, try creating
              // an individual context for the device.
              cl::Context ctx{dev, nullptr, nullptr, nullptr, &err};
              if (err == CL_SUCCESS)
                chosen_context = ctx;
              else {
                print_error(__acpp_here(),
                            error_info{"ocl_hardware_manager: Individual context "
                                      "creation failed",
                                      error_code{"CL", err}});
              }
            }
            if (chosen_context.has_value()) {
              ocl_hardware_context hw_ctx{dev, chosen_context.value(),
                                          static_cast<int>(_devices.size()),
                                          platform_id};

              if (device_matches(visibility_mask, backend_id::ocl,
                                global_device_index, platform_device_index,
                                platform_id, hw_ctx.get_device_name(),
                                platform_name)) {
                _devices.push_back(hw_ctx);
                // Allocator can only be initialized once the hardware context
                // is in the list, because the allocator may itself attempt to
                // access it using hardware_manager::get_device()
                _devices.back().init_allocator(this);
              }
            }
            ++global_device_index;
            ++platform_device_index;
          }
        }
      }
    }
  }
}

std::size_t ocl_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context* ocl_hardware_manager::get_device(std::size_t index) {
  return &(_devices[index]);
}

device_id ocl_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{backend_descriptor{_hw_platform, api_platform::ocl},
                   static_cast<int>(index)};
}

cl::Platform ocl_hardware_manager::get_platform(int platform_id) {
  return _platforms[platform_id];
}


cl::Context ocl_hardware_manager::get_context(device_id dev) {
  return static_cast<ocl_hardware_context *>(get_device(dev.get_id()))
      ->get_cl_context();
}

}
}
