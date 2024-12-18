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
#ifndef HIPSYCL_DEVICE_SELECTOR_HPP
#define HIPSYCL_DEVICE_SELECTOR_HPP

#include "hipSYCL/runtime/application.hpp"

#include "exception.hpp"
#include "device.hpp"
#include "hipSYCL/runtime/settings.hpp"

#include <limits>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <numeric>

namespace hipsycl {
namespace sycl {


enum class selection_policy {
  all,
  best
};

template <class Selector, selection_policy P = selection_policy::all>
class multi_device_selector {
public:
  constexpr multi_device_selector(const Selector& s = Selector{})
  : _s{s} {}

  int operator()(const device& dev) const {
    return _s(dev);
  }
private:
  Selector _s;
};


namespace detail {

template<class Selector>
struct selector_traits {
  static constexpr bool is_multi_device = false;
  static constexpr selection_policy policy = selection_policy::best;
};

template<class Selector, selection_policy P>
struct selector_traits<multi_device_selector<Selector, P>>{
  static constexpr bool is_multi_device = true;
  static constexpr selection_policy policy = P;
};


inline int select_gpu(const device& dev) {
  if (dev.is_gpu()) {
    // Would be good to prefer a device for which
    // we have actually compiled kernel code, because,
    // I don't know, a user might try to run kernels..
    if (dev.AdaptiveCpp_has_compiled_kernels())
      return 2;
    else
      return 1;
  }
  return -1;
}

inline int select_accelerator(const device& dev) {
  if(dev.is_accelerator()) {
    if(dev.AdaptiveCpp_has_compiled_kernels())
      return 2;
    else
      return 1;
  }
  return -1;
}

inline int select_cpu(const device& dev) {
  if(dev.is_cpu())
    return 1;
  return -1;
}

inline int select_host(const device& dev) {
  if(dev == detail::get_host_device())
    return 1;
  return -1;
}

inline int select_default(const device& dev) {
#if defined(__ACPP_ENABLE_CUDA_TARGET__) ||                                 \
    defined(__ACPP_ENABLE_HIP_TARGET__) ||                                  \
    defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
  // Add 2 to make sure that, if no GPU is found
  if(!dev.is_cpu() && dev.AdaptiveCpp_has_compiled_kernels()) {
    // Prefer GPUs (or other accelerators) that have been targeted
    // and have compiled kernels
    return 2;
  } else if(dev.is_cpu()) {
    // Prefer CPU over GPUs that don't have compiled kernels
    // and cannot run kernels.

    // Prefer non-OpenMP CPU device since the OpenMP backend cannot be disabled,
    // so there would be no way to select e.g. an OpenCL CPU device
    // using ACPP_VISIBILITY_MASK otherwise.
    if(dev.get_backend() != sycl::backend::omp)
      return 1;
    else
      return 0;
  } else {
    // Never select GPUs without compiled kernels
    return -1;
  }
#else
  return select_host(dev);
#endif
}

template <class Selector>
std::vector<device> select_devices(const Selector &s);

template<class T>
struct is_device_selector {
  static constexpr bool value =
      std::is_convertible_v<T, std::function<int(const device &)>>;
};

template<class T>
inline constexpr bool is_device_selector_v = is_device_selector<T>::value;

}

/// Provided only for backwards-compatibility with SYCL 1.2.1
/// so users can still derive custom selectors from device_selector
class device_selector
{
public:
  virtual ~device_selector(){};
  
  device select_device() const {
    auto res = detail::select_devices(*this);
    // detail::select_devices should throw if it finds
    // no matching devices
    assert(!res.empty());
    return res[0];
  }

  virtual int operator()(const device& dev) const = 0;

};

/// Old SYCL 1.2.1 types are still required for backwards compatibility
/// Note: SYCL 2020 does not mandate how they are implemented - in
/// particular, they don't have to be derived from device_selector!
class error_selector {
public:
  int operator()(const device &dev) const {
    throw exception{make_error_code(errc::runtime),
                    "error_selector device selection invoked"};
  }
};

class gpu_selector {
public:
  int operator()(const device &dev) const { return detail::select_gpu(dev); }
};

class accelerator_selector {
public:
  int operator()(const device &dev) const {
    return detail::select_accelerator(dev);
  }
};

class cpu_selector {
public:
  int operator()(const device &dev) const { return detail::select_cpu(dev); }
};

class host_selector {
public:
  int operator()(const device &dev) const { return detail::select_host(dev); }
};

class default_selector {
public:
  int operator()(const device &dev) const {
    return detail::select_default(dev);
  }
};


inline constexpr default_selector default_selector_v;
inline constexpr cpu_selector cpu_selector_v;
inline constexpr gpu_selector gpu_selector_v;
inline constexpr accelerator_selector accelerator_selector_v;

// Currently we don't distinguish between multiple CPUs anyway
// so it's unclear if this is even needed.
inline constexpr multi_device_selector<cpu_selector, selection_policy::best>
    multi_cpu_selector_v;

// Important to use best policy here to exclude GPUs
// that haven't been targeted.
inline constexpr multi_device_selector<gpu_selector, selection_policy::best>
    multi_gpu_selector_v;

// default_selector will never pick devices that cannot run kernels
// so we can implement system_selector_v by picking all devices
// for which the default selector does not return a negative number.
inline constexpr multi_device_selector<default_selector, selection_policy::all>
    system_selector_v;

inline auto aspect_selector(const std::vector<aspect> &aspectList,
                            const std::vector<aspect> &denyList = {}) {

  return [=](const device& dev) {
    if(aspectList.empty() && denyList.empty())
      return detail::select_default(dev);

    for(aspect a : aspectList) {
      if(!dev.has(a))
        return -1;
    }
    for(aspect a : denyList) {
      if(dev.has(a))
        return -1;
    }
    return 1;
  };
}

template <typename... aspectListTN>
auto aspect_selector(aspectListTN... aspectList) {
  return [=](const device& dev) {
    if(sizeof...(aspectList) == 0)
      return detail::select_default(dev);

    bool satisfies_all = (dev.has(aspectList) && ...);
    if(satisfies_all)
      return 1;
    return -1;
  };
}

template <aspect... aspectList>
auto aspect_selector() {
  return aspect_selector(aspectList...);
}

inline device::device()
  : device::device(default_selector_v)
{}
  
template <class DeviceSelector>
inline device::device(const DeviceSelector &deviceSelector) {
  this->_device_id = detail::select_devices(deviceSelector)[0]._device_id;
}
  
namespace detail {

template <class Selector>
std::vector<device> select_devices(const Selector &s) {

  if(std::is_same_v<default_selector, Selector>) {
    rt::default_selector_behavior b =
        rt::application::get_settings()
            .get<rt::setting::default_selector_behavior>();
    
    if(b == rt::default_selector_behavior::system)
      return select_devices(system_selector_v);
    else if(b == rt::default_selector_behavior::multigpu)
      return select_devices(multi_gpu_selector_v);
  }

  auto devices = device::get_devices();
  // There should always be at least a CPU device
  assert(devices.size() > 0);
  std::vector<int> dev_indices(devices.size());
  std::vector<int> dev_scores(devices.size());

  std::iota(dev_indices.begin(), dev_indices.end(), 0);
  std::transform(dev_indices.begin(), dev_indices.end(), dev_scores.begin(),
                 [&](int dev_index){ return s(devices[dev_index]); });

  std::sort(dev_indices.begin(), dev_indices.end(),
            [&](int a, int b) { return s(devices[a]) > s(devices[b]); });

  int max_devs = 1;

  if(selector_traits<Selector>::is_multi_device)
    max_devs = std::numeric_limits<int>::max();
  selection_policy policy = selector_traits<Selector>::policy;

  std::vector<device> result;
  assert(!dev_indices.empty());

  const int best_score = dev_scores[dev_indices[0]];
  for(int i = 0; i < dev_indices.size(); ++i) {
    // Only include devices with positive scores, no more than max_devs.
    // If we are not in multi device selection mode, max_devs is 1
    // so we will just select the best device.
    if (dev_scores[dev_indices[i]] >= 0 && result.size() < max_devs) {
      // If we are in best selection mode, we select all devices that
      // have the top score.
      // Otherwise, we select all devices that have positive score.
      if (policy != selection_policy::best ||
          dev_scores[dev_indices[i]] == best_score)
        result.push_back(devices[dev_indices[i]]);
    }
  }

  if (result.empty()) {
    throw exception{make_error_code(errc::runtime), "No matching device"};
  }

  return result;
}

}

}
}

#endif
