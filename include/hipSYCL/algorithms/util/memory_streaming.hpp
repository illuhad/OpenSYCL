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
#ifndef HIPSYCL_ALGORITHMS_MEMORY_STREAMING_HPP
#define HIPSYCL_ALGORITHMS_MEMORY_STREAMING_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/info/device.hpp"
#include <cstddef>


namespace hipsycl::algorithms::util {

class data_streamer {
public:
  data_streamer(rt::device_id dev, std::size_t problem_size,
                std::size_t group_size)
      : data_streamer{sycl::device{dev}, problem_size, group_size} {}

  data_streamer(const sycl::device &dev, std::size_t problem_size,
                std::size_t group_size)
      : _problem_size{problem_size}, _group_size{group_size} {
    std::size_t default_num_groups =
        (problem_size + group_size - 1) / group_size;

    std::size_t desired_num_groups = 0;
    if(!dev.is_host()) {
      desired_num_groups =
          dev.get_info<sycl::info::device::max_compute_units>() * 4;

    } else {
      desired_num_groups =
          (default_num_groups + cpu_work_per_item - 1) / cpu_work_per_item;
    }

    _num_groups = std::min(default_num_groups, desired_num_groups);
  }

  std::size_t get_required_local_size() const noexcept {
    return _group_size;
  }

  std::size_t get_required_global_size() const noexcept {
    return _num_groups * _group_size;
  }

  // Only to be called inside kernels.
  //
  // Ensures that f is broadcast across the entire problem space.
  // 
  // F is a callable of signature void(sycl::id<1>).
  template <class F>
  static void run(std::size_t problem_size, sycl::nd_item<1> idx,
                  F &&f) noexcept {
    namespace jit = sycl::AdaptiveCpp_jit;
    __acpp_if_target_sscp(
        jit::compile_if_else(
            jit::reflect<jit::reflection_query::compiler_backend>() ==
              jit::compiler_backend::host,
            [&]() { run_host(problem_size, idx, f); },
            [&]() { run_device(problem_size, idx, f); });

        return;);
    __acpp_if_target_device(
      run_device(problem_size, idx, f);
    );
    __acpp_if_target_host(
      run_host(problem_size, idx, f);
    );
  };

private:
  static constexpr int cpu_work_per_item = 8;

  template<class F>
  static void run_device(std::size_t problem_size, sycl::nd_item<1> idx, F&& f) noexcept {
    const std::size_t gid = idx.get_global_id(0);
    for (std::size_t i = gid; i < problem_size; i += idx.get_global_range(0)) {
      f(sycl::id<1>{i});
    }
  }

  template<class F>
  static void run_host(std::size_t problem_size, sycl::nd_item<1> idx, F&& f) noexcept {
    
    const std::size_t last_group = idx.get_group_range(0) - 1;
    const std::size_t gid = idx.get_global_id(0);

    if (idx.get_group_linear_id() != last_group) {
#pragma clang unroll
      for (int i = 0; i < cpu_work_per_item; ++i) {
        auto pos = cpu_work_per_item * gid + i;
        // if(pos < problem_size)
        f(sycl::id<1>{pos});
      }
    } else {
      for (int i = 0; i < cpu_work_per_item; ++i) {
        auto pos = cpu_work_per_item * gid + i;
        if (pos < problem_size)
          f(sycl::id<1>{pos});
      }
    }
  }

  std::size_t _num_groups;
  std::size_t _problem_size;
  std::size_t _group_size;
};

class abortable_data_streamer {
public:
  abortable_data_streamer(const sycl::device &dev, std::size_t problem_size,
                std::size_t group_size)
      : _problem_size{problem_size}, _group_size{group_size} {
    std::size_t default_num_groups =
        (problem_size + group_size - 1) / group_size;

    std::size_t desired_num_groups = 0;
    desired_num_groups =
        dev.get_info<sycl::info::device::max_compute_units>() * 4;

    _num_groups = std::min(default_num_groups, desired_num_groups);
  }

  std::size_t get_required_local_size() const noexcept {
    return _group_size;
  }

  std::size_t get_required_global_size() const noexcept {
    return _num_groups * _group_size;
  }

  // Only to be called inside kernels.
  //
  // Ensures that f is broadcast across the entire problem space.
  // If f() returns true, will attempt to abort execution as quickly
  // as possible.
  // 
  // F is a callable of signature bool(sycl::id<1>).
  template <class F>
  static void run(std::size_t problem_size, sycl::nd_item<1> idx,
                  F &&f) noexcept {


    const std::size_t gid = idx.get_global_id(0);
    for (std::size_t i = gid; i < problem_size; i += idx.get_global_range(0)) {
      if(f(sycl::id<1>{i}))
        return;
    }
  };

private:
  std::size_t _num_groups;
  std::size_t _problem_size;
  std::size_t _group_size;
};

}

#endif
