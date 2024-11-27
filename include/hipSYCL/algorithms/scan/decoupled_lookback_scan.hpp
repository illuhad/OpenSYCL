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

#ifndef ACPP_ALGORITHMS_DECOUPLED_LOOKBACK_SCAN_HPP
#define ACPP_ALGORITHMS_DECOUPLED_LOOKBACK_SCAN_HPP

#include <iterator>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <optional>
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/libkernel/atomic_ref.hpp"
#include "hipSYCL/sycl/libkernel/group_functions.hpp"
#include "hipSYCL/sycl/jit.hpp"
#include "hipSYCL/algorithms/util/allocation_cache.hpp"

namespace hipsycl::algorithms::scanning {

namespace detail {

enum class status : uint32_t {
  invalid = 0,
  aggregate_available = 1,
  prefix_available = 2
};

template<class T>
struct scratch_data {
  scratch_data(util::allocation_group &scratch, std::size_t num_groups) {
    group_aggregate = scratch.obtain<T>(num_groups);
    inclusive_prefix = scratch.obtain<T>(num_groups);
    group_status = scratch.obtain<status>(num_groups);
  }

  T* group_aggregate;
  T* inclusive_prefix;
  status* group_status;
};

template <class T, class BinaryOp>
T kogge_stone_scan(sycl::nd_item<1> idx, T my_element, BinaryOp op,
                   T *local_mem) {
  const int lid = idx.get_local_linear_id();
  const int local_size = idx.get_local_range().size();
  local_mem[lid] = my_element;

  for (unsigned stride = 1; stride < local_size; stride <<= 1) {
    sycl::group_barrier(idx.get_group());
    T current = my_element;
    if (lid >= stride) {
      current = op(local_mem[lid - stride], local_mem[lid]);
    }
    sycl::group_barrier(idx.get_group());
    
    if (lid >= stride) {
      local_mem[lid] = current;
    }
  }

  auto result = local_mem[lid];
  sycl::group_barrier(idx.get_group());
  return result;
}

template <class T, class BinaryOp>
T sequential_scan(sycl::nd_item<1> idx, T my_element, BinaryOp op,
                   T *local_mem) {
  int lid = idx.get_local_linear_id();
  local_mem[lid] = my_element;
  sycl::group_barrier(idx.get_group());

  if(lid == 0) {
    T current = local_mem[0];
    for(int i = 1; i < idx.get_local_range().size(); ++i) {
      current = op(current, local_mem[i]);
      local_mem[i] = current;
    }
  }
  sycl::group_barrier(idx.get_group());
  auto result = local_mem[lid];
  sycl::group_barrier(idx.get_group());
  return result;
}

template<class T, class BinaryOp>
constexpr bool can_use_group_algorithms() {
  // TODO
  return false;
}

template <class T, class BinaryOp>
T collective_inclusive_group_scan(sycl::nd_item<1> idx, T my_element,
                                  BinaryOp op, T *local_mem) {
  if constexpr(can_use_group_algorithms<T, BinaryOp>()) {
    // TODO
  } else {
    namespace jit = sycl::AdaptiveCpp_jit;
    __acpp_if_target_sscp(
      // For some reason, using the compile_if_else wrapper introduces
      // overheads for host JIT in this case :(
      // This seems to be unique to this particular case here though.
      if(jit::reflect<jit::reflection_query::compiler_backend>() ==
              jit::compiler_backend::host) {
        return sequential_scan<T, BinaryOp>(idx, my_element, op, local_mem);
      } else {
        return kogge_stone_scan<T, BinaryOp>(idx, my_element, op, local_mem);
      }
    );
    return kogge_stone_scan<T, BinaryOp>(idx, my_element, op, local_mem);
  }
}

template<class T, class BinaryOp>
T collective_broadcast(sycl::nd_item<1> idx, T x, int local_id, T* local_mem) {
  if constexpr(can_use_group_algorithms<T, BinaryOp>()) {
    // TODO
  } else {
    if(idx.get_local_linear_id() == local_id) {
      *local_mem = x;
    }
    sycl::group_barrier(idx.get_group());
    auto result = *local_mem;
    sycl::group_barrier(idx.get_group());
    return result;
  }
}

template <class T, class BinaryOp, class Generator, class Processor>
T joint_inclusive_group_scan(sycl::nd_item<1> idx, int num_scan_groups,
                             BinaryOp op, T *local_mem, Generator gen,
                             Processor result_processor) {
  const int lid = idx.get_local_linear_id();
  const int group_size = idx.get_local_range().size();
  T current_exclusive_prefix;
  for(int invocation = 0, invocation < num_scan_groups, ++invocation) {
    int current_id = invocation * group_size + lid;
    T my_element = gen(idx, invocation, current_id);
    T local_scan_result =
        collective_inclusive_group_scan(idx, my_element, op, local_mem);
    
    if(invocation != 0)
      local_scan_result = op(exclusive_prefix, local_scan_result);

    // need lookback here
    result_processor(idx, invocation, current_id, local_scan_result);
    
    current_exclusive_prefix = collective_broadcast<T, BinaryOp>(
        idx, local_scan_result, group_size - 1, local_mem);
  }
  // has local prefix here

  return current_exclusive_prefix;
}

template <class T, class BinaryOp>
T exclusive_prefix_look_back(const T &dummy_init, int effective_group_id,
                             detail::status *status, T *group_aggregate,
                             T *inclusive_prefix, BinaryOp op) {
  // dummy_init is a dummy value here; avoid relying on default constructor
  // in case T has none.
  T exclusive_prefix = dummy_init;
  bool exclusive_prefix_initialized = false;

  auto update_exclusive_prefix = [&](auto x){
    if(!exclusive_prefix_initialized) {
      exclusive_prefix = x;
      exclusive_prefix_initialized = true;
    } else {
      exclusive_prefix = op(x, exclusive_prefix);
    }
  };

  for(int lookback_group = effective_group_id - 1; lookback_group >= 0; --lookback_group) {
    uint32_t& status_ptr = reinterpret_cast<uint32_t&>(status[lookback_group]);
    sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> status_ref{status_ptr};

    detail::status lookback_status;
    while ((lookback_status = static_cast<detail::status>(status_ref.load())) ==
            detail::status::invalid)
      ;
    
    if(lookback_status == detail::status::prefix_available) {
      update_exclusive_prefix(inclusive_prefix[lookback_group]);
      return exclusive_prefix;
    } else {
      update_exclusive_prefix(group_aggregate[lookback_group]);
    }
  }

  return exclusive_prefix;
}

template <bool IsInclusive, class T, class Generator, class OptionalInitT,
          class BinaryOp>
T load_data_element(Generator &&gen, sycl::nd_item<1> idx, BinaryOp op,
                    uint32_t effective_group_id, std::size_t global_id,
                    std::size_t problem_size, OptionalInitT init) {
  if constexpr (IsInclusive) {
    auto elem = gen(idx, effective_group_id, global_id, problem_size);
    if constexpr(!std::is_same_v<OptionalInitT, std::nullopt_t>) {
      if(global_id == 0) {
        return op(init, elem);
      }
    }
    return elem;
  } else {
    if(global_id == 0)
      return init;
    return gen(idx, effective_group_id, global_id - 1, problem_size);
  }
}

template <bool IsInclusive, class T, class OptionalInitT, class BinaryOp,
          class Generator, class Processor>
void scan_kernel(sycl::nd_item<1> idx, T *local_memory, scratch_data<T> scratch,
                 uint32_t *group_counter, BinaryOp op, OptionalInitT init,
                 std::size_t problem_size, Generator &&gen,
                 Processor &&processor) {
  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
      group_id_counter{*group_counter};

  const int local_id = idx.get_local_linear_id();
  uint32_t effective_group_id = idx.get_group_linear_id();
  if(local_id == 0) {
    effective_group_id = group_id_counter.fetch_add(static_cast<uint32_t>(1));
  }
  effective_group_id = collective_broadcast<uint32_t, BinaryOp>(
      idx, effective_group_id, 0, reinterpret_cast<uint32_t*>(local_memory));

  const std::size_t global_id = effective_group_id * idx.get_local_range().size() +
                          local_id;
  
  int local_size = idx.get_local_range().size();
  
  std::size_t num_groups = idx.get_group_range().size();
  const bool is_last_group = effective_group_id == num_groups - 1;
  if(is_last_group) {
    std::size_t group_offset = effective_group_id * (num_groups - 1) + local_size;
    local_size = problem_size - group_offset;
  }

  // This invokes gen for the current work item to obtain our data element
  // for the scan. If we are dealing with an exclusive scan, load_data_element
  // shifts the data access by 1, thus allowing us to treat the scan as inclusive
  // in the subsequent algorithm.
  // It also applies init to the first data element, if provided.
  T my_element = load_data_element<IsInclusive, T>(
      gen, idx, op, effective_group_id, global_id, problem_size, init);
  
  // The exclusive scan case is handled in load_element() by accessing the element
  // at global_id-1 instead of global_id.
  T local_scan_result =
          collective_inclusive_group_scan(idx, my_element, op, local_memory);
  auto generator = [=](auto idx, int group_invocation, int current_id) {

  };


  uint32_t *status_ptr =
        reinterpret_cast<uint32_t *>(&scratch.group_status[effective_group_id]);
    sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> status_ref{*status_ptr};

  // Set group aggregate which we now know after scan. The first group
  // Can also set its prefix and is done.
  if(local_id == local_size - 1) {
    T group_aggregate = local_scan_result;
    
    if(effective_group_id == 0) {
      scratch.group_aggregate[effective_group_id] = group_aggregate;
      scratch.inclusive_prefix[effective_group_id] = group_aggregate;
      status_ref.store(static_cast<uint32_t>(status::prefix_available));
    } else {
      scratch.group_aggregate[effective_group_id] = group_aggregate;
      status_ref.store(static_cast<uint32_t>(status::aggregate_available));
    }
  }

  sycl::group_barrier(idx.get_group());

  // All groups except group 0 need to perform lookback to find their prefix
  if(effective_group_id != 0) {
    // my_element is a dummy value here; avoid relying on default constructor
    // in case T has none
    T exclusive_prefix = my_element;
    if(local_id == 0) {
      exclusive_prefix = exclusive_prefix_look_back(my_element, effective_group_id,
                          scratch.group_status, scratch.group_aggregate,
                          scratch.inclusive_prefix, op);
    }
    exclusive_prefix = collective_broadcast<T, BinaryOp>(
        idx, exclusive_prefix, 0, local_memory);
    local_scan_result = op(exclusive_prefix, local_scan_result);

    // All groups except first and last one need to update their prefix
    if(effective_group_id != num_groups - 1) {
      if(local_id == local_size - 1){
        scratch.inclusive_prefix[effective_group_id] = local_scan_result;
        status_ref.store(static_cast<uint32_t>(status::prefix_available));
      }
    }
  }

  processor(idx, effective_group_id, global_id, problem_size, local_scan_result);
}

} // detail


/// Implements the decoupled lookback scan algorithm -
/// See Merill, Garland (2016) for details.
///
/// This algorithm assumes that the hardware can support acquire/release
/// atomics.
/// It also assumes that work groups with smaller ids are either scheduled
/// before work groups with higher ids, or that work group execution may be
/// preempted. To provide this guarantee universally, our implementation
/// reassigns work group ids based on when they start executing.
///
/// \param gen A callable with signature \c T(nd_item<1>, uint32_t
/// effective_group_id, size_t effective_global_id, size_t problem_size)
///
/// \c gen is the generator that generates the data elements to run the scan.
/// Note that the scan implementation may reorder work-groups; \c gen should
/// therefore not rely on the group id and global id from the provided nd_item,
/// but instead use the provided \c effective_group_id and and \c
/// effective_global_id.
///
/// If the problem size is not divisible by the selected work group size, then
/// the last group might invoke \c gen with ids outside the bound. It is the
/// responsibility of \c gen to handle this case. For these work items, the
/// return value from \c gen can be an arbitrary dummy value (e.g. the last
/// valid element within bounds).
///
/// \param processor A callable with signature \c void(nd_item<1>,  uint32_t
/// effective_group_id, size_t effective_global_id, size_t problem_size, T
/// result)
///
/// \c processor is invoked at the end of the scan with the result of the global
/// scan for this particular work item. \c processor will be invoked once the
/// global result for the work item is available which might be before the scan
/// has completed for all work items. Do not assume global synchronization.
///
/// Note that the scan implementation may reorder work-groups; \c processor
/// should therefore not rely on the group id and global id from the provided
/// nd_item, but instead use the provided \c effective_group_id and and \c
/// effective_global_id.
///
/// If the problem size is not divisible by the selected work group size, then
/// the last group might invoke \c processor with ids outside the bound. It is
/// the responsibility of \c processor to handle this case. For these work
/// items, the result value passed into \c processor is undefined.
template <bool IsInclusive, class T, class WorkItemDataGenerator, class ResultProcessor,
          class BinaryOp, class OptionalInitT>
sycl::event
decoupled_lookback_scan(sycl::queue &q, util::allocation_group &scratch_alloc,
                        WorkItemDataGenerator gen,
                        ResultProcessor processor, BinaryOp op,
                        std::size_t problem_size, std::size_t group_size,
                        OptionalInitT init = std::nullopt,
                        const std::vector<sycl::event> &user_deps = {}) {

  if(problem_size == 0)
    return sycl::event{};

  static_assert(IsInclusive || std::is_convertible_v<OptionalInitT, T>,
                "Non-inclusive scans need an init argument of same type as the "
                "scan data element");
  static_assert(
      std::is_convertible_v<OptionalInitT, T> ||
          std::is_same_v<OptionalInitT, std::nullopt_t>,
      "Init argument must be of std::nullopt_t type or exact type of scan "
      "data elements");

  std::size_t num_groups = (problem_size + group_size - 1) / group_size;
  detail::scratch_data<T> scratch{scratch_alloc, num_groups};
  uint32_t* group_counter = scratch_alloc.obtain<uint32_t>(1);

  auto initialization_evt = q.parallel_for(num_groups, [=](sycl::id<1> idx){
    scratch.group_status[idx] = detail::status::invalid;
    if(idx.get(0) == 0) {
      *group_counter = 0;
    }
  });

  std::vector<sycl::event> deps = user_deps;
  if(!q.is_in_order())
    deps.push_back(initialization_evt);

  sycl::nd_range<1> kernel_range{num_groups * group_size, group_size};
  if constexpr(detail::can_use_group_algorithms<T, BinaryOp>()) {
    return q.parallel_for(kernel_range, deps, [=](auto idx) {
      detail::scan_kernel<IsInclusive>(idx, nullptr, scratch, group_counter, op,
                                       init, problem_size, gen, processor);
    });
  } else {
    // We need local memory:
    // - 1 data element per work item
    // - at least size for one uint32_t to broadcast group id
    std::size_t local_mem_elements =
        std::max(group_size, (sizeof(uint32_t) + sizeof(T) - 1) / sizeof(T));

    // This is not entirely correct since max local mem size can also depend
    // on work group size.
    // We also assume that there is no other local memory consumer.
    // TODO Improve this
    std::size_t max_local_size =
        q.get_device().get_info<sycl::info::device::local_mem_size>();

    bool has_sufficient_local_memory = static_cast<double>(max_local_size) >=
                                       1.5 * sizeof(T) * local_mem_elements;

    if(has_sufficient_local_memory) {
      return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        sycl::local_accessor<T, 1> local_mem{local_mem_elements, cgh};
        cgh.parallel_for(kernel_range, [=](auto idx) {
          detail::scan_kernel<IsInclusive>(idx, &(local_mem[0]),
                                           scratch, group_counter, op, init,
                                           problem_size, gen, processor);
        });
      });
    } else {
      // This is a super inefficient dummy algorithm for now that requires
      // large scratch storage
      T* emulated_local_mem = scratch_alloc.obtain<T>(num_groups * local_mem_elements);

      return q.parallel_for(kernel_range, deps, [=](auto idx) {
        detail::scan_kernel<IsInclusive>(
            idx,
            emulated_local_mem + local_mem_elements * idx.get_group_linear_id(),
            scratch, group_counter, op, init, problem_size, gen, processor);
      });
    }
  }
}
}

#endif
