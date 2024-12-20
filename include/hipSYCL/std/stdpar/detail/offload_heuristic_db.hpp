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
#ifndef HIPSYCL_PSTL_OFFLOAD_HEURISTIC_HPP
#define HIPSYCL_PSTL_OFFLOAD_HEURISTIC_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <fstream>
#include <optional>
#include <memory>
#include <vector>
#include <mutex>
#include "hipSYCL/runtime/settings.hpp"
#include "hipSYCL/std/stdpar/detail/allocation_map.hpp"
#include "hipSYCL/common/stable_running_hash.hpp"


namespace hipsycl::stdpar {

namespace algorithm_category {
struct for_each {};
struct for_each_n {};
struct transform {};
struct copy {};
struct copy_if {};
struct copy_n {};
struct fill {};
struct fill_n {};
struct generate {};
struct generate_n {};
struct replace {};
struct replace_if {};
struct replace_copy {};
struct replace_copy_if {};
struct find {};
struct find_if {};
struct find_if_not {};
struct all_of {};
struct any_of {};
struct none_of {};
struct sort {};
struct merge {};
struct inclusive_scan {};
struct exclusive_scan {};
struct transform_inclusive_scan {};
struct transform_exclusive_scan {};
struct transform_reduce {};
struct reduce {};
} // namespace algorithm_type

template<class AlgorithmCategory, class ExecPolicy>
struct algorithm {
  using algorithm_category = AlgorithmCategory;
  using execution_policy = ExecPolicy;

  algorithm(const AlgorithmCategory&, ExecPolicy&&) {}
};


namespace detail {

template <class K, class V>
using host_malloc_unordered_map =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       libc_allocator<std::pair<const K, V>>>;

class offload_heuristic_db_storage {
public:

  static std::shared_ptr<offload_heuristic_db_storage> get() {
    static std::shared_ptr<offload_heuristic_db_storage> instance =
        std::make_shared<offload_heuristic_db_storage>();
    return instance;
  }

  offload_heuristic_db_storage() {
    std::fstream f{get_dataset_filename().c_str(), std::ios::in};
    if(f.is_open()) {
      std::string line;
      while(std::getline(f, line)) {
        std::stringstream sstr{line};
        uint64_t hash;
        entry e;
        sstr >> hash;
        e.read(sstr);

        _entries[hash] = e;
      } 
    }
  }

  ~offload_heuristic_db_storage() {
    std::fstream f{get_dataset_filename().c_str(),
                   std::ios::out | std::ios::trunc};
    for (const auto &entry : _entries) {
      f << entry.first << " ";
      entry.second.write(f);
      f << "\n";
    }
  }

  using device_t = int;

  struct device_entry {
    device_t dev;
    uint64_t problem_size;
    double runtime = std::numeric_limits<double>::max();
    uint64_t num_samples = 0;

    void merge(const device_entry& other) {
      if(dev == other.dev && problem_size == other.problem_size) {
        uint64_t aggregated_num_samples = num_samples + other.num_samples;
        runtime = (num_samples * runtime + other.num_samples * other.runtime) /
                  aggregated_num_samples;
        num_samples = aggregated_num_samples;
      }
    }

    bool is_sampled() const {
      return runtime < std::numeric_limits<double>::max();
    }

    friend std::istream& operator>>(std::istream& in, device_entry& entry) {
      in >> entry.dev;
      in >> entry.problem_size;
      in >> entry.runtime;
      in >> entry.num_samples;

      return in;
    }

    friend std::ostream& operator<<(std::ostream& out, const device_entry& entry) {
      out << entry.dev << " ";
      out << entry.problem_size << " ";
      out << entry.runtime << " ";
      out << entry.num_samples << " ";

      return out;
    }
  };

  struct entry {
    
    std::vector<device_entry, libc_allocator<device_entry>> entries;

    void merge(const entry& other) {
      auto merge_or_insert = [this](auto& other_entry){
        for(auto& own_entry : entries) {
          if (own_entry.dev == other_entry.dev &&
              own_entry.problem_size == other_entry.problem_size) {
            own_entry.merge(other_entry);
            return;
          }
        }
        entries.push_back(other_entry);
      };

      
      for(auto& other_entry: other.entries) {
        merge_or_insert(other_entry);  
      }
    }

    void read(std::istream& istr) {
      
      int num_entries = 0;
      istr >> num_entries;
      for(int i = 0; i < num_entries; ++i) {
        device_entry dev_entry;
        istr >> dev_entry;
        entries.push_back(dev_entry);
      }
    }

    void write(std::ostream& ostr) const {
      
      ostr << entries.size() << " ";
      for(const auto &entry : entries)
        ostr << entry << " ";
    }
  };


  auto get_entries() const {
    std::lock_guard<std::mutex> lock{_lock};
    return _entries;
  }

  void update_entry_map(const host_malloc_unordered_map<uint64_t, entry>& entry_map) {
    std::lock_guard<std::mutex> lock{_lock};
    for(const auto& e : entry_map) {
      auto it = _entries.find(e.first);
      if(it == _entries.end()) {
        _entries[e.first] = e.second;
      } else {
        auto& own_entry = it->second;
        own_entry.merge(e.second);
      }
    }
  }
  
private:

  static std::string get_dataset_filename() {
    return ".acpp-stdpar-"+get_dataset_name();
  }

  static std::string get_dataset_name() {
    std::string dataset_name;
    if(rt::try_get_environment_variable("stdpar_dataset_name", dataset_name)) {
      return dataset_name;
    }
    return ".acpp-stdpar-profile";
  }

  host_malloc_unordered_map<uint64_t, entry> _entries;
  mutable std::mutex _lock;
};

class offload_heuristic_db {
public:
  offload_heuristic_db()
  : _storage{offload_heuristic_db_storage::get()} {
    _entries = _storage->get_entries();
  }

  ~offload_heuristic_db() {
    _storage->update_entry_map(_entries);
  }

  using device_t = offload_heuristic_db_storage::device_t;
  static constexpr device_t host_device_id = -1;
  static constexpr device_t offload_device_id = 0;

  double estimate_runtime(uint64_t op_hash, std::size_t problem_size, device_t dev) const {
    auto it = _entries.find(op_hash);
    if(it == _entries.end())
      return 0.0;

    // We attempt to find the closest measurements around the requested problem size,
    // and then interpolate linearly.

    // First find closest sampling point
    int64_t delta = std::numeric_limits<int>::max();
    std::size_t closest_problem_size = 0;
    double closest_runtime = 0.0;
    for(auto& e : it->second.entries) {
      if(e.dev == dev) {
        // If we find the required problem size exactly, we can just return it.
        if(e.problem_size == problem_size)
          return e.runtime;

        int64_t current_delta = std::abs((int64_t)e.problem_size - (int64_t)problem_size);
        if(current_delta < delta) {
          delta = current_delta;
          closest_problem_size = e.problem_size;
          closest_runtime = e.runtime;
        }
      }
    }
    // Find 2nd closest sampling point
    delta = std::numeric_limits<int>::max();
    std::size_t closest_problem_size2 = 0;
    double closest_runtime2 = 0.0;
    for(auto& e : it->second.entries) {
      if(e.dev == dev) {
        int64_t current_delta = std::abs((int64_t)e.problem_size - (int64_t)problem_size);
        if(current_delta < delta && e.problem_size != closest_problem_size) {
          delta = current_delta;
          closest_problem_size2 = e.problem_size;
          closest_runtime2 = e.runtime;
        }
      }
    }

    if(closest_problem_size2 < closest_problem_size) {
      std::swap(closest_problem_size, closest_problem_size2);
      std::swap(closest_runtime, closest_runtime2);
    }

    
    double runtime_lower = closest_runtime;
    double runtime_upper = closest_runtime2;
    std::size_t problem_size_lower = closest_problem_size;
    std::size_t problem_size_upper = closest_problem_size2;

    // If we have neither, then both lower bound and upper bound problem size might still be 0.
    // We need to catch this case to avoid division by zero.
    if(problem_size_lower == problem_size_upper) {
      return 0.0;
    }

    double delta_x = problem_size_upper - problem_size_lower;
    double delta_y = runtime_upper - runtime_lower;
    
    double m = delta_y / delta_x;

    double result = m * (problem_size - problem_size_lower) + runtime_lower;
    // If the interpolation results in a negative value, we cannot just return 0, as this
    // is generally interpreted as "couldn't estimate". In this case however, we could estimate,
    // but the estimate is just too low due to inaccuracies. So we just return a value that
    // is smaller than the measurement accuracy.
    if(result <= 0.0)
      return 1.e-20;
    return result;
  }

  void update_entry(uint64_t op_hash, std::size_t problem_size, device_t dev, double runtime) {

    uint64_t& op_invocation_count = _kernel_invocation_counts[op_hash];
    if(op_invocation_count == 0) {
      // Always ignore the first measurement due to potential JIT or other initialization
      // overheads
      ++op_invocation_count;
      return;
    } else {
      ++op_invocation_count;
    }

    auto& e = _entries[op_hash];

    offload_heuristic_db_storage::device_entry d_entry {dev, problem_size, runtime, 1};
    
    for(auto& measurement_entry : e.entries) {
      if(measurement_entry.dev == dev && measurement_entry.problem_size == problem_size) {
        measurement_entry.merge(d_entry);
        return;
      }
    }

    e.entries.push_back(d_entry);
  }


private:
  std::shared_ptr<offload_heuristic_db_storage> _storage;
  host_malloc_unordered_map<uint64_t, offload_heuristic_db_storage::entry>
      _entries;
  host_malloc_unordered_map<uint64_t, uint64_t> _kernel_invocation_counts;
};


template <class AlgorithmType, class Size, typename... Args>
inline uint64_t get_operation_hash(AlgorithmType /*arg*/, Size /*n*/, Args...) {
  common::stable_running_hash hash;

  auto add_to_hash = [&](const auto& x){
    hash(&x, sizeof(x));
  };
  add_to_hash(typeid(AlgorithmType).hash_code());
  (add_to_hash(typeid(Args).hash_code()), ...);
  return hash.get_current_hash();
}

} // namespace detail
} // namespace hipsycl::stdpar

#endif
