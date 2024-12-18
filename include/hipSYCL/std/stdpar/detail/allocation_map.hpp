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
#ifndef ACPP_STDPAR_ALLOCATION_DATA_STRUCTURES_HPP
#define ACPP_STDPAR_ALLOCATION_DATA_STRUCTURES_HPP


#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <atomic>
#include <algorithm>
#include <set>
#include <array>
#include <cassert>

#include "hipSYCL/common/allocation_map.hpp"


extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void*);

namespace hipsycl::stdpar {

template <class T>
class libc_allocator{
public:
  using value_type    = T;

  libc_allocator() noexcept {}
  template <class U> libc_allocator(libc_allocator<U> const&) noexcept {}

  value_type*
  allocate(std::size_t n)
  {
    void* ptr = __libc_malloc(sizeof(T) * n);
    return static_cast<value_type*>(ptr);
  }

  void
  deallocate(value_type* p, std::size_t) noexcept {
    __libc_free(p);
  }
};

struct libc_untyped_allocator {
  static void* allocate(size_t n) {
    return __libc_malloc(n);
  }

  static void deallocate(void* ptr) {
    __libc_free(ptr);
  }
};


template <class T, class U>
bool operator==(libc_allocator<T> const &, libc_allocator<U> const &) noexcept {
  return true;
}
template <class T, class U>
bool operator!=(libc_allocator<T> const &x,
                libc_allocator<U> const &y) noexcept {
  return !(x == y);
}

template <class Payload>
using allocation_map =
    common::allocation_map<Payload, libc_untyped_allocator>;

class free_space_map {
public:
  free_space_map(std::size_t max_assignable_space)
  : _max_assignable_space{max_assignable_space}, _lock{0} {
    // Register all address space (starting at 0) as free
    _sorted_free_blocks_in_level[max_allocation_space_in_bits-1].insert(0ull);
  }


  bool claim(std::size_t size, uint64_t& address) {
    spin_lock lock{_lock};
    return claim(get_desired_level(size), size, address);
  }

  bool release(uint64_t address, std::size_t size) {
    assert(address % get_block_size(get_desired_level(size)) == 0);
    spin_lock lock{_lock};
    return release_block(address, get_desired_level(size));
  }
private:

  static constexpr uint64_t get_block_size(int level) {
    return 1ull << level;
  }

  class spin_lock {
  public:
    spin_lock(std::atomic<int>& lock)
    : _lock{lock} {
      int expected = 0;
      while (!_lock.compare_exchange_strong(
          expected, 1, std::memory_order_release,
          std::memory_order_relaxed))
        expected = 0;
    }

    ~spin_lock() {
      _lock.store(0, std::memory_order_release);
    }
  private:
    std::atomic<int>& _lock;
  };
  

  int get_desired_level(std::size_t allocation_size) {
    for(int i = 0; i < max_allocation_space_in_bits; ++i) {
      if(get_block_size(i) >= allocation_size)
        return i;
    }
    return max_allocation_space_in_bits-1;
  }

  bool claim(int desired_level, std::size_t size, uint64_t& address) {

    auto& target_block_set = _sorted_free_blocks_in_level[desired_level];
    
    if(target_block_set.empty()) {
      if(!generate_new_free_blocks(desired_level)) {
        return false;
      } 
    }

    assert(!target_block_set.empty());

    for (auto it = target_block_set.rbegin(); it != target_block_set.rend();
         ++it) {
      address = *it;
      if(address + size < _max_assignable_space) {
        assert(address % get_block_size(desired_level) == 0);
        target_block_set.erase(address);
        return true;
      }
    }

    return false;
  
  }

  bool generate_new_free_blocks(int level) {
    int next_available_level = find_lowest_level_with_free_blocks(level + 1);
    if(next_available_level < level) {
      return false;
    }

    assert(!_sorted_free_blocks_in_level[next_available_level].empty());

    auto begin_it = _sorted_free_blocks_in_level[next_available_level].begin();
    uint64_t address_to_split = *begin_it;
    _sorted_free_blocks_in_level[next_available_level].erase(begin_it);

    assert(address_to_split % get_block_size(next_available_level) == 0);

    for(int i = next_available_level-1; i >= level; --i) {
      if(i == level)
        _sorted_free_blocks_in_level[i].insert(address_to_split);
      _sorted_free_blocks_in_level[i].insert(address_to_split+get_block_size(i));
    }

    return true;
  }

  int find_lowest_level_with_free_blocks(int min_level) {
    for(std::size_t i = min_level; i < _sorted_free_blocks_in_level.size(); ++i) {
      if(!_sorted_free_blocks_in_level[i].empty())
        return i;
    }
    return -1;
  }

  template<class It>
  auto get_merge_candidate_iterator(const It& current, uint64_t current_address, int level) {
    if(current_address % get_block_size(level+1) == 0) {
      return current+1;
    } else {
      return current-1;
    }
  }

  template<class Iterator>
  void try_merge_blocks(Iterator it, uint64_t address, int level) {
    auto merge_candidate = it;
    assert(address % get_block_size(level) == 0);

    if(address % get_block_size(level + 1) == 0) {
      ++merge_candidate;
    } else {
      --merge_candidate;
    }

    auto& current_level_free_blocks = _sorted_free_blocks_in_level[level];
    if(merge_candidate != current_level_free_blocks.end()) {
      uint64_t first_block_address = (*merge_candidate < address) ? *merge_candidate : address;
      uint64_t second_block_address = (*merge_candidate > address) ? *merge_candidate : address;

      if(second_block_address - first_block_address == get_block_size(level)) {
        current_level_free_blocks.erase(first_block_address);
        current_level_free_blocks.erase(second_block_address);
        auto next = _sorted_free_blocks_in_level[level+1].insert(first_block_address);

        try_merge_blocks(next.first, first_block_address, level + 1);
      }
    }
  }

  bool release_block(uint64_t address, int target_level) {
    auto& target_block_set = _sorted_free_blocks_in_level[target_level];
    auto res = target_block_set.insert(address);
    
    if(!res.second)
      return false;
    
    try_merge_blocks(res.first, address, target_level);

    return true;
  }
  
  static constexpr int max_allocation_space_in_bits = 48;
  
  const std::size_t _max_assignable_space;
  std::atomic<int> _lock;

  using block_set_type = std::set<uint64_t, std::less<uint64_t>, libc_allocator<uint64_t>>;
  std::array<block_set_type, max_allocation_space_in_bits> _sorted_free_blocks_in_level;
};

}

#endif
