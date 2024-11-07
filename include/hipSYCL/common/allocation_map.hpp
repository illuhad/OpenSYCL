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
#ifndef ACPP_ALLOCATION_MAP_HPP
#define ACPP_ALLOCATION_MAP_HPP


#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <atomic>
#include <algorithm>
#include <set>
#include <array>
#include <cassert>


namespace hipsycl::common {

struct stdlib_untyped_allocator {
  static void* allocate(size_t n) {
    return std::malloc(n);
  }

  static void deallocate(void* ptr) {
    std::free(ptr);
  }
};

template<class UntypedAllocatorT, class Int_type, int... Bit_sizes>
class bit_tree {
protected:
  bit_tree(){}
  
  static constexpr int num_levels = sizeof...(Bit_sizes);
  static constexpr int root_level_idx = num_levels - 1;
  static constexpr int bitsizes[num_levels] = {Bit_sizes...};

  static constexpr int get_num_entries_in_level(int level) {
    return 1ull << bitsizes[level];
  }

  static constexpr int get_bitoffset_in_level(int level) {
    int result = 0;
    for(int i = 0; i < level; ++i) {
      result += bitsizes[i];
    }
    return result;
  }

  static constexpr int get_index_in_level(Int_type address, int level) {
    Int_type bitmask = get_n_low_bits_set(bitsizes[level]);
    return (address >> get_bitoffset_in_level(level)) & bitmask;
  }

  static constexpr uint64_t get_n_low_bits_set(int n) {
    if(n == 64)
      return ~0ull;
    return (1ull << n) - 1;
  }

  static constexpr uint64_t get_space_spanned_by_node_in_level(int level) {
    uint64_t result = 1;
    for(int i = 0; i < level; ++i)
      result *= get_num_entries_in_level(level);
    return result;
  }

  template<class T>
  static T* alloc(int count) {
    return static_cast<T*>(UntypedAllocatorT::allocate(sizeof(T) * count));
  }

  static void free(void* ptr) {
    UntypedAllocatorT::deallocate(ptr);
  }
};

template<class UntypedAllocatorT>
using allocation_map_bit_tree_config = bit_tree<UntypedAllocatorT, uint64_t, 
  4, 4, 4, 4,  4, 4, 4, 4,
  4, 4, 4, 4,  4, 4, 4, 4>;

template <class UserPayload, class UntypedAllocatorT = stdlib_untyped_allocator>
class allocation_map : public allocation_map_bit_tree_config<UntypedAllocatorT> {
public:
  using bit_tree_t = allocation_map_bit_tree_config<UntypedAllocatorT>;

  static_assert(sizeof(void*) == 8, "Unsupported pointer size");
  static_assert(std::is_trivial_v<UserPayload>, "UserPayload must be trivial type");

  allocation_map()
  : _num_in_progress_operations{0} {}

  struct value_type : public UserPayload {
    std::size_t allocation_size;
  };

  // Access entry of allocation that address belongs to, or nullptr if the address
  // does not belong to a known allocation.
  value_type* get_entry(uint64_t address, uint64_t& root_address) noexcept {
    insert_or_get_entry_lock lock{_num_in_progress_operations};
    root_address = 0;
    int num_leaf_attempts = 0;
    return get_entry(_root, address, num_leaf_attempts, root_address);
  }

  // Access entry of allocation that has the given address. Unlike get_entry(),
  // this does not succeed if the address does not point to the base of the allocation.
  value_type* get_entry_of_root_address(uint64_t address) noexcept {
    insert_or_get_entry_lock lock{_num_in_progress_operations};
    return get_entry_of_root_address(_root, address);
  }

  // Insert new element. Element's allocation range must be
  // non-overlapping w.r.t existing entries.
  // ~0ull is unsupported, because then non-zero allocation
  // ranges cannot be expressed.
  bool insert(uint64_t address, const value_type& v) {
    insert_or_get_entry_lock lock{_num_in_progress_operations};
    return insert(_root, address, v);
  }

  bool erase(uint64_t address) {
    erase_lock lock{_num_in_progress_operations};
    return erase(_root, address);
  }

  ~allocation_map() {
    for(int i = 0; i < get_num_entries_in_level(bit_tree_t::root_level_idx); ++i) {
      auto* ptr = _root.children[i].load(std::memory_order_acquire);
      if(ptr)
        release(*ptr);
    }
  }
    
private:
  // Useful for debugging/printing
  template<class F>
  void with_decomposed_address(uint64_t address, int current_level, F&& handler) {
    for(int i = this->root_level_idx; i >= current_level; --i) {
      handler(this->get_index_in_level(address, i));
    }
    for(int i = current_level - 1; i >= 0; --i) {
      handler(-1);
    }
  }

  template<class Ostream>
  void print(Ostream& ostr, uint64_t address, int level) {
    with_decomposed_address(address, level, [&](int x){
      if(x >= 0)
        ostr << x << ".";
      else
        ostr << "x";
    });
    ostr << "\n";
  }

  struct leaf_node {
    leaf_node()
    : num_entries {} {
      for(int i = 0; i < bit_tree_t::get_num_entries_in_level(0); ++i) {
        entries[i].allocation_size = 0;
      }
    }

    value_type entries [bit_tree_t::get_num_entries_in_level(0)];
    std::atomic<int> num_entries;
  };

  template<int Level>
  struct intermediate_node {
  private:
    static constexpr auto make_child() {
      if constexpr (Level > 1) return 
        intermediate_node<Level - 1>{};
      else return leaf_node{};
    }
  public:
    intermediate_node()
    : children{}, num_entries{} {}

    using child_type = decltype(make_child());

    std::atomic<child_type*> children [bit_tree_t::get_num_entries_in_level(Level)];
    std::atomic<int> num_entries;
  };

  value_type *get_entry(leaf_node &current_node, uint64_t address,
                        int &/*num_leaf_attempts*/,
                        uint64_t &root_address) noexcept {
    int start_address = 0;

    uint64_t max_local_address =
        root_address | (bit_tree_t::get_num_entries_in_level(0) - 1);
    
    if(max_local_address <= address)
      start_address = bit_tree_t::get_num_entries_in_level(0) - 1;
    else
      start_address = bit_tree_t::get_index_in_level(address, 0);

    for (int local_address = start_address; local_address >= 0;
         --local_address) {
      
      auto& element = current_node.entries[local_address];

      std::size_t allocation_size =
          __atomic_load_n(&(element.allocation_size), __ATOMIC_ACQUIRE);
      if(allocation_size > 0) {

        uint64_t root_address_candidate =
            root_address | (static_cast<uint64_t>(local_address)
                            << bit_tree_t::get_bitoffset_in_level(0));

        uint64_t allocation_end = root_address_candidate + allocation_size;
        if(address >= root_address_candidate && address < allocation_end) {
          root_address = root_address_candidate;
          return &element;
        } else {
          return nullptr;
        }
        
      }
    }
    return nullptr;
  }

  template <int Level>
  value_type *get_entry(intermediate_node<Level> &current_node,
                        uint64_t address,
                        int& num_leaf_attempts,
                        uint64_t& root_address) noexcept {
    // If the queried address is too close to the next allocation,
    // it can happen that the search converges on the next allocation.
    // Therefore, to exclude that case, if a search fails, we also
    // need to try again with the next allocation before that.
    // This variable counts how many leaves we have accessed. If it
    // reaches two, we can abort.
    if constexpr(Level == bit_tree_t::root_level_idx) {
      num_leaf_attempts = 0;
    }

    uint64_t max_local_address =
        root_address |
        get_n_low_bits_set(bit_tree_t::get_bitoffset_in_level(Level) +
                           bit_tree_t::bitsizes[Level]);

    // We are always looking for the next allocation preceding the
    // current address. If the maximum local address in this node
    // cannot reach the search address, (e.g. if we are looking in
    // a preceding node at the same level), we need to start from 
    // the maximum address. Otherwise, we need to look at the bits
    // set in this address.
    int start_address = 0;
    if(max_local_address <= address)
      start_address = bit_tree_t::get_num_entries_in_level(Level) - 1;
    else
      start_address = bit_tree_t::get_index_in_level(address, Level);

    for (int local_address = start_address;
         local_address >= 0; --local_address) {
      
      auto *ptr = current_node.children[local_address].load(
          std::memory_order_acquire);
      
      if(ptr) {
        uint64_t root_address_candidate =
            root_address | (static_cast<uint64_t>(local_address)
                            << bit_tree_t::get_bitoffset_in_level(Level));

        auto* ret = get_entry(*ptr, address, num_leaf_attempts,
                              root_address_candidate);
        // If we are in level 1, ret refers to a leaf node
        if constexpr(Level == 1) {
          ++num_leaf_attempts;
        }

        if(ret) {
          root_address = root_address_candidate;
          return ret;
        } else if(num_leaf_attempts >= 2) {
          // We can abort if we have looked at the first hit leaf node,
          // and the one before that.
          return nullptr;
        }
      }
    }
    return nullptr;
  }

  value_type *get_entry_of_root_address(leaf_node &current_node, uint64_t address) noexcept {
    int local_address = bit_tree_t::get_index_in_level(address, 0);
  
    auto& element = current_node.entries[local_address];
    std::size_t allocation_size =
        __atomic_load_n(&(element.allocation_size), __ATOMIC_ACQUIRE);

    if (allocation_size > 0) {
      return &element;
    }

    return nullptr;
  }

  template <int Level>
  value_type *get_entry_of_root_address(intermediate_node<Level> &current_node,
                                        uint64_t address) noexcept {
    int local_address = bit_tree_t::get_index_in_level(address, Level);
  
    auto *ptr = current_node.children[local_address].load(
          std::memory_order_acquire);
      
    if(ptr) {
      return get_entry_of_root_address(*ptr, address);
    }
    return nullptr;
  }

  bool insert(leaf_node &current_node, uint64_t address, const value_type &v) {

    int local_address = bit_tree_t::get_index_in_level(address, 0);

    std::size_t *allocation_size_ptr =
        &(current_node.entries[local_address].allocation_size);

    std::size_t allocation_size = __atomic_load_n(allocation_size_ptr, __ATOMIC_ACQUIRE);
    if(allocation_size > 0) {
      // Entry is already occupied
      return false;
    }
    
    __atomic_store_n(allocation_size_ptr, v.allocation_size, __ATOMIC_RELEASE);
    current_node.entries[local_address].UserPayload::operator=(v);
    
    current_node.num_entries.fetch_add(
        1, std::memory_order_acq_rel);

    return true;
  }

  template <int Level>
  bool insert(intermediate_node<Level> &current_node, uint64_t address,
              const value_type &v) {
    using child_t = typename intermediate_node<Level>::child_type;

    int local_address = bit_tree_t::get_index_in_level(address, Level);
    
    auto *ptr = current_node.children[local_address].load(
        std::memory_order_acquire);
    
    if(!ptr) {
      child_t* new_child = this->template alloc<child_t>(1);
      new (new_child) child_t{};

      if (!current_node.children[local_address].compare_exchange_strong(
              ptr /* == nullptr*/, new_child, std::memory_order_acq_rel)) {
        // Assigning new child has failed because child is no longer nullptr
        // -> free new child again
        destroy(*new_child);
        this->free(new_child);
      } else {
        current_node.num_entries.fetch_add(
            1, std::memory_order_acq_rel);
        ptr = new_child;
      }
    }

    return insert(*ptr, address, v);
  }

  bool erase(leaf_node& current_node, uint64_t address) {
    int local_address = bit_tree_t::get_index_in_level(address, 0);

    std::size_t *allocation_size_ptr =
        &(current_node.entries[local_address].allocation_size);
    // Entry was already deleted or does not exist
    if(__atomic_load_n(allocation_size_ptr, __ATOMIC_ACQUIRE) == 0)
      return false;

    __atomic_store_n(allocation_size_ptr, 0, __ATOMIC_RELEASE);

    current_node.num_entries.fetch_sub(
        1, std::memory_order_acq_rel);
    
    return true;
  }

  template <int Level>
  bool erase(intermediate_node<Level> &current_node, uint64_t address) {

    int local_address = bit_tree_t::get_index_in_level(address, Level);
    auto *ptr = current_node.children[local_address].load(
        std::memory_order_acquire);
    if(!ptr)
      return false;
    
    bool result = erase(*ptr, address);
    if(result) {
      if(ptr->num_entries.load(std::memory_order_acquire) == 0) {
        auto *current_ptr = current_node.children[local_address].exchange(
            nullptr, std::memory_order_acq_rel);
        // TODO: We could potentially get erase() lock-free
        // by counting by how many ops each node is currently used,
        // and waiting here until the count turns to 0.
        if(current_ptr) {
          destroy(*current_ptr);
          this->free(current_ptr);
          current_node.num_entries.fetch_sub(
              1, std::memory_order_acq_rel);
        }
      }
    }
    return result;
  }

  void release(leaf_node& current_node) {
    destroy(current_node);
  }

  template<int Level>
  void release(intermediate_node<Level>& current_node) {
    for(int i = 0; i < bit_tree_t::get_num_entries_in_level(Level); ++i){
      if (auto *ptr = current_node.children[i].load(
              std::memory_order_acquire)) {
        release(*ptr);
        this->free(ptr);
      }
    }
    destroy(current_node);
  }

  void destroy(leaf_node& node) {
    node.~leaf_node();
  }

  template<int Level>
  void destroy(intermediate_node<Level>& node) {
    node.~intermediate_node<Level>();
  }

  struct erase_lock {
  public:
    erase_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = 0;
      while (!_op_counter.compare_exchange_strong(
          expected, -1, std::memory_order_release, std::memory_order_relaxed)) {
        expected = 0;
      }
    }

    ~erase_lock() {
      _op_counter.store(0, std::memory_order_release);
    }
  private:
    std::atomic<int>& _op_counter;
  };

  struct insert_or_get_entry_lock {
  public:
    insert_or_get_entry_lock(std::atomic<int>& op_counter)
    : _op_counter{op_counter} {
      int expected = std::max(0, _op_counter.load(std::memory_order_acquire));
      while (!_op_counter.compare_exchange_strong(
          expected, expected+1, std::memory_order_release,
          std::memory_order_relaxed)) {
        if(expected < 0)
          expected = 0;
      }
    }

    ~insert_or_get_entry_lock() {
      _op_counter.fetch_sub(1, std::memory_order_acq_rel);
    }
  private:
   std::atomic<int>& _op_counter;
  };

  intermediate_node<bit_tree_t::root_level_idx> _root;
  std::atomic<int> _num_in_progress_operations;
};


}

#endif
