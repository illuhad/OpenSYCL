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
#ifndef HIPSYCL_DATA_HPP
#define HIPSYCL_DATA_HPP

#include <limits>
#include <mutex>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "dag_node.hpp"
#include "device_id.hpp"
#include "util.hpp"
#include "allocator.hpp"

namespace hipsycl {
namespace rt {

class range_store
{
public:
  using rect = std::pair<id<3>, range<3>>;

  enum class data_state : char
  {
    empty = 0,
    available,
  };

  range_store(range<3> size);

  void add(const rect& r);
  
  void remove(const rect& r);

  range<3> get_size() const;

  void intersections_with(const rect& r, 
                          data_state desired_state,
                          std::vector<rect>& out) const;

  void intersections_with(const rect& r, 
                          std::vector<rect>& out) const
  { intersections_with(r, data_state::available, out); }

  void inverted_intersections_with(const rect& r, 
                                  std::vector<rect>& out) const
  { intersections_with(r, data_state::empty, out); }                             
    
  bool entire_range_equals(const rect&, data_state desired_state) const;

  bool entire_range_filled(const rect& r) const
  { return entire_range_equals(r, data_state::available); }

  bool entire_range_empty(const rect& r) const
  { return entire_range_equals(r, data_state::empty); }

private:
  template<class Entry_selection_predicate>
  range<3> find_max_contiguous_rect_extent(
    id<3> begin,
    id<3> search_range_end,
    Entry_selection_predicate p) const
  {
    // Find out length of longest contiguous row
    size_t z_size = 
      find_z_size(begin, search_range_end, p);

    // Now go to 2D and find out width of surface
    size_t y_size = 1;
    if(_size[1] > 1)
      y_size = find_y_size(begin, search_range_end, z_size, p);

    // Now 3D
    size_t x_size = 1;
    if(_size[0] > 1)
      x_size = find_x_size(begin, search_range_end, z_size, y_size, p);

    return range<3>{x_size, y_size, z_size};
    
  }

  template<class Entry_selection_predicate>
  size_t find_x_size(id<3> begin,
                     id<3> search_range_end,
                     size_t z_size,
                     size_t y_size,
                     Entry_selection_predicate p) const
  {
    for(size_t x_offset = 0; x_offset < search_range_end[0]-begin[0]; ++x_offset){
      id<3> surface_begin = begin;
      surface_begin[0] += x_offset;

      if(find_y_size(surface_begin, search_range_end, z_size, p) != y_size){
        return x_offset;
      }
    }
    return search_range_end[0] - begin[0];
  }

  template<class Entry_selection_predicate>
  size_t find_y_size(id<3> begin,
                    id<3> search_range_end,
                    size_t z_size,
                    Entry_selection_predicate p) const
  {
    for(size_t y_offset = 0; y_offset < search_range_end[1]-begin[1]; ++y_offset){
      id<3> row_begin = begin;
      row_begin[1] += y_offset;

      if(find_z_size(row_begin, search_range_end, p) != z_size){
        return y_offset;
      }
    }
    return search_range_end[1] - begin[1];
  }

  template<class Entry_selection_predicate>
  size_t find_z_size(id<3> begin,
                    id<3> search_range_end,
                    Entry_selection_predicate p) const
  {
    size_t base_pos = get_index(begin);
    size_t max_length = search_range_end[2] - begin[2];
    for(size_t offset = 0; offset < max_length; ++offset)
    {
      if(!p(base_pos + offset))
        return offset;
    }
    return max_length;
  }

  template<class Function>
  void for_each_element_in_range(const rect& r,
    std::vector<data_state>& v,
    Function f) const
  {
    for(size_t x = r.first[0]; x < r.second[0]+r.first[0]; ++x){
      for(size_t y = r.first[1]; y < r.second[1]+r.first[1]; ++y){
        for(size_t z = r.first[2]; z < r.second[2]+r.first[2]; ++z){

          id<3> idx{x,y,z};
          size_t pos = get_index(idx);
          assert(pos < v.size());
          f(id<3>{x,y,z}, v[pos]);
        }
      }
    }
  }

  template<class Function>
  void for_each_element_in_range(const rect& r, 
    const std::vector<data_state>& v, 
    Function f) const
  {
    for(size_t x = r.first[0]; x < r.second[0]+r.first[0]; ++x){
      for(size_t y = r.first[1]; y < r.second[1]+r.first[1]; ++y){
        for(size_t z = r.first[2]; z < r.second[2]+r.first[2]; ++z){

          id<3> idx{x,y,z};
          size_t pos = get_index(idx);
          assert(pos < v.size());
          f(id<3>{x,y,z}, v[pos]);
        }
      }
    }
  }

  template<class Function>
  void for_each_element_in_range(const rect& r, Function f) const
  { for_each_element_in_range(r, _contained_data, f); }

  template<class Function>
  void for_each_element_in_range(const rect& r, Function f)
  { for_each_element_in_range(r, _contained_data, f); }

  size_t get_index(id<3> pos) const
  {
    return pos[0] * _size[1] * _size[2] + pos[1] * _size[2] + pos[2]; 
  }

  range<3> _size;
  std::vector<data_state> _contained_data;
};


struct data_user
{
  std::weak_ptr<dag_node> user;
  sycl::access::mode mode;
  sycl::access::target target;
  id<3> offset;
  rt::range<3> range;
};


class data_user_tracker
{
public:
  using user_iterator = std::vector<data_user>::iterator;
  using const_user_iterator = std::vector<data_user>::const_iterator;

  data_user_tracker() = default;
  data_user_tracker(const data_user_tracker& other);
  data_user_tracker(data_user_tracker&& other);
  data_user_tracker& operator=(data_user_tracker other);
  data_user_tracker& operator=(data_user_tracker&& other);

  const std::vector<data_user> get_users() const;

  template<class F>
  void for_each_user(F f){
    std::lock_guard<std::mutex> lock{_lock};
    // Iterate in reverse order over the users since
    // this will iterate over the newest users first.
    // This is a more advantageous pattern e.g. during
    // DAG construction as it allows finding the relevant users
    // quicker.
    for(int i = _users.size() - 1; i >= 0; --i) {
      f(_users[i]);
    }
  }

  bool has_user(dag_node_ptr user) const;

  void release_dead_users();

  template<class Predicate>
  void add_user(dag_node_ptr user, 
                sycl::access::mode mode, 
                sycl::access::target target, 
                id<3> offset,
                range<3> range,
                Predicate replaces_user) {
    std::lock_guard<std::mutex> lock{_lock};

    _users.erase(std::remove_if(_users.begin(), _users.end(), replaces_user), 
      _users.end());

    _users.push_back(
      data_user{std::weak_ptr<dag_node>(user), mode, target, offset, range});
  }

private:
  std::vector<data_user> _users;
  mutable std::mutex _lock;
};

template <class Memory_descriptor>
struct data_allocation {

  using allocation_function = std::function<Memory_descriptor(
      range<3> num_elements, std::size_t element_size)>;
  
  device_id dev;
  Memory_descriptor memory;
  range_store invalid_pages;
  bool is_owned;
  backend_allocator* managing_allocator;
};

template <class Memory_descriptor> class allocation_list {
public:
  template<class BinaryPredicate>
  bool add_if_unique(BinaryPredicate&& comparator, data_allocation<Memory_descriptor> &&new_alloc) {
    std::lock_guard<std::mutex> lock{_mutex};

    for (const auto &alloc : _allocations) {
      if (comparator(alloc, new_alloc))
        return false;
    }

    _allocations.push_back(new_alloc);
    return true;
  }

  template <class Handler> void for_each_allocation_while(Handler &&h) const {
    std::lock_guard<std::mutex> lock{_mutex};

    for (const auto &alloc : _allocations) {
      if (!h(alloc))
        break;
    }
  }

  template <class Handler> void for_each_allocation_while(Handler &&h) {
    std::lock_guard<std::mutex> lock{_mutex};

    for (auto &alloc : _allocations) {
      if (!h(alloc))
        break;
    }
  }

  template <class UnaryPredicate, class Handler>
  bool select_and_handle(UnaryPredicate &&selector, Handler &&h) {
    std::lock_guard<std::mutex> lock{_mutex};
    for (auto &alloc : _allocations) {
      if (selector(alloc)) {
        h(alloc);
        return true;
      }
    }
    return false;
  }

  template <class UnaryPredicate, class Handler>
  bool select_and_handle(UnaryPredicate &&selector, Handler &&h) const {
    std::lock_guard<std::mutex> lock{_mutex};
    for (const auto &alloc : _allocations) {
      if (selector(alloc)) {
        h(alloc);
        return true;
      }
    }
    return false;
  }

  template <class UnaryPredicate>
  bool has_match(UnaryPredicate &&selector) const {
    return select_and_handle(selector, [](const auto&){});
  }
private:
  std::vector<data_allocation<Memory_descriptor>> _allocations;
  mutable std::mutex _mutex;
};


/// Manages data regions on different devices under
/// the assumptions:
/// * different devices may have copies of the same data regions
/// * data is managed at a "page" granularity (a 3D subrange of the data region)
/// * data accesses cover a 3D (sub-)range of the whole buffer
/// * data ranges on some devices may be outdated due to writes on
///   other devices
///
/// The interface of this class works in terms of numbers of elements, not
/// bytes!
template<class Memory_descriptor = void*>
class data_region
{
public:
  /// Controls when two allocations are considered equal in order
  /// to maintain the requirement that allocations are unique.
  class default_allocation_comparator {
  public:
    bool operator()(const data_allocation<Memory_descriptor> &a1,
                    const data_allocation<Memory_descriptor> &a2) const {
      return a1.dev == a2.dev;
    }
  };
  /// Controls which allocation is selected when looking for an
  /// allocation to use on a given device.
  /// Together with \c default_allocation_comparator, this currently
  /// enforces the policy that each device has its own dedicated allocation.
  /// However in the future it may be desirable to share allocations
  /// between multiple devices, e.g. different CPU backends.
  class default_allocation_selector {
  public:
    default_allocation_selector(rt::device_id dev) : _dev{dev} {}

    bool operator()(const data_allocation<Memory_descriptor> &alloc) const {
      return alloc.dev == _dev;
    }
  private: device_id _dev;
  };

  using page_range = std::pair<id<3>, range<3>>;
  using allocation_function =
      typename data_allocation<Memory_descriptor>::allocation_function;


  /// Construct object
  /// \param num_elements The 3D number of elements in each dimension. Each
  /// dimension must be a multiple of the page size
  /// \param page_size The size (numbers of elements) of the granularity of data
  /// management
  data_region(
      range<3> num_elements, std::size_t element_size, range<3> page_size)
      : _element_size{element_size}, _page_size{page_size},
        _num_elements{num_elements} {

    for(std::size_t i = 0; i < 3; ++i){
      assert(page_size[i] > 0);
      _num_pages[i] = (num_elements[i] + page_size[i] - 1) / page_size[i];
      assert(_num_pages[i] > 0);
    }

    HIPSYCL_DEBUG_INFO << "data_region: constructed with page table dimensions "
                       << _num_pages[0] << " " << _num_pages[1] << " "
                       << _num_pages[2] << std::endl;
  }

  ~data_region() {
    _allocations.for_each_allocation_while([](auto& alloc) {
      if(alloc.memory && alloc.is_owned) {
        device_id dev = alloc.dev;
        HIPSYCL_DEBUG_INFO << "data_region::~data_region: Freeing allocation "
                           << alloc.memory << std::endl;
        if(!alloc.managing_allocator) {
          HIPSYCL_DEBUG_WARNING
              << "data_region::~data_region: Cannot free owned allocation "
              << alloc.memory
              << " because no managing allocator was provided. This is likely "
                 "a memory leak."
              << std::endl;
        } else {
          rt::deallocate(alloc.managing_allocator, alloc.memory);
        }
      }
      return true;
    });
  }

  /// Iterate over all allocations, abort as soon as \c h() returns false.
  /// \param h A callable of signature \c bool(const data_allocation&)
  template <class Handler> void for_each_allocation_while(Handler &&h) const {
    _allocations.for_each_allocation_while(h);
  }

  /// Iterate over all allocations, abort as soon as \c h() returns false.
  /// \param h A callable of signature \c bool(data_allocation&)
  template <class Handler> void for_each_allocation_while(Handler &&h) {
    _allocations.for_each_allocation_while(h);
  }

  bool has_allocation(const device_id &d) const {
    return _allocations.has_match(default_allocation_selector{d});
  }

  void add_empty_allocation(const device_id &d,
                            Memory_descriptor memory_context,
                            backend_allocator* allocator,
                            bool takes_ownership = true) {
    // Make sure that there isn't already an allocation on the given device
    assert(!has_allocation(d));

    this->add_allocation<initial_data_state::invalid>(d, memory_context,
                                                      takes_ownership,
                                                      allocator);
  }

  void add_nonempty_allocation(const device_id &d,
                               Memory_descriptor memory_context,
                               backend_allocator* allocator,
                               bool takes_ownership = false) {
    // Make sure that there isn't already an allocation on the given device
    assert(!has_allocation(d));

    // TODO in principle we would also need to invalidate other allocations,
    // if we get a *new* allocation that should now be considered valid.
    // In practice this is not really needed because this function
    // is only invoked at the initialization of a buffer if constructed
    // with an existing pointer (e.g. host pointer).
    this->add_allocation<initial_data_state::valid>(d, memory_context,
                                                    takes_ownership,
                                                    allocator);
  }

  /// Converts an offset into the data buffer (in element numbers) and the
  /// data length (in element numbers) into an equivalent \c page_range.
  page_range get_page_range(id<3> data_offset, range<3> data_range) const {
    // Is thread safe without lock because it doesn't modify internal state
    // and doesn't access mutable members.
    id<3> page_begin{0,0,0};
    
    for(int i = 0; i < 3; ++i)
      page_begin[i] =  data_offset[i] / _page_size[i];

    id<3> page_end = page_begin;

    for (int i = 0; i < 3; ++i)
      page_end[i] =
          (data_offset[i] + data_range[i] + _page_size[i] - 1) / _page_size[i];
    

    range<3> page_range{1,1,1};
    for (int i = 0; i < 3; ++i)
      page_range[i] = page_end[i] - page_begin[i];

    return std::make_pair(page_begin, page_range);
  }

  /// Marks an allocation range on a give device as not invalidated
  void mark_range_valid(const device_id &d, id<3> data_offset,
                        range<3> data_size)
  {
    page_range pr = get_page_range(data_offset, data_size);

    assert(has_allocation(d));

    _allocations.select_and_handle(default_allocation_selector{d},
                                   [&](auto &alloc) {
      alloc.invalid_pages.remove(pr);              
    });
  }
  
  /// Marks an allocation range on a given device most recent
  void mark_range_current(const device_id& d,
      id<3> data_offset,
      range<3> data_size)
  {
    page_range pr = get_page_range(data_offset, data_size);

    default_allocation_selector argument_match{d};

    _allocations.for_each_allocation_while([&](auto &alloc) {
      if (argument_match(alloc)) {
        alloc.invalid_pages.remove(pr);
      } else {
        alloc.invalid_pages.add(pr);
      }
      return true;
    });
  }

  void get_outdated_regions(const device_id& d,
                            id<3> data_offset,
                            range<3> data_size,
                            std::vector<range_store::rect>& out) const
  {
    assert(has_allocation(d));

    // Convert byte offsets/sizes to page ranges
    page_range pr = get_page_range(data_offset, data_size);
    id<3> first_page = pr.first;
    range<3> num_pages = pr.second;

    // Find outdated regions among pages
    [[maybe_unused]] bool was_found = _allocations.select_and_handle(
        default_allocation_selector{d}, [&](auto &alloc) {
          alloc.invalid_pages.intersections_with(
              std::make_pair(first_page, num_pages), out);
        });
    
    assert(was_found);
    
    // Convert back to num elements
    for(range_store::rect& r : out) {
      for(int i = 0; i < 3; ++i) {
        r.first[i] *= _page_size[i];
        r.second[i] *= _page_size[i];

        // Clamp result range to data range. This is necessary
        // if the number of elements is not divisible by the page
        // size, in which case we can end up out of bounds when mapping
        // pages back to elements.
        r.first[i] = std::min(r.first[i], _num_elements[i]);

        std::size_t max_range = _num_elements[i] - r.first[i];
        r.second[i] = std::min(r.second[i], max_range);

        assert(r.first[i]+r.second[i] <= _num_elements[i]);
      }
    }
  }

  void get_update_source_candidates(
              const device_id& d,
              const range_store::rect& data_range,
              std::vector<std::pair<device_id, range_store::rect>>& update_sources) const
  {
    update_sources.clear();

    page_range pr = get_page_range(data_range.first, data_range.second);

    default_allocation_selector selector{d};
    _allocations.for_each_allocation_while([&](const auto &alloc) {
      // Find all valid pages that are *not* accessible on the given device
      if (!selector(alloc)) {
        if(alloc.invalid_pages.entire_range_empty(pr)) {
          update_sources.push_back(std::make_pair(alloc.dev, data_range));
        }
      }
      return true;
    });
    if(update_sources.empty()){
      assert(false && "Could not find valid data source for updating data buffer - "
              "this can happen if several data transfers are required to update accessed range, "
              "which is not yet supported.");
    }
  }

  data_user_tracker& get_users()
  { return _user_tracker; }

  const data_user_tracker& get_users() const
  { return _user_tracker; }

  std::size_t get_element_size() const { return _element_size; }

  range<3> get_num_elements() const { return _num_elements; }

  Memory_descriptor get_memory(device_id dev) const
  {
    assert(has_allocation(dev));

    Memory_descriptor mem{};

    [[maybe_unused]] bool was_found = _allocations.select_and_handle(
        default_allocation_selector{dev}, [&](const auto &alloc) {
          mem = alloc.memory;
    });

    assert(was_found);

    return mem;
  }

  data_allocation<Memory_descriptor> get_allocation(device_id dev) const {
    assert(has_allocation(dev));

    data_allocation<Memory_descriptor> found_alloc;

    [[maybe_unused]] bool was_found = _allocations.select_and_handle(
        default_allocation_selector{dev}, [&](const auto &alloc) {
          found_alloc = alloc;
    });

    assert(was_found);

    return found_alloc;
  }

  template <class Handler>
  bool find_and_handle_allocation(device_id dev, Handler &&h) const {
    return _allocations.select_and_handle(default_allocation_selector{dev}, h);
  }

  template <class Handler>
  bool find_and_handle_allocation(device_id dev, Handler &&h) {
    return _allocations.select_and_handle(default_allocation_selector{dev}, h);
  }

  template <class Handler>
  bool find_and_handle_allocation(Memory_descriptor mem, Handler &&h) const {
    return _allocations.select_and_handle([mem](const auto &alloc) {
      return alloc.memory == mem;
    }, h);
  }

  template <class Handler>
  bool find_and_handle_allocation(Memory_descriptor mem, Handler &&h) {
    return _allocations.select_and_handle([mem](const auto &alloc) {
      return alloc.memory == mem;
    }, h);
  }
  
  bool has_initialized_content(id<3> data_offset,
                               range<3> data_range) const {
    page_range pr = get_page_range(data_offset, data_range);

    bool found_valid_pages = false;

    _allocations.for_each_allocation_while([&](const auto &alloc) {
      if (!alloc.invalid_pages.entire_range_filled(pr)) {
        found_valid_pages = true;
        return false;
      }
      return true;
    });

    return found_valid_pages;
  }

private:
  std::size_t _element_size;

  allocation_list<Memory_descriptor> _allocations;

  enum class initial_data_state {
    valid,
    invalid
  };

  template <initial_data_state InitialState>
  void add_allocation(const device_id &d, Memory_descriptor memory_context,
                      bool takes_ownership = true,
                      backend_allocator *allocator = nullptr) {
    // Make sure that there isn't already an allocation on the given device
    assert(!has_allocation(d));

    data_allocation<Memory_descriptor> new_alloc{
        d, memory_context, range_store{_num_pages}, takes_ownership, allocator};

    if constexpr(InitialState == initial_data_state::invalid) {
      new_alloc.invalid_pages.add(std::make_pair(id<3>{0, 0, 0}, _num_pages));
    } else {
      new_alloc.invalid_pages.remove(std::make_pair(id<3>{0, 0, 0}, _num_pages));
    }

    [[maybe_unused]] bool was_inserted = _allocations.add_if_unique(
        default_allocation_comparator{}, std::move(new_alloc));

    // If another thread has added an allocation for this device in the meantime
    // this may fail. The API however currently does not allow for this to
    // happen as allocations are either added at buffer construction, or
    // later on by the scheduler.
    assert(was_inserted);
  }

  range<3> _page_size;
  range<3> _num_pages;
  range<3> _num_elements;

  data_user_tracker _user_tracker;
};

using buffer_data_region = data_region<void*>;



}
}

#endif
