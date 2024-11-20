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
#ifndef HIPSYCL_ACCESSOR_HPP
#define HIPSYCL_ACCESSOR_HPP

#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <cassert>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/embedded_pointer.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/data.hpp"

#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/sycl/extensions.hpp"
#include "hipSYCL/sycl/buffer.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/buffer_allocator.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/property.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include "hipSYCL/sycl/property.hpp"
#include "range.hpp"
#include "item.hpp"
#include "multi_ptr.hpp"
#include "atomic.hpp"
#include "../specialized.hpp"
#include "detail/local_memory_allocator.hpp"
#include "detail/mobile_shared_ptr.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

template<access_mode M, target T = target::device, bool IsRaw = false>
struct mode_tag_descriptor {
  static constexpr sycl::access_mode mode = M;
  static constexpr sycl::target target = T;
  static constexpr bool is_raw = IsRaw;
};

template<class Descriptor>
struct mode_tag {
  static constexpr sycl::access_mode mode = Descriptor::mode;
  static constexpr sycl::target target = Descriptor::target;
  static constexpr bool is_raw = Descriptor::is_raw;
};

using read_only_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::read, target::device>>;
using read_write_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::read_write, target::device>>;
using write_only_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::write, target::device>>;

using read_only_host_task_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::read, target::host_task>>;
using read_write_host_task_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::read_write, target::host_task>>;
using write_only_host_task_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::write, target::host_task>>;

using read_only_raw_tag_t  = mode_tag<
  mode_tag_descriptor<access_mode::read, target::device, true>>;
using read_write_raw_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::read_write, target::device, true>>;
using write_only_raw_tag_t = mode_tag<
  mode_tag_descriptor<access_mode::write, target::device, true>>;

template <typename TagDescT>
constexpr accessor_variant deduce_accessor_variant(mode_tag<TagDescT> tag,
                                                   accessor_variant fallback) {
  if constexpr (tag.is_raw) {
    return accessor_variant::raw;
  } else {
    return fallback;
  }
}

// Defined in buffer.hpp
template <class BufferT>
std::shared_ptr<rt::buffer_data_region>
extract_buffer_data_region(const BufferT &buff);

template <class T, int dimensions, class AllocatorT>
sycl::range<dimensions>
extract_buffer_range(const buffer<T, dimensions, AllocatorT> &buff);

template <typename T, int Dimensions,
          typename Accessor>
class accessor_iterator {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type  = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  accessor_iterator() = default;

  reference operator*() const {
    return (*acc_ptr)[id_from_linear()];
  }

  accessor_iterator &operator++() {
    ++linear_id;
    return *this;
  }

  accessor_iterator operator++(int) {
    auto old = *this;
    ++(*this);
    return old;
  }

  accessor_iterator &operator--() {
    --linear_id;
    return *this;
  }

  accessor_iterator operator--(int) {
    auto old = *this;
    --(*this);
    return old;
  }

  accessor_iterator &operator+=(difference_type diff) {
    linear_id += diff;
    return *this;
  }

  accessor_iterator operator+(difference_type diff) const {
    auto ret = *this;
    ret += diff;
    return ret;
  }

  friend accessor_iterator operator+(difference_type diff,
                                     const accessor_iterator &rhs) {
    auto ret = rhs;
    ret += diff;
    return ret;
  }

  accessor_iterator &operator-=(difference_type diff) {
    linear_id -= diff;
    return *this;
  }

  accessor_iterator operator-(difference_type diff) const {
    auto ret = *this;
    ret -= diff;
    return ret;
  }

  friend accessor_iterator operator-(difference_type diff,
                                     const accessor_iterator &rhs) {
    auto ret = rhs;
    ret -= diff;
    return ret;
  }

  reference &operator[](difference_type diff) const {
    auto ret = *this;
    ret += diff;
    return *ret;
  }

  bool operator==(const accessor_iterator &other) const {
    return linear_id == other.linear_id;
  }

  bool operator!=(const accessor_iterator &other) const {
    return !(*this == other);
  }

  bool operator<(const accessor_iterator &other) const {
    return linear_id < other.linear_id;
  }

  bool operator>(const accessor_iterator &other) const {
    return other < *this; 
  }

  bool operator <=(const accessor_iterator &other) const {
    return !(*this > other);
  }

  bool operator >=(const accessor_iterator &other) const {
    return !(*this < other);
  }

  difference_type operator-(const accessor_iterator &rhs) const {
    return linear_id - rhs.linear_id;
  }
  
private:
  template <typename, int,
            sycl::access_mode,
            sycl::target,
            sycl::accessor_variant> friend class sycl::accessor;
  template <typename, int,
            sycl::access_mode> friend class sycl::host_accessor;

  const Accessor *acc_ptr = nullptr;

  /* Linear id relative to the offset, i.e., linear_id = 0
     corresponds to the element at the offset. */
  size_t linear_id;

  // Constructs an iterator pointing to the beginning of the accessor's data
  accessor_iterator(const Accessor* acc_ptr)
    : acc_ptr{acc_ptr}, linear_id{0} {}
  
  static accessor_iterator make_begin(const Accessor *acc_ptr) {
    return accessor_iterator{acc_ptr};
  }

  static accessor_iterator make_end(const Accessor *acc_ptr) {
    auto end = accessor_iterator{acc_ptr};
    end.linear_id = acc_ptr->get_range().size();    
    return end;
  }

  /* Computes a sycl::id corresponding to the current linear id.
     Since the linear id is always relative to the offset, we can
     compute the coordinates w.r.t. the sub range and ignore the
     size of the whole underlying buffer range.

     When dereferencing the iterator, these coordinates will then
     be converted back to a linear id using /the buffer range/
     (and not the sub range) which will then give the correct
     linear id within the whole accessor.
   */
  sycl::id<Dimensions> id_from_linear() const {
    if constexpr (Dimensions == 1)
      return linear_id;

    if constexpr (Dimensions == 2) {
      const auto range = acc_ptr->get_range();
      auto y = linear_id / range[1];
      auto z = linear_id % range[1];

      return {y, z};
   }

    if constexpr (Dimensions == 3) {
      const auto range = acc_ptr->get_range();
      auto m_linear_id = linear_id;

      auto x = linear_id / (range[1] * range[2]);
      m_linear_id -= x * (range[1] * range[2]);
      auto y = m_linear_id / range[2];
      auto z = linear_id % range[2];

      return {x, y, z};
    }
  }
};
}

inline constexpr detail::read_only_tag_t read_only;
inline constexpr detail::read_write_tag_t read_write;
inline constexpr detail::write_only_tag_t write_only;

inline constexpr detail::read_only_host_task_tag_t read_only_host_task;
inline constexpr detail::read_write_host_task_tag_t read_write_host_task;
inline constexpr detail::write_only_host_task_tag_t write_only_host_task;

inline constexpr detail::read_only_raw_tag_t read_only_raw;
inline constexpr detail::read_write_raw_tag_t read_write_raw;
inline constexpr detail::write_only_raw_tag_t write_only_raw;

class handler;

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget,
         access::placeholder isPlaceholder>
class accessor;

namespace detail::handler {

template<class T>
inline
detail::local_memory::address allocate_local_mem(sycl::handler&,
                                                 size_t num_elements);

} // detail::handler

namespace detail::accessor {

template<class T, access_mode M>
struct accessor_data_type {
  using value = T;
};

template<class T>
struct accessor_data_type<T, access_mode::read> {
  using value = const T;
};

template <typename dataT, int dimensions, access_mode accessmode,
          target accessTarget, access::placeholder isPlaceholder,
          int current_dimension = 1>
class subscript_proxy
{
  ACPP_UNIVERSAL_TARGET
  static constexpr bool can_invoke_access(int current_dim, int dim) {
    return current_dim == dim - 1;
  }
public:
  static_assert(dimensions > 1, "dimension must be > 1");
  
  using accessor_type = sycl::accessor<dataT, dimensions, accessmode,
                                       accessTarget, isPlaceholder>;
  using reference = typename accessor_type::reference;

  using next_subscript_proxy =
      subscript_proxy<dataT, dimensions, accessmode, accessTarget,
                      isPlaceholder, current_dimension+1>;

  ACPP_UNIVERSAL_TARGET
  subscript_proxy(const accessor_type *original_accessor,
                  sycl::id<dimensions> current_access_id)
      : _original_accessor{original_accessor}, _access_id{current_access_id} {}


  template <int D = dimensions,
            int C = current_dimension,
            std::enable_if_t<!can_invoke_access(C, D), bool> = true>
  ACPP_UNIVERSAL_TARGET
  next_subscript_proxy operator[](size_t index) const {
    return create_next_proxy(index);
  }

  template <int D = dimensions,
            int C = current_dimension,
            access_mode M = accessmode,
            std::enable_if_t<can_invoke_access(C, D) && (M != access_mode::atomic), bool> = true>
  ACPP_UNIVERSAL_TARGET
  reference operator[](size_t index) const {
    return invoke_value_access(index);
  }

  template <int D = dimensions,
            int C = current_dimension,
            access_mode M = accessmode,
            std::enable_if_t<can_invoke_access(C, D) && (M == access_mode::atomic), bool> = true>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] ACPP_UNIVERSAL_TARGET
  auto operator[](size_t index) const {
    return invoke_atomic_value_access(index);
  }

private:
  ACPP_UNIVERSAL_TARGET
  reference invoke_value_access(size_t index) const {
    // Set the last index
    _access_id[dimensions - 1] = index;
    return (*_original_accessor)[_access_id];
  }

  ACPP_UNIVERSAL_TARGET
  auto invoke_atomic_value_access(size_t index) const {
    // Set the last index
    _access_id[dimensions - 1] = index;
    return (*_original_accessor)[_access_id];
  }

  ACPP_UNIVERSAL_TARGET
  next_subscript_proxy create_next_proxy(size_t next_id) const {
    _access_id[current_dimension] = next_id;
    return next_subscript_proxy{_original_accessor, _access_id};
  }

  const accessor_type *_original_accessor;
  mutable sycl::id<dimensions> _access_id;
};


// This function is defined in handler.hpp
template<class AccessorType>
void bind_to_handler(AccessorType& acc, sycl::handler& cgh);

template <class AccessorType, int Dim>
void bind_to_handler(AccessorType &acc, sycl::handler &cgh,
                     std::shared_ptr<rt::buffer_data_region> mem,
                     sycl::id<Dim> offset, sycl::range<Dim> range,
                     bool is_no_init);


template<class AccessorType>
glue::unique_id get_unique_id(const AccessorType& acc);

struct accessor_properties {
public:
  accessor_properties()
  : _flags{0} {}

  accessor_properties(bool is_placeholder, bool is_no_init)
  : _flags {0} {
    if(is_placeholder)
      _flags |= bit_placeholder;
    if(is_no_init)
      _flags |= bit_no_init;
  }

  bool is_placeholder() const {
    return (_flags & bit_placeholder) != 0;
  }

  bool is_no_init() const {
    return (_flags & bit_no_init) != 0;
  }

private:
  unsigned char _flags;

  static constexpr int bit_placeholder = 1 << 0;
  static constexpr int bit_no_init     = 1 << 1;
};

template<int Dim>
struct access_range {
  sycl::id<Dim> offset;
  sycl::range<Dim> range;
};

inline constexpr bool stores_buffer_pointer(accessor_variant V) {
  if(V == accessor_variant::ranged || V == accessor_variant::unranged ||
    V == accessor_variant::raw)
    return false;
  return true;
}

inline constexpr bool stores_buffer_range(accessor_variant V) {
  return V != accessor_variant::raw;
}

inline constexpr bool stores_access_range(accessor_variant V) {
  if (V == accessor_variant::unranged ||
      V == accessor_variant::unranged_placeholder || V == accessor_variant::raw)
    return false;
  return true;
}

inline constexpr bool stores_accessor_properties(accessor_variant V) {
  // We need to store placeholder and no init flags whenever we are forced into
  // standard SYCL 2020 sycl::access::placeholder semantics instead
  // of the hipSYCL-specific other values of the accessor_variant enum.
  return V == accessor_variant::false_t || V == accessor_variant::true_t;
}

} // detail::accessor

namespace detail {

template<class TagT, bool Enable, class T>
class conditional_storage {
public:

  conditional_storage() = default;
  conditional_storage(const T&) {}

  conditional_storage(const conditional_storage&) = default;
  // We are copying from a type that stores while we don't - can't do anything.
  conditional_storage(const conditional_storage<TagT, true, T>&) {}

protected:
  using value_type = T;

  static constexpr bool value_exists() {
    return Enable;
  }

  bool attempt_set(const T& val) { return false; }
  T get(T default_val = T{}) const { return default_val; }

  T* ptr() { return nullptr; }
  const T* ptr() const { return nullptr; }
};

template<class TagT, class T>
class conditional_storage<TagT, true, T> {
public:

  conditional_storage() = default;
  conditional_storage(const T& v) 
  : _val{v} {}

  conditional_storage(const conditional_storage&) = default;
  // We are copying from a type that doesn't store
  conditional_storage(const conditional_storage<TagT, false, T>&)
  : _val{} {}

protected:
  using value_type = T;

  static constexpr bool value_exists() {
    return true;
  }

  bool attempt_set(const T &val) {
    _val = val;
    return true;
  }

  T get(T default_val = T{}) const { return _val; }

  T* ptr() { return &_val; }
  const T* ptr() const { return &_val; }
private:
  T _val;
};

namespace accessor {

struct conditional_storage_buffer_pointer_tag_t {};
struct conditional_storage_access_range_tag_t {};
struct conditional_storage_buffer_range_tag_t {};
struct conditional_storage_accessor_properties_tag_t {};

template <bool Enable>
using conditional_buffer_pointer_storage =
    conditional_storage<conditional_storage_buffer_pointer_tag_t, Enable,
                        detail::mobile_shared_ptr<rt::buffer_data_region>>;

template <bool Enable, int Dim>
using conditional_access_range_storage =
    conditional_storage<conditional_storage_access_range_tag_t, Enable,
                        access_range<Dim>>;

template <bool Enable, int Dim>
using conditional_buffer_range_storage =
    conditional_storage<conditional_storage_buffer_range_tag_t, Enable,
                        sycl::range<Dim>>;

template <bool Enable>
using conditional_accessor_properties_storage =
    conditional_storage<conditional_storage_accessor_properties_tag_t, Enable,
                        accessor_properties>;

}

inline access_mode get_effective_access_mode(access_mode accessmode,
                                             bool is_no_init) {
  access_mode mode = accessmode;

  if(mode == access_mode::atomic){
    mode = access_mode::read_write;
  }

  if(is_no_init) {
    if(mode == access_mode::write) {
      mode = access_mode::discard_write;
    } else if(mode == access_mode::read_write) {
      mode = access_mode::discard_read_write;
    }
  }

  return mode;
}

template <class T, int Dim>
rt::range<3> get_effective_range(const std::shared_ptr<rt::buffer_data_region> &mem_region,
                                  const rt::range<Dim> range, const rt::range<Dim> buffer_shape,
                                  const bool has_access_range) {
  // todo: optimize range / offset (would have to calculate a bounding box for reshaped buffers..)
  if(!has_access_range || sizeof(T) != mem_region->get_element_size() 
      || rt::embed_in_range3(buffer_shape) != mem_region->get_num_elements())
    return mem_region->get_num_elements();

  return rt::embed_in_range3(range);
}

template <class T, int Dim>
rt::id<3> get_effective_offset(const std::shared_ptr<rt::buffer_data_region> &mem_region,
                                  const rt::id<Dim> offset, const rt::range<Dim> buffer_shape, 
                                  const bool has_access_range) {
  // todo: optimize range / offset (would have to calculate a bounding box for reshaped buffers..)
  if(!has_access_range || sizeof(T) != mem_region->get_element_size()
      || rt::embed_in_range3(buffer_shape) != mem_region->get_num_elements())
    return {};
  return rt::embed_in_id3(offset);
}

/// The accessor base allows us to retrieve the associated buffer
/// for the accessor.
template<class T>
class accessor_base
{
protected:
  // Will hold the actual USM pointer after scheduling
  glue::embedded_pointer<T> _ptr;
};

template<class T>
inline constexpr auto default_access_tag(){
  if constexpr(std::is_const_v<T>)
    return sycl::read_only;
  else
    return sycl::read_write;
}

template<class T>
inline constexpr sycl::access_mode default_access_mode() {
  return std::is_const_v<T> ? access_mode::read : access_mode::read_write;
}

template <target accessTarget, typename value_type>
struct multi_ptr_for_target {
  using type = void*;
};

template <typename value_type>
struct multi_ptr_for_target<target::device, value_type> {
  using type = global_ptr<value_type>;
};

template <typename value_type>
struct multi_ptr_for_target<target::local, value_type> {
  using type = local_ptr<value_type>;
};
    
} // detail

namespace property {

struct no_init : public detail::property {};

} // property

inline constexpr property::no_init no_init;

// manually align layout with Itanium ABI, until MS ABI default changes
// in a "future major version"
// https://devblogs.microsoft.com/cppblog/optimizing-the-layout-of-empty-base-classes-in-vs2015-update-2-3/
#if !defined(HIPSYCL_EMPTY_BASES) && defined(_WIN32)
#define HIPSYCL_EMPTY_BASES __declspec(empty_bases)
#else
#define HIPSYCL_EMPTY_BASES
#endif // HIPSYCL_EMPTY_BASES

template <typename dataT, int dimensions = 1,
          access_mode accessmode = detail::default_access_mode<dataT>(),
          target accessTarget = target::device,
          accessor_variant AccessorVariant = accessor_variant::false_t>
class HIPSYCL_EMPTY_BASES accessor
    : public detail::accessor_base<std::remove_const_t<dataT>>,
      public detail::accessor::conditional_buffer_pointer_storage<
          detail::accessor::stores_buffer_pointer(AccessorVariant)>,
      public detail::accessor::conditional_access_range_storage<
          detail::accessor::stores_access_range(AccessorVariant), dimensions == 0 ? 1 : dimensions>,
      public detail::accessor::conditional_buffer_range_storage<
          detail::accessor::stores_buffer_range(AccessorVariant), dimensions == 0 ? 1 : dimensions>,
      public detail::accessor::conditional_accessor_properties_storage<
          detail::accessor::stores_accessor_properties(AccessorVariant)> {

  static constexpr int adj_dimensions = dimensions == 0 ? 1 : dimensions;

  static constexpr bool has_buffer_pointer =
      detail::accessor::stores_buffer_pointer(AccessorVariant);
  static constexpr bool has_access_range =
      detail::accessor::stores_access_range(AccessorVariant);
  static constexpr bool has_buffer_range =
      detail::accessor::stores_buffer_range(AccessorVariant);
  static constexpr bool has_accessor_properties =
      detail::accessor::stores_accessor_properties(AccessorVariant);
  
  static constexpr bool is_raw_accessor =
      AccessorVariant == accessor_variant::raw;
  static constexpr bool has_placeholder_constructors =
      !is_raw_accessor && AccessorVariant != accessor_variant::ranged &&
      AccessorVariant != accessor_variant::unranged;
  static constexpr bool has_ranged_constructors =
      AccessorVariant != accessor_variant::unranged &&
      AccessorVariant != accessor_variant::unranged_placeholder;
  static constexpr bool has_size_queries = !is_raw_accessor || (dimensions == 0);
  static constexpr bool has_subscript_operators =
      has_size_queries || (dimensions <= 1);

  static_assert(!std::is_const_v<dataT> || accessmode == access_mode::read,
    "const accessors are only allowed for read-only accessors");

  // Need to be friends with other accessors for implicit
  // conversion rules
  template <class Data2, int Dim2, access_mode M2, target Tgt2,
            access::placeholder P2>
  friend class accessor;

  template <class AccessorType>
  friend glue::unique_id
  detail::accessor::get_unique_id(const AccessorType &acc);

  friend class handler;

  static constexpr bool is_std_accessor_variant(accessor_variant v) {
    return v == accessor_variant::false_t || v == accessor_variant::true_t;
  }

  static constexpr bool is_ranged_variant(accessor_variant v){
    return v == accessor_variant::ranged ||
           v == accessor_variant::ranged_placeholder;
  }

  static constexpr bool is_unranged_variant(accessor_variant v){
    return v == accessor_variant::unranged ||
           v == accessor_variant::unranged_placeholder;
  }

  static constexpr bool is_placeholder_variant(accessor_variant v){
    return v == accessor_variant::unranged_placeholder ||
           v == accessor_variant::ranged_placeholder;
  }

  static constexpr bool is_nonplaceholder_variant(accessor_variant v){
    return v == accessor_variant::unranged ||
           v == accessor_variant::ranged;
  }

  static constexpr bool is_variant_convertible(accessor_variant src, accessor_variant dest) {
    if(src == dest)
      // can always convert between two accessors of same variant
      return true;
    else if(src == accessor_variant::raw)
      // Cannot turn raw accessor into a non-raw accessor
      // because of lack of information
      return false;
    else if(is_std_accessor_variant(dest))
      // Can always convert to regular accessors from non-raw accessors
      return true;
    else if (is_unranged_variant(dest) &&
             (is_ranged_variant(src) || is_std_accessor_variant(src)))
      // Allowing this would expose an accessor that has forgotten its
      // correct access range, using the subscript operators would potentially
      // access memory outside of the allowed range
      return false;
    else if(is_placeholder_variant(dest) && is_nonplaceholder_variant(src))
      // should both directions be forbidden? reversed?
      return false;
    
    return true;
  }
public:
  static constexpr access_mode mode = accessmode;
  static constexpr target access_target = accessTarget;

  using value_type =
      typename detail::accessor::accessor_data_type<dataT, accessmode>::value;
  using reference = value_type &;
  using const_reference = const dataT &;
  template <access::decorated IsDecorated>
  using accessor_ptr = typename detail::multi_ptr_for_target<accessTarget, value_type>::type;
  using iterator = detail::accessor_iterator<value_type, dimensions, accessor>;
  using const_iterator = detail::accessor_iterator<const value_type, dimensions, accessor>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
    typename std::iterator_traits<iterator>::difference_type;
  using size_type = size_t;

  using pointer_type = value_type*;

  accessor() = default;

  // 0D accessors
  template <typename AllocatorT, int D = dimensions,
            bool AllowPlaceholders = has_placeholder_constructors,
            std::enable_if_t<(D == 0 && AllowPlaceholders), int> = 0>
  accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
           const property_list &prop_list = {}) {
    // TODO: 0D accessor accesses only first element
    this->init(bufferRef, prop_list);
  }

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<D == 0> * = nullptr>
  accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef,
           const property_list &prop_list = {}) {
    // TODO: 0D accessor accesses only first element
    this->init(bufferRef, commandGroupHandlerRef, prop_list);
  }

  // Non 0-dimensional accessors
  template <typename AllocatorT, int D = dimensions,
            bool AllowPlaceholders = has_placeholder_constructors,
            std::enable_if_t<(D > 0 && AllowPlaceholders), int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           const property_list &prop_list = {}) {
    this->init(bufferRef, prop_list);
  }

  template <typename AllocatorT, typename TagDescriptorT, int D = dimensions,
            bool AllowPlaceholders = has_placeholder_constructors,
            std::enable_if_t<(D > 0 && AllowPlaceholders), int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef, detail::mode_tag<TagDescriptorT> tag,
           const property_list &prop_list = {})
  : accessor{bufferRef, prop_list} {}

  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef,
           const property_list &prop_list = {}) {

    this->init(bufferRef, commandGroupHandlerRef, prop_list);
  }

  template <typename AllocatorT, typename TagDescT, int D = dimensions,
            std::enable_if_t<(D > 0)> * = nullptr>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, detail::mode_tag<TagDescT> tag,
           const property_list &prop_list = {})
      : accessor{bufferRef, commandGroupHandlerRef, prop_list} {}

  /* Ranged accessors */

  template <
      typename AllocatorT, int D = dimensions,
      bool AllowPlaceholders = has_placeholder_constructors,
      bool AllowRanged = has_ranged_constructors,
      std::enable_if_t<(D > 0 && AllowPlaceholders && AllowRanged), int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, const property_list &propList = {})
      : accessor{bufferRef, accessRange, id<dimensions>{}, propList} {}

  template <
      typename AllocatorT, typename TagDescT, int D = dimensions,
      bool AllowPlaceholders = has_placeholder_constructors,
      bool AllowRanged = has_ranged_constructors,
      std::enable_if_t<(D > 0 && AllowPlaceholders && AllowRanged), int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, detail::mode_tag<TagDescT> tag,
           const property_list &propList = {})
      : accessor{bufferRef, accessRange, id<dimensions>{}, tag, propList} {}

  template <
      typename AllocatorT, int D = dimensions,
      bool AllowPlaceholders = has_placeholder_constructors,
      bool AllowRanged = has_ranged_constructors,
      std::enable_if_t<(D > 0 && AllowPlaceholders && AllowRanged), int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, id<dimensions> accessOffset,
           const property_list &propList = {}) {

    this->init(bufferRef, accessOffset, accessRange, propList);
  }

  template <
      typename AllocatorT, typename TagDescT, int D = dimensions,
      bool AllowPlaceholders = has_placeholder_constructors,
      bool AllowRanged = has_ranged_constructors,
      std::enable_if_t<(D > 0 && AllowPlaceholders && AllowRanged), int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           range<dimensions> accessRange, id<dimensions> accessOffset,
           detail::mode_tag<TagDescT> tag, const property_list &propList = {})
      : accessor{bufferRef, accessRange, accessOffset, propList} {}

  template <typename AllocatorT, int D = dimensions,
            bool AllowRanged = has_ranged_constructors,
            std::enable_if_t<(D > 0) && AllowRanged, int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           const property_list &propList = {})
      : accessor{bufferRef, commandGroupHandlerRef, accessRange,
                 id<dimensions>{}, propList} {}

  template <typename AllocatorT, typename TagDescT, int D = dimensions,
            bool AllowRanged = has_ranged_constructors,
            std::enable_if_t<(D > 0) && AllowRanged, int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           detail::mode_tag<TagDescT> tag, const property_list &propList = {})
      : accessor{bufferRef, commandGroupHandlerRef, accessRange, propList} {}

  template <typename AllocatorT, int D = dimensions,
            bool AllowRanged = has_ranged_constructors,
            std::enable_if_t<(D > 0) && AllowRanged, int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset, const property_list &propList = {}) {

    this->init(bufferRef, commandGroupHandlerRef, accessOffset, accessRange,
               propList);
  }

  template <typename AllocatorT, typename TagDescT, int D = dimensions,
            bool AllowRanged = has_ranged_constructors,
            std::enable_if_t<(D > 0) && AllowRanged, int> = 0>
  accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<dimensions> accessRange,
           id<dimensions> accessOffset, detail::mode_tag<TagDescT> tag,
           const property_list &propList = {})
      : accessor{bufferRef, commandGroupHandlerRef, accessRange, accessOffset,
                 propList} {}

  ACPP_UNIVERSAL_TARGET
  accessor(const accessor& other) = default;

  ACPP_UNIVERSAL_TARGET
  accessor& operator=(const accessor& other) = default;

  // Implicit conversion from read-write accessor to const
  // and non-const read-only accessor
  template <access::placeholder P, access_mode M = accessmode,
            std::enable_if_t<M == access_mode::read &&
                                 is_variant_convertible(P, AccessorVariant),
                             int> = 0>
  ACPP_UNIVERSAL_TARGET
  accessor(const accessor<std::remove_const_t<dataT>, dimensions,
                          access_mode::read_write, accessTarget, P> &other)
      : detail::accessor_base<std::remove_const_t<dataT>>{other},
        detail::accessor::conditional_buffer_pointer_storage<
            has_buffer_pointer>{other},
        detail::accessor::conditional_access_range_storage<has_access_range,
                                                           dimensions>{
            detail::accessor::access_range<dimensions>{other.get_offset(),
                                                       other.get_range()}},
        detail::accessor::conditional_buffer_range_storage<has_buffer_range,
                                                           dimensions>{other},
        detail::accessor::conditional_accessor_properties_storage<
            has_accessor_properties>{other} {}

  // Conversion between different accessor variants
  template <accessor_variant OtherV,
            std::enable_if_t<is_variant_convertible(OtherV, AccessorVariant) &&
                                 OtherV != AccessorVariant,
                             int> = 0>
  ACPP_UNIVERSAL_TARGET
  accessor(const accessor<dataT, dimensions, accessmode,
                          accessTarget, OtherV> &other)
      : detail::accessor_base<std::remove_const_t<dataT>>{other},
        detail::accessor::conditional_buffer_pointer_storage<
            has_buffer_pointer>{other},
        detail::accessor::conditional_access_range_storage<has_access_range,
                                                           dimensions>{
            detail::accessor::access_range<dimensions>{other.get_offset(),
                                                       other.get_range()}},
        detail::accessor::conditional_buffer_range_storage<has_buffer_range,
                                                           dimensions>{other},
        detail::accessor::conditional_accessor_properties_storage<
            has_accessor_properties>{other} {}

  /* -- common interface members -- */

  template <accessor_variant OtherV>
  ACPP_UNIVERSAL_TARGET friend bool
  operator==(const accessor &lhs,
             const accessor<dataT, dimensions, accessmode, accessTarget, OtherV>
                 &rhs) noexcept {
    if (detail::accessor::get_unique_id(lhs) !=
        detail::accessor::get_unique_id(rhs))
      return false;

    if constexpr (AccessorVariant != accessor_variant::raw &&
                  OtherV != accessor_variant::raw &&
                  dimensions > 0) {
      if (lhs.get_offset() != rhs.get_offset())
        return false;
      if (lhs.get_range() != rhs.get_range())
        return false;
    }

    return true;
  }

  template <accessor_variant OtherV>
  ACPP_UNIVERSAL_TARGET friend bool
  operator!=(const accessor &lhs,
             const accessor<dataT, dimensions, accessmode, accessTarget, OtherV>
                 &rhs) noexcept {
    return !(lhs == rhs);
  }

  std::size_t AdaptiveCpp_hash_code() const {
    // TODO: This only really guarantees unique hash codes
    // outside of kernels, on the host side.
    // Once on device, the UID will contain the device pointer
    // which might be aliased by multiple accessors.
    return get_uid().AdaptiveCpp_hash_code();
  }

  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

  ACPP_UNIVERSAL_TARGET
  bool is_placeholder() const noexcept
  {
    if constexpr (has_accessor_properties) {
      return this->detail::accessor::conditional_accessor_properties_storage<
          has_accessor_properties>::get().is_placeholder();
    } else if constexpr (AccessorVariant ==
                             accessor_variant::ranged_placeholder ||
                         AccessorVariant ==
                             accessor_variant::unranged_placeholder) {
      return true;
    } else {
      return false;
    }
  }

  template<bool IsAllowed = has_size_queries,
          std::enable_if_t<IsAllowed, int> = 0>
  ACPP_UNIVERSAL_TARGET
  size_t get_size() const noexcept
  {
    return get_count() * sizeof(dataT);
  }

  template <int D = dimensions, bool IsAllowed = has_size_queries,
            std::enable_if_t<(D > 0 && IsAllowed), int> = 0>
  ACPP_UNIVERSAL_TARGET size_t get_count() const noexcept {
    return get_range().size();
  }

  template<int D = dimensions, bool IsAllowed = has_size_queries,
           std::enable_if_t<D == 0 && IsAllowed, int> = 0>
  ACPP_UNIVERSAL_TARGET
  size_t get_count() const noexcept
  { return 1; }

  /* void swap(accessor &other); */

  template <bool IsAllowed = has_size_queries,
            std::enable_if_t<IsAllowed, int> = 0>
  size_t byte_size() const noexcept {
    return size();
  }

  template <bool IsAllowed = has_size_queries,
            std::enable_if_t<IsAllowed, int> = 0>
  size_t size() const noexcept {
    return get_count();
  }

  size_type max_size() const noexcept {
    return std::numeric_limits<difference_type>::max();
  }

  template <bool IsAllowed = has_size_queries,
            std::enable_if_t<IsAllowed, int> = 0>
  bool empty() const noexcept {
    return size() == 0;
  }

  /* Available only when: dimensions > 0 */
  template <int D = dimensions, bool IsAllowed = has_size_queries,
            std::enable_if_t<(D > 0 && IsAllowed), int> = 0>
  ACPP_UNIVERSAL_TARGET range<dimensions> get_range() const noexcept {
    if constexpr(has_access_range) {
      return this->detail::accessor::conditional_access_range_storage<
          has_access_range, dimensions>::ptr()->range;
    } else if constexpr(has_buffer_range) {
      return this->detail::accessor::conditional_buffer_range_storage<
          has_buffer_range, dimensions>::get();
    } else {
      return sycl::range<dimensions>{};
    }
  }

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           std::enable_if_t<(D > 0), int> = 0>
  ACPP_UNIVERSAL_TARGET
  id<dimensions> get_offset() const noexcept
  {
    if constexpr(!has_access_range) {
      return sycl::id<dimensions>{};
    } else {
      return this->detail::accessor::conditional_access_range_storage<
          has_access_range, dimensions>::ptr()->offset;
    }
  }
  
  template<int D = dimensions,
            std::enable_if_t<(D == 0), bool> = true>
  ACPP_UNIVERSAL_TARGET
  operator reference() const noexcept
  {
    return *(this->_ptr.get());
  }

  template <
      int D = dimensions, access::mode M = accessmode,
      bool IsAllowed = has_subscript_operators,
      // Note: spec says D==1 here, however this can lead to ambiguity when item<1>
      // is passed as argument
      // with respect to the SYCL 2020 implicit conversion from id<1>/item<1> to size_t
      // since we also have operator[](size_t).
      // Because of this, we only enable this for D > 1.
      std::enable_if_t<(D > 1) && IsAllowed && (M != access::mode::atomic),
                       bool> = true>
  ACPP_UNIVERSAL_TARGET reference
  operator[](id<dimensions> index) const noexcept {
    return (this->_ptr.get())[get_linear_id(index + get_offset())];
  }

  template <
      int D = dimensions, access::mode M = accessmode,
      bool IsAllowed = has_subscript_operators,
      std::enable_if_t<(D == 1) && IsAllowed && (M != access::mode::atomic),
                       bool> = true>
  ACPP_UNIVERSAL_TARGET reference operator[](size_t index) const noexcept {
    return (this->_ptr.get())[index + get_offset()];
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions == 0*/
  template<int D = dimensions,
           access::mode M = accessmode,
           typename = std::enable_if_t<M == access::mode::atomic && D == 0>>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] ACPP_UNIVERSAL_TARGET
  operator atomic<dataT, access::address_space::global_space> () const noexcept
  {
    return atomic<dataT, access::address_space::global_space>{
        global_ptr<dataT>(this->_ptr.get())};
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions > 0*/
  template <int D = dimensions, access::mode M = accessmode,
            bool IsAllowed = has_subscript_operators,
            typename = std::enable_if_t<(D > 0) && IsAllowed &&
                                        (M == access::mode::atomic)>>
  [[deprecated("Atomic accessors are deprecated as of SYCL "
               "2020")]] ACPP_UNIVERSAL_TARGET
      atomic<dataT, access::address_space::global_space>
      operator[](id<dimensions> index) const noexcept {
    return atomic<dataT, access::address_space::global_space>{global_ptr<dataT>(
        this->_ptr.get() + get_linear_id(index + get_offset()))};
  }

  template <int D = dimensions, access::mode M = accessmode,
            bool IsAllowed = has_subscript_operators,
            typename = std::enable_if_t<(D == 1) && IsAllowed &&
                                        (M == access::mode::atomic)>>
  [[deprecated("Atomic accessors are deprecated as of SYCL "
               "2020")]] ACPP_UNIVERSAL_TARGET
      atomic<dataT, access::address_space::global_space>
      operator[](size_t index) const noexcept {
    return atomic<dataT, access::address_space::global_space>{
        global_ptr<dataT>(this->_ptr.get() + index + get_offset())};
  }

  /* Available only when: dimensions > 1 */
  template <int D = dimensions, bool IsAllowed = has_subscript_operators,
            std::enable_if_t<(D > 1) && IsAllowed, int> = 0>
  ACPP_UNIVERSAL_TARGET
      detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                        accessTarget, AccessorVariant>
      operator[](size_t index) const noexcept {

    sycl::id<dimensions> initial_index;
    initial_index[0] = index;

    return detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                             accessTarget, AccessorVariant> {
      this, initial_index
    };
  }

  /* Available only when: accessTarget == access::target::host_buffer */
  // TODO: T == host_buffer is deprecated in SYCL2020
  template<access::target T = accessTarget,
           typename = std::enable_if_t<
	     T == access::target::host_buffer ||
	     T == access::target::host_task>>
  dataT *get_pointer() const noexcept
  {
    return const_cast<dataT*>(this->_ptr.get());
  }

  /* Available only when: accessTarget == access::target::global_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::global_buffer>>
  ACPP_UNIVERSAL_TARGET
  global_ptr<dataT> get_pointer() const noexcept
  {
    return global_ptr<dataT>{const_cast<dataT*>(this->_ptr.get())};
  }

  /* Available only when: accessTarget == access::target::constant_buffer */
  template<access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::constant_buffer>>
  ACPP_UNIVERSAL_TARGET
  constant_ptr<dataT> get_pointer() const noexcept
  {
    return constant_ptr<dataT>{const_cast<dataT*>(this->_ptr.get())};
  }

  /* Available only when: accessTarget == access::target::device */
  template<access::decorated IsDecorated,
           access::target T = accessTarget,
           typename = std::enable_if_t<T == access::target::device>>
  HIPSYCL_UNIVERSAL_TARGET
  accessor_ptr<IsDecorated> get_multi_ptr() const noexcept
  {
    return accessor_ptr<IsDecorated>{this->_ptr.get()};
  }

  iterator begin() const noexcept {
    return iterator::make_begin(this);
  }

  iterator end() const noexcept {
    return iterator::make_end(this);
  }

  const_iterator cbegin() const noexcept {
    return const_iterator::make_begin(this);
  }

  const_iterator cend() const noexcept {
    return const_iterator::make_end(this);
  }

  reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }

  reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }
private:
  template <typename, int, typename> friend class detail::accessor_iterator;

  ACPP_UNIVERSAL_TARGET
  static constexpr int get_dimensions() noexcept{
    return dimensions;
  }

  // Only valid until the embedded pointer has been initialized
  ACPP_HOST_TARGET
  glue::unique_id get_uid() const noexcept {
    return this->_ptr.get_uid();
  }

  template <class BufferT>
  void init(BufferT &buff, id<adj_dimensions> offset,
            range<adj_dimensions> access_range, const property_list &prop_list) {
    
    bool is_no_init_access = false;
    bool is_placeholder_access = true;

    if constexpr (has_accessor_properties) {
      is_no_init_access = this->is_no_init(prop_list);

      this->detail::accessor::conditional_accessor_properties_storage<
          has_accessor_properties>::
          attempt_set(detail::accessor::accessor_properties{
              is_placeholder_access, is_no_init_access});
    }

    bind_to_buffer(buff, offset, access_range);

    if (accessTarget == access::target::host_buffer) {
      init_host_buffer(buff.AdaptiveCpp_runtime(), is_no_init_access);
    }
  }
  
  template <class BufferT>
  void init(BufferT& buff, const property_list& prop_list) {
    init(buff, id<adj_dimensions>{}, detail::extract_buffer_range(buff), prop_list);
  }

  template <class BufferT>
  void init(BufferT &buff, handler &cgh, id<adj_dimensions> offset,
            range<adj_dimensions> access_range, const property_list &prop_list) {

    bool is_no_init_access = this->is_no_init(prop_list);
    bool is_placeholder_access = false;

    if constexpr (has_accessor_properties) {
      this->detail::accessor::conditional_accessor_properties_storage<
          has_accessor_properties>::
          attempt_set(detail::accessor::accessor_properties{
              is_placeholder_access, is_no_init_access});
    }

    bind_to_buffer(buff, offset, access_range);
    detail::accessor::bind_to_handler(*this, cgh,
                                      detail::extract_buffer_data_region(buff),
                                      offset, access_range, is_no_init_access);
  }

  template <class BufferT>
  void init(BufferT& buff, handler& cgh, const property_list& prop_list) {
    init(buff, cgh, id<adj_dimensions>{}, detail::extract_buffer_range(buff),
         prop_list);
  }
  
  

  ACPP_UNIVERSAL_TARGET
  size_t get_linear_id(id<dimensions> idx) const noexcept {
    if constexpr (dimensions == 0) {
      return 0;
    } else if constexpr (dimensions == 1) {
      return idx[0];
    } else {
      return detail::linear_id<dimensions>::get(idx, get_buffer_shape());
    }
  }

  ACPP_UNIVERSAL_TARGET
  range<adj_dimensions> get_buffer_shape() const noexcept {
    if constexpr(has_buffer_range) {
      return this->detail::accessor::conditional_buffer_range_storage<
          has_buffer_range, adj_dimensions>::get();
    } else {
      return range<adj_dimensions>{};
    }
  }

  ACPP_HOST_TARGET
  std::shared_ptr<rt::buffer_data_region> get_data_region() const noexcept {
    if constexpr(has_buffer_pointer) {
      return this
          ->detail::accessor::conditional_buffer_pointer_storage<
              has_buffer_pointer>::get()
          .get_shared_ptr();
    }
    return nullptr;
  }

  template <class BufferType>
  void bind_to_buffer(BufferType &buff,
                      sycl::id<adj_dimensions> accessOffset,
                      sycl::range<adj_dimensions> accessRange) {
    __acpp_if_target_host(
      auto buffer_region = detail::extract_buffer_data_region(buff);
      this->detail::accessor::
          conditional_buffer_pointer_storage<has_buffer_pointer>::attempt_set(
              detail::mobile_shared_ptr{buffer_region});
    );

    this->detail::accessor::conditional_access_range_storage<
        has_access_range,
        adj_dimensions>::attempt_set(detail::accessor::access_range<adj_dimensions>{
        accessOffset, accessRange});

    this->detail::accessor::conditional_buffer_range_storage<
        has_buffer_range,
        adj_dimensions>::attempt_set(detail::extract_buffer_range(buff));
  }

  void init_host_buffer(rt::runtime* rt, bool is_no_init) {
    // TODO: Maybe unify code with handler::update_host()?
    HIPSYCL_DEBUG_INFO << "accessor [host]: Initializing host access" << std::endl;

    auto data_mobile_ptr =
        this->detail::accessor::conditional_buffer_pointer_storage<
            has_buffer_pointer>::get();
    auto data = data_mobile_ptr.get_shared_ptr();
    assert(data);

    rt::dag_node_ptr node;
    {
      rt::dag_build_guard build{rt->dag()};

      // get_offset and get_range are only defined for dimensions > 0
      id<adj_dimensions> acc_offset;
      if constexpr (dimensions == 0)
        acc_offset = id<1>{0};
      else
        acc_offset = get_offset();

      range<adj_dimensions> acc_range;
      if constexpr (dimensions == 0)
        acc_range = range<1>{1};
      else
        acc_range = get_range();
      
      const rt::range<adj_dimensions> buffer_shape = rt::make_range(get_buffer_shape());
      auto explicit_requirement = rt::make_operation<rt::buffer_memory_requirement>(
        data,
        detail::get_effective_offset<dataT, adj_dimensions>(data,
                                                           rt::make_id(acc_offset),
                                                           buffer_shape,
                                                           has_access_range),
        detail::get_effective_range<dataT, adj_dimensions>(data,
                                                          rt::make_range(acc_range),
                                                          buffer_shape,
                                                          has_access_range),
        detail::get_effective_access_mode(accessmode, is_no_init),
        accessTarget
      );

      rt::cast<rt::buffer_memory_requirement>(explicit_requirement.get())
          ->bind(this->get_uid());

      rt::execution_hints enforce_bind_to_host;
      enforce_bind_to_host.set_hint(
          rt::hints::bind_to_device{detail::get_host_device()});

      node = build.builder()->add_explicit_mem_requirement(
          std::move(explicit_requirement), rt::requirements_list{rt},
          enforce_bind_to_host);
      
      HIPSYCL_DEBUG_INFO << "accessor [host]: forcing DAG flush for host access..." << std::endl;
      rt->dag().flush_sync();
    }
    if(rt::application::errors().num_errors() == 0){
      HIPSYCL_DEBUG_INFO << "accessor [host]: Waiting for completion of host access..." << std::endl;

      assert(node);
      node->wait();

      rt::buffer_memory_requirement *req =
          static_cast<rt::buffer_memory_requirement *>(node->get_operation());
      assert(req->has_device_ptr());
      void* host_ptr = req->get_device_ptr();
      assert(host_ptr);
      
      // For host accessors, we need to manually trigger the initialization
      // of the embedded pointer
      this->_ptr.explicit_init(host_ptr);
    } else {
      HIPSYCL_DEBUG_ERROR << "accessor [host]: Aborting synchronization, "
                             "runtime error list is non-empty"
                          << std::endl;
      glue::throw_asynchronous_errors([](sycl::exception_list errors) {
        glue::print_async_errors(errors);
        // Additionally throw the first exception to create synchronous
        // error behavior
        if (errors.size() > 0) {
          std::rethrow_exception(errors[0]);
        }
      });
    }
    // TODO Need to lock execution of DAG
  }

  constexpr bool is_no_init_accessmode() const {
    if constexpr (accessmode == access_mode::discard_write ||
                  accessmode == access_mode::discard_read_write) {
      return true;
    } else {
      return false;
    }
  }

  bool is_no_init(const property_list& prop_list) {
    if(prop_list.has_property<property::no_init>()) {
      return true;
    }
    return is_no_init_accessmode();
  }

  bool is_no_init() {
    if(has_accessor_properties) {
      bool flag = this->detail::accessor::conditional_accessor_properties_storage<
              has_accessor_properties>::ptr()
          ->is_no_init();
      if(flag)
        return true;
    }
    return is_no_init_accessmode();
  }

};

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using raw_accessor = accessor<T, Dim, M, Tgt, accessor_variant::raw>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using ranged_accessor = accessor<T, Dim, M, Tgt, accessor_variant::ranged>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using unranged_accessor = accessor<T, Dim, M, Tgt, accessor_variant::unranged>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using ranged_placeholder_accessor =
    accessor<T, Dim, M, Tgt, accessor_variant::ranged_placeholder>;

template <class T, int Dim = 1,
          access_mode M = (std::is_const_v<T> ? access_mode::read
                                              : access_mode::read_write),
          target Tgt = target::device>
using unranged_placeholder_accessor =
    accessor<T, Dim, M, Tgt, accessor_variant::unranged_placeholder>;

// Accessor deduction guides
#ifdef ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION
 #define HIPSYCL_ACCESSOR_VARIANT_SELECTOR(tag, fallback, optimized) \
  detail::deduce_accessor_variant(tag, optimized)
#else
 #define HIPSYCL_ACCESSOR_VARIANT_SELECTOR(tag, fallback, optimized) fallback
#endif

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, detail::mode_tag<TagDescT> tag,
         const property_list &prop_list = {})
    -> accessor<T, Dim, detail::mode_tag<TagDescT>::mode,
                detail::mode_tag<TagDescT>::target,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::mode_tag<TagDescT>{}, accessor_variant::true_t,
                    accessor_variant::unranged_placeholder)>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         detail::mode_tag<TagDescT> tag, const property_list &prop_list = {})
    -> accessor<T, Dim, detail::mode_tag<TagDescT>::mode,
                detail::mode_tag<TagDescT>::target,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(detail::mode_tag<TagDescT>{},
                                                  accessor_variant::false_t,
                                                  accessor_variant::unranged)>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         detail::mode_tag<TagDescT> tag, const property_list &propList = {})
    -> accessor<T, Dim, detail::mode_tag<TagDescT>::mode,
                detail::mode_tag<TagDescT>::target,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::mode_tag<TagDescT>{}, accessor_variant::true_t,
                    accessor_variant::ranged_placeholder)>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, detail::mode_tag<TagDescT> tag,
         const property_list &propList = {})
    -> accessor<T, Dim, detail::mode_tag<TagDescT>::mode,
                detail::mode_tag<TagDescT>::target,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::mode_tag<TagDescT>{}, accessor_variant::true_t,
                    accessor_variant::ranged_placeholder)>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, detail::mode_tag<TagDescT> tag,
         const property_list &propList = {})
    -> accessor<T, Dim, detail::mode_tag<TagDescT>::mode,
                detail::mode_tag<TagDescT>::target,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::mode_tag<TagDescT>{}, accessor_variant::false_t,
                    accessor_variant::ranged)>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, id<Dim> accessOffset, 
         detail::mode_tag<TagDescT> tag, const property_list &propList = {})
    -> accessor<T, Dim, detail::mode_tag<TagDescT>::mode,
                detail::mode_tag<TagDescT>::target,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::mode_tag<TagDescT>{}, accessor_variant::false_t,
                    accessor_variant::ranged)>;

// Non-TagT deduction guides


template <typename T, int Dim, typename AllocatorT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef,
         const property_list &prop_list = {})
    -> accessor<T, Dim, detail::default_access_mode<T>(),
                target::device,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::default_access_tag<T>(), accessor_variant::true_t,
                    accessor_variant::unranged_placeholder)>;

template <typename T, int Dim, typename AllocatorT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         const property_list &prop_list = {})
    -> accessor<T, Dim, detail::default_access_mode<T>(),
                target::device,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(detail::default_access_tag<T>(),
                                                  accessor_variant::false_t,
                                                  accessor_variant::unranged)>;

template <typename T, int Dim, typename AllocatorT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         const property_list &propList = {})
    -> accessor<T, Dim, detail::default_access_mode<T>(),
                target::device,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::default_access_tag<T>(), accessor_variant::true_t,
                    accessor_variant::ranged_placeholder)>;

template <typename T, int Dim, typename AllocatorT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, const property_list &propList = {})
    -> accessor<T, Dim, detail::default_access_mode<T>(),
                target::device,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::default_access_tag<T>(), accessor_variant::true_t,
                    accessor_variant::ranged_placeholder)>;

template <typename T, int Dim, typename AllocatorT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, const property_list &propList = {})
    -> accessor<T, Dim, detail::default_access_mode<T>(),
                target::device,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::default_access_tag<T>(), accessor_variant::false_t,
                    accessor_variant::ranged)>;

template <typename T, int Dim, typename AllocatorT>
accessor(buffer<T, Dim, AllocatorT> &bufferRef, handler &commandGroupHandlerRef,
         range<Dim> accessRange, id<Dim> accessOffset,
         const property_list &propList = {})
    -> accessor<T, Dim, detail::default_access_mode<T>(),
                target::device,
                HIPSYCL_ACCESSOR_VARIANT_SELECTOR(
                    detail::default_access_tag<T>(), accessor_variant::false_t,
                    accessor_variant::ranged)>;

//host_accessor implementation

template <typename dataT, int dimensions = 1,
          access_mode accessMode =
              (std::is_const_v<dataT> ? access_mode::read
                                      : access_mode::read_write)>
class host_accessor {
  using accessor_type =
      accessor<dataT, dimensions, accessMode, target::host_buffer,
               access::placeholder::false_t>;

  template<typename DataT2, int Dim2, access_mode M2>
  friend class host_accessor;

  template<class TagT>
  void validate_host_accessor_tag(TagT tag) {
    static_assert(std::is_same_v<TagT, detail::read_only_tag_t> ||
                  std::is_same_v<TagT, detail::write_only_tag_t> ||
                  std::is_same_v<TagT, detail::read_write_tag_t>,
                  "Invalid tag for host_accessor");
  }
public:
  using value_type = typename accessor_type::value_type;
  using reference = typename accessor_type::reference;
  using const_reference = typename accessor_type::const_reference;

  using iterator = detail::accessor_iterator<value_type, dimensions, host_accessor>;
  using const_iterator = detail::accessor_iterator<const value_type, dimensions, host_accessor>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
    typename std::iterator_traits<iterator>::difference_type;
  using size_type = typename accessor_type::size_type;

  host_accessor() = default;

  /* Available only when: (dimensions == 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<D == 0, bool> = true>
  host_accessor(buffer<dataT, 1, AllocatorT> &bufferRef,
                const property_list &propList = {})
      : _impl{bufferRef, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                const property_list &propList = {})
      : _impl{bufferRef, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, typename TagDescT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef, 
                detail::mode_tag<TagDescT> tag, 
                const property_list &propList = {})
      : _impl{bufferRef, tag, propList} {
    validate_host_accessor_tag(tag);
  }

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, typename TagDescT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange,
                detail::mode_tag<TagDescT> tag,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, tag, propList} {
    validate_host_accessor_tag(tag);
  }

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange, id<dimensions> accessOffset,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, accessOffset, propList} {}

  /* Available only when: (dimensions > 0) */
  template <typename AllocatorT, typename TagDescT, int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  host_accessor(buffer<dataT, dimensions, AllocatorT> &bufferRef,
                range<dimensions> accessRange, id<dimensions> accessOffset,
                detail::mode_tag<TagDescT> tag,
                const property_list &propList = {})
      : _impl{bufferRef, accessRange, accessOffset, tag, propList} {
    validate_host_accessor_tag(tag);
  }

  // Conversion read-write -> read-only accessor
  template <access_mode M = accessMode,
            std::enable_if_t<M == access_mode::read, bool> = true>
  host_accessor(const host_accessor<std::remove_const_t<dataT>, dimensions,
                                    access_mode::read_write> &other)
      : _impl{other._impl} {}

  friend bool operator==(const host_accessor &lhs, const host_accessor &rhs) {
    return lhs._impl == rhs._impl;
  }

  friend bool operator!=(const host_accessor &lhs, const host_accessor &rhs) {
    return lhs._impl != rhs._impl;
  }

  void swap(host_accessor &other) {
    using std::swap;
    swap(_impl, other._impl);
  }

  size_type byte_size() const noexcept {
    return _impl.get_size();
  }

  size_type size() const noexcept {
    return _impl.get_count();
  }

  size_type max_size() const noexcept {
    return std::numeric_limits<difference_type>::max();
  }

  bool empty() const noexcept {
    return size() == 0;
  }

  /* Available only when: (dimensions > 0) */
  template<int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  range<dimensions> get_range() const {
    return _impl.get_range();
  }

  /* Available only when: (dimensions > 0) */
  template<int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  id<dimensions> get_offset() const {
    return _impl.get_offset();
  }

  /* Available only when: (dimensions == 0) */
  template<int D = dimensions,
            std::enable_if_t<(D == 0), bool> = true>
  operator reference() const {
    return *_impl.get_pointer();
  }

  /* Available only when: (dimensions > 0) */
  template<int D = dimensions,
            std::enable_if_t<(D > 0), bool> = true>
  reference operator[](id<dimensions> index) const {
    return _impl[index];
  }

  /* Available only when: (dimensions > 1) */
  template<int D = dimensions,
            std::enable_if_t<(D > 1), bool> = true>
  auto operator[](size_t index) const {
    return _impl[index];
  }

  /* Available only when: (dimensions == 1) */
  template<int D = dimensions,
            std::enable_if_t<(D == 1), bool> = true>
  reference operator[](size_t index) const {
    return _impl[index];
  }

  std::add_pointer_t<value_type> get_pointer() const noexcept {
    return _impl.get_pointer();
  }

  iterator begin() const noexcept {
    return iterator::make_begin(this);
  }

  iterator end() const noexcept {
    return iterator::make_end(this);
  }

  const_iterator cbegin() const noexcept {
    return const_iterator::make_begin(this);
  }

  const_iterator cend() const noexcept {
    return const_iterator::make_end(this);
  }

  reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }

  reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }

  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

  std::size_t AdaptiveCpp_hash_code() const {
    return _impl.AdaptiveCpp_hash_code();
  }

  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

private:
  accessor_type _impl;
};


// host_accessor deduction guides

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef,
  detail::mode_tag<TagDescT> tag, const property_list &prop_list = {})
    -> host_accessor<T, Dim, detail::mode_tag<TagDescT>::mode>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
              detail::mode_tag<TagDescT> tag, const property_list &propList = {})
    -> host_accessor<T, Dim, detail::mode_tag<TagDescT>::mode>;

template <typename T, int Dim, typename AllocatorT, typename TagDescT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, detail::mode_tag<TagDescT> tag,
         const property_list &propList = {})
    -> host_accessor<T, Dim, detail::mode_tag<TagDescT>::mode>;

// Non-TagT guides

template <typename T, int Dim, typename AllocatorT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef,
          const property_list &prop_list = {})
    -> host_accessor<T, Dim, 
          detail::default_access_tag<T>().mode>;

template <typename T, int Dim, typename AllocatorT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
              const property_list &propList = {})
    -> host_accessor<T, Dim, 
          detail::default_access_tag<T>().mode>;

template <typename T, int Dim, typename AllocatorT>
host_accessor(buffer<T, Dim, AllocatorT> &bufferRef, range<Dim> accessRange,
         id<Dim> accessOffset, const property_list &propList = {})
    -> host_accessor<T, Dim, 
          detail::default_access_tag<T>().mode>;

/// Accessor specialization for local memory
template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::placeholder isPlaceholder>
class accessor<
    dataT,
    dimensions,
    accessmode,
    access::target::local,
    isPlaceholder>
{
  using address = detail::local_memory::address;
public:

  using value_type =
      typename detail::accessor::accessor_data_type<dataT, accessmode>::value;
  using reference = value_type &;
  using const_reference = const dataT &;
  using iterator = detail::accessor_iterator<value_type, dimensions, accessor>;
  using const_iterator = detail::accessor_iterator<const value_type, dimensions, accessor>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
    typename std::iterator_traits<iterator>::difference_type;
  // TODO difference_type
  using size_type = size_t;


  accessor() = default;

  /* Available only when: dimensions == 0 */
  template<int D = dimensions,
           typename std::enable_if_t<D == 0>* = nullptr>
  accessor(handler &commandGroupHandlerRef, const property_list& p = {})
    : _addr{detail::handler::allocate_local_mem<dataT>(
              commandGroupHandlerRef,1)}
  {}

  /* Available only when: dimensions > 0 */
  template<int D = dimensions,
           typename std::enable_if_t<(D > 0)>* = nullptr>
  accessor(range<dimensions> allocationSize,
           handler &commandGroupHandlerRef, const property_list& p = {})
    : _addr{detail::handler::allocate_local_mem<dataT>(
              commandGroupHandlerRef,
              allocationSize.size())},
      _num_elements{allocationSize}
  {}

  accessor(const accessor &) = default;
  accessor &operator=(const accessor &) = default;

  void swap(accessor &other)
  {
    using std::swap;
    swap(_addr, other._addr);
    swap(_num_elements, other._num_elements);
  }

  friend bool operator==(const accessor& lhs, const accessor& rhs)
  {
    return lhs._addr == rhs._addr && lhs._num_elements == rhs._num_elements;
  }

  friend bool operator!=(const accessor& lhs, const accessor& rhs)
  {
    return !(lhs == rhs);
  }

  std::size_t AdaptiveCpp_hash_code() const {
    return _addr;
  }

  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

  [[deprecated("get_size() was removed for SYCL 2020, use byte_size() instead")]]
  ACPP_KERNEL_TARGET
  size_t get_size() const
  {
    return get_count() * sizeof(dataT);
  }

  [[deprecated("get_count() was removed for SYCL 2020, use size() instead")]]
  ACPP_KERNEL_TARGET
  size_t get_count() const
  {
    return _num_elements.size();
  }

  ACPP_KERNEL_TARGET
  size_t byte_size() const noexcept
  {
    return size() * sizeof(dataT);
  }

  ACPP_KERNEL_TARGET
  size_t size() const noexcept
  {
    return _num_elements.size();
  }

  size_type max_size() const noexcept
  {
    return std::numeric_limits<difference_type>::max();
  }

  range<dimensions> get_range() const
  {
    return _num_elements;
  }

  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 0) && (M != access_mode::atomic), bool> = false>
  ACPP_KERNEL_TARGET
  operator reference() const
  {
    return *detail::local_memory::get_ptr<dataT>(_addr);
  }

  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D > 0) && (M != access_mode::atomic), bool> = false>
  ACPP_KERNEL_TARGET
  reference operator[](id<dimensions> index) const
  {
    return *(detail::local_memory::get_ptr<dataT>(_addr) +
        detail::linear_id<dimensions>::get(index, _num_elements));
  }

  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 1) && (M != access_mode::atomic), bool> = false>
  ACPP_KERNEL_TARGET
  reference operator[](size_t index) const
  {
    return *(detail::local_memory::get_ptr<dataT>(_addr) + index);
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions == 0 */
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 0) && (M == access_mode::atomic), bool> = false>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] ACPP_KERNEL_TARGET
  operator atomic<dataT,access::address_space::local_space>() const
  {
    return atomic<dataT, access::address_space::local_space>{
            local_ptr<dataT>{detail::local_memory::get_ptr<dataT>(_addr)}};
  }


  /* Available only when: accessMode == access::mode::atomic && dimensions > 0 */
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D > 0) && (M == access_mode::atomic), bool> = false>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] ACPP_KERNEL_TARGET
  atomic<dataT, access::address_space::local_space> operator[](
       id<dimensions> index) const
  {
    return atomic<dataT, access::address_space::local_space>{local_ptr<dataT>{
            detail::local_memory::get_ptr<dataT>(_addr) +
            detail::linear_id<dimensions>::get(index, _num_elements)}};
  }

  /* Available only when: accessMode == access::mode::atomic && dimensions == 1 */
  template<int D = dimensions,
           access_mode M = accessmode,
           std::enable_if_t<(D == 1) && (M == access_mode::atomic), bool> = false>
  [[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] ACPP_KERNEL_TARGET
  atomic<dataT, access::address_space::local_space> operator[](size_t index) const
  {
    return atomic<dataT, access::address_space::local_space>{local_ptr<dataT>{
            detail::local_memory::get_ptr<dataT>(_addr) + index}};
  }

  /* Available only when: dimensions > 1 */
  template<int D = dimensions,
           std::enable_if_t<(D > 1)>* = nullptr>
  ACPP_KERNEL_TARGET
  detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                      access::target::local, isPlaceholder>
  operator[](size_t index) const
  {
    sycl::id<dimensions> initial_index;
    initial_index[0] = index;
    
    return detail::accessor::subscript_proxy<dataT, dimensions, accessmode,
                                             access::target::local, isPlaceholder> {
      this, initial_index
    };
  }

  ACPP_KERNEL_TARGET
  local_ptr<dataT> get_pointer() const
  {
    return local_ptr<dataT>{
      detail::local_memory::get_ptr<dataT>(_addr)
    };
  }

  iterator begin() const noexcept {
    return iterator::make_begin(this);
  }

  iterator end() const noexcept {
    return iterator::make_end(this);
  }

  const_iterator cbegin() const noexcept {
    return const_iterator::make_begin(this);
  }

  const_iterator cend() const noexcept {
    return const_iterator::make_end(this);
  }

  reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }

  reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }

  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }
private:
  ACPP_KERNEL_TARGET
  accessor(address addr, range<dimensions> r)
    : _addr{addr}, _num_elements{r}
  {}

  specialized<address> _addr;
  specialized<range<dimensions>> _num_elements;
};

namespace detail::accessor {

template<class AccessorType>
glue::unique_id get_unique_id(const AccessorType& acc) {
  return acc.get_uid();
}

}

} // sycl
} // hipsycl

namespace std {

template <typename dataT, int dimensions, hipsycl::sycl::access_mode accessmode,
          hipsycl::sycl::target accessTarget,
          hipsycl::sycl::accessor_variant AccessorVariant>
struct hash<hipsycl::sycl::accessor<dataT, dimensions, accessmode, accessTarget,
                                    AccessorVariant>> {
  std::size_t
  operator()(const hipsycl::sycl::accessor<dataT, dimensions, accessmode, accessTarget,
                                    AccessorVariant> &acc) const {
    return acc.AdaptiveCpp_hash_code();
  }
};
// accessor also covers the local_accessor specialization

template<typename dataT, int dimensions, hipsycl::sycl::access_mode A>
struct hash<hipsycl::sycl::host_accessor<dataT, dimensions, A>> {

  std::size_t operator()(
      const hipsycl::sycl::host_accessor<dataT, dimensions, A> &acc) const {
    return acc.AdaptiveCpp_hash_code();
  }
};

}

#endif
