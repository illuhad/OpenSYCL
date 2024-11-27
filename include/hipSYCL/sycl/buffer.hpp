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
#ifndef HIPSYCL_BUFFER_HPP
#define HIPSYCL_BUFFER_HPP

#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <atomic>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/extensions.hpp"
#include "property.hpp"
#include "types.hpp"
#include "context.hpp"
#include "buffer_allocator.hpp"

#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"

#include "../common/reinterpret_pointer_cast.hpp"

#include "libkernel/accessor.hpp"

namespace hipsycl {
namespace sycl {


namespace detail::buffer_policy {

class destructor_waits : public buffer_property
{ 
public: 
  destructor_waits(bool v): _v{v}{} 
  bool value() const {return _v;}
private:
  bool _v;
};

class writes_back : public buffer_property
{ 
public: 
  writes_back(bool v): _v{v}{} 
  bool value() const {return _v;}
private:
  bool _v;
};

class use_external_storage : public buffer_property
{ 
public: 
  use_external_storage(bool v): _v{v}{} 
  bool value() const {return _v;}
private:
  bool _v;
};

}

namespace property::buffer {

class use_host_ptr : public detail::buffer_property
{
public:
  use_host_ptr() = default;
};

class use_mutex : public detail::buffer_property
{
public:
  use_mutex(mutex_class& ref);
  mutex_class* get_mutex_ptr() const;
};

class context_bound : public detail::buffer_property
{
public:
  context_bound(context bound_context)
    : _ctx{bound_context}
  {}

  context get_context() const
  {
    return _ctx;
  }
private:
  context _ctx;
};

class use_optimized_host_memory : public detail::buffer_property
{};

template<int Dim>
class AdaptiveCpp_page_size : public detail::buffer_property
{
public:
  AdaptiveCpp_page_size(const sycl::range<Dim>& page_size)
  : _page_size{page_size} {}

  sycl::range<Dim> get_page_size() const
  {
    return _page_size;
  }
private:
  sycl::range<Dim> _page_size;
};

class AdaptiveCpp_write_back_node_group : public detail::buffer_property
{
public:
  AdaptiveCpp_write_back_node_group(std::size_t group)
  : _node_group{group} {}

  std::size_t get_node_group() const {
    return _node_group;
  }
private:
  std::size_t _node_group;
};

using AdaptiveCpp_buffer_uses_external_storage =
    detail::buffer_policy::use_external_storage;
using AdaptiveCpp_buffer_writes_back =
    detail::buffer_policy::writes_back;
using AdaptiveCpp_buffer_destructor_blocks =
    detail::buffer_policy::destructor_waits;


// backwards-compatibiliy

template<int Dim>
using hipSYCL_page_size = AdaptiveCpp_page_size<Dim>;

using hipSYCL_buffer_uses_external_storage =
    detail::buffer_policy::use_external_storage;
using hipSYCL_buffer_writes_back =
    detail::buffer_policy::writes_back;
using hipSYCL_buffer_destructor_blocks =
    detail::buffer_policy::destructor_waits;

} // property::buffer


namespace detail {

template <class BufferT>
std::shared_ptr<rt::buffer_data_region>
extract_buffer_data_region(const BufferT &buff);

struct buffer_impl
{
  rt::runtime_keep_alive_token requires_runtime;

  std::mutex lock;
  // Only used if a shared_ptr is passed to set_final_data()
  std::shared_ptr<void> writeback_buffer;
  // Only used if writeback is enabled
  void* writeback_ptr;
  // Only used if a shared_ptr is passed to the buffer constructor
  std::shared_ptr<void> shared_host_data;
  
  std::size_t write_back_node_group;

  std::shared_ptr<rt::buffer_data_region> data;

  bool writes_back;
  bool destructor_waits;
  bool use_external_storage;

  buffer_impl() {
    static std::atomic<bool> was_warning_emitted = false;
    if(!was_warning_emitted) {
      HIPSYCL_DEBUG_WARNING << "This application uses SYCL buffers; the SYCL "
	      "buffer-accessor model is well-known to introduce unnecessary "
	      "overheads. Please consider migrating to the SYCL2020 USM model, "
	      "in particular device USM (sycl::malloc_device) combined with "
	      "in-order queues for more performance. See the AdaptiveCpp "
	      "performance guide for more information: \n"
	      "https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md"
	<< std::endl;
      was_warning_emitted = true;
    }
  }

  ~buffer_impl() {
    if (writes_back) {
      if (!writeback_ptr) {
        HIPSYCL_DEBUG_WARNING
            << "buffer_impl::~buffer_impl: Writeback was requested but "
               "writeback pointer is null. Skipping write-back."
            << std::endl;
      } else {
        HIPSYCL_DEBUG_INFO
            << "buffer_impl::~buffer_impl: Preparing submission of writeback..."
            << std::endl;
        
        if (data->has_allocation(get_host_device()) &&
            (data->get_memory(get_host_device()) != this->writeback_ptr)) {
          // We are writing back to an external location, i.e. a location
          // set with set_final_data()
          // TODO Currently, we are requesting an host update and then
          // submit an explicit copy to writeback_ptr.
          // We could directly copy from device if
          // there is a device that has up-to-date data.
          submit_copy(detail::get_host_device(), writeback_ptr);
        } else {
          rt::dag_build_guard build{requires_runtime.get()->dag()};

          auto explicit_requirement =
              rt::make_operation<rt::buffer_memory_requirement>(
                  data, rt::id<3>{}, data->get_num_elements(),
                  sycl::access::mode::read, sycl::access::target::host_buffer);

          rt::execution_hints hints;
          add_writeback_hints(detail::get_host_device(), hints);

          build.builder()->add_explicit_mem_requirement(
              std::move(explicit_requirement),
              rt::requirements_list{requires_runtime.get()}, hints);
        }
      }
    }
    if(destructor_waits) {
      HIPSYCL_DEBUG_INFO
          << "buffer_impl::~buffer_impl: Waiting for operations to complete..."
          << std::endl;

      auto buffer_users = data->get_users().get_users();
      for (auto &user : buffer_users) {
        auto user_ptr = user.user.lock();
        if(user_ptr) {
          if(!user_ptr->is_submitted()) {
            HIPSYCL_DEBUG_INFO
                << "buffer_impl::~buffer_impl: dag node is registered as user "
                  "but not marked as submitted, performing emergency DAG flush."
                << std::endl;

            requires_runtime.get()->dag().flush_sync();
          }
          assert(user_ptr->is_submitted());
          user_ptr->wait();
        }
      }
    }
  }
private:

  bool has_writeback_node_group() const {
    return write_back_node_group != std::numeric_limits<std::size_t>::max();
  }

  void add_writeback_hints(rt::device_id dev, rt::execution_hints& hints) {
    hints.set_hint(rt::hints::bind_to_device{dev});
    if (has_writeback_node_group()) {
      hints.set_hint(rt::hints::node_group{
          write_back_node_group});
    }
  }
  
  rt::dag_node_ptr submit_copy(rt::device_id source_dev, void* dest) {

    std::shared_ptr<rt::buffer_data_region> data_src = this->data;

    rt::dag_build_guard build{requires_runtime.get()->dag()};
    rt::execution_hints hints;
    add_writeback_hints(source_dev, hints);

    rt::requirements_list reqs{requires_runtime.get()};

    auto req = std::make_unique<rt::buffer_memory_requirement>(
        data_src, rt::id<3>{}, data_src->get_num_elements(), access_mode::read,
        target::device);

    reqs.add_requirement(std::move(req));

    rt::memory_location source_location{source_dev, rt::id<3>{},
                                        data_src};
    
    rt::memory_location dest_location{detail::get_host_device(), dest,
                                      rt::id<3>{}, data_src->get_num_elements(),
                                      data_src->get_element_size()};

    auto explicit_copy = rt::make_operation<rt::memcpy_operation>(
        source_location, dest_location, data_src->get_num_elements());

    rt::dag_node_ptr node = build.builder()->add_command_group(
        std::move(explicit_copy), reqs, hints);

    return node;

  }

};

template <typename, typename = void>
struct has_data : std::false_type {};

template <typename Container>
struct has_data<Container, std::void_t<decltype(std::data(std::declval<Container>()))>>
  : std::true_type {};

template <typename, typename = void>
struct has_size : std::false_type {};

template <typename Container>
struct has_size<Container, std::void_t<decltype(std::size(std::declval<Container>()))>>
  : std::true_type {};

template <typename Container, typename T>
using enable_if_contiguous = std::void_t<std::enable_if_t<
  has_data<Container>::value &&
  has_size<Container>::value &&
  std::is_convertible_v<decltype(std::data(std::declval<Container>())),
                        const T*>>>;
}


namespace buffer_allocation {

// This class is part of the USM-buffer interop API
//
// TODO Currently cannot represent allocations from
// malloc_host()
template <class T> struct descriptor {
  // the USM allocation used for this device
  T *ptr;
  // the device for which this allocation is used.
  // Note that the runtime may only maintain
  // a single allocation for all host devices.
  device dev;

  // Whether the runtime will delete this allocation
  // when the buffer is released.
  bool is_owned;
};

template<class T>
struct tracked_descriptor {
  descriptor<T> desc;
  bool is_recent;
};

enum class management_mode {
  owning,
  non_owning
};

inline constexpr management_mode take_ownership = management_mode::owning;
inline constexpr management_mode no_ownership = management_mode::non_owning;

/// Construct an allocation descriptor for outdated data
/// that needs to be updated by the runtime when accessed
template <class T>
tracked_descriptor<T> empty_view(T *ptr, device dev,
                                    management_mode m = no_ownership) {
  tracked_descriptor<T> d;
  bool is_owned = (m == take_ownership) ? true : false;

  d.desc = descriptor<T>{ptr, dev, is_owned};
  d.is_recent = false;
  return d;
}

/// Construct an allocation descriptor for an allocation
/// already holding live data that is up-to-date and does
/// not need updating before use.
template <class T>
tracked_descriptor<T> view(T *ptr, device dev,
                           management_mode m = no_ownership) {
  tracked_descriptor<T> d;
  bool is_owned = (m == take_ownership) ? true : false;

  d.desc = descriptor<T>{ptr, dev, is_owned};
  d.is_recent = true;
  return d;
}
}


template <typename T, int dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
class buffer : public detail::property_carrying_object
{
public:
  template <class OtherT, int OtherDim, typename OtherAllocator>
  friend class buffer;

  template <class BufferT>
  friend std::shared_ptr<rt::buffer_data_region>
  detail::extract_buffer_data_region(const BufferT &buff);

  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  static constexpr int buffer_dim = dimensions;

  /// buffer USM interop constructor
  buffer(const std::vector<buffer_allocation::tracked_descriptor<T>>
             &input_allocations,
         const range<dimensions> &r,
         const property_list &propList = {})
      : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = true;
    dpol.writes_back = false;

    init_policies_from_properties_or_default(dpol);

    if(_impl->writes_back) {
      HIPSYCL_DEBUG_WARNING
          << "buffer: Explicit writeback policy was requested, but buffers "
             "using USM interoperability cannot enable writeback at "
             "construction. Disabling writeback."
          << std::endl;
      _impl->writes_back = false;
    }
    if(!_impl->use_external_storage) {
      HIPSYCL_DEBUG_WARNING
          << "buffer: No external storage policy was explicitly requested, but "
             "this does not make sense for USM interoperability buffers. "
             "Enabling using external storage."
          << std::endl;
      _impl->writes_back = false;
    }

    this->init(r, input_allocations);
  }

  buffer(const std::vector<buffer_allocation::tracked_descriptor<T>>
             &input_allocations,
         const range<dimensions> &r,
         AllocatorT allocator, const property_list &propList = {})
      : buffer(r, input_allocations, propList)
  {
    _alloc = allocator;
  }

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    
    init_policies_from_properties_or_default(dpol);

    this->init(bufferRange);

    if(_impl->use_external_storage) {
      HIPSYCL_DEBUG_WARNING
          << "buffer: was constructed with use_external_storage but no host "
             "pointer was supplied. Cannot initialize this buffer with "
             "external storage."
          << std::endl;
    }
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : buffer(bufferRange, propList)
  { _alloc = allocator; }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = true;

    if constexpr (std::is_const_v<T>)
      dpol.writes_back = false;
    else
      dpol.writes_back = true;
    
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage)
      this->init(bufferRange, hostData);
    else {
      this->init(bufferRange);
      copy_host_content(hostData);
    }
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
      : buffer{hostData, bufferRange, propList} {
    _alloc = allocator;
  }

  template <class t = T, std::enable_if_t<!std::is_const_v<t>, bool> = true>
  buffer(const T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : detail::property_carrying_object{propList} {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    if(!_impl->use_external_storage) {
      // Construct buffer
      this->init(bufferRange);
      // Only use hostData for initialization
      copy_host_content(hostData);
    } else {
      HIPSYCL_DEBUG_WARNING
          << "buffer: constructed with property use_external_storage, but user "
             "passed a const pointer to buffer constructor. Removing const to enforce "
             "requested view semantics."
          << std::endl;
      this->init(bufferRange, const_cast<T*>(hostData));
    }
  }

  template <class t = T, std::enable_if_t<!std::is_const_v<t>, bool> = true>
  buffer(const T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
    : buffer{hostData, bufferRange, propList}
  { _alloc = allocator; }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();
    _alloc = allocator;

    default_policies dpol;
    dpol.destructor_waits = true;

    if (hostData.use_count() != 0) {
      dpol.use_external_storage = true;
      dpol.writes_back = true;
    } else {
      dpol.use_external_storage = false;
      dpol.writes_back = false;
    }
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage) {
      _impl->shared_host_data = hostData;
      this->init(bufferRange, hostData.get());
    } else {
      this->init(bufferRange);

      if (hostData.use_count() != 0)
	this->copy_host_content(hostData.get());
    }
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
  : buffer(hostData, bufferRange, AllocatorT(), propList)
  {}

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();
    _alloc = allocator;

    default_policies dpol;
    dpol.destructor_waits = true;

    if (hostData.use_count() != 0) {
      dpol.use_external_storage = true;
      dpol.writes_back = true;
    } else {
      dpol.use_external_storage = false;
      dpol.writes_back = false;
    }
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage) {
      _impl->shared_host_data = hostData;
      this->init(bufferRange, hostData.get());
    } else {
      this->init(bufferRange);

      if (hostData.use_count() != 0)
	this->copy_host_content(hostData.get());
    }
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
  : buffer(hostData, bufferRange, AllocatorT(), propList)
  {}

  template <class InputIterator,
            int D = dimensions,
            typename = std::enable_if_t<D==1>>
  buffer(InputIterator first, InputIterator last,
         AllocatorT allocator,
         const property_list &propList = {})
  : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();

    default_policies dpol;
    dpol.destructor_waits = true;
    dpol.use_external_storage = false;
    dpol.writes_back = false;
    init_policies_from_properties_or_default(dpol);

    if(_impl->use_external_storage)
      // TODO This could be allowed for special cases, e.g. if iterators are pointers
      throw exception{make_error_code(errc::invalid),
                      "buffer: Cannot comply: User requested to using " 
                      "external storage, but this is not yet possible "
                      "with iterators."};

    _alloc = allocator;

    std::size_t num_elements = std::distance(first, last);

    // Construct buffer
    this->init(range<1>{num_elements});

    // Work around vector<bool> specialization..
    if constexpr(std::is_same_v<bool, std::remove_const_t<T>>){
      std::vector<char> contiguous_buffer(num_elements);
      std::copy(first, last, reinterpret_cast<T*>(&(contiguous_buffer[0])));
      copy_host_content(reinterpret_cast<T*>(contiguous_buffer.data()));
    } else {
      std::vector<T> contiguous_buffer;
      contiguous_buffer.reserve(num_elements);
      std::copy(first, last, contiguous_buffer.begin());
      copy_host_content(contiguous_buffer.data());
    }
  }

  template <class InputIterator, int D = dimensions,
            typename = std::enable_if_t<D == 1>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {})
  : buffer(first, last, AllocatorT(), propList) 
  {}

  template <typename Container,
            int D = dimensions,
            typename = std::enable_if_t<D == 1>,
            typename = detail::enable_if_contiguous<Container, T>>
  buffer(Container& container, AllocatorT allocator,
         const property_list& propList = {})
    : detail::property_carrying_object{propList}
  {
    _impl = std::make_shared<detail::buffer_impl>();
    _alloc = allocator;
    
    constexpr bool is_const_container = std::is_const_v<
      std::remove_pointer_t<decltype(std::data(container))>>;

    default_policies dpol;
    dpol.destructor_waits = true;
    // If std::data returns non-const pointer, enable write_back
    if constexpr (is_const_container) {
      dpol.writes_back = false;
      dpol.use_external_storage = false;
    } else {
      dpol.writes_back = true;
      dpol.use_external_storage = true;
    }

    init_policies_from_properties_or_default(dpol);

    const range<1> bufferRange(std::size(container));

    if constexpr (is_const_container) {
      if (_impl->use_external_storage) {
         HIPSYCL_DEBUG_WARNING
          << "buffer: constructed with property use_external_storage, but user "
             "passed a const container to buffer constructor. Removing const to enforce "
             "requested view semantics."
          << std::endl;
         this->init(bufferRange, const_cast<T*>(std::data(container)));
      } else {
        this->init(bufferRange);
        copy_host_content(std::data(container));
      }
    } else {
      this->init(bufferRange, std::data(container));
    }
  }

  template <typename Container,
            int D = dimensions,
            typename = std::enable_if_t<D == 1>,
            typename = detail::enable_if_contiguous<Container, T>>
  buffer(Container& container, const property_list& propList = {})
    : buffer(container, AllocatorT(), propList) {}

  buffer(buffer<T, dimensions, AllocatorT> b,
         const id<dimensions> &baseIndex,
         const range<dimensions> &subRange)
  {
    assert(false && "subbuffer is unimplemented");
  }

  // Allow conversion to buffer<const T> from buffer<T>
  template <class t = T, std::enable_if_t<std::is_const_v<t>, bool> = true>
  buffer(const buffer<std::remove_const_t<T>, dimensions, AllocatorT> &other)
      : _alloc{other._alloc}, _range{other._range}, _impl{other._impl},
        detail::property_carrying_object{other} {}

  range<dimensions> get_range() const
  {
    return _range;
  }

  std::size_t size() const noexcept
  {
    return _range.size();
  }

  std::size_t byte_size() const noexcept
  {
    return size() * sizeof(T);
  }

  std::size_t get_size() const
  {
    return byte_size();
  }

  std::size_t get_count() const
  {
    return size();
  }

  AllocatorT get_allocator() const
  {
    return _alloc;
  }

  template <access_mode mode = access_mode::read_write,
            access::target target = access::target::device>
  auto get_access(handler &commandGroupHandler) {
#ifdef ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION
    constexpr accessor_variant variant = accessor_variant::unranged;
#else
    constexpr accessor_variant variant = accessor_variant::false_t;
#endif
    return accessor<T, dimensions, mode, target, variant>{
        *this, commandGroupHandler};
  }

  // Deprecated
  template <access::mode mode>
  auto get_access()
  {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    accessor_variant::false_t>{*this};
  }

  template <access_mode mode = access_mode::read_write,
            access::target target = access::target::device>
  auto get_access(handler &commandGroupHandler, range<dimensions> accessRange,
             id<dimensions> accessOffset = {}) {

#ifdef ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION
    constexpr accessor_variant variant = accessor_variant::ranged;
#else
    constexpr accessor_variant variant = accessor_variant::false_t;
#endif

    return accessor<T, dimensions, mode, target, variant>{
      *this, commandGroupHandler, accessRange, accessOffset
    };
  }

  // Deprecated
  template <access::mode mode>
  auto get_access(
      range<dimensions> accessRange, id<dimensions> accessOffset = {})
  {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    accessor_variant::false_t>{*this, accessRange,
                                               accessOffset};
  }

  template<typename... Args>
  auto get_access(Args&&... args) {
    return accessor{*this, std::forward<Args>(args)...};
  }

  template<typename... Args>
  auto get_host_access(Args&&... args) {
    return host_accessor{*this, std::forward<Args>(args)...};
  }

  void set_final_data(std::shared_ptr<T> finalData)
  {
    std::lock_guard<std::mutex> lock {_impl->lock};
    set_write_back_target(finalData.get());
    
    _impl->writeback_buffer = finalData;
  }

  // TODO Add special handling of iterators for set_final_data()
  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr)
  {
    std::lock_guard<std::mutex> lock {_impl->lock};
    set_write_back_target(finalData);
  }

  void set_write_back(bool flag = true)
  {
    std::lock_guard<std::mutex> lock {_impl->lock};
    this->enable_write_back(flag);
  }

  // ToDo Subbuffers are unsupported
  bool is_sub_buffer() const
  { return false; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim,
        typename std::allocator_traits<AllocatorT>
                 ::template rebind_alloc<ReinterpretT>>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    if(_range.size() * sizeof(T) != reinterpretRange.size() * sizeof(ReinterpretT))
      throw exception{make_error_code(errc::invalid),
                      "reinterpret must preserve the byte count of the buffer"};

    buffer<ReinterpretT, ReinterpretDim,
            typename std::allocator_traits<AllocatorT>::template rebind_alloc<
            ReinterpretT>> new_buffer;
    static_cast<detail::property_carrying_object&>(new_buffer) = *this;
    new_buffer._alloc = _alloc;
    new_buffer._impl = _impl;
    new_buffer._range = reinterpretRange;
    
    return new_buffer;
  }

  template <typename ReinterpretT, int ReinterpretDim = dimensions,
    std::enable_if_t<ReinterpretDim == 1 ||
      (ReinterpretDim == dimensions && sizeof(ReinterpretT) == sizeof(T)), int> = 0>
  buffer<ReinterpretT, ReinterpretDim,
        typename std::allocator_traits<AllocatorT>
                 ::template rebind_alloc<ReinterpretT>>
  reinterpret() const {
    if constexpr (ReinterpretDim == 1) {
      return reinterpret<ReinterpretT, 1>(range<1>{
        (_range.size() * sizeof(T)) / sizeof(ReinterpretT)});
    } else {
      return reinterpret<ReinterpretT, ReinterpretDim>(_range);
    }
  }

  friend bool operator==(const buffer& lhs, const buffer& rhs)
  {
    return lhs._impl == rhs._impl;
  }

  friend bool operator!=(const buffer& lhs, const buffer& rhs)
  {
    return !(lhs == rhs);
  }

  std::size_t AdaptiveCpp_hash_code() const {
    return std::hash<void*>{}(_impl.get());
  }

  rt::runtime* AdaptiveCpp_runtime() const {
    return _impl->requires_runtime.get();
  }

  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

  [[deprecated("Use AdaptiveCpp_runtime()")]]
  auto hipSYCL_runtime() const {
    return AdaptiveCpp_runtime();
  }
  
  // --- The following methods are part the hipSYCL buffer introspection API
  // which is part of the hipSYCL buffer-USM interoperability framework.

  /// Iterate over each allocation.
  /// \param h Handler that will be invoked for each allocation.
  ///  Signature: void(const buffer_allocation::descriptor<T>&)
  template <class Handler>
  void for_each_allocation(Handler &&h) const{
    _impl->data->for_each_allocation_while(
        [&h](const rt::data_allocation<void *> &alloc) {
          buffer_allocation::descriptor<T> a =
              rt_data_allocation_to_buffer_alloc(alloc);
          h(a);

          return true;
        });
  }

  /// Instruct buffer to free the allocation on the specified device at buffer
  /// destruction.
  /// Throws \c errc::invalid exception if no allocation for specified device
  /// exists.
  void own_allocation(const device &dev) {

    rt::device_id rt_dev = detail::extract_rt_device(dev);

    bool found = _impl->data->find_and_handle_allocation(
        rt_dev,
        [](rt::data_allocation<void *> &alloc) { alloc.is_owned = true; });

    if (!found)
      throw exception{make_error_code(errc::invalid),
                      "Buffer does not contain allocation for specified device"};
  }
  /// Instruct buffer to free the allocation at buffer destruction.
  /// \c ptr must be an existing allocation managed by the buffer.
  /// If \c ptr cannot be found among the managed memory allocations,
  /// \c errc::invalid exception is thrown.
  void own_allocation(const T *ptr) {
    bool found = _impl->data->find_and_handle_allocation(
        static_cast<void *>(const_cast<T*>(ptr)),
        [&](rt::data_allocation<void *> &rt_allocation) {
          rt_allocation.is_owned = true;
        });

    if (!found) {
      throw exception{make_error_code(errc::invalid),
                      "Provided pointer was not found among the "
                      "managed buffer allocations."};
    }
  }

  /// Instruct buffer to not free the allocation on the specified device at buffer
  /// destruction.
  /// Throws \c errc::invalid exception if no allocation for specified device
  /// exists.
  void disown_allocation(const device &dev) {
    rt::device_id rt_dev = detail::extract_rt_device(dev);

    bool found = _impl->data->find_and_handle_allocation(
        rt_dev,
        [](rt::data_allocation<void *> &alloc) { alloc.is_owned = false; });

    if (!found)
      throw exception{make_error_code(errc::invalid),
                      "Buffer does not contain allocation for specified device"};
  }

  /// Instruct buffer to not free the allocation associated with the provided
  /// pointer.
  /// Throws \c errc::invalid exception if no allocation managed by the buffer
  /// is described by \c ptr.
  void disown_allocation(const T *ptr) {
    bool found = _impl->data->find_and_handle_allocation(
        static_cast<void *>(const_cast<T*>(ptr)),
        [&](rt::data_allocation<void *> &rt_allocation) {
          rt_allocation.is_owned = false;
        });

    if (!found) {
      throw exception{make_error_code(errc::invalid),
                      "Provided pointer was not found among the "
                      "managed buffer allocations."};
    }
  }

  /// Get USM pointer for the buffer allocation of the specified device.
  /// \return The USM pointer associated with the device, or nullptr if
  /// the buffer does not contain an allocation for the device.
  T *get_pointer(const device &dev) const {
    rt::device_id rt_dev = detail::extract_rt_device(dev);

    if (!_impl->data->has_allocation(rt_dev))
      return nullptr;

    // Because the hipSYCL buffer-accessor model spec guarantees that
    // allocations are never freed before buffer destruction,
    // it is not a race condition to assume that the allocation still
    // exists after the check above.
    return static_cast<T*>(_impl->data->get_memory(rt_dev));
  }

  /// \return Whether the buffer contains an allocation for the given device.
  bool has_allocation(const device &dev) const {
    rt::device_id rt_dev = detail::extract_rt_device(dev);

    return _impl->data->has_allocation(rt_dev);
  }

  /// \return the buffer allocation object associated with the provided
  /// device. If the buffer does not contain an allocation for the specified
  /// device, throws \c errc::invalid exception.
  buffer_allocation::descriptor<T> get_allocation(const device &dev) const {
    rt::device_id rt_dev = detail::extract_rt_device(dev);

    if (!_impl->data->has_allocation(rt_dev))
      throw exception{make_error_code(errc::invalid),
                      "No allocation for the given device was found"};

    auto rt_allocation = _impl->data->get_allocation(rt_dev);
    return rt_data_allocation_to_buffer_alloc(rt_allocation);
  }

  /// \return the buffer allocation object associated with the provided pointer.
  /// If the buffer does not contain an allocation described by ptr,
  /// throws \c errc::invalid exception.
  buffer_allocation::descriptor<T> get_allocation(const T *ptr) const {

    buffer_allocation::descriptor<T> result = null_allocation();
    bool found = _impl->data->find_and_handle_allocation(
        static_cast<void *>(const_cast<T*>(ptr)), 
        [&](const auto &rt_allocation) {
      result = rt_data_allocation_to_buffer_alloc(rt_allocation);
    });

    if (!found) {
      throw exception{make_error_code(errc::invalid),
                      "Provided pointer was not found among the "
                      "managed buffer allocations."};
    }
    return result;
  }

  // -- End of hipSYCL buffer-USM introspection API
private:
  
  struct default_policies
  {
    bool destructor_waits;
    bool writes_back;
    bool use_external_storage; 
  };

  static buffer_allocation::descriptor<T>
  rt_data_allocation_to_buffer_alloc(const rt::data_allocation<void *> &alloc) {

    buffer_allocation::descriptor<T> buffer_alloc;
    buffer_alloc.dev = sycl::device{alloc.dev};
    buffer_alloc.is_owned = alloc.is_owned;
    buffer_alloc.ptr = static_cast<T *>(alloc.memory);

    return buffer_alloc;
  }

  static buffer_allocation::descriptor<T> null_allocation() {
    buffer_allocation::descriptor<T> result;
    result.ptr = nullptr;
    result.is_owned = false;
    result.dev = detail::get_host_device();

    return result;
  }
  
  
  template <typename Destination = std::nullptr_t>
  void set_write_back_target(Destination finalData = nullptr)
  {
    if constexpr(std::is_pointer_v<Destination> || std::is_null_pointer_v<Destination>){
      if (finalData) {
        enable_write_back(true);
      }
      else {
        enable_write_back(false);
      }

      _impl->writeback_ptr = finalData;
    } else {
      // Assume it is an iterator to contiguous memory
      enable_write_back(true);
      _impl->writeback_ptr = &(*finalData);
    }
  }

  void enable_write_back(bool flag) {

    if(this->has_property<detail::buffer_policy::writes_back>()){
      if (_impl->writes_back != flag){
        // Deny changing policy if it has previously been explicitly requested
        // by the user
        throw exception{make_error_code(errc::invalid),
                        "buffer::set_write_back(): buffer was constructed "
                        "explicitly with writeback policy, denying changing "
                        "the policy as this likely indicates a bug in user code"};
      }
    }
    if(_impl->writes_back != flag) {
      HIPSYCL_DEBUG_INFO << "buffer: Changing write back policy to: " << flag
                         << std::endl;
      _impl->writes_back = flag;
    }
  }

  void copy_host_content(const T* data)
  {
    assert(_impl);
    auto host_device = detail::get_host_device();
    preallocate_host_buffer();

    std::memcpy(_impl->data->get_memory(host_device), data,
                sizeof(T) * _range.size());
    // Mark the modified range current so that the runtime
    // knows that it needs to transfer this data if it is
    // accessed on device
    _impl->data->mark_range_current(host_device,
                                    rt::embed_in_id3(sycl::id<3>{}),
                                    rt::embed_in_range3(get_range()));
  }

  void init_policies_from_properties_or_default(default_policies dpol)
  {
    _impl->destructor_waits = get_policy_from_property_or_default<
        detail::buffer_policy::destructor_waits>(dpol.destructor_waits);
    
    _impl->writes_back =
        get_policy_from_property_or_default<detail::buffer_policy::writes_back>(
            dpol.writes_back);

    _impl->use_external_storage = get_policy_from_property_or_default<
        detail::buffer_policy::use_external_storage>(dpol.use_external_storage);

    if(this->has_property<property::buffer::AdaptiveCpp_write_back_node_group>()){
      _impl->write_back_node_group =
          this->get_property<property::buffer::AdaptiveCpp_write_back_node_group>()
              .get_node_group();
    } else {
      this->_impl->write_back_node_group =
          std::numeric_limits<std::size_t>::max();
    }
  }

  template<class Policy>
  bool get_policy_from_property_or_default(bool default_value)
  {
    if(this->has_property<Policy>())
      return this->get_property<Policy>().value();
    return default_value;
  }

  
  buffer()
  : detail::property_carrying_object {property_list {}}
  {}
  
  void init_data_backend(const range<dimensions>& range)
  {
    this->_range = range;

    rt::range<3> page_size = rt::embed_in_range3(range);
    if (this->has_property<property::buffer::AdaptiveCpp_page_size<dimensions>>()) {
      page_size = rt::embed_in_range3(
          this->get_property<property::buffer::AdaptiveCpp_page_size<dimensions>>()
              .get_page_size());
    }

    _impl->data = std::make_shared<rt::buffer_data_region>(
        rt::embed_in_range3(range), sizeof(T), page_size);
  }

  void preallocate_host_buffer()
  {
    void* host_ptr = nullptr;
    rt::device_id host_device = detail::get_host_device();
    rt::runtime* rt = _impl->requires_runtime.get();

    if(!_impl->data->has_allocation(host_device)){
      if(this->has_property<property::buffer::use_optimized_host_memory>()){
        // TODO: Actually may need to use non-host backend here...
        host_ptr =
            rt->backends().get(host_device.get_backend())
                ->get_allocator(host_device)
                ->allocate_optimized_host(
                    alignof(T), _impl->data->get_num_elements().size() * sizeof(T));
      } else {
        host_ptr =
            rt->backends().get(host_device.get_backend())
                ->get_allocator(host_device)
                ->allocate(
                    alignof(T), _impl->data->get_num_elements().size() * sizeof(T));
      }

      if(!host_ptr)
        throw exception{make_error_code(errc::runtime),
                        "buffer: host memory allocation failed"};

      _impl->data->add_empty_allocation(
          host_device, host_ptr,
          rt->backends().get(host_device.get_backend())
              ->get_allocator(host_device),
          true /*takes_ownership*/);
    }
  }

  void init(const range<dimensions>& range)
  {
    if(range.size() > 0) {
      this->init_data_backend(range);
      // necessary to preallocate to make sure potential optimized memory
      // can be allocated
      preallocate_host_buffer();
    } else {
      sycl::range<dimensions> default_range;
      for(int i = 0; i < dimensions; ++i)
        default_range[i] = 1;
      this->init_data_backend(default_range);
    }
  }

  void init(const range<dimensions>& range, T* host_memory)
  {
    if(!host_memory)
      throw exception{make_error_code(errc::invalid),
                      "buffer: Supplied host pointer is null."};

    if(this->has_property<property::buffer::use_optimized_host_memory>()){
      HIPSYCL_DEBUG_INFO
          << "buffer: was constructed with use_optimized_host_memory property, "
             "but also as view for existing host memory. "
             "use_optimized_host_memory will have no effect, since "
             "the buffer will rely on existing memory instead of allocating itself."
          << std::endl;
    }

    this->init_data_backend(range);

    rt::device_id host_device = detail::get_host_device();
    _impl->data->add_nonempty_allocation(detail::get_host_device(),
					 const_cast<std::remove_const_t<T>*>(host_memory),
                                         _impl->requires_runtime.get()
                                             ->backends()
                                             .get(host_device.get_backend())
                                             ->get_allocator(host_device),
                                         false /*takes_ownership*/);
    // Remember host_memory in case of potential write back
    _impl->writeback_ptr = const_cast<std::remove_const_t<T>*>(host_memory);
  }

  void init(const range<dimensions> &range,
            const std::vector<buffer_allocation::tracked_descriptor<T>>
                &input_allocations) {
    this->init_data_backend(range);

    if(input_allocations.size() == 0) {
      throw exception{make_error_code(errc::invalid),
                      "buffer: USM constructor was used, but no "
                      "USM allocations to work with were used."};
    }

    // TODO: We should check here for duplicate USM pointers or duplicate devices
    for (const buffer_allocation::tracked_descriptor<T> &desc :
         input_allocations) {
      if(!desc.desc.ptr) {
        throw exception{make_error_code(errc::invalid),
                        "buffer: Invalid USM input pointer"};
      }

      rt::device_id dev = detail::extract_rt_device(desc.desc.dev);
      rt::backend_allocator *allocator = _impl->requires_runtime.get()
                                             ->backends()
                                             .get(dev.get_backend())
                                             ->get_allocator(dev);

      if (desc.is_recent) {
        _impl->data->add_nonempty_allocation(
            dev, desc.desc.ptr, allocator, desc.desc.is_owned);
      } else {
        _impl->data->add_empty_allocation(
            dev, desc.desc.ptr, allocator, desc.desc.is_owned);
      }
    }
  }

  AllocatorT _alloc;
  range<dimensions> _range;

  std::shared_ptr<detail::buffer_impl> _impl;
};

// Deduction guides
template <class InputIterator, class AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list & = {}) 
-> buffer<typename std::iterator_traits<InputIterator>::value_type, 1, AllocatorT>;

template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
-> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;

template <class Container, class AllocatorT>
buffer(Container&, AllocatorT, const property_list& = {})
    -> buffer<typename Container::value_type, 1, AllocatorT>;

template <class Container>
buffer(Container&, const property_list& = {})
    -> buffer<typename Container::value_type, 1>;

template <class T, int dimensions, class AllocatorT>
buffer(const T *, const range<dimensions> &, AllocatorT, const property_list & = {})
-> buffer<T, dimensions, AllocatorT>;

template <class T, int dimensions>
buffer(const T *, const range<dimensions> &, const property_list & = {}) 
-> buffer<T, dimensions, buffer_allocator<std::remove_const_t<T>>>;

template <class T, int dimensions>
buffer(const range<dimensions> &r,
       const std::vector<buffer_allocation::tracked_descriptor<T>>
           &input_allocations,
       const property_list &propList = {})
    -> buffer<T, dimensions, buffer_allocator<std::remove_const_t<T>>>;

template <class T, int dimensions, class AllocatorT>
buffer(const range<dimensions> &r,
       const std::vector<buffer_allocation::tracked_descriptor<T>>
           &input_allocations,
       AllocatorT allocator, const property_list &propList = {})
    -> buffer<T, dimensions, AllocatorT>;

namespace detail {

template <class BufferT>
std::shared_ptr<rt::buffer_data_region>
extract_buffer_data_region(const BufferT &buff) {
  return buff._impl->data;
}

template <class T, int dimensions, class AllocatorT>
sycl::range<dimensions>
extract_buffer_range(const buffer<T, dimensions, AllocatorT> &buff) {
  return buff.get_range();
}

}


} // sycl
} // hipsycl

namespace std {

template <typename T, int dimensions,
          typename AllocatorT>
struct hash<hipsycl::sycl::buffer<T, dimensions, AllocatorT>>
{
  std::size_t
  operator()(const hipsycl::sycl::buffer<T, dimensions, AllocatorT> &b) const {
    return b.AdaptiveCpp_hash_code();
  }
};

}

#endif
