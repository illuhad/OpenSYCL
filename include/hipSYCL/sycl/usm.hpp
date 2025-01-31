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
#ifndef HIPSYCL_USM_HPP
#define HIPSYCL_USM_HPP

#include <cstdint>
#include <cassert>
#include <exception>

#include "context.hpp"
#include "device.hpp"
#include "property.hpp"
#include "queue.hpp"
#include "exception.hpp"
#include "usm_query.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/hints.hpp"

namespace hipsycl {
namespace sycl {

// Wrapper namespace to declare all usm properties
namespace property::usm {

}

// Explicit USM

inline void *malloc_device(size_t num_bytes, const device &dev,
                           const context &ctx,
                           const property_list &propList = {}) {
  rt::allocation_hints hints{};
  return rt::allocate_device(detail::select_device_allocator(dev), 0,
                             num_bytes, hints);
}

template <typename T>
T* malloc_device(std::size_t count, const device &dev,
                 const context &ctx,
                 const property_list &propList = {}) {
  return static_cast<T*>(malloc_device(count * sizeof(T), dev, ctx, propList));
}

inline void *malloc_device(size_t num_bytes, const queue &q,
                           const property_list &propList = {}) {
  return malloc_device(num_bytes, q.get_device(), q.get_context(), propList);
}

template <typename T>
T* malloc_device(std::size_t count, const queue &q,
                 const property_list &propList = {}) {
  return malloc_device<T>(count, q.get_device(), q.get_context(), propList);
}

inline void *aligned_alloc_device(std::size_t alignment, std::size_t num_bytes,
                                  const device &dev, const context &ctx,
                                  const property_list &propList = {}) {
  rt::allocation_hints hints{};
  return rt::allocate_device(detail::select_device_allocator(dev), alignment,
                             num_bytes, hints);
}

template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const device &dev, const context &ctx,
                        const property_list &propList = {}) {
  return static_cast<T *>(
      aligned_alloc_device(alignment, count * sizeof(T), dev, ctx, propList));
}

inline void *aligned_alloc_device(std::size_t alignment, std::size_t size,
                                  const queue &q,
                                  const property_list &propList = {}) {
  return aligned_alloc_device(alignment, size, q.get_device(), q.get_context(),
                              propList);
}

template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const queue &q,
                        const property_list &propList = {}) {
  return aligned_alloc_device<T>(alignment, count, q.get_device(),
                                 q.get_context(), propList);
}

// Restricted USM

inline void *malloc_host(std::size_t num_bytes, const context &ctx,
                         const property_list &propList = {}) {
  rt::allocation_hints hints{};
  return rt::allocate_host(detail::select_usm_allocator(ctx), 0, num_bytes,
                           hints);
}

template <typename T> T *malloc_host(std::size_t count, const context &ctx,
                                     const property_list &propList = {}) {
  return static_cast<T*>(malloc_host(count * sizeof(T), ctx, propList));
}

inline void *malloc_host(std::size_t num_bytes, const queue &q,
                         const property_list &propList = {}) {
  return malloc_host(num_bytes, q.get_context(), propList);
}

template <typename T> T *malloc_host(std::size_t count, const queue &q,
                                     const property_list &propList = {}) {
  return malloc_host<T>(count, q.get_context(), propList);
}

inline void *malloc_shared(std::size_t num_bytes, const device &dev,
                           const context &ctx,
                           const property_list &propList = {}) {
  rt::allocation_hints hints{};
  return rt::allocate_shared(detail::select_usm_allocator(ctx, dev), num_bytes,
                             hints);
}

template <typename T>
T *malloc_shared(std::size_t count, const device &dev,
                 const context &ctx,
                 const property_list &propList = {}) {
  return static_cast<T*>(malloc_shared(count * sizeof(T), dev, ctx, propList));
}

inline void *malloc_shared(std::size_t num_bytes, const queue &q,
                           const property_list &propList = {}) {
  return malloc_shared(num_bytes, q.get_device(), q.get_context(), propList);
}

template <typename T> T *malloc_shared(std::size_t count, const queue &q,
                                       const property_list &propList = {}) {
  return malloc_shared<T>(count, q.get_device(), q.get_context(), propList);
}

inline void *aligned_alloc_host(std::size_t alignment, std::size_t num_bytes,
                                const context &ctx,
                                const property_list &propList = {}) {
  rt::allocation_hints hints{};
  return rt::allocate_host(detail::select_usm_allocator(ctx), alignment,
                           num_bytes, hints);
}

template <typename T>
T *aligned_alloc_host(std::size_t alignment, size_t count, const context &ctx,
                      const property_list &propList = {}) {
  return static_cast<T*>(aligned_alloc_host(alignment, count * sizeof(T), ctx,
                         propList));
}

inline void *aligned_alloc_host(size_t alignment, size_t num_bytes,
                                const queue &q,
                                const property_list &propList = {}) {
  return aligned_alloc_host(alignment, num_bytes, q.get_context(), propList);
}

template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count,
                         const queue &q,
                         const property_list &propList = {}) {
  return static_cast<T *>(
      aligned_alloc_host(alignment, count * sizeof(T), q.get_context(),
                         propList));
}

inline void *aligned_alloc_shared(std::size_t alignment, std::size_t num_bytes,
                                  const device &dev, const context &ctx,
                                  const property_list &propList = {}) {
  rt::allocation_hints hints{};
  return rt::allocate_shared(detail::select_usm_allocator(ctx, dev), num_bytes,
                             hints);
}

template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const device &dev, const context &ctx,
                        const property_list &propList = {}) {
  return static_cast<T*>(
      aligned_alloc_shared(alignment, count * sizeof(T), dev, ctx, propList));
}

inline void *aligned_alloc_shared(std::size_t alignment, std::size_t num_bytes,
                                  const queue &q,
                                  const property_list &propList = {}) {
  return aligned_alloc_shared(alignment, num_bytes, q.get_device(),
                              q.get_context(), propList);
}

template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const queue &q,
                        const property_list &propList = {}) {
  return static_cast<T *>(
      aligned_alloc_shared(alignment, count * sizeof(T), q.get_device(),
                           q.get_context(), propList));
}


// General

inline void *malloc(std::size_t num_bytes, const device &dev,
                    const context &ctx, usm::alloc kind,
                    const property_list &propList = {}) {

  if (kind == usm::alloc::device) {
    return malloc_device(num_bytes, dev, ctx, propList);
  } else if (kind == usm::alloc::host) {
    return malloc_host(num_bytes, ctx, propList);
  } else if (kind == usm::alloc::shared) {
    return malloc_shared(num_bytes, dev, ctx, propList);
  }
  return nullptr;
}

template <typename T>
T *malloc(std::size_t count, const device &dev, const context &ctx,
          usm::alloc kind,
          const property_list &propList = {}) {
  return static_cast<T*>(malloc(count * sizeof(T), dev, ctx, kind, propList));
}

inline void *malloc(std::size_t num_bytes, const queue &q, usm::alloc kind,
                    const property_list &propList = {}) {
  return malloc(num_bytes, q.get_device(), q.get_context(), kind, propList);
}

template <typename T>
T *malloc(std::size_t count, const queue &q, usm::alloc kind,
          const property_list &propList = {}) {
  return static_cast<T *>(
      malloc(count * sizeof(T), q.get_device(), q.get_context(), kind,
             propList));
}

inline void *aligned_alloc(std::size_t alignment, std::size_t num_bytes,
                           const device &dev, const context &ctx,
                           usm::alloc kind,
                           const property_list &propList = {}) {
  if (kind == usm::alloc::device) {
    return aligned_alloc_device(alignment, num_bytes, dev, ctx, propList);
  } else if (kind == usm::alloc::host) {
    return aligned_alloc_host(alignment, num_bytes, ctx, propList);
  } else if (kind == usm::alloc::shared) {
    return aligned_alloc_shared(alignment, num_bytes, dev, ctx, propList);
  }
  return nullptr;
}

template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count, const device &dev,
                 const context &ctx, usm::alloc kind,
                 const property_list &propList = {}) {
  return static_cast<T *>(
      aligned_alloc(alignment, count * sizeof(T), dev, ctx, kind, propList));
}

inline void *aligned_alloc(std::size_t alignment, std::size_t num_bytes,
                           const sycl::queue &q, usm::alloc kind,
                           const property_list &propList = {}) {
  return aligned_alloc(alignment, num_bytes, q.get_device(), q.get_context(),
                       kind, propList);
}

template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count, const sycl::queue &q,
                 usm::alloc kind,
                 const property_list &propList = {}) {
  return static_cast<T *>(aligned_alloc(alignment, count * sizeof(T),
                                        q.get_device(), q.get_context(), kind,
                                        propList));
}

inline void free(void *ptr, const sycl::context &ctx) {
  return rt::deallocate(detail::select_usm_allocator(ctx), ptr);
}

inline void free(void *ptr, const sycl::queue &q) {
  free(ptr, q.get_context());
}

// hipSYCL synchronous mem_advise extension
inline void mem_advise(const void *ptr, std::size_t num_bytes, int advise,
                       const context &ctx, const device &dev) {

  rt::backend_allocator* b = detail::select_usm_allocator(ctx, dev);
  assert(b);

  rt::result r = b->mem_advise(ptr,  num_bytes, advise);

  if(!r.is_success())
    std::rethrow_exception(glue::throw_result(r));
}

inline void mem_advise(const void *ptr, std::size_t num_bytes, int advise,
                       const queue& q) {
  mem_advise(ptr, num_bytes, advise, q.get_context(), q.get_device());
}

// USM allocator
template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  static_assert(
      AllocKind != usm::alloc::device,
      "usm_allocator does not support AllocKind == usm::alloc::device");

  usm_allocator() noexcept = delete;
  usm_allocator(const context &ctx, const device &dev,
                const property_list &propList = {}) noexcept
      : _ctx{ctx}, _dev{dev}, _propList{propList} {}


  usm_allocator(const queue &q,
                const property_list &propList = {}) noexcept
      : _ctx{q.get_context()}, _dev{q.get_device()}, _propList{propList} {}
  
  usm_allocator(const usm_allocator &) noexcept = default;
  usm_allocator(usm_allocator &&) noexcept = default;

  usm_allocator &operator=(const usm_allocator &) = delete;
  usm_allocator &operator=(usm_allocator &&) = default;

  template <class U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept
      : _ctx{other._ctx}, _dev{other._dev}, _propList{other._propList} {}

  T *allocate(std::size_t num_elements) {

    T *ptr = aligned_alloc<T>(Alignment, num_elements, _dev, _ctx, AllocKind,
                              _propList);

    if (!ptr)
      throw exception{make_error_code(errc::memory_allocation),
                      "usm_allocator: Allocation failed"};

    return ptr;
  }

  void deallocate(T *ptr, std::size_t size) {
    if (ptr)
      free(ptr, _ctx);
  }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment> &a,
                         const usm_allocator<U, AllocKindU, AlignmentU> &b) {
    return a._dev == b._dev && a._ctx == b._ctx && AllocKindU == AllocKind &&
           AlignmentU == Alignment;
  }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment> &a,
                         const usm_allocator<U, AllocKindU, AlignmentU> &b) {
    return !(a == b);
  }

private:
  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend class usm_allocator;
  context _ctx;
  device _dev;
  property_list _propList;
};
}
} // namespace hipsycl

#endif
