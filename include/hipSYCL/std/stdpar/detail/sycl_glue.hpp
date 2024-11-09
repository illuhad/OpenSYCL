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
#ifndef HIPSYCL_PSTL_SYCL_GLUE_HPP
#define HIPSYCL_PSTL_SYCL_GLUE_HPP




#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <unistd.h>

#include <hipSYCL/algorithms/util/allocation_cache.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <hipSYCL/sycl/device.hpp>
#include <hipSYCL/sycl/context.hpp>
#include <hipSYCL/sycl/usm.hpp>
#include <hipSYCL/sycl/usm_query.hpp>
// Fetch builtin declarations to aid SSCP StdBuiltinRemapperPass for
// std:: math function support inside kernels.
#include <hipSYCL/sycl/libkernel/builtin_interface.hpp>


#include "allocation_map.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "offload_heuristic_db.hpp"
#include "hipSYCL/runtime/settings.hpp"
#include "hipSYCL/sycl/info/device.hpp"

extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void*);

namespace hipsycl::stdpar::detail {


inline uint64_t get_time_now() {
  uint64_t now =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
  return now;
}

inline sycl::queue construct_default_queue() {
  return sycl::queue{hipsycl::sycl::property_list{
        hipsycl::sycl::property::queue::in_order{},
        hipsycl::sycl::property::queue::AdaptiveCpp_coarse_grained_events{}}};
}

class stdpar_tls_runtime {
private:
  stdpar_tls_runtime()
      : _queue{construct_default_queue()},
        _device_scratch_cache{algorithms::util::allocation_type::device},
        _shared_scratch_cache{algorithms::util::allocation_type::shared},
        _host_scratch_cache{algorithms::util::allocation_type::host} {
          auto dev = _queue.get_device().AdaptiveCpp_device_id();
          auto* be = _queue.get_context().AdaptiveCpp_runtime()->backends().get(
              dev.get_backend());
          if (be->get_hardware_manager()
                  ->get_device(dev.get_id())
                  ->has(rt::device_support_aspect::
                            work_item_independent_forward_progress))
            _has_independent_work_item_forward_progress = true;
        }

  ~stdpar_tls_runtime() {
    _device_scratch_cache.purge();
    _shared_scratch_cache.purge();
    _host_scratch_cache.purge();
  }

  sycl::queue _queue;
  algorithms::util::allocation_cache _device_scratch_cache;
  algorithms::util::allocation_cache _shared_scratch_cache;
  algorithms::util::allocation_cache _host_scratch_cache;
  int _outstanding_offloaded_operations = 0;
  bool _has_independent_work_item_forward_progress = false;

  offload_heuristic_db _offload_db;
  std::vector<uint64_t, libc_allocator<uint64_t>> _instrumented_ops_in_batch;
  std::vector<std::size_t, libc_allocator<std::size_t>> _instrumented_op_problem_sizes_in_batch;
  uint64_t _batch_start_timestamp = 0;

  static std::atomic<std::size_t>& offloading_batch_counter() {
    static std::atomic<std::size_t> batch_counter = 0;
    return batch_counter;
  }

  void reset_num_outstanding_operations() {
    _outstanding_offloaded_operations = 0;
  }
public:
  const offload_heuristic_db& get_offload_db() const {
    return _offload_db;
  }

  offload_heuristic_db& get_offload_db() {
    return _offload_db;
  }
  
  sycl::queue& get_queue() {
    return _queue;
  }

  bool device_has_work_item_independent_forward_progress() const {
    return _has_independent_work_item_forward_progress;
  }

  int get_num_outstanding_operations() const {
    return _outstanding_offloaded_operations;
  }

  void increment_num_outstanding_operations() {
    ++_outstanding_offloaded_operations;
  }

  void instrument_offloaded_operation(uint64_t op_hash, std::size_t problem_size) {
    if(_outstanding_offloaded_operations == 0)
      _batch_start_timestamp = get_time_now();
    _instrumented_ops_in_batch.push_back(op_hash);
    _instrumented_op_problem_sizes_in_batch.push_back(problem_size);
  }

  std::size_t get_current_offloading_batch_id() const {
    return offloading_batch_counter().load(std::memory_order_acquire);
  }

  void finalize_offloading_batch() noexcept {
#ifndef __ACPP_STDPAR_UNCONDITIONAL_OFFLOAD__
    uint64_t batch_end = get_time_now();
    double mean_time = static_cast<double>(batch_end - _batch_start_timestamp) /
                       _instrumented_ops_in_batch.size();
    
    assert(_instrumented_ops_in_batch.size() ==
           _instrumented_op_problem_sizes_in_batch.size());
    
    for(std::size_t i = 0; i < _instrumented_ops_in_batch.size(); ++i) {
      std::size_t problem_size = _instrumented_op_problem_sizes_in_batch[i];
      _offload_db.update_entry(_instrumented_ops_in_batch[i], problem_size,
                               offload_heuristic_db::offload_device_id,
                               mean_time);
    }
    _instrumented_ops_in_batch.clear();
    _instrumented_op_problem_sizes_in_batch.clear();
#endif
    reset_num_outstanding_operations();
    ++offloading_batch_counter();
  }

  template<algorithms::util::allocation_type AT>
  algorithms::util::allocation_cache& get_scratch_cache() {
    if constexpr(AT == algorithms::util::allocation_type::device)
      return _device_scratch_cache;
    else if constexpr(AT == algorithms::util::allocation_type::shared)
      return _shared_scratch_cache;
    else
      return _host_scratch_cache;
  }

  template<algorithms::util::allocation_type AT>
  algorithms::util::allocation_group make_scratch_group() {
    algorithms::util::allocation_cache& cache = get_scratch_cache<AT>();
    return algorithms::util::allocation_group{
        &cache, get_queue().get_device().AdaptiveCpp_device_id()};
  }

  static stdpar_tls_runtime& get() {
    static thread_local stdpar_tls_runtime rt;
    return rt;
  }
};

class single_device_dispatch {
public:
  static hipsycl::sycl::queue& get_queue() {
    return stdpar_tls_runtime::get().get_queue();
  }
};

}

#if defined(__clang__) && defined(ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST) &&    \
    !defined(__ACPP_STDPAR_ASSUME_SYSTEM_USM__)

namespace hipsycl::stdpar::detail {

struct usm_context {
  usm_context() {
    _ctx = single_device_dispatch::get_queue().get_context();
    _is_alive = true;
  }

  ~usm_context() {
    _is_alive = false;
  }

  static bool is_alive() {
    return _is_alive;
  }

  hipsycl::sycl::context& get() {
    return _ctx;
  }
private:
  hipsycl::sycl::context _ctx;
  inline static std::atomic<bool> _is_alive;
};

}


namespace hipsycl::stdpar {

class memory_pool {
private:
  uint64_t ceil_division(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
  }

  uint64_t next_multiple_of(uint64_t a, uint64_t b) {
    return ceil_division(a, b) * b;
  }
public:
  memory_pool(std::size_t size)
      : _pool_size{size}, _pool{nullptr},
        _free_space_map{size > 0 ? size : 1024},
        _page_size{static_cast<std::size_t>(sysconf(_SC_PAGESIZE))} {
    init();
  }

  void* claim(std::size_t size) {
    if(_pool_size == 0)
      return nullptr;

    if(size < _page_size)
      size = _page_size;

    uint64_t address = 0;
    if(_free_space_map.claim(size, address)) {
      
      void* ptr = static_cast<void*>((char*)_base_address + address);
      assert(is_from_pool(ptr));
      assert(is_from_pool((char*)ptr+size));
      assert((uint64_t)ptr % _page_size == 0);

      // Inform the runtime that there is a new user allocation
      // by invoking the runtime hook. We need to do this manually
      // because memory pool directly uses raw backend allocation commands.
      rt::application::event_handler_layer().on_new_allocation(
          ptr, size,
          rt::allocation_info{_dev,
                              rt::allocation_info::allocation_type::shared});

      return ptr;
    }

    return nullptr;
  }

  void release(void* ptr, std::size_t size) {
    if(_pool && is_from_pool(ptr)) {
      uint64_t address = reinterpret_cast<uint64_t>(ptr)-reinterpret_cast<uint64_t>(_base_address);
      _free_space_map.release(address, size);

      rt::application::event_handler_layer().on_deallocation(ptr);
    }
  }

  ~memory_pool() {
    // Memory pool might be destroyed after runtime shutdown, so rely on OS
    // to clean up for now
  }

  std::size_t get_size() const {
    return _pool_size;
  }

  bool is_from_pool(void* ptr) const {
    if(!_pool)
      return false;

    void* pool_end = (char*)_base_address + _pool_size;
    return ptr >= _base_address && ptr < pool_end;
  }
private:

  void* raw_malloc_shared(std::size_t bytes, sycl::queue& q) {
    auto *allocator = sycl::detail::select_usm_allocator(q.get_context(),
                                                         q.get_device());
    return allocator->raw_allocate_usm(bytes);
  }

  void init() {
    HIPSYCL_DEBUG_INFO << "[stdpar] Building a memory pool of size "
                       << static_cast<double>(_pool_size) / (1024 * 1024 * 1024)
                       << " GB" << std::endl;
    auto& q = detail::single_device_dispatch::get_queue();
    _dev = q.get_device().AdaptiveCpp_device_id();

    // We need to call raw_allocate_usm so that we can inform the runtime's allocation
    // tracking mechanism of actual user allocations, not just of the memory pool as a 
    // whole.
    // Make sure to allocate an additional page so that we can fix alignment if needed
    _pool = raw_malloc_shared(_pool_size + _page_size, q);

    uint64_t aligned_pool_base = next_multiple_of((uint64_t)_pool, _page_size);
    _base_address = (void*)aligned_pool_base;
    assert(aligned_pool_base % _page_size == 0);
  }


  std::size_t _pool_size;
  void* _pool;
  void* _base_address;
  free_space_map _free_space_map;
  std::size_t _page_size;
  rt::device_id _dev;
};

class unified_shared_memory {
  
  struct allocation_map_payload {
    // Note: This gets updated when logic, such as the prefetch
    // heuristic, touches this value - so it may not be up to date
    // if there is no prefetch!
    int64_t most_recent_offload_batch;
  };

  using allocation_map_t = allocation_map<allocation_map_payload>;
public:

  static void pop_disabled() {
    thread_local_storage::get().disabled_stack--;
  }

  static void push_disabled() {
    thread_local_storage::get().disabled_stack++;
  }
  
  static void* malloc(std::size_t n, std::size_t alignment = 0) {
    // Seems some apps rely on n==0 still returning a valid pointer
    if(n == 0)
      n = 1;
    
    if(thread_local_storage::get().disabled_stack == 0) {
      
      void* ptr = nullptr;
      push_disabled();
      if (alignment != 0) {
        ptr = sycl::aligned_alloc_shared(alignment, n,
                                          detail::single_device_dispatch::get_queue());
      } else {
        auto& usm_manager = get();
        // We need to lazily construct the memory pool because our free()
        // will be called early on during program startup. Querying devices
        // to determine an appropriate pool size might cause recursive initialization.
        auto* mem_pool = usm_manager.get_memory_pool();
        if(!mem_pool){
          usm_manager.init_mem_pool();
          mem_pool = usm_manager.get_memory_pool();
        }

        if(n < mem_pool->get_size() / 2) {
          ptr = mem_pool->claim(n);
        }
        // ptr will still be nullptr if pool was not used, or pool allocation
        // failed.
        if(!ptr) {
          ptr = sycl::malloc_shared(n, detail::single_device_dispatch::get_queue());
        }
      }
      get()._is_initialized = true;
      pop_disabled();

      if(ptr) {
        allocation_map_t::value_type v;
        v.allocation_size = n;
        v.most_recent_offload_batch = -1;
        get()._allocation_map.insert(reinterpret_cast<uint64_t>(ptr), v);
      }

      return ptr;

    } else {
      if(alignment != 0) {
        void *ptr = 0;
        posix_memalign(&ptr, alignment, n);
        return ptr;
      } else
        return __libc_malloc(n);
    }
  }

  static void free(void* ptr) {
    if(!get()._is_initialized) {
      __libc_free(ptr);
      return;
    }

    if (thread_local_storage::get().disabled_stack == 0) {
          
      push_disabled();
      static detail::usm_context ctx;
      pop_disabled();

      if(!detail::usm_context::is_alive())
        // If the runtime has already shut down, we cannot really
        // reliably free things anymore :( Currently, we just ignore
        // the request.
        return;

      push_disabled();
      auto* map_entry = get()._allocation_map.get_entry_of_root_address(
              reinterpret_cast<uint64_t>(ptr));
      if (!map_entry) {
        __libc_free(ptr);
      } else {
        uint64_t allocation_size = map_entry->allocation_size;

        get()._allocation_map.erase(reinterpret_cast<uint64_t>(ptr));
        memory_pool* mem_pool = get().get_memory_pool();
        if(mem_pool && mem_pool->is_from_pool(ptr)) {
          mem_pool->release(ptr, allocation_size);
        } else {
          sycl::free(ptr, ctx.get());
        }
      }
      pop_disabled();
    } else {
      __libc_free(ptr);
    }
  }

  struct allocation_lookup_result {
    void* root_address;
    allocation_map_t::value_type* info;
  };

  static bool allocation_lookup(void* ptr, allocation_lookup_result& result) {
    uint64_t root_address;
    auto* ret = get()._allocation_map.get_entry(reinterpret_cast<uint64_t>(ptr), root_address);
    if(!ret)
      return false;

    result.root_address = reinterpret_cast<void*>(root_address);
    result.info = ret;
    return true;
  }
private:
  memory_pool* get_memory_pool() const {
    return __atomic_load_n(&_memory_pool, __ATOMIC_ACQUIRE);
  }
  
  void init_mem_pool() {
    std::lock_guard<std::mutex> pool_construction_lock{_pool_construction_lock};
    if (!get_memory_pool()) {
      memory_pool* mem_pool = (memory_pool *)__libc_malloc(sizeof(memory_pool));
      std::size_t pool_size = get_mem_pool_size_gb() * 1024 * 1024 * 1024;

      new (mem_pool) memory_pool{pool_size};
      __atomic_store_n(&_memory_pool,
                       mem_pool,
                       __ATOMIC_RELEASE);
    }
  }

  double get_mem_pool_size_gb() {
    auto dev = detail::single_device_dispatch::get_queue().get_device();

    double user_defined_mem_pool_size = 0.0;
    if (rt::try_get_environment_variable("stdpar_mem_pool_size",
                                         user_defined_mem_pool_size))
      return user_defined_mem_pool_size;
    
    // If we have system allocations, mem pool is not really needed.
    // Note: This also excludes OpenMP backend from the following calculations,
    // which might be important since it return 2^64 for both queries.
    if(dev.has(sycl::aspect::usm_system_allocations))
      return 0.0;

    std::size_t max_alloc_size = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    std::size_t global_mem_size = dev.get_info<sycl::info::device::global_mem_size>();

    return 0.4 * static_cast<double>((max_alloc_size < global_mem_size)
                                   ? max_alloc_size
                                   : global_mem_size) /
           (1024 * 1024 * 1024);
  }

  unified_shared_memory()
      : _is_initialized{false}, _memory_pool{nullptr} {}
  
  ~unified_shared_memory() {
    if(_memory_pool) {
      _memory_pool->~memory_pool();
      __libc_free(_memory_pool);
    }
  }

  static unified_shared_memory& get() {
    static unified_shared_memory usm_state;
    return usm_state;
  }

  std::atomic<bool> _is_initialized;
  allocation_map_t _allocation_map;
  memory_pool* _memory_pool;
  std::mutex _pool_construction_lock;

  class thread_local_storage {
  public:
    static thread_local_storage& get() {
      static thread_local thread_local_storage s;
      return s;
    }

#ifdef HIPSYCL_STDPAR_MEMORY_MANAGEMENT_DEFAULT_DISABLED
    int disabled_stack = 1;
#else
    int disabled_stack = 0;
#endif
  private:
    thread_local_storage(){}
  };
};

}


#define HIPSYCL_STDPAR_ALLOC [[clang::annotate("hipsycl_stdpar_alloc")]] __attribute__((noinline))
#define HIPSYCL_STDPAR_FREE [[clang::annotate("hipsycl_stdpar_free")]]



HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n);
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n, std::align_val_t a) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
  if(!ptr)
    throw std::bad_alloc{};
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n, const std::nothrow_t&) noexcept  {
  return hipsycl::stdpar::unified_shared_memory::malloc(n);
}

HIPSYCL_STDPAR_ALLOC
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n);
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n, std::align_val_t a) {
  auto* ptr = hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
  if(!ptr) {
    throw std::bad_alloc{};
  }
  return ptr;
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n);
}

HIPSYCL_STDPAR_ALLOC
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return hipsycl::stdpar::unified_shared_memory::malloc(n, static_cast<std::size_t>(a));
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::align_val_t ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::align_val_t ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::size_t ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::size_t ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::size_t,
                        std::align_val_t ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::size_t,
                        std::align_val_t ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, const std::nothrow_t& ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, const std::nothrow_t& ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete  ( void* ptr, std::align_val_t,
                        const std::nothrow_t& ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

HIPSYCL_STDPAR_FREE
void operator delete[]( void* ptr, std::align_val_t,
                        const std::nothrow_t& ) noexcept {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

/* Both libc++ and libstdc++ define std::malloc as ::malloc and similarly
 * for std::calloc, std::aligned_alloc, and std::free, so it is enough to 
 * implement the global functions here. */
HIPSYCL_STDPAR_ALLOC
void* malloc(std::size_t size) {
  return hipsycl::stdpar::unified_shared_memory::malloc(size);
}

HIPSYCL_STDPAR_ALLOC
void* aligned_alloc(std::size_t alignment, std::size_t size) {
  return hipsycl::stdpar::unified_shared_memory::malloc(size, alignment);
}

HIPSYCL_STDPAR_FREE
void free(void* ptr) {
  hipsycl::stdpar::unified_shared_memory::free(ptr);
}

#endif

#endif
