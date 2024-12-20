# AdaptiveCpp parallel algorithms library

AdaptiveCpp ships with a library for common parallel primitives. This library is supported on all backends, with all compiler-based compilation flows. The library-only compilation flows `omp.library-only` and `cuda-nvcxx` are currently unsupported.

The main support target is the generic JIT compiler (`--acpp-targets=generic`).

## Example

```c++
#include <sycl/sycl.hpp>
#include <AdaptiveCpp/algorithms/numeric.hpp>

void run_scan(sycl::queue& q, int* device_data_ptr, int* device_output_ptr, 
            std::size_t problem_size) {
  // Setup handling for temporary scratch memory. Note: In production work-loads,
  // the allocation cache should be reused by multiple algorithm invocations for
  // optimal performance.
  acpp::algorithms::util::allocation_cache cache{
    acpp::algorithms::util::allocation_type::device};
  // Create a handle for the current invocation to manage its allocation requests
  acpp::algorithms::util::allocation_group scratch{&cache, q.get_device()};
  // Invoke inclusive_scan
  auto evt = acpp::algorithms::inclusive_scan(q, scratch, device_data_ptr,
    device_data_ptr + problem_size, device_output_ptr, sycl::plus<int>{});
  
  evt.wait();
}

```

## Basic concepts

* All algorithms are exclusively supported for the SYCL 2020 USM memory management model (either `device`, `host` or `shared` allocations). The old SYCL `buffer` model is unsupported.
* All algorithms take a `sycl::queue` to which they submit their operations. Both out-of-order and in-order queues are supported, but we recommend in-order queues for performance and since the library is better tested with in-order queues.
* All algorithms operate asynchronously, i.e. it is the user's resposibility to synchronize appropriately before results are accessed.
* All algorithms take an optional `const std::vector<sycl::event>&` argument that can be used to express dependencies.
* All algorithms return a `sycl::event` which can be used for synchronization. Note: If an algorithm is invoked for a problem size of 0, then for performance reasons it immediately returns a default-constructed `sycl::event` which has a `completed` status. This is the case even if the algorithms has dependencies that are not yet complete!
* Some algorithms require temporary scratch memory. For performance reasons, this scratch memory is cached. The AdaptiveCpp algorithms library exposes control over allocation lifetime and allocation kind for this scratch memory to users (see below).
* The iterators passed into the algorithms need to be valid on the target device.

## Allocation cache for scratch memory


```c++

namespace acpp::algorithms::util {

/// Encodes which kind of allocations the allocation cache manages
enum class allocation_type {
  device, // device USM (sycl::malloc_device())
  shared, // shared USM (sycl::malloc_shared())
  host // host USM (sycl::malloc_host())
};


/// The allocation_cache serves as an allocation pool which can serve the
/// need of algorithms. It releases its memory upon destruction or when purge()
/// is called. It is the user's responsibility to ensure that neither event
/// occurs while an algorithm using the allocation cache is still running!
///
/// This class is thread-safe, although it might be a good idea to check
/// whether thread-local allocation caches might result in better performance.
class allocation_cache {
public:
  /// Construct an allocation_cache for a specified memory type
  allocation_cache(allocation_type alloc_type);

  /// When the allocation_cache is destroyed, all allocations that it manages
  /// are freed. Users must ensure that the lifetime of the object extends until all operations
  /// using it have completed.
  ~allocation_cache();

  /// Explicitly free allocations. Users must ensure that this is not invoked before all
  /// operations using it have completed.
  void purge();
};

/// An allocation_group represents a handle for an algorithm invocation
/// to manage its temporary scratch memory needs.
/// In typical scenarios, you will want to use one allocation_group object
/// per algorithm invocation.
///
/// When the allocation_group is destroyed, the allocations that were requested
/// through it are returned to the parent cache, which might then use them
/// to serve other requests.
/// Therefore, users need to
/// * either ensure that the allocation_group is not destroyed before all algorithms
///   using it have completed
/// * or guarantee that allocations may be safely reassigned to other operations while
///   they are still running, e.g. because all submitted algorithms using the same
///   allocation_cache are ordered such that no race condition on the scratch allocations may
///   occur. (Imagine e.g. if all algorithms sharing one allocation_cache are submitted to a single
///   in-order queue)
///
/// This class is not thread-safe.
class allocation_group {
public:
  /// Construct allocation_group for the given cache and device.
  ///
  /// The user is responsible to ensure that the lifetime of the provided parent cache
  /// exceeds the lifetime of this allocation_group.
  ///
  /// The device will be used to provide the memory allocation context; for
  /// typical practical applications it will be the same device that the
  /// algorithm is submitted to.
  /// If the memory from this device is not accessible to the device to which
  /// the algorithm is submitted, the behavior is undefined.
  allocation_group(allocation_cache *parent_cache, const sycl::device &dev);

  allocation_group() = default;
  allocation_group(const allocation_group&) = delete;
  allocation_group& operator=(const allocation_group&) = delete;

  /// Releases all managed allocations to the parent cache to be reassigned to
  /// other operations.
  ~allocation_group();

  /// Explicitly releases all managed allocations to the parent cache to be reassigned to
  /// other operations.
  ///
  /// It is the user's responsibility to ensure that this function is not called
  /// before all managed allocations can be safely returned to the parent cache.
  void release();

  /// Request new allocation with the specified number of elements of type T.
  ///
  /// If the parent allocation cache has an allocation of sufficient size available,
  /// then it will be returned and made unavailable for other allocation requests.
  /// Otherwise, a new allocation will be created.
  template<class T>
  T* obtain(std::size_t count);
};


}
```

## Algorithms

The following algorithms are currently supported. Their definition aligns with their definition in the C++ STL. Please refer to the C++ reference of your choice for more information on them.

Here, we will only describe AdaptiveCpp-specific behavior.

### Header `<AdaptiveCpp/algorithms/algorithm.hpp>`

```c++
namespace acpp::algorithms {


template <class ForwardIt, class UnaryFunction2>
sycl::event for_each(sycl::queue &q, ForwardIt first, ForwardIt last,
                     UnaryFunction2 f,
                     const std::vector<sycl::event> &deps = {});

template <class ForwardIt, class Size, class UnaryFunction2>
sycl::event for_each_n(sycl::queue &q, ForwardIt first, Size n,
                       UnaryFunction2 f,
                       const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class ForwardIt2, class UnaryOperation>
sycl::event transform(sycl::queue &q, ForwardIt1 first1, ForwardIt1 last1,
                      ForwardIt2 d_first, UnaryOperation unary_op,
                      const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class ForwardIt2, class ForwardIt3,
          class BinaryOperation>
sycl::event transform(sycl::queue &q, ForwardIt1 first1, ForwardIt1 last1,
                      ForwardIt2 first2, ForwardIt3 d_first,
                      BinaryOperation binary_op,
                      const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class ForwardIt2>
sycl::event copy(sycl::queue &q, ForwardIt1 first, ForwardIt1 last,
                 ForwardIt2 d_first, const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate>
sycl::event copy_if(sycl::queue &q, util::allocation_group &scratch_allocations,
                    ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first,
                    UnaryPredicate pred,
                    std::size_t *num_elements_copied = nullptr,
                    const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class Size, class ForwardIt2>
sycl::event copy_n(sycl::queue &q, ForwardIt1 first, Size count,
                   ForwardIt2 result,
                   const std::vector<sycl::event> &deps = {});

template <class ForwardIt, class T>
sycl::event fill(sycl::queue &q, ForwardIt first, ForwardIt last,
                 const T &value, const std::vector<sycl::event> &deps = {});

template<class ForwardIt, class Size, class T >
sycl::event fill_n(sycl::queue& q,
                  ForwardIt first, Size count, const T& value,
                  const std::vector<sycl::event> &deps = {});

template <class ForwardIt, class Generator>
sycl::event generate(sycl::queue &q, ForwardIt first, ForwardIt last,
                     Generator g, const std::vector<sycl::event> &deps = {});

template <class ForwardIt, class Size, class Generator>
sycl::event generate_n(sycl::queue &q, ForwardIt first, Size count, Generator g,
                       const std::vector<sycl::event> &deps = {});

template <class ForwardIt, class T>
sycl::event replace(sycl::queue &q, ForwardIt first, ForwardIt last,
                    const T &old_value, const T &new_value,
                    const std::vector<sycl::event> &deps = {});

template <class ForwardIt, class UnaryPredicate, class T>
sycl::event replace_if(sycl::queue &q, ForwardIt first, ForwardIt last,
                       UnaryPredicate p, const T &new_value,
                       const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate, class T>
sycl::event replace_copy_if(sycl::queue &q, ForwardIt1 first, ForwardIt1 last,
                            ForwardIt2 d_first, UnaryPredicate p,
                            const T &new_value,
                            const std::vector<sycl::event> &deps = {});

template <class ForwardIt1, class ForwardIt2, class T>
sycl::event replace_copy(sycl::queue &q, ForwardIt1 first, ForwardIt1 last,
                         ForwardIt2 d_first, const T &old_value,
                         const T &new_value,
                         const std::vector<sycl::event> &deps = {});

/// The result of the operation will be stored in out.
///
/// out must point to device-accessible memory, and will be set to 0
/// for a negative result, and 1 for a positive result.
template <class ForwardIt, class UnaryPredicate>
sycl::event all_of(sycl::queue &q,
                   ForwardIt first, ForwardIt last, int* out,
                   UnaryPredicate p, const std::vector<sycl::event>& deps = {});

/// The result of the operation will be stored in out.
///
/// out must point to device-accessible memory, and will be set to 0
/// for a negative result, and 1 for a positive result.
template <class ForwardIt, class UnaryPredicate>
sycl::event any_of(sycl::queue &q,
                   ForwardIt first, ForwardIt last, int* out,
                   UnaryPredicate p, const std::vector<sycl::event>& deps = {});

/// The result of the operation will be stored in out.
///
/// out must point to device-accessible memory, and will be set to 0
/// for a negative result, and 1 for a positive result.
template <class ForwardIt, class UnaryPredicate>
sycl::event none_of(sycl::queue &q,
                   ForwardIt first, ForwardIt last, int* out,
                   UnaryPredicate p, const std::vector<sycl::event>& deps = {});

template <class RandomIt, class Compare>
sycl::event sort(sycl::queue &q, RandomIt first, RandomIt last,
                 Compare comp = std::less<>{},
                 const std::vector<sycl::event>& deps = {});

template< class ForwardIt1, class ForwardIt2,
          class ForwardIt3, class Compare >
sycl::event merge(sycl::queue& q,
                  util::allocation_group &scratch_allocations,
                  ForwardIt1 first1, ForwardIt1 last1,
                  ForwardIt2 first2, ForwardIt2 last2,
                  ForwardIt3 d_first, Compare comp = std::less<>{},
                  const std::vector<sycl::event>& deps = {});

}


```

### Header `<AdaptiveCpp/algorithms/numeric.hpp>`

```c++
namespace acpp::algorithms {

/// The result of the reduction will be written to out.
///
/// out must point to memory that is accessible on the target device.
template <class ForwardIt1, class ForwardIt2, class T, class BinaryReductionOp,
          class BinaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T *out,
                 T init, BinaryReductionOp reduce,
                 BinaryTransformOp transform,
                 const std::vector<sycl::event>& deps = {});

/// The result of the reduction will be written to out.
///
/// out must point to memory that is accessible on the target device.
template <class ForwardIt, class T, class BinaryReductionOp,
          class UnaryTransformOp>
sycl::event
transform_reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                 ForwardIt first, ForwardIt last, T* out, T init,
                 BinaryReductionOp reduce, UnaryTransformOp transform,
                 const std::vector<sycl::event>& deps = {});

/// The result of the reduction will be written to out.
///
/// out must point to memory that is accessible on the target device.
template <class ForwardIt1, class ForwardIt2, class T>
sycl::event transform_reduce(sycl::queue &q,
                             util::allocation_group &scratch_allocations,
                             ForwardIt1 first1, ForwardIt1 last1,
                             ForwardIt2 first2, T *out, T init,
                             const std::vector<sycl::event>& deps = {});


/// The result of the reduction will be written to out.
///
/// out must point to memory that is accessible on the target device.
template <class ForwardIt, class T, class BinaryOp>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last, T *out, T init,
                   BinaryOp binary_op,
                   const std::vector<sycl::event>& deps = {});

/// The result of the reduction will be written to out.
///
/// out must point to memory that is accessible on the target device.
template <class ForwardIt, class T>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last, T *out, T init,
                   const std::vector<sycl::event>& deps = {});

/// The result of the reduction will be written to out.
///
/// out must point to memory that is accessible on the target device.
template <class ForwardIt>
sycl::event reduce(sycl::queue &q, util::allocation_group &scratch_allocations,
                   ForwardIt first, ForwardIt last,
                   typename std::iterator_traits<ForwardIt>::value_type *out,
                   const std::vector<sycl::event>& deps = {});


template <class InputIt, class OutputIt, class BinaryOp>
sycl::event
inclusive_scan(sycl::queue &q, util::allocation_group &scratch_allocations,
               InputIt first, InputIt last, OutputIt d_first, BinaryOp op,
               const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt, class BinaryOp, class T>
sycl::event
inclusive_scan(sycl::queue &q, util::allocation_group &scratch_allocations,
               InputIt first, InputIt last, OutputIt d_first, BinaryOp op,
               T init, const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt>
sycl::event inclusive_scan(sycl::queue &q,
                           util::allocation_group &scratch_allocations,
                           InputIt first, InputIt last, OutputIt d_first,
                           const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt, class T, class BinaryOp>
sycl::event
exclusive_scan(sycl::queue &q, util::allocation_group &scratch_allocations,
               InputIt first, InputIt last, OutputIt d_first, T init,
               BinaryOp op, const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt, class T>
sycl::event exclusive_scan(sycl::queue &q,
                           util::allocation_group &scratch_allocations,
                           InputIt first, InputIt last, OutputIt d_first,
                           T init, const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt, class BinaryOp, class UnaryOp>
sycl::event transform_inclusive_scan(
    sycl::queue &q, util::allocation_group &scratch_allocations, InputIt first,
    InputIt last, OutputIt d_first, BinaryOp binary_op, UnaryOp unary_op,
    const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt, class BinaryOp, class UnaryOp, class T>
sycl::event transform_inclusive_scan(
    sycl::queue &q, util::allocation_group &scratch_allocations, InputIt first,
    InputIt last, OutputIt d_first, BinaryOp binary_op, UnaryOp unary_op,
    T init, const std::vector<sycl::event> &deps = {});

template <class InputIt, class OutputIt, class T, class BinaryOp, class UnaryOp>
sycl::event transform_exclusive_scan(
    sycl::queue &q, util::allocation_group &scratch_allocations, InputIt first,
    InputIt last, OutputIt d_first, T init, BinaryOp binary_op,
    UnaryOp unary_op, const std::vector<sycl::event> &deps = {});


}
```