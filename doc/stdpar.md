# C++ standard parallelism support

AdaptiveCpp supports automatic offloading of C++ standard algorithms.

## Installation & dependencies

C++ standard parallelism offload requires LLVM >= 14. It is automatically enabled when a sufficiently new LLVM is detected. Requires `cmake -DACPP_COMPILER_FEATURE_PROFILE=full` (this is the default setting) at cmake configure time.
C++ standard parallelism offload currently is only supported in conjunction with `libstdc++` >= 11. Other standard C++ standard library versions may or may not work. Support for `libc++` is likely easy to add if there is demand.

## Using accelerated C++ standard parallelism

Offloading of C++ standard parallelism is enabled using `--acpp-stdpar`. This flag does not by itself imply a target or compilation flow, which will have to be provided in addition using the normal `--acpp-targets` argument. C++ standard parallelism is expected to work with any of our clang compiler-based compilation flows, such as `omp.accelerated`, `cuda`, `hip` or the generic SSCP compiler (`--acpp-targets=generic`). It is not currently supported in library-only compilation flows. The focus of testing currently is the generic SSCP compiler.
AdaptiveCpp by default uses some experimental heuristics to determine if a problem is worth offloading. These heuristics are currently very simplistic and might not work well for you. They can be disabled using `--acpp-stdpar-unconditional-offload`.


## Algorithms and policies supported for offloading

Currently, the following execution policies qualify for offloading:
* `par_unseq`
* `par` (experimental; will only be offloaded on hardware that provides independent work item forward progress guarantees such as recent NVIDIA GPUs)

Offloading is implemented for the following STL algorithms:

| Algorithm | Notes |
|------------------|-------------------|
|`for_each`| |
|`for_each_n`| |
|`transform` | both unary and binary operator overloads |
|`copy` | |
|`copy_n` | |
|`copy_if` | |
|`fill` | |
|`fill_n` | |
|`generate` | |
|`generate_n` | |
|`replace` | |
|`replace_if` | |
|`replace_copy` | |
|`replace_copy_if` | |
|`transform_reduce` | all overloads |
|`reduce` | all overloads |
|`any_of` | |
|`all_of` | |
|`none_of` | |
|`merge` | |
|`sort` | may not scale optimally for large problems |
|`inclusive_scan` | |
|`exclusive_scan` | |
|`transform_inclusive_scan` | |
|`transform_exclusive_scan` | |

For all other execution policies or algorithms, the algorithm will compile and execute correctly, however the regular host implementation of the algorithm provided by the C++ standard library implementation will be invoked and no offloading takes place.


## Performance

Performance can generally be expected to be on par with comparable SYCL kernels, although there are some optimizations specific to the C++ standard parallelism model. See the sections on execution and memory model below for details. However, because the implementation of C++ standard parallelism depends heavily on SYCL shared USM (unified shared memory) allocations, the implementation quality of USM at the driver and hardware level can have a great impact on performance, especially for memory-intensive applications.
In particular, on some AMD GPUs USM is known to not perform well due to hardware and driver limitations.
In general, USM relies on memory pages automatically migrating between host and device, depending on where they are accessed. Consequently, patterns where the same memory region is accessed by host and offloaded C++ standard algorithms in alternating fashion should be avoided as much as possible, as this will trigger memory transfers behind the scenes.

## Execution model

### Queues and devices

Each thread in the user application maintains a dedicated thread-local in-order SYCL queue that will be used to dispatch STL algorithms. Thus, concurrent operations can be expressed by launching them from separate threads.
The selected device is currently the device returned from the default selector. Use `ACPP_VISIBILITY_MASK` and/or backend-specific environment variables such as `HIP_VISIBLE_DEVICES` to control which device this is. Because `sycl::event` objects are not needed in the C++ standard parallelism model, queues are set up to rely exclusively on the hipSYCL coarse grained events extension. This means that offloading a C++ standard parallel algorithm can potentially have lower overhead compared to submitting a regular SYCL kernel.

### Synchronous and asynchronous execution

The C++ STL algorithms are all designed around the assumption of being synchronous. This can become a performance issue especially when multiple algorithms are executed in succession, as in principle a `wait()` must be executed after each algorithm is submitted to device.

To address this issue, a dedicated compiler optimization tries to remove `wait()` calls in between successive calls to offloaded algorithms, such that a `wait()` will only be executed for the last algorithm invocation. This is possible without side effects if no instructions (particularly loads and stores) between the algorithm invocations are present.
Currently, the analysis is very simplistic and the compiler gives up the optimization attempt early - therefore, it is recommended for now to make it as easy as possible for the compiler to spot this opportunity by removing any code between calls to C++ algorithms if possible. This also includes code in the call arguments, such as calls to `begin()` and `end()`, which currently should better be moved to before the algorithm invocation. Example:

```c++

auto first = data.begin();
auto last = data.end();
auto dest = dest.begin();
std::for_each(std::execution::par_unseq, first, last, ...);
std::transform(std::execution::par_unseq, first, last, dest, ...);

```

## Memory model

### Automatic migration of heap allocations to USM shared allocations

C++ is unaware of separate devices with their own device memory. In order to retain C++ semantics, when offloading C++ standard algorithms AdaptiveCpp tries to move all memory allocations that the application performs in translation units compiled with `--acpp-stdpar` to SYCL shared USM allocations. To this end, `operator new` and `operator delete` as well as the C-Style functions `malloc`, `aligned_alloc` and `free` are replaced by our own implementations (`calloc` and `realloc` are not yet implemented).
**Note that pointers to host stack memory cannot be used in offloaded C++ algorithms, because we cannot move stack allocations to USM memory! AdaptiveCpp detects whether such pointers are used, and will not offload in this case!**

This replacement is performed using a special compiler transformation. This compiler transformation also enforces that the SYCL headers perform regular allocations instead. This is important because in general the SYCL headers construct complex objects such as `std::vector` or `std::shared_ptr` which then get handed over to the SYCL runtime library. The runtime library however cannot rely on SYCL USM pointers -- in short: The runtime as the code responsible for managing these allocations cannot itself sit on them. Therefore, the compiler performs non-trivial operations to only selectively replace memory allocations.

The backend used to perform USM allocations is the backend managing the executing device as described in the previous section.


## Scope and visibility of replaced functions

Functions for memory allocation are only exchanged for USM variants within translation units compiled with `--acpp-stdpar`. Our USM functions for releasing memory are however overriding the standard functions within the entire application. This is motivated by the expectation that pointers may be shared within the application, and the place where they are released may not be the place where they are created. As our functions for freeing memory can handle both regular and USM allocations, making them more widely available seems like the safer choice.

Note that in C++ due to the one definition rule (ODR) the linker may in certain circumstances pick one definition of a symbol when multiple definitions are available. This can potentially be a problem if a user-defined function is both defined in a translation unit compiled with `--acpp-stdpar` and one without it. In this case, there is no guarantee that the linker will pick the variant that does USM allocations. Be aware that the most vulnerable code for this issue might not only be user code directly, but also header-only library code such as `std::` functions (think of e.g. the allocations performed by `std::vector` of common types) as these functions may be commonly used in multiple translation units.
**We therefore recommend that if you enable `--acpp-stdpar` for one translation unit, you also enable it for the other translation units in your project!**

Such issues are not present for the functions defined in the SYCL headers, because the compiler inserts special ABI tags into their symbol names when compiled with `--acpp-stdpar` to distinguish them from the regular variants, thus preventing such linking issues. Unfortunately, we cannot do the same for client code because we cannot know if other translation or linkage units will attempt to link against the user code, and expect the unaltered symbol names.

### User-controlled USM device pointers

Of course, if you wish to have greater control over memory, USM device pointers from user-controlled USM memory management function calls can also be used, as in any regular SYCL kernel. The buffer-accessor model is not supported; memory stored in `sycl::buffer` objects can only be used when converting it to a USM pointer using our buffer-USM interoperability extension.
Note that you may need to invoke SYCL functions to explicitly copy memory to device and back if you use explicit SYCL device USM allocations.
It is not recommended to use USM shared allocations from direct calls to `sycl::malloc_shared` in C++ standard algorithms. Such allocations will not be tracked by the stdpar runtime, and as such might not benefit from optimizations such as automatic prefetching.

### Systems with system-level USM support

If you are on a system that supports system-level USM, i.e. a system where every CPU pointer returned from regular memory allocations or even stack pointers can directly be used on GPUs (such as on AMD MI300 or Grace-Hopper), the compiler transformation to turn heap allocations to SYCL USM shared allocations is unnecessary. In this case, you may want to request the compiler to assume system-level USM and disable the compiler transformations regarding SYCL shared USM allocations using `--acpp-stdpar-system-usm`.

## Functionality supported in device code

The functionality supported in device code aligns with the kernel restrictions from SYCL. This means that no exceptions, dynamic polymorphism, dynamic memory management, or calls to external shared libraries are allowed. Note that this functionality might already be prohibited in the C++ `par_unseq` model anyway.

The `std::` math functions are supported in device code in an experimental state when using the generic SSCP compilation flow (`--acpp-targets=generic`). This is accomplished using a dedicated compiler pass that maps standard functions to our SSCP builtins.

When using the `par` execution policy, `std::atomic` and `std::atomic_ref` support in device code is available when using the generic SSCP compilation flow (`--acpp-targets=generic`), but experimental.
