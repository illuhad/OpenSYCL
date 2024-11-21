# AdaptiveCpp performance guide


AdaptiveCpp has been repeatedly shown to deliver very competitive performance compared to other SYCL implementations or proprietary solutions like CUDA. See for example:

* *Sohan Lal, Aksel Alpay, Philip Salzmann, Biagio Cosenza, Nicolai Stawinoga, Peter Thoman, Thomas Fahringer, and Vincent Heuveline. 2020. SYCL-Bench: A Versatile Single-Source Benchmark Suite for Heterogeneous Computing. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 10, 1. DOI:https://doi.org/10.1145/3388333.3388669*
* *Brian Homerding and John Tramm. 2020. Evaluating the Performance of the hipSYCL Toolchain for HPC Kernels on NVIDIA V100 GPUs. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 16, 1–7. DOI:https://doi.org/10.1145/3388333.3388660*
* *Tom Deakin and Simon McIntosh-Smith. 2020. Evaluating the performance of HPC-style SYCL applications. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 12, 1–11. DOI:https://doi.org/10.1145/3388333.3388643*


## LLVM stack

* Building AdaptiveCpp against newer LLVM generally results in better performance for backends that are relying on LLVM.
* Note that when comparing AdaptiveCpp against other LLVM-based compilers, the comparison is usually only really fair if both are based on the same LLVM version.

## Use the right compilation flow: Generic is typically fastest

* Despite its name, **generic compilation does not imply that the compiler compiles to a less target-specific and thus less performant code representation**.
* The `generic` compiler instead relies on runtime JIT compilation & optimization, i.e. it optimizes when it sees the exact target hardware at runtime.
* `generic` also implements many optimizations that are not available in the other compilers.

`generic` always compiles *faster*, usually produces *faster* binaries and creates binaries that are more *portable* as they can directly offload to all supported devices.

The other compilation flows `omp`, `cuda`, `hip` should mainly be used when *interoperability* with vendor programming languages like CUDA or HIP is more important than benefitting from the latest compiler features.

## Different compilers may have different compilation flag defaults

* Some other compilers like `nvcc` or `hipcc` compile with `-O3` by default. AdaptiveCpp, like regular g++ or clang++ compilers, does not do this and instead expects optimization flags to be provided by the user.
* The oneAPI SYCL compiler `icpx` compiles with `-ffast-math -ffp-contract=fast` by default. AdaptiveCpp does not use `-ffast-math` by default. Make sure to align these flags to obtain fair comparisons.
* AdaptiveCpp's clang-based compilation flows (`omp`, `generic`, `hip`, `cuda`) compile with `-ffp-contract=fast` by default. Non-clang based flows do not use that flag. If you use compilation flows that do not rely on clang (e.g. `cuda-nvcxx` or `omp.library-only`), it is your own responsibility to apply comparable compilation flags!
* The oneAPI SYCL compiler by default does not correctly round `sqrt` calls, using approximate builtins instead **even when using `-fno-fast-math`**. AdaptiveCpp (both with `generic` and `hip`/`cuda` targets) by default correctly rounds `sqrt` builtin calls and divisions. This behavior also aligns with the defaults of e.g. `hipcc`. You may use the `-fsycl-fp32-prec-sqrt` to force DPC++ to correctly round `sqrt`.
* If you are unsure, the compilation flags used by clang-based compilers under the hood can be inspected using `<clang invocation> -###`. This also works for AdaptiveCpp's clang-based compilation flows. When in doubt, use this mechanism to align compilation flags between compilers.
* The compiler invocation that `acpp` generates can be printed and its flags inspected with `--acpp-dryrun`.

## SYCL memory management: USM vs buffers

There are three kinds of unified shared memory (USM) in SYCL:

* host USM (`sycl::malloc_host`). This is device-accessible host memory, similarly to CUDA pinned memory. It is primarily useful when the memory is only rarely accessed on GPU and a full data copy might be unnecessary.
* device USM (`sycl::malloc_device`). This is device-resident memory that is unavailable on the host or other devices. It always stays on that device, and is similar to CUDA's `cudaMalloc`. It provides very low overhead. Explicit data transfers mechanisms need to be invoked by the user to migrate data between host and device. Device USM is usually the most efficient memory for usage on device in SYCL.
* shared USM (`sycl::malloc_shared`). This is memory that can automatically migrate between host and device, or potentially other devices, similarly to e.g. CUDA's cudaMallocManaged.

AdaptiveCpp supports all forms of USM universally on all backends and supported hardware.

Additionally, SYCL provides the `sycl::buffer`/`sycl::accessor` model.

**All types of USM have significantly lower host-side runtime overhead than buffers**, and can substantially outperform buffers, especially for short running kernels where submission latencies matter. This is especially true when in-order queues are used. See e.g. this paper for details: https://dl.acm.org/doi/fullHtml/10.1145/3648115.3648120
This is because with USM, the programmer can express dependencies statically, while the SYCL buffer-accessor model must figure out dependencies at runtime. Similarly, USM allows to statically express allocation/deallocation and data transfers, while with buffers non-trivial mechanisms in the SYCL runtime need to automatically manage these operations. This can add overhead.

Furthermore, buffer accessors are not lightweight objects, and can increase register pressure in kernels compared to USM pointers.

Performance of shared USM typically depends on memory access patterns and driver quality. Depending on the operating system and hardware, very good performance is also possible with shared USM (e.g. on NVIDIA GPUs). For some configurations, in particular AMD GPUs without the XNACK feature, performance may be degraded. 
Performance can be improved using the `queue::prefetch` performance hint.
Shared USM is the most productive memory management model that SYCL has, and can be a great solution for e.g. rapid prototyping or porting CPU code.

In summary:
* **When control and maximum performance is needed, use device USM (`sycl::malloc_device`)**
* **When maximum productivity is needed, use shared USM (`sycl::malloc_shared`)**
* **Never use buffers. They do not bring significant advantages compared to USM, but can introduce substantial drawbacks**

## Ahead-of-time vs JIT compilation

The compilation targets `omp`, `hip` and `cuda` perform ahead-of-time compilation. This means they depend strongly on the user to provide correct optimization flags when compiling.
You should thus always compile with e.g. `-O3`, and additionally `-march=native` or comparable for the CPU backend `omp`.

The `generic` target on the other hand relies on JIT compilation at runtime, and mainly optimizes kernels at runtime. Its kernel performance is less sensitive to user-provided optimization flags.
However, `generic` has a slight overhead the first time it launches a kernel since it carries out JIT compilation at that point.
For future application runs, this initial overhead is reduced as it leverages an on-disk persistent kernel cache.

## Generic target

### Adaptivity

The generic compilation target will try to create increasingly more optimized kernels based on information gathered at runtime. This can allow for faster kernels in future application runs. Because of this, to obtain peak performance, an application may have to be run multiple times.

This optimization process is complete when the following warning is no longer printed:
```
[AdaptiveCpp Warning] kernel_cache: This application run has resulted in new binaries being JIT-compiled. This indicates that the runtime optimization process has not yet reached peak performance. You may want to run the application again until this warning no longer appears to achieve optimal performance.
```

The extent of this can be controlled using the environment variable `ACPP_ADAPTIVITY_LEVEL`. A value of 0 disables the feature. The default is 1. Higher levels are expected to result in higher peak performance, but may require more application runs to converge to this performance. The default level of 1 usually guarantees peak performance for the second application run.

At adaptivity level >= 2, AdaptiveCpp will enable additional, aggressive optimizations.
In particular, AdaptiveCpp will attempt to detect invariant kernel arguments, and hardwire those as constants during JIT time. In some cases, this can result in substantial performance increases. It is thus advisable to try setting `ACPP_ADAPTIVITY_LEVEL=2` and running the application a couple of times (typically 3-4 times).

Note: Applications that are highly latency-sensitive may notice a slightly increased kernel launch latency at adaptivity level >= 2 due to the additional analysis steps at runtime.

**For peak performance, you should not disable adaptivity, and run the application until the warning above is no longer printed.**

*Note: Adaptivity levels higher than 2 are currently not implemented.*

### Empty the kernel cache when upgrading the stack

The generic compiler also relies on an on-disk persistent kernel cache to speed up kernel JIT compilation. This cache usually resides in `$HOME/.acpp/apps`.
If you have made any changes to the stack that the AdaptiveCpp runtime is not aware of (e.g. upgrade AdaptiveCpp itself, or other lower-level components of the stack like drivers), you may want to force recompilation of kernels. Otherwise it might still use the old kernels from the cache, and you may thus not benefit e.g. from compiler upgrades.
Clearing the cache can be accomplished by simply clearing the cache directory, e.g. `rm -rf ~/.acpp/apps/*`

*Note: Changes in user application code do not require clearing the kernel cache.*

## CPU backend

* When comparing CPU performance to icpx/DPC++, please note that DPC++ relies on either the Intel CPU OpenCL implementation or oneAPI construction kit to target CPUs. AdaptiveCpp can target CPUs either through OpenMP, or through OpenCL. In the latter case, it can use exactly the same OpenCL implementations that DPC++ uses for CPUs as well. So, if you notice that DPC++ performs better on CPU in some scenario, it might be a good idea to try the Intel OpenCL CPU implementation or the oneAPI construction kit with AdaptiveCpp! Drawing e.g. the conclusion that DPC++ is faster than AdaptiveCpp on CPU but only testing AdaptiveCpp's OpenMP backend is *not* correct reasoning!
* When targeting the Intel OpenCL CPU implementation, you might also want to take into account [Intel's vectorizer tuning knobs](https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-guide-core-xeon/2018/vectorizer-knobs.html).
* For the OpenMP backend, enable OpenMP thread pinning (e.g. `OMP_PROC_BIND=true`). AdaptiveCpp uses asynchronous worker threads for some light-weight tasks such as garbage collection, and these additional threads can interfere with kernel execution if OpenMP threads are not bound to cores.

### With omp.* compilation flow
* When using `OMP_PROC_BIND`, there have been observations that performance suffers substantially, if AdaptiveCpp's OpenMP backend has been compiled against a different OpenMP implementation than the one used by `acpp` under the hood. For example, if `omp.accelerated` is used, `acpp` relies on clang and typically LLVM `libomp`, while the AdaptiveCpp runtime library may have been compiled with gcc and `libgomp`. The easiest way to resolve this is to appropriately use `cmake -DCMAKE_CXX_COMPILER=...` when building AdaptiveCpp to ensure that it is built using the same compiler. **If you observe substantial performance differences between AdaptiveCpp and native OpenMP, chances are your setup is broken.**

### With omp.library-only compilation flow

* Don't use `nd_range` parallel for unless you absolutely have to, as it is difficult to map efficiently to CPUs. 
* If you don't need barriers or local memory, use `parallel_for` with `range` argument.
* If you need local memory or barriers, scoped parallelism or hierarchical parallelism models may perform better on CPU than `parallel_for` kernels using `nd_range` argument and should be preferred. Especially scoped parallelism also works well on GPUs.
* If you *have* to use `nd_range parallel_for` with barriers on CPU, the `omp.accelerated`  or `generic` compilation flow will most likely provide substantially better performance than the `omp.library-only` compilation target. See the [documentation on compilation flows](compilation.md) for details.

## Strong-scaling/latency-bound problems

* Eager submission can be forced by setting the environment variable `ACPP_RT_MAX_CACHED_NODES=0`. By default AdaptiveCpp performs batched submission.
* The alternative instant task submission mode can be used, which can substantially lower task launch latencies. Define the macro `ACPP_ALLOW_INSTANT_SUBMISSION=1` before including `sycl.hpp` to enable it. Instant submission is possible with operations that do not use buffers (USM only), have no dependencies on non-instant tasks, do not use SYCL 2020 reductions and use in-order queues. In the stdpar model, instant submission is active by default.
* SYCL 2020 `in_order` queues bypass certain scheduling layers and may thus display lower submission latency.
* The USM pointer-based memory management model typically has less overheads and lower latency compared to SYCL's traditional buffer-accessor model.
* Consider using the `ACPP_EXT_COARSE_GRAINED_EVENTS` [(extension documentation)](extensions.md) extension if you rarely use events returned from the `queue`. This extension allows the runtime to elide backend event creation.
* Stdpar kernels typically have lower submission latency compared to SYCL kernels.
* If you are using `ACPP_ADAPTIVITY_LEVEL >= 2`, try also with lower adaptivity levels. The aggressive optimizations enabled at `ACPP_ADAPTIVITY_LEVEL >= 2` may come with a slight increase in kernel launch latency.

## Stdpar

* Take note of the environment variables and compiler flags that serve as tuning knobs for stdpar. See e.g. the `ACPP_STDPAR_*` environment variables [here](env_variables.md).
* Drivers for discrete Intel GPUs currently migrate allocations not at page granularity, but at granularity of an entire allocation at a time. This means that the memory pool that AdaptiveCpp uses by default will have severely negative performance impact, since every data access causes the entire memory pool to be migrated. Use the environment variable `ACPP_STDPAR_MEM_POOL_SIZE=0` to disable the memory pool on these devices.
* On hardware that is not discrete Intel GPUs, the stdpar memory pool is an important optimization to reduce costs and overheads of memory allocations. By default, the memory pool size is 40% of the device global memory. If your application needs more memory, you might want to increase the memory pool size.
* AdaptiveCpp by default tries to prefetch allocations that are used in kernels. This is usually beneficial for performance. In latency-bound scenarios however, enqueuing these additional operations may result in additional undesired overheads. You may want to disable memory prefetching using `ACPP_STDPAR_PREFETCH_MODE=never` in these cases.
* In general it may be a good idea to try out the different prefetch modes, as different devices and applications may react differently to different prefetch modes (even devices from the same backend may not behave the same!)
* AdaptiveCpp is the only stdpar implementation that can detect and elide unnecessary synchronization for stdpar kernels, and execute them asynchronously if possible. This is however only possible if it can prove that asynchronous execution is safe and correct. This analysis currently does not work beyond the boundaries of one translation unit. I.e. invoking code where AdaptiveCpp does not see the definition when compiling a TU prevents eliding synchronization of previously submitted stdpar operations. Concentrating kernels and stdpar code in as few as possible translation units may thus be beneficial.
* For more details on performance in the C++ parallelism model specifically, see also [here](stdpar.md).
