# Using AdaptiveCpp in projects

It is recommended that the CMake integration be used for larger projects (see the section on the CMake integration below). Alternatively, `acpp` can be used directly as a compiler.

## Using `acpp`

`acpp` can be invoked like a regular compiler (e.g. `acpp -O3 -o test test.cpp`). It supports multiple compilation flows. A typical installation (i.e. when AdaptiveCpp was built against LLVM >= 14 and the generic SSCP compiler was not explicitly disabled) uses the `generic` compilation flow by default. This compilation flow usually compiles the quickest, produces the fastest binaries, and its generated binaries can run on all supported devices. **Unless you have very specific needs, you should probably use the default `generic` compiler.**

Advanced users or users with more specific needs may want to specify compilation flows explicitly. This is achieved with the `--acpp-targets="compilation-flow1:target1,target2,...;compilation-flow2:..."` command line argument, the `ACPP_TARGETS` environment variable or the `ACPP_TARGETS` CMake variable.

**Other compilation flows like `omp`, `cuda`, and `hip` are typically mostly attractive for backend interoperability use cases, not when performance is the primary concern.**

## AdaptiveCpp targets specification

Both `acpp` and the CMake integration can optionally be provided with an AdaptiveCpp targets specification. This specification defines which compilation flows AdaptiveCpp should enable and which devices from a compilation flow AdaptiveCpp should target. In general, it has the form:

```
"flow1:target1,target2,...;flow2:...;..."
```
and can be passed either as an `acpp` command line argument (`--acpp-targets=...`), environment variable (`ACPP_TARGETS=...`) or CMake argument (`-DACPP_TARGETS=...`) depending on whether `acpp` or `cmake` is used.

"Compilation flow" refers to one of the available compilation flows defined in the [compilation documentation](compilation.md).


### Requirements for specifying targets of individual compilation flows

Whether a compilation flow needs to be followed by a target list or not varies between the available flows and is described below.

For the following compilation flows, targets cannot be specified:

* `omp.*`
* `generic`

For the following compilation flows, targets can optionally be specified:

* `cuda-nvcxx` - Targets take the format of `ccXY` where `XY` stands for the compute capability of the device.

For the following compilation flows, targets must be specified:

* `cuda.*` - The target format is defined by `clang` and takes the format of `sm_XY`. For example:
    * `sm_52`: NVIDIA Maxwell GPUs (e.g. GeForce GTX 980, TITAN X)
    * `sm_61`: NVIDIA Pascal GPUs (e.g. GeForce GTX 1080, TITAN Xp)
    * `sm_70`: NVIDIA Volta GPUs  (e.g. Tesla V100, TITAN V)
* `hip.*` - The target format is defined by `clang` and takes the format of `gfxXYZ`. For example:
    * `gfx900`: AMD Vega 10 GPUs (e.g. Radeon Vega 56, Vega 64)
    * `gfx906`: AMD Vega 20 GPUs (e.g. Radeon VII, Instinct MI50)
    * `gfx908`: AMD CDNA GPUs (e.g. Instinct MI100)

### Abbreviations

For some compilation flows, abbreviations exist that will be resolved by AdaptiveCpp to one of the available compilation flows:

* `omp` will be translated
    * into `omp.accelerated`
        * if AdaptiveCpp has been built with support for accelerated CPU and the host compiler is the `clang` that AdaptiveCpp has been built with or
        * if `--acpp-use-accelerated-cpu` is set. If the accelerated CPU compilation flow is not available (e.g. AdaptiveCpp has been compiled without support for it), compilation will abort with an error.
    * into `omp.library-only` otherwise.
* `cuda` will be translated
    * into `cuda.explicit-multipass`
        * if another integrated multipass has been requested, or another backend that would conflict with `cuda.integrated-multipass`. AdaptiveCpp will emit a warning in this case, since switching to explicit multipass can change interoperability guarantees (see the [compilation documentation](compilation.md)).
        * if `--acpp-explicit-multipass` is set explicitly.
    * into `cuda.integrated-multipass` otherwise.
* `hip` will be translated into `hip.integrated-multipass`.

Of course, the desired flows can also always be specified explicitly.

### Examples

* `"generic"` - creates a binary that can run on all backends. This also typically creates the fastest binaries.
* `"omp.library-only;cuda.explicit-multipass:sm_61;sm_70"` - compiles for the CPU backend and Pascal- and Volta-era GPUs.
* `"omp;cuda:sm_70;hip:gfx906"` - compiles for the CPU backend (library or accelerated), NVIDIA Volta-era GPUs via explicit multipass and AMD Vega 20 GPUs.
* `"omp.accelerated;cuda:sm_70"` - compiles for the CPU backend (compiler-accelerated) and NVIDIA Volta-era GPUs.
* `"omp;cuda-nvcxx"` - compiles for the CPU backend and NVIDIA GPUs using `nvc++`.

### Offloading C++ standard parallelism

See [here](stdpar.md) for details on how to offload C++ standard STL algorithms using AdaptiveCpp.

## All the flags: `acpp --help`

The full output obtained when running `acpp --help` is provided below. Note that the options can also be set via environment variables or the corresponding CMake options. Default values can be set in the `/acpp/install/path/etc/AdaptiveCpp/*.json` files.
```
acpp [AdaptiveCpp compilation driver], Copyright (C) 2018-2024 Aksel Alpay and the AdaptiveCpp project
  AdaptiveCpp version: 24.06.0+git.8cf7a902.20241001.branch.develop
  Installation root: /install/path
  Plugin LLVM version: <version>, can accelerate CPU: <bool>
  Available runtime backends:
     librt-backend-<name>.so
     librt-backend-<name>.so
     librt-backend-<name>.so
     librt-backend-<name>.so
     librt-backend-<name>.so
Usage: acpp <options>

Options are:
--acpp-platform=<value>
  [can also be set with environment variable: ACPP_PLATFORM=<value>]
  [default value provided by field 'default-platform' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  (deprecated) The platform that AdaptiveCpp should target. Valid values:
    * cuda: Target NVIDIA CUDA GPUs
    * rocm: Target AMD GPUs running on the ROCm platform
    * cpu: Target only CPUs

--acpp-clang=<value>
  [can also be set with environment variable: ACPP_CLANG=<value>]
  [default value provided by field 'default-clang' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  The path to the clang executable that should be used for compilation
    (Note: *must* be compatible with the clang version that the
     AdaptiveCpp clang plugin was compiled against!)

--acpp-nvcxx=<value>
  [can also be set with environment variable: ACPP_NVCXX=<value>]
  [default value provided by field 'default-nvcxx' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  The path to the nvc++ executable that should be used for compilation
    with the cuda-nvcxx backend.

--acpp-cuda-path=<value>
  [can also be set with environment variable: ACPP_CUDA_PATH=<value>]
  [default value provided by field 'default-cuda-path' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  The path to the CUDA toolkit installation directory

--acpp-rocm-path=<value>
  [can also be set with environment variable: ACPP_ROCM_PATH=<value>]
  [default value provided by field 'default-rocm-path' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  The path to the ROCm installation directory

--acpp-gpu-arch=<value>
  [can also be set with environment variable: ACPP_GPU_ARCH=<value>]
  [default value provided by field 'default-gpu-arch' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  (deprecated) The GPU architecture that should be targeted when compiling for GPUs.
    For CUDA, the architecture has the form sm_XX, e.g. sm_60 for Pascal.
    For ROCm, the architecture has the form gfxYYY, e.g. gfx900 for Vega 10, gfx906 for Vega 20.

--acpp-cpu-cxx=<value>
  [can also be set with environment variable: ACPP_CPU_CXX=<value>]
  [default value provided by field 'default-cpu-cxx' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  The compiler that should be used when targeting only CPUs.

--acpp-clang-include-path=<value>
  [can also be set with environment variable: ACPP_CLANG_INCLUDE_PATH=<value>]
  [default value provided by field 'default-clang-include-path' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  The path to clang's internal include headers. Typically of the form $PREFIX/include/clang/<version>/include. Only required by ROCm.

--acpp-sequential-link-line=<value>
  [can also be set with environment variable: ACPP_SEQUENTIAL_LINK_LINE=<value>]
  [default value provided by field 'default-sequential-link-line' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the linker for the sequential backend

--acpp-sequential-cxx-flags=<value>
  [can also be set with environment variable: ACPP_SEQUENTIAL_CXX_FLAGS=<value>]
  [default value provided by field 'default-sequential-cxx-flags' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the sequential backend

--acpp-omp-link-line=<value>
  [can also be set with environment variable: ACPP_OMP_LINK_LINE=<value>]
  [default value provided by field 'default-omp-link-line' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the linker for the OpenMP backend.

--acpp-omp-cxx-flags=<value>
  [can also be set with environment variable: ACPP_OMP_CXX_FLAGS=<value>]
  [default value provided by field 'default-omp-cxx-flags' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the OpenMP backend

--acpp-rocm-link-line=<value>
  [can also be set with environment variable: ACPP_ROCM_LINK_LINE=<value>]
  [default value provided by field 'default-rocm-link-line' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the linker for the ROCm backend.

--acpp-rocm-cxx-flags=<value>
  [can also be set with environment variable: ACPP_ROCM_CXX_FLAGS=<value>]
  [default value provided by field 'default-rocm-cxx-flags' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the ROCm backend

--acpp-cuda-link-line=<value>
  [can also be set with environment variable: ACPP_CUDA_LINK_LINE=<value>]
  [default value provided by field 'default-cuda-link-line' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the linker for the CUDA backend.

--acpp-cuda-cxx-flags=<value>
  [can also be set with environment variable: ACPP_CUDA_CXX_FLAGS=<value>]
  [default value provided by field 'default-cuda-cxx-flags' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the CUDA backend

--acpp-config-file-dir=<value>
  [can also be set with environment variable: ACPP_CONFIG_FILE_DIR=<value>]
  [default value provided by field 'default-config-file-dir' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  Select an alternative path for the config files containing the default AdaptiveCpp settings.
    It is normally not necessary for the user to change this setting.

--acpp-targets=<value>
  [can also be set with environment variable: ACPP_TARGETS=<value>]
  [default value provided by field 'default-targets' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  Specify backends and targets to compile for. Example: --acpp-targets='omp;hip:gfx900,gfx906'
    Available backends:
      * omp - OpenMP CPU backend
               Backend Flavors:
               - omp.library-only: Works with any OpenMP enabled CPU compiler.
                                   Uses Boost.Fiber for nd_range parallel_for support.
               - omp.accelerated: Uses clang as host compiler to enable compiler support
                                  for nd_range parallel_for (see --acpp-use-accelerated-cpu).
      * cuda - CUDA backend
               Requires specification of targets of the form sm_XY,
               e.g. sm_70 for Volta, sm_60 for Pascal
               Backend Flavors:
               - cuda.explicit-multipass: CUDA backend in explicit multipass mode
                                          (see --acpp-explicit-multipass)
               - cuda.integrated-multipass: Force CUDA backend to operate in integrated
                                           multipass mode.
      * cuda-nvcxx - CUDA backend with nvc++. Target specification is optional;
               if given requires the format ccXY.
      * hip  - HIP backend
               Requires specification of targets of the form gfxXYZ,
               e.g. gfx906 for Vega 20, gfx900 for Vega 10
               Backend Flavors:
               - hip.explicit-multipass: HIP backend in explicit multipass mode
                                         (see --acpp-explicit-multipass)
               - hip.integrated-multipass: Force HIP backend to operate in integrated
                                           multipass mode.
      * generic - use generic LLVM SSCP compilation flow, and JIT at runtime to target device

--acpp-stdpar-prefetch-mode=<value>
  [can also be set with environment variable: ACPP_STDPAR_PREFETCH_MODE=<value>]
  [default value provided by field 'default-stdpar-prefetch-mode' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  AdaptiveCpp supports issuing automatic USM prefetch operations for allocations used inside offloaded C++ PSTL
    algorithms. This flags determines the strategy for submitting such prefetches.
    Supported values are:
      * always      - Prefetches every allocation used by every stdpar kernel
      * never       - Disables prefetching
      * after-sync  - Prefetch all allocations used by the first kernel submitted after each synchronization point.
                      (Prefetches running on non-idling queues can be expensive!)
      * first       - Prefetch allocations only the very first time they are used in a kernel
      * auto        - Let AdaptiveCpp decide (default)

--acpp-use-accelerated-cpu
  [can also be set by setting environment variable ACPP_USE_ACCELERATED_CPU to any value other than false|off|0 ]
  [default value provided by field 'default-use-accelerated-cpu' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  If set, Clang is used for host compilation and explicit compiler support
  is enabled for accelerating the nd-range parallel_for on CPU.
  Uses continuation-based synchronization to execute all work-items
  of a work-group in a single thread, eliminating scheduling overhead
  and enabling enhanced vectorization opportunities compared to the fiber variant.

--acpp-dryrun
  [can also be set by setting environment variable ACPP_DRYRUN to any value other than false|off|0 ]
  [default value provided by field 'default-is-dryrun' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  If set, only shows compilation commands that would be executed,
  but does not actually execute it.

--acpp-explicit-multipass
  [can also be set by setting environment variable ACPP_EXPLICIT_MULTIPASS to any value other than false|off|0 ]
  [default value provided by field 'default-is-explicit-multipass' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  If set, executes device passes as separate compiler invocation and lets AdaptiveCpp control embedding device
  images into the host binary. This allows targeting multiple backends simultaneously that might otherwise be
  incompatible. In this mode, source code level interoperability may not be supported in the host pass.
  For example, you cannot use the CUDA kernel launch syntax[i.e. kernel <<< ... >>> (...)] in this mode.

--acpp-save-temps
  [can also be set by setting environment variable ACPP_SAVE_TEMPS to any value other than false|off|0 ]
  [default value provided by field 'default-save-temps' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  If set, do not delete temporary files created during compilation.

--acpp-stdpar
  [can also be set by setting environment variable ACPP_STDPAR to any value other than false|off|0 ]
  [default value provided by field 'default-is-stdpar' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  If set, enables SYCL offloading of C++ standard parallel algorithms.

--acpp-stdpar-system-usm
  [can also be set by setting environment variable ACPP_STDPAR_SYSTEM_USM to any value other than false|off|0 ]
  [default value provided by field 'default-is-stdpar-system-usm' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  If set, assume availability of system-level unified shared memory where every pointer from regular
  malloc() is accessible on GPU. This disables automatic hijacking of memory allocations at the compiler
  level by AdaptiveCpp.

--acpp-stdpar-unconditional-offload
  [can also be set by setting environment variable ACPP_STDPAR_UNCONDITIONAL_OFFLOAD to any value other than false|off|0 ]
  [default value provided by field 'default-is-stdpar-unconditional-offload' in JSON files from directories: ['/install/path/etc/AdaptiveCpp'].]
  [current value: NOT SET]
  Normally, heuristics are employed to determine whether algorithms should be offloaded.
  This particularly affects small problem sizes. If this flag is set, supported parallel STL
  algorithms will be offloaded unconditionally.

--acpp-version
  Print AdaptiveCpp version and configuration

--help
  Print this help message


Any other options will be forwarded to the compiler.

Note: Command line arguments take precedence over environment variables.
```

## Using the CMake integration

Setting up a project using the AdaptiveCpp CMake integration is fairly straightforward.
The main points are adding `find_package(AdaptiveCpp REQUIRED)` and, after defining the targets to build, adding `add_sycl_to_target(TARGET <target_name>)` to have the compilation handled by the AdaptiveCpp toolchain (see the [example CMake project](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/examples/CMakeLists.txt)).

A typical configure command might look like this: `cmake .. -DAdaptiveCpp_DIR=/acpp/install/dir/lib/cmake/AdaptiveCpp -DACPP_TARGETS="<targets>"`.
`ACPP_TARGETS` has to be set either as an environment variable or through the command line for the `find_package` call to succeed. See the documentation of this flag above.

If the accelerated CPU flow has been built, `-DACPP_USE_ACCELERATED_CPU=ON/OFF` can be used to override whether `omp` should refer to the `omp.library-only` or `omp.accelerated` compilation flow.
