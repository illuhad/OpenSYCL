# Macros used by AdaptiveCpp

# Read-only macros

## General macros
* `__ACPP__`, `__ADAPTIVECPP__` - defined if compiling with AdaptiveCpp

## Macros to specialize code paths based on backend

* `__acpp_if_target_host(code)` - `code` will only be compiled for the host backend.
* `__acpp_if_target_device(code)` - `code` will only be compiled for device backends.
* `__acpp_if_target_cuda(code)` - `code` will only be compiled for the CUDA backend.
* `__acpp_if_target_hip(code)` - `code` will only be compiled for the HIP backend.
* `__acpp_if_target_hiplike(code)` - `code` will only be compiled for the CUDA and HIP backend.

## Information about current compiler

### `ACPP_LIBKERNEL_COMPILER_SUPPORTS_<backend>`

* `ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA` - Set to 1 if the compiler currently compiling the code supports CUDA language extensions. 0 otherwise.
* `ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP` - Set to 1 if the compiler currently compiling the code supports HIP language extensions. 0 otherwise.
* `ACPP_LIBKERNEL_COMPILER_SUPPORTS_HOST` - Always set to 1, since every compiler supports the host language, which is just C++.
* `ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP` - Set to 1 if the generic SSCP compiler is enabled.

## Information about compilation passes

### `ACPP_LIBKERNEL_IS_DEVICE_PASS_<backend>`
Note: Some compiler drivers that AdaptiveCpp supports can compile for multiple backends in a single pass. Therefore, the following macros should not be seen as mutually exclusive in general. Currently, this affects the `cuda-nvcxx` driver which can target CUDA and host in a single pass.

* `ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST` - Set to 1 if the current compilation pass targets host. 0 otherwise. 
* `ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA` - Set to 1 if the current compilation pass targets CUDA. 0 otherwise. 
* `ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP` - Set to 1 if the current compilation pass targets host. 0 otherwise. 
* `ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP` - Set to 1 if the current compilation pass targets the generic SSCP compiler.

### Properties of current compilation pass

* `ACPP_LIBKERNEL_IS_DEVICE_PASS` - Set to 1 if the current compilation pass targets at least one device backend. 0 otherwise.
* `ACPP_LIBKERNEL_IS_EXCLUSIVE_PASS(backend)` - returns 1 if the current compilation pass targets the provided backend (`CUDA|HIP|HOST`) and no other backend.
* `ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS` - Set to 1 if the current compilation pass compiles for both host and device in a single, unified compilation pass.
* `SYCL_DEVICE_ONLY` - defined if the current compilation pass targets a device backend and `ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS` is 0. **Note**: `SYCL_DEVICE_ONLY` is not defined for `cuda-nvcxx` where host and device are compiled in a single pass. This is therefore in general not suitable to implement specialized code paths for host and device in a portable way
* `__ACPP_CLANG__` - defined by `acpp` when compiling with the clang plugin

## Information about targeted backends

* `__ACPP_ENABLE_HIP_TARGET__` - defined during host and device passes if HIP is targeted with the old SMCP compiler
* `__ACPP_ENABLE_CUDA_TARGET__` - defined during host and device passes if CUDA is targeted with the old SMCP compiler
* `__ACPP_ENABLE_OMPHOST_TARGET__` - defined if OpenMP is targeted
* `__ACPP_ENABLE_LLVM_SSCP_TARGET__` - defined the modern SSCP generic JIT compiler is targeted

## Extension feature test macros

* `ACPP_EXT_<NAME>` - defined if the AdaptiveCpp extension `<NAME>` is available.

## Deprecated macros

* (deprecated) `HIPSYCL_PLATFORM_CUDA` - defined when CUDA language extensions are available
* (deprecated) `HIPSYCL_PLATFORM_ROCM`, `HIPSYCL_PLATFORM_HIP` - defined when HIP language extensions are available
* (deprecated) `HIPSYCL_PLATFORM_CPU` - defined if compiling for the host


## Mainly for AdaptiveCpp developers
* `ACPP_UNIVERSAL_TARGET` - expands to `__host__ __device__`. Use for functions that should be available everywhere.
* `ACPP_KERNEL_TARGET` - currently expands to `__host__ __device__`. Use for functions that should be available in kernels.

# Configuration macros

* `HIPSYCL_DEBUG_LEVEL` - sets the output verbosity. `0`: none, `1`: error, `2`: warning, `3`: info, `4`: verbose, default is warning for Release and info for Debug builds.
* `ACPP_STRICT_ACCESSOR_DEDUCTION` - define when building your SYCL implementation to enforce strict SYCL 2020 accessor type deduction rules. While this might be required for the correct compilation of certain SYCL code, it also disables parts of the AdaptiveCpp accessor variants performance optimization extension. As such, it can have a negative performance impact for code bound by register pressure.
* `ACPP_ALLOW_INSTANT_SUBMISSION` - define to `1` before including `sycl.hpp` to allow submission of USM operations to in-order queues via the low-latency instant submission mechanism. Set to `0` to prevent the runtime from utilizing the instant submission mechanism. If C++ standard parallelism offloading is enabled, instant submissions are always allowed.
* `ACPP_FORCE_INSTANT_SUBMISSION` - define to `1` before including `sycl.hpp` to imply `ACPP_ALLOW_INSTANT_SUBMISSION=1` and throw an exception when instant submission is not possible.
* `ACPP_NO_SHORT_NAMESPACE` - if defined, disables exposing AdaptiveCpp functionality in the `acpp` namespace.