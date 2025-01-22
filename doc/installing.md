
# Building and installing AdaptiveCpp


## Operating system support

Operating system support currently strongly focuses on Linux. On Mac, only the CPU backend is expected to work. Windows support with CPU and CUDA backends is experimental, see [Using AdaptiveCpp on Windows](https://github.com/OpenSYCL/OpenSYCL/wiki/Using-Open-SYCL-on-Windows).

## Installation from source (Linux)

### Software dependencies
In order to successfully build and install AdaptiveCpp, the following dependencies must be installed for all backends:

* python 3 (for the `acpp` compiler driver)
* `cmake`
* the Boost C++ libraries (in particular `boost.fiber`, `boost.context` and for the unit tests `boost.test`)
    * it may be helpful to set the `BOOST_ROOT` `cmake` variable to the path to the root directory of Boost you wish to use if `cmake` does not find it automatically
    * **Note for boost 1.78 users:** There seems to be a bug in the build system for boost 1.78, causing the compiled fiber and context libraries not to be copied to the installation directory. You will have to copy these libraries manually to the installation directory. In binary packages from some distribution repositories this issue is fixed. You might be only affected when building boost manually from source.

In addition, the various supported [compilation flows](compilation.md) and programming models have additional requirements:

### A standard installation

For a standard installation that has the most important features enabled, you will additionally need to install an official LLVM release >= 14. Please do not use a development version or vendor-specific fork of LLVM.
This can be very conveniently be achieved e.g. using https://apt.llvm.org [(detailed instructions)](install-llvm.md).

Next, ensure that you have the stacks installed that you want to target (e.g. CUDA, ROCm, OpenCL etc).

AdaptiveCpp will automatically enable all backends that it finds on the system, so in typical scenarios, the following is sufficient:

```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp
cd AdaptiveCpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/your/desired/install/location ..
make install
```
If it does not find some backends or does not pick up the right LLVM, please look at the documentation for the individual components linked below.


### Advanced installation

Advanced users may want to customize their installation more, or use features that are not so commonly used. The following sections describe requirements for individual components in more detail.

#### Compilation flows

| Compilation flow | Target hardware | Short description | Requirements |
|------------------|-------------------|-------------------|-------------------|
| `omp.library-only` | Any CPU | OpenMP CPU backend | Any OpenMP compiler |
| `omp.accelerated` | Any CPU supported by LLVM | OpenMP CPU backend (compiler-accelerated)| LLVM >= 14 |
| `cuda.integrated-multipass` | NVIDIA GPUs | CUDA backend (clang)| CUDA >= 10, LLVM >= 14 |
| `cuda.explicit-multipass` | NVIDIA GPUs | CUDA backend (clang, can be targeted simultaneously with other backends) | CUDA >= 10, LLVM >= 14 |
| `cuda-nvcxx` | NVIDIA GPUs | CUDA backend (nvc++) | Latest NVIDIA HPC SDK |
| `hip.integrated-multipass` | AMD GPUs (supported by ROCm) | HIP backend (clang) | ROCm >= 4.0, LLVM >= 14 |
| `generic` | NVIDIA, AMD, Intel GPUs, OpenCL SPIR-V devices | Generic single-pass compiler | LLVM >= 14. When dispatching kernels to AMD hardware, ROCm >= 5.3 is recommended and LLVM must be <= the ROCm LLVM version. When dispatching to NVIDIA, clang needs nvptx64 backend enabled. AdaptiveCpp runtime backends for the respective target hardware need to be available. |

Note: Building against `libc++` instead of `libstdc++` is only expected to work for the `generic` target. Additionally, AdaptiveCpp must have been built using the same standard library that the user code is linked against.
`libc++` is currently not supported for the C++ standard parallelism offloading model.

#### Models

* SYCL: (No SYCL-specific requirements)
* C++ standard parallelism: See [here](stdpar.md) for dependencies.

Please make sure to read the instructions below for the dependencies that apply to your use case.

#### LLVM (skip if you only want flows without LLVM dependency)

Follow [these](install-llvm.md) instructions.

#### CUDA (skip if you don't need CUDA support)

Follow [these](install-cuda.md) instructions

#### ROCm (skip if you don't need ROCm support)

Follow [these](install-rocm.md) instructions

#### SPIR-V/Level Zero (skip if you don't need SPIR-V/Level Zero support)

Follow [these](install-spirv.md) instructions.

#### SPIR-V/OpenCL (skip if you don't need SPIR-V/OpenCL support)

Follow [these](install-ocl.md) instructions.

#### Building and installing 

Once the software requirements mentioned above are met, clone the repository:
```
$ git clone https://github.com/AdaptiveCpp/AdaptiveCpp
```
Then, create a build directory and compile AdaptiveCpp. As described below, some backends and compilation flows must be configured with specific cmake arguments which should be passed during the cmake step.

```
$ cd <build directory>
$ cmake -DCMAKE_INSTALL_PREFIX=<installation prefix> <more optional options, e.g. to configure the LLVM dependency> <AdaptiveCpp source directory>
$ make install
```

The default installation prefix is `/usr/local`. Change this to your liking.
**Note: AdaptiveCpp needs to be installed to function correctly; don't replace "make install" with just "make"!**

##### CMake options to configure the AdaptiveCpp build

###### General
*  `-DCMAKE_CXX_COMPILER` should be pointed to the C++ compiler to compile AdaptiveCpp with. Note that this also sets the default C++ compiler for the CPU backend when using acpp once AdaptiveCpp is installed. This can however also be modified later using `HIPSYCL_CPU_CXX`.
* `-DACPP_COMPILER_FEATURE_PROFILE` can be used to configure the desired degree of compiler support. Supported values:
    * `full` (default and recommended): Enables all AdaptiveCpp features, requires a compatible LLVM installation as described [here](install-llvm.md). This is recommended for both functionality and performance.
    * `minimal`: Only enables the older interoperability-focused compilation flows for CUDA and HIP (`--acpp-targets=cuda` and `--acpp-targets=hip`). No OpenCL or Level Zero support, no C++ standard parallelism offloading support, no generic JIT compiler (`generic` target), no compiler acceleration for SYCL constructs on CPU device. **Should only be selected in specific circumstances.**
    * `none`: Disables all compiler support and dependencies on LLVM. In addition to `minimal`, also disables the support for `--acpp-targets=cuda` and `--acpp-targets=hip`. In this mode, AdaptiveCpp operates purely as a library for third-party compilers. **Should only be selected in specific circumstances.**

###### generic

* Requires `-DACPP_COMPILER_FEATURE_PROFILE=full`

###### omp.library-only

* `-DCMAKE_CXX_COMPILER` can be used to set the default OpenMP compiler.

###### omp.accelerated

* Requires `-DACPP_COMPILER_FEATURE_PROFILE=full`

###### cuda.*

* See the CUDA [installation instructions](install-cuda.md) instructions (section on clang).

###### cuda-nvcxx

* See the CUDA [installation instructions](install-cuda.md) instructions (section on nvc++).

###### hip.*

* See the ROCm [installation instructions](install-rocm.md) instructions.


#### Building an LLVM toolchain with AdaptiveCpp linked in (experimental, but also for Windows)

Another advanced installation procedure is to build AdaptiveCpp as part of LLVM.
This makes it easy to ship a full AdaptiveCpp installation with all dependencies.
Additionally, it enables systems (such as Windows), where LLVM plugins are supported in a limited manner, to use most of the compiler features AdaptiveCpp has to offer.
When building AdaptiveCpp as part of LLVM, you can choose to link the AdaptiveCpp compiler components into the LLVM tools (`clang`, `opt`, ...).
Therefore, it is no longer necessary to separately build AdaptiveCpp's LLVM plugins.

To get started, select an appropriate LLVM version that you want to use.
AdaptiveCpp only actively supports released LLVM versions.
Typically, the second newest released version is a solid choice.

Then, clone LLVM and AdaptiveCpp and install (set the environment variables on top, before copy-pasting).
You will want to use the `ninja` build tool for the build step, install it, if you don't have it, yet.
For Windows instructions see the [wiki](https://github.com/AdaptiveCpp/AdaptiveCpp/wiki/Using-AdaptiveCpp-on-Windows).
```bash
export LLVM_VERSION=18 # set me!
export LLVM_PARALLEL_LINK_JOBS=8 # set me (when using the default GNU ld, allow around 4GB of RAM per link job)
export ACPP_INSTALL_PREFIX=`pwd`/../../install # set me
export USE_CCACHE=ON # leave set to on, if you have ccache/sccache installed for faster rebuild times
git clone https://github.com/llvm/llvm-project --single-branch -b release/${LLVM_VERSION}.x llvm
cd llvm
git clone https://github.com/AdaptiveCpp/AdaptiveCpp AdaptiveCpp
mkdir -p build && cd build
cmake ../llvm -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_PREFIX} -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DLLVM_ENABLE_PROJECTS="clang;openmp;lld" -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS} -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_CCACHE_BUILD=${USE_CCACHE} -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp -DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=`pwd`/../AdaptiveCpp -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON
ninja install
```

Now, you should have a full LLVM installation with AdaptiveCpp built-in.
You can use it just, as you would normally: using the CMake integration or using the `acpp` tool directly.

##### Caveats
- On Windows, this requires LLVM 18 or above.
- When using the `omp` target plus the `hip` target with this LLVM 17+ is required. Any other target specification should work with versions lower than this.
- It is experimental. So test your usecase well :)

## Installation from source (Mac)

On Mac, only the CPU backends are supported. The required steps are analogous to Linux.

## Installation from source (Windows)

For experimental building on Windows (CPU and CUDA backends) see the corresponding [wiki](https://github.com/AdaptiveCpp/AdaptiveCpp/wiki/Using-AdaptiveCpp-on-Windows).
The `omp.accelerated` and `generic` compilation flows are only supported when building AdaptiveCpp as part of LLVM (see above).

