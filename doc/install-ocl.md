# AdaptiveCpp installation instructions for SPIR-V/OpenCL

You will need an OpenCL implementation, and the OpenCL icd loader. The OpenCL library can be specified using `cmake -DOpenCL_LIBRARY=/path/to/libOpenCL.so`.

In order to generate correct code, AdaptiveCpp needs to apply a patch to the Khronos llvm-spirv translator.
You *must* have the `patch` command installed and available when running the AdaptiveCpp `cmake` configuration. If you have run `cmake` without the `patch` command available, please *clean your build directory* before trying again.

The OpenCL backend can be enabled using `cmake -DWITH_OPENCL_BACKEND=ON` when building AdaptiveCpp.
In order to run code successfully on an OpenCL device, it must support SPIR-V ingestion and the Intel USM (unified shared memory) extension. In a degraded mode, devices supporting OpenCL fine-grained system SVM (shared virtual memory) may work as well.

