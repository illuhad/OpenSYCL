# Environment variables used by AdaptiveCpp

* `ACPP_DEBUG_LEVEL`: if set, overrides the output verbosity. `0`: none, `1`: error, `2`: warning, `3`: info, `4`: verbose, default is the value of `HIPSYCL_DEBUG_LEVEL` [macro](macros.md).
* `ACPP_VISIBILITY_MASK`: can be used to activate only a subset of backends. Syntax: `backend;backend2;..`. Possible values are `omp` (OpenMP), `cuda`, `hip`, `ocl` (OpenCL) and `ze` (Level Zero). `omp` will always be active as a CPU backend is required. For most backends, device level visibility has to be set via vendor specific variables for now, including `{CUDA,HIP}_VISIBLE_DEVICES` and `ZE_AFFINITY_MASK`. Certain backends, particularly `ocl`, support device level visibility specifications: For example, `omp;ocl:0,4` exposes OpenCL device 0 and 4, `omp;ocl:0.0,3.0` exposes device 0 from platform 0 and device 0 from platform 3. Instead of numbers, strings can also be passed, in which case a device will match if the platform/device name contains the given string. `*` acts as wildcard. Examples: `omp;ocl:Intel.0` (first device from platforms containing "Intel" in the name), `omp;ocl:Graphics.*` (All devices from platforms containing "Graphics" in their name), `omp;ocl:CPU` (All devices containing CPU in their name)
* `ACPP_RT_DAG_REQ_OPTIMIZATION_DEPTH`: maximum depth when descending the DAG requirement tree to look for DAG optimization opportunities, such as eliding unnecessary dependencies.
* `ACPP_RT_MQE_LANE_STATISTICS_MAX_SIZE`: For the `multi_queue_executor`, the maximum size of entries in the lane statistics, i.e. the maximum number of submissions to retain statistical information about. This information is used to estimate execution lane utilization.
* `ACPP_RT_MQE_LANE_STATISTICS_DECAY_TIME_SEC`: The time in seconds (floating point value) after which to forget information about old submissions.
* `ACPP_RT_SCHEDULER`: Set scheduler type. Allowed values: 
    * `direct` is a low-latency direct-submission scheduler. 
    * `unbound` is the default scheduler and supports automatic work distribution across multiple devices. If the `ACPP_EXT_MULTI_DEVICE_QUEUE` extension is used, the scheduler must be `unbound`.
* `ACPP_DEFAULT_SELECTOR_BEHAVIOR`: Set behavior of default selector. Allowed values:
    * `strict` (default): Strictly behave as defined by the SYCL specification
    * `multigpu`: Makes default selector behave like a multigpu selector from the `ACPP_EXT_MULTI_DEVICE_QUEUE` extension
    * `system`: Makes default selector behave like a system selector from the `ACPP_EXT_MULTI_DEVICE_QUEUE` extension
* `ACPP_HCF_DUMP_DIRECTORY`: If set, hipSYCL will dump all embedded HCF data files in this directory. HCF is hipSYCL's container format that is used by all compilation flows that are fully controlled by hipSYCL to store kernel code.
* `ACPP_PERSISTENT_RUNTIME`: If set to 1, hipSYCL will use a persistent runtime that will continue to live even if no SYCL objects are currently in use in the application. This can be helpful if the application consists of multiple distinct phases in which SYCL is used, and multiple launches of the runtime occur.
* `ACPP_RT_MAX_CACHED_NODES`: Maximum number of nodes that the runtime buffers before flushing work.
* `ACPP_SSCP_FAILED_IR_DUMP_DIRECTORY`: If non-empty, hipSYCL will dump the IR of code that fails SSCP JIT into this directory.
* `ACPP_RT_GC_TRIGGER_BATCH_SIZE`: Number of nodes in flight that trigger a garbage collection job to be spawned
* `ACPP_RT_OCL_NO_SHARED_CONTEXT`: If set to `1`, instructs the OpenCL backend to not attempt to construct a shared context across devices within a platform. This can be necessary on OpenCL implementations that do not support this. Note that if shared contexts are unavailable, support for data transfers between devices might be limited as the devices can no longer directly talk to each other.
* `ACPP_RT_OCL_SHOW_ALL_DEVICES`: If set to `1`, instructs the OpenCL backend to expose all found devices, even if those might be incompatible with AdaptiveCpp or unable to execute kernels.
* `ACPP_STDPAR_MEM_POOL_SIZE`: Determines the size of USM memory pool in GB to be used in stdpar allocations. The memory pool can substantially improve performance for applications that rely on frequent memory allocations or frees. If set to 0, the memory pool optimization is disabled. If not set, a default logic is used to determine a suitable size of the memory pool.
* `ACPP_STDPAR_HOST_SAMPLING`: If set to to `1` and the application was not compiled with `--acpp-stdpar-unconditional-offload`, will cause this application run to be carried out on the host. The stdpar runtime will measure the runtime of the execution of host parallel STL calls in-order to automatically determine the offload viability in future runs. If host execution is too slow to run production problem sizes, it is recommended to make multiple application runs with `ACPP_STDPAR_HOST_SAMPLING` with various smaller problem sizes. AdaptiveCpp will then interpolate/extrapolate from those measurements.
* `ACPP_STDPAR_OFFLOAD_SAMPLING`: If set to `1` and the application was not compiled with `--acpp-stdpar-unconditional-offload`, will cause this application to be carried out through the offloading mechanism. The stdpar runtime will measure the performance of offloaded STL algorithms, and make this information available for future application runs which can then benefit from potentially better information to decide whether offloading is viable.
* `ACPP_STDPAR_DATASET_NAME`: If set, is used as an identifier in the filename of the application profile constructed by the stdpar offloading heuristic engine. This can be used to distinguish different application profiles (e.g., if different compiler flags were used, or different hardware was targeted).
* `ACPP_STDPAR_PREFETCH_MODE`: Can be used to specify the desired prefetch mode (see `acpp --help` for details) if the compiler flag `--acpp-stdpar-prefetch-mode` was not set. If `--acpp-stdpar-prefetch-mode` was set, has no effect.
* `ACPP_STDPAR_OHC_MIN_OPS`: stdpar offload heuristic configuration (ohc): If set, offloading decisions will only be reevaluated after at least this many stdpar algorithms have been dispatched. This also configures, how many operations the offload heuristic will attempt to predict when estimating performance.
* `ACPP_STDPAR_OHC_MIN_TIME`: stdpar offload heuristic configuration (ohc): If set, offloading decisions will only be reevaluated after at least this much time in seconds has passed.
* `ACPP_RT_NO_JIT_CACHE_POPULATION`: If set to `1`, prevents the kernel cache from storing SSCP JIT-compiled binaries in the persistent on-disk cache. This can be useful e.g. in an MPI context, where it is sufficient that only one process among many populates the cache.
* `ACPP_ADAPTIVITY_LEVEL`: Controls the optimization level of the adaptivity engine. This is currently only relevant for the generic SSCP target. A higher value implies JIT-compiling more specialized kernels at the expense of more frequent JIT compilations. A value of 0 disables all adaptivity (not recommended). The default is 1; the maximum implemented adaptivity level is 2.
* `ACPP_APPDB_DIR`: By default, AdaptiveCpp stores its application db (which in particular includes the per-app JIT cache) in `$HOME/.acpp`. This environment variable can be used to override the location.
* `ACPP_JITOPT_IADS_RELATIVE_THRESHOLD`: JIT-time optimization *invariant argument detection & specialization* (active if `ACPP_ADAPTIVITY_LEVEL >= 2`): When the same argument has been passed into the kernel for this fraction of all invocations of the kernel, a new kernel will be JIT-compiled with the argument value hard-wired as constant. Not taken into account for the first application run. Default: 0.8.
* `ACPP_JITOPT_IADS_RELATIVE_THRESHOLD_MIN_DATA`: JIT-time optimization *invariant argument detection & specialization* (active if `ACPP_ADAPTIVITY_LEVEL >= 2`): Only consider kernels with at least many invocations for the relative threshold described above. Default: 1024.
* `ACPP_JITOPT_IADS_RELATIVE_EVICTION_THRESHOLD`: JIT-time optimization *invariant argument detection & specialization* (active if `ACPP_ADAPTIVITY_LEVEL >= 2`): If the relative frequency of a kernel argument value falls below this threshold, the statistics entry for the the argument value may be evicted if space for other values is needed.
* `ACPP_ALLOCATION_TRACKING`: If set to 1, allows the AdaptiveCpp runtime to track and register the allocations that it manages. This enables additional JIT-time optimizations. Set to 0 to disable. (Default: 0)

## Environment variables to control dumping IR during JIT compilation

AdaptiveCpp can dump the IR of the code during stage 2 compilation (JIT compilation) at various stages in the processing and optimization pipeline.
This feature only applies to the AdaptiveCpp generic JIT SSCP compiler (`--acpp-targets=generic`).

It is primarily helpful for AdaptiveCpp developers for debugging or expert users who wish to understand how their input code is translated and processed in LLVM IR.

These environment variables take the shape `ACPP_S2_DUMP_IR_<Stage>` for various stages in the optimization process.
* If the variable is set to `1`, the IR will be stored in `<original-sourcefile>.ll`.
* Otherwise, the content is interpreted as a filepath were the IR will be written to.

Within one application run, AdaptiveCpp appends IR dumps to the dump file. When a new application run results in new dumps being generated to the same file, the file will be truncated first.

Available stages for dumping:

* `ACPP_S2_DUMP_IR_INPUT` - dumps the raw, unoptimized generic input LLVM IR
* `ACPP_S2_DUMP_IR_INITIAL_OUTLINING` - After initial kernel outlining
* `ACPP_S2_DUMP_IR_SPECIALIZATION` - After applying specializations to the kernel
* `ACPP_S2_DUMP_IR_REFLECTION` - After processing JIT-time reflection queries
* `ACPP_S2_DUMP_IR_JIT_OPTIMIZATIONS` - After processing optimizations that rely on JIT-time information
* `ACPP_S2_DUMP_IR_BACKEND_FLAVORING` - After applying the "backend flavor", i.e. turning generic LLVM IR into IR that targets a specific backend.
* `ACPP_S2_DUMP_IR_BUILTIN_REFLECTION` - After second run of JIT-time reflection pass; particularly affects reflection use inside AdaptiveCpp builtins.
* `ACPP_S2_DUMP_IR_FULL_OPTIMIZATIONS` - After running the full LLVM optimization pipeline on the code.
* `ACPP_S2_DUMP_IR_FINAL` - Final state of the LLVM IR before handing it off to lowering it to backend-specific formats (e.g. PTX, amdgcn ISA, SPIR-V).
* `ACPP_S2_DUMP_IR_ALL` - Dump all stages.


A dump section for a stage in the dump file will take the following form:
```
;---------------- Begin AdaptiveCpp IR dump --------------
; AdaptiveCpp SSCP S2 IR dump; Compiling kernels: (KERNELS), stage: (STAGENAME)

(LLVM code here)
;----------------- End AdaptiveCpp IR dump ---------------
```
`(STAGENAME)` refers to to one of the stages listed above. `(KERNELS)` is an identifier that describes which kernels AdaptiveCpp is compiling in this IR. It contains the mangled function name of the kernels.

In general, the dump file will contain multiple dump sections if dumping is enabled for multiple stages, or if multiple JIT compilations are triggered (e.g. multiple kernels are launched).

If `ACPP_S2_DUMP_IR_FILTER` filter is non-empty, AdaptiveCpp will only dump IR if the kernel identifier corresponds to the one specified in this variable.
Note that this can still lead to multiple JIT compilation dumps, e.g. if AdaptiveCpp generates multiple specialized kernels based on runtime information for one C++ kernel.


