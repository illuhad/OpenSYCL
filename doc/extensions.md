# SYCL extensions in AdaptiveCpp

AdaptiveCpp implements several extensions that are not defined by the specification.

## Supported extensions

### `ACPP_EXT_RESTRICT_PTR`

Provides a wrapper type that hints to the compiler that a pointer kernel argument does not alias other pointer arguments.


Example:

```c++

sycl::queue q;
float* data = ...
sycl::AdaptiveCpp_restrict_ptr<float> restrict_data = data;

q.parallel_for(range, [=](auto idx){
  // Converts implicitly to the underlying pointer type - float* in this
  // example.
  restrict_data[idx] *= 1.5f;
});

```

### `ACPP_EXT_DYNAMIC_FUNCTIONS`

This extension allows users to provide functions used in kernels with definitions selected at runtime. We call such functions *dynamic functions*, since their definition will be determined at runtime using the JIT compiler. Once a kernel using dynamic functions has been JIT-compiled, there are no runtime overheads as dynamic functions are hardwired at JIT-time.

This can be used to assemble custom kernels at runtime, or to obtain kernel-fusion-like semantics with a high degree of user control.

**This functionality relies on JIT compilation to provide correct semantics. It is thus only available with `--acpp-targets=generic`.** For other compilation flows, code using this functionality will not compile.

The extension works by replacing all function calls in the kernel to a target function with function calls to a replacement function. The original function may be just a declaration, or a function with an existing definition.

The following example demonstrates how this feature could be used in kernel-fusion-like style to decide at runtime that the kernel should consist of both calls to `myfunction1` followed by `myfunction2`.
```c++
// SYCL_EXTERNAL ensures that these functions are emitted to device code,
// even though they are not referenced by the kernel at compile time.
// SYCL_EXTERNAL may be optional in future versions of this extension.
SYCL_EXTERNAL void myfunction1(sycl::item<1> idx) {
  // code
}

SYCL_EXTERNAL void myfunction2(sycl::item<1> idx) {
  // code
}

void execute_operations(sycl::item<1> idx);

int main() {
  sycl::queue q;

  // The dynamic_function_config object stores the JIT-time function mapping information.
  sycl::jit::dynamic_function_config dyn_function_config;
  // Requests calls to execute_operations to be replaced at JIT time
  // with {myfunction1(idx); myfunction2(idx);}
  dyn_function_config.define_as_call_sequence(&execute_operations, {&myfunction1, &myfunction2});
  q.parallel_for(sycl::range{1024}, dyn_function_config.apply([=](sycl::item<1> idx){
    execute_operations(idx);
  }));

  q.wait();
}
```


The AdaptiveCpp runtime maintains a kernel cache that automatically distinguishes the same kernel invoked with different dynamic function configuration. JIT compilation is only triggered when a new configuration is requested that is not yet present in the cache.

**Important notes**
* `dynamic_function_config::apply()` is a very light-weight operation, but constructing a new `dynamic_function_config` object may have some overhead due to initializing the required data structures. It is therefore recommended to reuse a preexisting `dynamic_function_config` object when the same kernel is submitted multiple times with the same configuration.
* Only a single `dynamic_function_config` object may be applied at a given kernel launch.
* It is the user's responsibility to ensure that the `dynamic_function_config` object is kept alive at least until all kernels using it have completed.
* `dynamic_function_config` is not thread-safe; if one object is shared across multiple threads, it is the user's responsibility to ensure appropriate synchronization.
* With this extension, the user can exchange kernel code at runtime. This means that in general, the compiler cannot know at compile time anymore which parts of the code need to be part of device code. Therefore, functions  providing the definitions have to be marked as `SYCL_EXTERNAL` to ensure that they are emitted to device code. This can be omitted if the function is invoked from the kernel already at compile time.
* It is possible to provide a "default definition" for dynamic functions by not just declaring them, but also providing a definition (e.g. in the example above, provide a definition for `execute_operations`). However, in this case, we recommend that the function is marked with `__attribute__((noinline))`. Otherwise, in some cases the compiler might decide to already inline the function early on during the optimization process -- and once, inlined, the JIT compiler no loner sees the function and therefore can no longer find function calls to replace. The `noinline` attribute will have no performance implications once the replacement function definition has been put in place by the JIT compiler. Additionally, if the default function does not actually use the function arguments, the frontend might not actually emit function calls to the dynamic function. It is thus a good idea to use `sycl::jit::arguments_are_used()` to assert that these arguments might e.g. be used by a dynamic function replacement function.

With a default function definition, the example above might look like so:
```c++
SYCL_EXTERNAL void myfunction1(int* data, sycl::item<1> idx) {
  // code
}

SYCL_EXTERNAL void myfunction2(int* data, sycl::item<1> idx) {
  // code
}

__attribute__((noinline))
void execute_operations(int* data, sycl::item<1> idx) {
  // This prevents the compiler from removing calls to execute_operations if it
  // sees that the function cannot actually have any side-effects.
  sycl::jit::arguments_are_used(data, idx);
}

int main() {
  sycl::queue q;
  int* data = ...;

  // The dynamic_function_config object stores the JIT-time function mapping information.
  sycl::jit::dynamic_function_config dyn_function_config;
  // Requests calls to execute_operations to be replaced at JIT time
  // with {myfunction1(idx); myfunction2(idx);}
  // If this is removed, the regular function definition of execute_operations
  // will be executed instead.
  dyn_function_config.define_as_call_sequence(&execute_operations, {&myfunction1, &myfunction2});
  q.parallel_for(sycl::range{1024}, dyn_function_config.apply([=](sycl::item<1> idx){
    execute_operations(data, idx);
  }));

  q.wait();
}
```


#### API Reference

A more detailed API reference follows:

```c++
namespace sycl::jit {

// This function can be used in dynamic functions with a definition
// to prevent the compiler from performing early optimizations if it finds
// that the function does not actually use the arguments because it cannot know
// that the definition may be replaced at runtime.
template<class T, typename... Args>
void arguments_are_used(Args&&... args);

// Represents a function id. Objects of this class can be obtained from
// dynamic_function or dynamice_function_definition. dynamic_function_id objects
// can be passed to dynamice_function_config to control the dynamic function mapping.
class dynamic_function_id {
public:
  dynamic_function_id() = default;
  explicit dynamic_function_id(__unspecified_handle_type__);

  const __unspecified_handle_type__ get_handle() const;
};

// Represents a dynamic function, where the definition might be replaced at runtime.
template<class Ret, typename... Args>
class dynamic_function {
public:
  // Construct object. IMPORTANT: The function pointer it is initialized with
  // must directly point to the target function in the source code. This is
  // because the compiler needs to understand at compile-time which functions
  // are dynamic functions. When a variable however is passed in, this
  // can no longer be guaranteed. Example:
  //
  // Allowed: dynamic_function df{&myfunc};
  // Not allowed: auto* myfuncptr = &myfunc; dynamic_function{myfuncptr};
  dynamic_function(Ret (*func)(Args...));

  // Obtain dynamic_function_id object.
  dynamic_function_id id() const;
};


// Represents a dynamic function definition, i.e. a function whose definition might replace
// the definition of a dynamic_function at runtime.
template<class Ret, typename... Args>
class dynamic_function_definition {
public:

  // Construct object. The same restrictions apply as with the dynamic_function constructor
  // regarding the function pointer argument. See above for details.
  dynamic_function_definition(Ret (*func)(Args...));

  // Obtain dynamic_function_id object.
  dynamic_function_id id() const;
};

// Represents the dynamic function configuration that may be applied to a kernel.
// Per kernel launch, only a single dynamic_function_config may be applied.
class dynamic_function_config {
public:

  // Set the definition of `func` to be provided by `definition`.
  // IMPORTANT: The function pointers passed as arguments
  // must directly point to the target function in the source code. This is
  // because the compiler needs to understand at compile-time which functions
  // are dynamic functions. When a variable however is passed in, this
  // can no longer be guaranteed.
  //
  // This function can be seen as shorthand for
  // `define(dynamic_function{func}, dynamic_function_definition{definition})`
  template<class Ret, typename... Args>
  void define(Ret (*func)(Args...), Ret(*definition)(Args...));

  // Set the definition of `func` to be provided by `definition`.
  //
  // This is a type-safer, but semantically equivalent shorthand for
  // define(df.id(), definition.id()).
  template <class Ret, typename... Args>
  void define(dynamic_function<Ret, Args...> df,
              dynamic_function_definition<Ret, Args...> definition) {
    define(df.id(), definition.id());
  }

  // Set the definition of `func` to be provided by `definition`.
  // In most cases, the type-safe other overloads should be used instead of this one.
  // However, this function can be useful when type-erasure of the functions is explicitly
  // desired; e.g. when in a larger framework many function ids need to be stored
  // in a central location.
  void define(dynamic_function_id function, dynamic_function_id definition)

  // Set the definition of `func` to be provided by a sequence of calls to the functions
  // provided in `definitions`. Note that `define_as_call_sequence` is only supported
  // for functions of void return type.
  //
  // IMPORTANT: The function pointers passed as arguments
  // must directly point to the target function in the source code. This is
  // because the compiler needs to understand at compile-time which functions
  // are dynamic functions. When a variable however is passed in, this
  // can no longer be guaranteed.
  template <typename... Args>
  void define_as_call_sequence(void (*func)(Args...),
                          const std::vector<void (*)(Args...)> &definitions);

  // Set the definition of `func` to be provided by a sequence of calls to the functions
  // provided in `definitions`. Note that `define_as_call_sequence` is only supported
  // for functions of void return type.
  template <typename... Args>
  void define_as_call_sequence(
      dynamic_function<void, Args...> call,
      const std::vector<dynamic_function_definition<void, Args...>>
          &definitions)

  // Set the definition of `func` to be provided by a sequence of calls to the functions
  // provided in `definitions`. Note that `define_as_call_sequence` is only supported
  // for functions of void return type.
  //
  // In most cases, the type-safe other overloads should be used instead of this one.
  // However, this function can be useful when type-erasure of the functions is explicitly
  // desired; e.g. when in a larger framework many function ids need to be stored
  // in a central location.
  void
  define_as_call_sequence(dynamic_function_id func,
                          const std::vector<dynamic_function_id> &definitions) 

  // Returns a kernel object that has this configuration applied. The resulting object
  // can then be passed e.g. to parallel_for().
  template<class Kernel>
  auto apply(Kernel k);
};

}
```

### `ACPP_EXT_SPECIALIZED`

This extension adds a mechanism to hint to the SSCP JIT compiler that a kernel specialization should be generated. That is, when `sycl::specialized<T>` is passed as a kernel argument, the compiler will generate a kernel with the value of the object stored in the `specialized` wrapper hardcoded as a constant. This addresses the same problem as SYCL 2020 specialization constants, however it provides two major benefits:

* It is much, much easier to use for users due to a far more intuitive API
* It is a true zero-cost abstraction that does not add any overhead when the code is not compiled with a JIT compiler. This is in contrast to SYCL 2020 specialization constants which can actively *hurt* performance when they need to be emulated (e.g. if the compiler uses ahead-of-time compilation).

As a consequence, unlike SYCL 2020 specialization constants, the `sycl::specialized` mechanism can be used as an optimization hint that has no drawbacks when the compiler cannot capitalize on it.

Example:

```c++

sycl::queue q;

sycl::specialized<float> scaling_factor = // some runtime value
float* data = ...

q.parallel_for(range, [=](auto idx){
  // The JIT compiler will treat the value of scaling_factor as a constant at JIT-time.
  // E.g, if scaling_factor is 1 at runtime, the compiler may generate an empty kernel.
  data *= scaling_factor;
});

```

`sycl::specialized` currently only affects the code generation of the SSCP JIT compiler (`--acpp-targets=generic`), and only if `ACPP_ADAPTIVITY_LEVEL` is set to any value larger than 0 (the default is 1).

### `ACPP_EXT_SCOPED_PARALLELISM_V2`
This extension provides the scoped parallelism kernel invocation and programming model. This extension does not need to be enabled explicitly and is always available.
See [here](scoped-parallelism.md) for more details. **Scoped parallelism is the recommended way in AdaptiveCpp to write programs that are performance portable between CPU and GPU backends.**

### `ACPP_EXT_ENQUEUE_CUSTOM_OPERATION`

This extension allows to enqueue custom device operations for efficient interoperability with backends. This extension does not need to be enabled explicitly and is always available.
See [here](enqueue-custom-operation.md) for more details.

### `ACPP_EXT_BUFFER_USM_INTEROP`

AdaptiveCpp supports interoperability between `sycl::buffer` and USM pointers. See [here](buffer-usm-interop.md) for details.

### `ACPP_EXT_EXPLICIT_BUFFER_POLICIES`

An extension that allows to explicitly set view/non-view semantics for buffers as well as enable some behaviors that cannot be expressed in regular SYCL such as buffers that do not block in the destructor. See [here](explicit-buffer-policies.md) for details.

### `ACPP_EXT_MULTI_DEVICE_QUEUE`

Allows constructing a queue that automatically distributes work across multiple devices, or even the entire system. See [here](multi-device-queue.md) for details.

### `ACPP_EXT_ACCESSOR_VARIANTS` and `ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION`

AdaptiveCpp supports various flavors of accessors that encode the purpose and feature set of the accessor (e.g. placeholder, ranged, unranged) in the accessor type. Based on this information, the size of the accessor is optimized by eliding unneeded information at compile time. This can be beneficial for performance in kernels bound by register pressure.
If `ACPP_EXT_ACCESSOR_VARIANT_DEDUCTION` is enabled, the SYCL 2020 CTAD deduction guides automatically construct optimized accessor types.
See [here](accessor-variants.md) for more details.

### `ACPP_EXT_UPDATE_DEVICE`

An extension that adds `handler::update()` for device accessors in analogy to `update_host()`. While `update_host()` makes sure that the host allocation of the buffer is updated, `update()` updates the allocation on the device to which the operation is submitted. This can be used
* To preallocate memory if the buffer is uninitialized;
* To separate potential data transfers from kernel execution, e.g. for benchmarking;
* To control buffer data state when using buffer-USM interoperability(`ACPP_EXT_BUFFER_USM_INTEROP`);
* To inform the runtime earlier of expected data usage in order to optimize data transfers or overlap of compute and data transfers

#### API Reference

```c++
namespace sycl {
class handler {
public:
  template <typename T, int dim, access::mode mode, access::target tgt,
          accessor_variant variant>
  void update(accessor<T, dim, mode, tgt, variant> acc);
};
}
```

### `ACPP_EXT_QUEUE_WAIT_LIST`

Adds a `queue::get_wait_list()` method that returns a vector of `sycl::event` in analogy to `event::get_wait_list()`, such that waiting for all returned events guarantees that all operations submitted to the queue have completed. This can be used to express asynchronous barrier-like semantics when passing the returned vector into handler::depends_on().
If the queue is an in-order queue, the returned vector will contain at most one event.

Note that `queue::get_wait_list()` might not return an event for all submitted operations, e.g. completed operations or operations that are dependencies of others in the the dependency graphs may be optimized away in the returned set of events.

#### API Reference

```c++
namespace sycl {
class queue {
public:
  std::vector<event> get_wait_list() const;
};
}
```

### `ACPP_EXT_COARSE_GRAINED_EVENTS`

This extension allows to hint to AdaptiveCpp that events associated with command groups can be more coarse-grained and are allowed to synchronize with potentially more operations.
This can allow AdaptiveCpp to trade less synchronization performance for lighter-weight events, and hence lower kernel launch latency. The main benefit are situations where the returned event from `submit` is not of particular interest, e.g. in in-order queues when no user synchronization with those events are expected.

For example, a coarse grained event for a backend based on in-order queues (e.g. CUDA or HIP) might just synchronize with the entire HIP or CUDA stream - thereby completely eliding the need to create a new backend event.

Coarse-grained events support the same functionality as regular events.

Coarse-grained events can be requested in two ways: 
1. By passing a property to `queue` which instructs the `queue` to construct coarse-grained events for all operations that it processes, and 
2. by passing in a property to an individual command group (see `ACPP_EXT_CG_PROPERTY_*`). In this case, coarse-grained events can be enabled selectively only for some command groups submitted to a queue.

#### API Reference

```c++

namespace sycl::property::queue {
class AdaptiveCpp_coarse_grained_events {};
}

namespace sycl::property::command_group {
class AdaptiveCpp_coarse_grained_events {};
}

```

### `ACPP_EXT_CG_PROPERTY_*`: Command group properties

AdaptiveCpp supports attaching special command group properties to individual command groups. This is done by passing a property list to the queue's `submit` member function:

```cpp
template <typename T>
event queue::submit(const property_list& prop_list, T cgf)
```

A list of supported command group properties follows:

#### `ACPP_EXT_CG_PROPERTY_PREFER_GROUP_SIZE`

##### API reference

```cpp
namespace sycl::property::command_group {

template<int Dim>
struct AdaptiveCpp_prefer_group_size {
  AdaptiveCpp_prefer_group_size(range<Dim> group_size);
};

}
```

##### Description

If this property is added to a command group property list, it instructs the backend to prefer a particular work group size for kernels for models where a SYCL implementation has the freedom to decide on a work group size.

In the current implementation, this property only affects the selected local size for basic parallel for on HIP and CUDA backends.

*Note:* The property only affects kernel launches of the same dimension. If you want to set the group size for 2D kernels, you need to attach a `AdaptiveCpp_prefer_group_size<2>` property.

#### `ACPP_EXT_CG_PROPERTY_RETARGET`

##### API reference

```cpp
namespace sycl::property::command_group {

template<int Dim>
struct AdaptiveCpp_retarget {
  AdaptiveCpp_retarget(const device& dev);
};

}

namespace sycl::property::queue {
struct AdaptiveCpp_retargetable {};
}
```

##### Description

If this property is added to a command group property list, it instructs the AdaptiveCpp runtime to execute the submitted operation on a different device than the one the queue was bound to. This can be useful as a more convenient mechanism to dispatch to multiple devices compared to creating multiple queues.

Using this property does *not* introduce additional overheads compared to using multiple queues. In particular, it does *not* silently lead to the creation of additional backend execution resources such as CUDA streams.

In order to understand this, it is important to realize that because of the design of the AdaptiveCpp runtime, a queue is decoupled from backend objects. Instead, the AdaptiveCpp runtime internally manages a pool of backend execution resources such as CUDA streams, and automatically distributes work across those resources.
In this design, a queue is nothing more than an interface to AdaptiveCpp runtime functionality. This allows us to efficiently retarget operations submitted to a queue arbitrarily.

Compared to using multiple queues bound to different devices, using a single queue and submitting with the `AdaptiveCpp_retarget` property comes with some minor semantic differences:

* A single `queue::wait()` call guarantees that all operations submitted to the queue, no matter to which device they were retargeted, have completed. With multiple queues on the other hand, multiple `wait()` calls are necessary which can add some overhead.
* If the queue is an in-order queue, the in-order property is *preserved even if the operations are retargeted to run on different devices*. This can be a highly convenient way to formulate in-order USM algorithms that require processing steps on different devices.

The `AdaptiveCpp_retarget` property can only be used with queues that have been constructed with the `AdaptiveCpp_retargetable` property.


#### `ACPP_EXT_CG_PROPERTY_PREFER_EXECUTION_LANE`

##### API reference

```c++
namespace sycl::property::command_group {

struct AdaptiveCpp_prefer_execution_lane {
  AdaptiveCpp_prefer_execution_lane(std::size_t lane_id);

};

}
```

##### Description

Provides a hint to the runtime on which *execution lane* to execute the operation. Execution lanes refer to a generalization of resources that can execute kernels and data transfers, such as a CUDA or HIP stream.

Many backends in AdaptiveCpp such as CUDA or HIP maintain a pool of inorder queues as execution lanes. By default, the scheduler will already automatically attempt to distribute work across all those queues.
If this distribution turns out to be not optimal, the `AdaptiveCpp_prefer_execution_lane` property can be used to influence the distribution, for example in order to achieve better overlap of data transfers and kernels, or to make sure that certain kernels execute concurrently if supported by backend and hardware.

Execution lanes for a device are enumerated starting from 0. If a non-existent execution lane is provided, it is mapped back to the permitted range using a modulo operation. Therefore, the execution lane id provided by the property can be seen as additional information on *potential* and desired parallelism that the runtime can exploit.

### `ACPP_EXT_BUFFER_PAGE_SIZE`

A property that can be attached to the buffer to set the buffer page size. See the AdaptiveCpp buffer model [specification](runtime-spec.md) for more details.

#### API reference

```c++
namespace sycl::property::buffer {

template<int Dim>
class AdaptiveCpp_page_size
{
public:
  // Set page size of buffer in units of number of elements.
  AdaptiveCpp_page_size(const sycl::range<Dim>& page_size);
};

}
````


### `ACPP_EXT_PREFETCH_HOST`

Provides `handler::prefetch_host()` (and corresponding queue shortcuts) to prefetch data from shared USM allocations to the host.
This is a more convenient alternative to constructing a host queue and executing regular `prefetch()` there.

#### API reference

```c++
/// Prefetches num_bytes from the USM pointer ptr to host memory
void handler::prefetch_host(const void *ptr, std::size_t num_bytes);

/// Queue shortcuts
event queue::prefetch_host(const void *ptr, std::size_t num_bytes);

event queue::prefetch_host(const void *ptr, std::size_t num_bytes, 
                          event dependency);

event queue::prefetch_host(const void *ptr, std::size_t num_bytes,
                          const std::vector<event> &dependencies);
```

### `ACPP_EXT_SYNCHRONOUS_MEM_ADVISE`

Provides a synchronous, free `sycl::mem_advise()` function as an alternative to the asynchronous `handler::mem_advise()`. In AdaptiveCpp, the synchronous variant is expected to be more efficient.

#### API reference

```c++
void sycl::mem_advise(const void *ptr, std::size_t num_bytes, int advise,
                       const context &ctx, const device &dev);

void sycl::mem_advise(const void *ptr, std::size_t num_bytes, int advise,
                       const queue& q);

```

### `ACPP_EXT_FP_ATOMICS`
This extension allows atomic operations on floating point types. Since this is not in the spec, this may break portability. Additionally, not all AdaptiveCpp backends may support the same set of FP atomics. It is the user's responsibility to ensure that the code remains portable and to implement fallbacks for platforms that don't support this. This extension must be enabled explicitly by `#define ACPP_EXT_FP_ATOMICS` before including `sycl.hpp`

### `ACPP_EXT_AUTO_PLACEHOLDER_REQUIRE`
This SYCL extension allows to `require()` placeholder accessors automatically. This extension does not need to be enabled explicitly and is always available.

The following example illustrates the use of this extension:

```cpp

cl::sycl::queue q;
cl::sycl::buffer<int, 1> buff{/* Initialize somehow */};
cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, 
      cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> acc{buff};

// This will call handler::require(acc) for each command group subsequently
// launched in the queue, until the `automatic_requirement` object is destroyed
// or `automatic_requirement.release()` is called.
auto automatic_requirement = 
   cl::sycl::vendor::hipsycl::automatic_require(q, acc);
// The member function `is_required()` can be used to check if
// if the automatic requirement object is active:
assert(automatic_requirement.is_required());

q.submit([&] (cl::sycl::handler& cgh) {
  // No call to require() is necessary here!
  cgh.parallel_for<class kernel1>([=] (cl::sycl::item<1> idx){
    /* use acc according to the requested access mode here */
  });
});

q.submit([&] (cl::sycl::handler& cgh) {
  cgh.parallel_for<class kernel2>([=] (cl::sycl::item<1> idx){
    /* use acc according to the requested access mode here */
  });
});

// After release(), acc will not be required anymore by kernels submitted to the queue
// afterwards.
automatic_requirement.release();
assert(!automatic_requirement.is_required());

/* Kernel submissions that do not need acc could go here */

// acc will now be required again
automatic_requirement.reacquire();

q.submit([&] (cl::sycl::handler& cgh) {
  cgh.parallel_for<class kernel3>([=] (cl::sycl::item<1> idx){
    /* use acc according to the requested access mode here */
  });
});

// If the automatic_requirement object goes out of scope, it will release the auto requirement
// if it is active.
```

This extension serves two purposes:
1. Avoid having to call `require()` again and again if the same accessor is used in many subsequent kernels. This can lead to a significant reduction of boilerplate code.
2. Simplify code when working with SYCL libraries that accept lambda functions or function objects. For example, for a `sort()` function in a SYCL library a custom comparator may be desired. Currently, there is no easy way to access some independent data in that comparator because accessors must be requested in the command group handler. This would not be possible in that case since the command group would be spawned internally by `sort`, and the user has no means of accessing it.

### `ACPP_EXT_CUSTOM_PFWI_SYNCHRONIZATION`
This extension allows for the user to specify what/if synchronization should happen at the end of a `parallel_for_work_item` call.
This extension is always enabled and does not need to be enabled explicitly.

Example:
```cpp
// By default, a barrier() will be executed at the end of
// a parallel for work item call, as defined by the spec and shown here:
group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
  ...
});
// The extension allows the user to specify custom 'finalizers' to
// alter synchronization behavior:
namespace sync = cl::sycl::vendor::hipsycl::synchronization;
group.parallel_for_work_item<sync::none>(
  [&](cl::sycl::h_item<1> item) {
  // No synchronization will be done after this call
});

group.parallel_for_work_item<sync::local_mem_fence>(
  [&](cl::sycl::h_item<1> item) {
  // local mem_fence will be done after this call
});
```
The following 'finalizers' are supported:
* `cl::sycl::vendor::hipsycl::synchronization::none` - no Operation
* `cl::sycl::vendor::hipsycl::synchronization::barrier<access::fence_space>` - barrier
* `cl::sycl::vendor::hipsycl::synchronization::local_barrier` - same as `barrier<access::fence_space::local_space>`
* `cl::sycl::vendor::hipsycl::synchronization::mem_fence<access::fence_space, access::mode = access::mode::read_write>` - memory fence
* `cl::sycl::vendor::hipsycl::synchronization::local_mem_fence` - same as `mem_fence<access::fence_space::local_space>`
* `cl::sycl::vendor::hipsycl::synchronization::global_mem_fence` - same as `mem_fence<access::fence_space::global_space>`
* `cl::sycl::vendor::hipsycl::synchronization::global_and_local_mem_fence` - same as `mem_fence<access::fence_space::global_and_local>`
