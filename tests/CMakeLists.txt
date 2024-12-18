cmake_minimum_required(VERSION 3.12)
project(adaptivecpp-tests)

set(Boost_USE_STATIC_LIBS off)
set(BUILD_SHARED_LIBS on)
set(REDUCED_LOCAL_MEM_USAGE OFF CACHE BOOL "Only run tests with reduced local memory usage to allow running on hardware with little local memory.")

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

find_package(AdaptiveCpp REQUIRED)

find_package(Threads REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
    FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
      FORCE)
  endif()
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja" AND
    ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9) OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)))
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

if(REDUCED_LOCAL_MEM_USAGE)
  add_definitions(-DREDUCED_LOCAL_MEM_USAGE)
endif()


#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})
if(HIPSYCL_DISABLE_UNNAMED_LAMBDA_TESTS)
  add_definitions(-DHIPSYCL_DISABLE_UNNAMED_LAMBDA_TESTS)
endif()
if(WIN32)
add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_USE_MATH_DEFINES)
endif()

#OMP_ROOT and/or OpenMP_ROOT is not defined by default on Mac
if (APPLE)
  if (NOT DEFINED OpenMP_ROOT AND NOT DEFINED OMP_ROOT)
    execute_process(COMMAND brew list libomp
    COMMAND grep libomp.a 
    COMMAND sed -E "s/\\/lib\\/.*//"
    OUTPUT_VARIABLE DefaultOMP_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # fater linkage will error out if it is not found
    set(OpenMP_ROOT ${DefaultOMP_ROOT})
    set(OMP_ROOT ${DefaultOMP_ROOT})
  endif()
endif()
find_package(OpenMP REQUIRED)

add_subdirectory(platform_api)

add_subdirectory(dump_test)


add_executable(device_compilation_tests device_compilation_tests.cpp)
target_include_directories(device_compilation_tests PRIVATE ${Boost_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
add_sycl_to_target(TARGET device_compilation_tests)

add_executable(sycl_tests
  sycl/smoke/task_graph.cpp
  sycl/accessor.cpp
  sycl/atomic.cpp
  sycl/buffer.cpp
  sycl/explicit_copy.cpp
  sycl/extensions.cpp
  sycl/fill.cpp
  sycl/group_functions/group_functions_misc.cpp
  sycl/group_functions/group_functions_binary_reduce.cpp
  sycl/group_functions/group_functions_reduce.cpp
  sycl/group_functions/group_functions_scan.cpp
  sycl/half.cpp
  sycl/id_range.cpp
  sycl/info_queries.cpp
  sycl/interop_handle.cpp
  sycl/item.cpp
  sycl/kernel_invocation.cpp
  sycl/math.cpp
  sycl/marray.cpp
  sycl/profiler.cpp
  sycl/reduction.cpp
  sycl/reference_semantics.cpp
  sycl/relational.cpp
  sycl/sub_group.cpp
  sycl/sycl_test_suite.cpp 
  sycl/usm.cpp
  sycl/vec.cpp
  sycl/queue.cpp
  sycl/multi_ptr.cpp
)

# Also test instant submission mode
target_compile_definitions(sycl_tests PRIVATE -DACPP_ALLOW_INSTANT_SUBMISSION=1)
target_include_directories(sycl_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${OpenMP_CXX_INCLUDE_DIRS})
add_sycl_to_target(TARGET sycl_tests)

add_executable(rt_tests 
  runtime/runtime_test_suite.cpp 
  runtime/dag_builder.cpp
  runtime/data.cpp)

target_include_directories(rt_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(rt_tests PRIVATE Threads::Threads)
add_sycl_to_target(TARGET rt_tests)

# We cannot enable building them unconditionally at the moment,
# because --acpp-stdpar is not compatible with all --acpp-targets
# values. Enabling them in all cases would break some existing test flows.
if(WITH_PSTL_TESTS)
  add_executable(pstl_tests 
    pstl/pstl_test_suite.cpp
    pstl/std_math.cpp
    pstl/std_atomic.cpp
    pstl/all_of.cpp
    pstl/any_of.cpp
    pstl/copy.cpp
    pstl/copy_if.cpp
    pstl/copy_n.cpp
    pstl/exclusive_scan.cpp
    pstl/fill.cpp
    pstl/fill_n.cpp
    pstl/for_each.cpp
    pstl/for_each_n.cpp
    pstl/generate.cpp
    pstl/generate_n.cpp
    pstl/inclusive_scan.cpp
    pstl/memory.cpp
    pstl/merge.cpp
    pstl/none_of.cpp
    pstl/reduce.cpp
    pstl/replace.cpp
    pstl/replace_if.cpp
    pstl/replace_copy.cpp
    pstl/replace_copy_if.cpp
    pstl/sort.cpp
    pstl/transform.cpp
    pstl/transform_reduce.cpp
    pstl/transform_inclusive_scan.cpp
    pstl/transform_exclusive_scan.cpp
    pstl/pointer_validation.cpp
    pstl/allocation_map.cpp
    pstl/free_space_map.cpp)

  target_compile_options(pstl_tests PRIVATE --acpp-stdpar --acpp-stdpar-unconditional-offload)
  # pstl tests cannot run with global memory allocation hijacking, because apparently
  # the boost test headers allocate memory that is then transferred to libboost_unit_tests
  # which then cannot free it anymore. We we need to selectively enable it only for the actual test code.
  target_compile_definitions(pstl_tests PRIVATE -DHIPSYCL_STDPAR_MEMORY_MANAGEMENT_DEFAULT_DISABLED)
  target_include_directories(pstl_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${OpenMP_CXX_INCLUDE_DIRS})
  target_link_libraries(pstl_tests PRIVATE Threads::Threads -ltbb)
  add_sycl_to_target(TARGET pstl_tests)
endif()

add_subdirectory(compiler)
