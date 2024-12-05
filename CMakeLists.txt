cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074) # Since CMake 3.12
  cmake_policy(SET CMP0074 NEW) # Don't complain about using BOOST_ROOT...
endif()
if(POLICY CMP0167) # Since CMake 3.30
    # Suppress warning: "Policy CMP0167 is not set: The FindBoost module is removed."
    # Set CMP0167 to OLD to avoid breakage in the Windows & macOS CI tests.
    cmake_policy(SET CMP0167 OLD)
endif()


set(ACPP_VERSION_MAJOR 24)
set(ACPP_VERSION_MINOR 06)
set(ACPP_VERSION_PATCH 0)

execute_process(
        COMMAND git status
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_STATUS
)

if(NOT ACPP_VERSION_SUFFIX)
  # With a release tarball, "git status" will fail (return non zero)
  if(GIT_STATUS)
    set(ACPP_VERSION_SUFFIX "")
  else()
    # Get the latest abbreviated commit hash of the working branch
    execute_process(
      COMMAND git log -1 --format=%h
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE ACPP_GIT_COMMIT_HASH
      RESULT_VARIABLE GIT_HASH_RETURN_CODE
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # git branch
    execute_process(
      COMMAND git rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE ACPP_GIT_BRANCH
      RESULT_VARIABLE GIT_BRANCH_RETURN_CODE
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # date of commit
    execute_process(COMMAND git show -s --format=%cd --date=short HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE ACPP_GIT_DATE
      RESULT_VARIABLE GIT_DATE_RETURN_CODE
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Convert YYYY-MM-DD to YYYYMMDD
    # We can't use --date=format:%Y%m%d to be compatible with git pre-2.6
    string(REPLACE "-" "" ACPP_GIT_DATE ${ACPP_GIT_DATE})
    # check whether there are local changes
    execute_process(COMMAND git diff-index --name-only HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE ACPP_LOCAL_CHANGES
      RESULT_VARIABLE GIT_LOCAL_CHANGES_RETURN_CODE
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (GIT_HASH_RETURN_CODE EQUAL 0 AND GIT_BRANCH_RETURN_CODE EQUAL 0 AND
        GIT_DATE_RETURN_CODE EQUAL 0 AND GIT_LOCAL_CHANGES_RETURN_CODE EQUAL 0)

      if(NOT "${ACPP_LOCAL_CHANGES}" STREQUAL "")
        set(DIRTY_STR ".dirty")
      else()
        set(DIRTY_STR "")
      endif()

      set(ACPP_VERSION_SUFFIX "+git.${ACPP_GIT_COMMIT_HASH}.${ACPP_GIT_DATE}.branch.${ACPP_GIT_BRANCH}${DIRTY_STR}")
    endif()
  endif()
endif()

project(hipSYCL VERSION ${ACPP_VERSION_MAJOR}.${ACPP_VERSION_MINOR}.${ACPP_VERSION_PATCH})


set(HIPSYCL_SOURCE_DIR ${PROJECT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT HIPSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(HIPSYCL_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(HIPSYCL_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

if (NOT ${BOOST_ROOT} STREQUAL "")
  set(ENV{BOOST_ROOT} ${BOOST_ROOT})
endif()

find_package(Boost COMPONENTS context fiber )

if(NOT Boost_FOUND)
  message(FATAL_ERROR "Please provide the path to the root directory of the boost installation using the -DBOOST_ROOT option")
endif()

# Check for CUDA/ROCm and clang
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/")
find_package(CUDA QUIET)
find_package(HIP QUIET HINTS ${ROCM_PATH} ${ROCM_PATH}/lib/cmake)

#set(WITH_CUDA_NVCXX_ONLY FALSE CACHE BOOL "(Deprecated) Whether to target CUDA exclusively with nvc++")

if(APPLE OR WIN32)
  set(DEFAULT_COMPILER_FEATURES_ENABLED false)
  set(ACPP_COMPILER_FEATURE_PROFILE "none" CACHE STRING "The level of AdaptiveCpp compiler features to enable. If not set to 'full', AdaptiveCpp will run in a degraded mode that may have implications for certain functionalities or performance in certain cases.")
else()
  set(DEFAULT_COMPILER_FEATURES_ENABLED true)
  set(ACPP_COMPILER_FEATURE_PROFILE "full" CACHE STRING "The level of AdaptiveCpp compiler features to enable. If not set to 'full', AdaptiveCpp will run in a degraded mode that may have implications for certain functionalities or performance in certain cases.")
endif()
set_property(CACHE ACPP_COMPILER_FEATURE_PROFILE PROPERTY STRINGS full  minimal none)


if (DEFINED WITH_SSCP_COMPILER OR DEFINED WITH_STDPAR_COMPILER OR DEFINED WITH_ACCELERATED_CPU OR DEFINED WITH_CUDA_NVCXX_ONLY)
  message(DEPRECATION "WITH_SSCP_COMPILER, WITH_STDPAR_COMPILER, WITH_ACCELERATED_CPU and WITH_CUDA_NVCXX_ONLY are deprecated and will be removed in a future version; please use ACPP_COMPILER_FEATURE_PROFILE instead.")
  set(ACPP_COMPILER_FEATURE_PROFILE "custom-deprecated")
endif()


set(ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER ${DEFAULT_COMPILER_FEATURES_ENABLED} CACHE INTERNAL "(deprecated) custom profile")
set(ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER ${DEFAULT_COMPILER_FEATURES_ENABLED} CACHE INTERNAL "(deprecated) custom profile")
set(ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU ${DEFAULT_COMPILER_FEATURES_ENABLED} CACHE INTERNAL "(deprecated) custom profile")

if(ACPP_COMPILER_FEATURE_PROFILE STREQUAL "full")
  set(WITH_SSCP_COMPILER true)
  set(WITH_STDPAR_COMPILER true)
  set(WITH_ACCELERATED_CPU true)
elseif(ACPP_COMPILER_FEATURE_PROFILE STREQUAL "minimal")
  set(WITH_SSCP_COMPILER false)
  set(WITH_STDPAR_COMPILER false)
  set(WITH_ACCELERATED_CPU false)
elseif(ACPP_COMPILER_FEATURE_PROFILE STREQUAL "none")
  # CUDA_NVCXX is handled further down.
  set(WITH_SSCP_COMPILER false)
  set(WITH_STDPAR_COMPILER false)
  set(WITH_ACCELERATED_CPU false)
elseif(ACPP_COMPILER_FEATURE_PROFILE STREQUAL "custom-deprecated")
  if(DEFINED WITH_SSCP_COMPILER)
    set(ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER ${WITH_SSCP_COMPILER} CACHE INTERNAL "(deprecated) custom profile")
  endif()
  if(DEFINED WITH_STDPAR_COMPILER)
    set(ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER ${WITH_STDPAR_COMPILER} CACHE INTERNAL "(deprecated) custom profile")
  endif()
  if(DEFINED WITH_ACCELERATED_CPU)
    set(ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU ${WITH_ACCELERATED_CPU} CACHE INTERNAL "(deprecated) custom profile")
  endif()

  # Ensure that these argument do not enter cache, otherwise the if(defined) checks above
  # will no longer work.
  unset(WITH_SSCP_COMPILER CACHE)
  unset(WITH_STDPAR_COMPILER CACHE)
  unset(WITH_ACCELERATED_CPU CACHE)

  set(WITH_SSCP_COMPILER ${ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER})
  set(WITH_STDPAR_COMPILER ${ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER})
  set(WITH_ACCELERATED_CPU ${ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU})


  message(DEPRECATION "Configuring DEPRECATED custom compiler feature profile with SSCP=${WITH_SSCP_COMPILER}, stdpar=${WITH_STDPAR_COMPILER}, accelerated cpu=${WITH_ACCELERATED_CPU}")
else()
  message(SEND_ERROR "Invalid value for ACPP_COMPILER_FEATURE_PROFILE: '${ACPP_COMPILER_FEATURE_PROFILE}'. Allowed values are: full, minimal, none")
endif()


message(STATUS "Building with compiler feature profile: ${ACPP_COMPILER_FEATURE_PROFILE}")

# Check for OpenCL unless user has explicitly set WITH_SSCP_COMPILER to False
if (DEFINED WITH_SSCP_COMPILER)
  if (WITH_SSCP_COMPILER)
    find_package(OpenCL QUIET)
  endif()
else()
  find_package(OpenCL QUIET)
endif()

# In case HIP is not found via the cmake pkgs we fall back to the legacy rocm discovery:
if (NOT HIP_FOUND)
  message(STATUS "Could not find HIP cmake integration, falling back to finding hipcc")
  find_program(HIPCC_COMPILER NAMES hipcc HINTS ${ROCM_PATH} ${ROCM_PATH}/bin)
  set(ROCM_PATH /opt/rocm CACHE PATH "Path to ROCm installation")
  if(HIPCC_COMPILER MATCHES "-NOTFOUND")
    message(STATUS "Could not find ROCm installation by looking for hipcc. ROCm support unavailable.")
    set(ROCM_FOUND false)
  else()
    set(ROCM_FOUND true)
  endif()
else()
  set(ROCM_FOUND true)
endif()

if(WITH_CUDA_BACKEND)
  if(NOT CUDA_FOUND)
    message(SEND_ERROR "CUDA was not found")
  endif()
endif()
if(WITH_ROCM_BACKEND)
  if(NOT ROCM_FOUND)
    #  message(SEND_ERROR "hipcc was not found")

    # User has requested ROCm, but we could not find hipcc.
    # this is not necessarily a reason to abort,
    # since we only need libhip_hcc, the HIP includes,
    # and the ROCm device headers. It could be that we
    # are faced with a minimal/incomplete ROCm installation
    # that could still be enough for us.
    # Let's assume the user knows what he/she is doing.
  endif()
endif()

if(WITH_OPENCL_BACKEND)
  if(NOT OpenCL_FOUND)
    message(SEND_ERROR "OpenCL was not found")
  endif()
endif()

set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "Build AdaptiveCpp support for NVIDIA GPUs with CUDA")
set(WITH_ROCM_BACKEND ${ROCM_FOUND} CACHE BOOL "Build AdaptiveCpp support for AMD GPUs with ROCm")
set(WITH_OPENCL_BACKEND ${OpenCL_FOUND} CACHE BOOL "Build AdaptiveCpp support for OpenCL SPIR-V devices supporting USM")

set(WITH_CPU_BACKEND true)

if(WITH_CUDA_BACKEND)
  find_program(NVCXX_COMPILER NAMES nvc++)
endif()

if(WITH_CUDA_NVCXX_ONLY)
  if(NVCXX_COMPILER MATCHES "-NOTFOUND")
      message(SEND_ERROR "Could not find nvc++ executable, specifiy with -DNVCXX_COMPILER=")
  endif()
endif()


if(WITH_ACCELERATED_CPU) # sanitize input
  set(WITH_ACCELERATED_CPU true)
  set(BUILD_CLANG_PLUGIN true)
else()
  set(WITH_ACCELERATED_CPU false)
endif()

if(WITH_SSCP_COMPILER) # sanitize input
  set(WITH_SSCP_COMPILER true)
  set(BUILD_CLANG_PLUGIN true)
else()
  set(WITH_SSCP_COMPILER false)
endif()

if(WITH_STDPAR_COMPILER) # sanitize input
  set(WITH_STDPAR_COMPILER true)
  set(BUILD_CLANG_PLUGIN true)
else()
  set(WITH_STDPAR_COMPILER false)
endif()

if(WITH_CUDA_BACKEND)
  if(NOT WITH_CUDA_NVCXX_ONLY AND NOT ACPP_COMPILER_FEATURE_PROFILE STREQUAL "none")
    set(BUILD_CLANG_PLUGIN true)
  endif()
endif()
if(WITH_ROCM_BACKEND)
  set(BUILD_CLANG_PLUGIN true)
endif()
if(WITH_OPENCL_BACKEND OR WITH_LEVEL_ZERO_BACKEND)
  if(NOT ACPP_COMPILER_FEATURE_PROFILE STREQUAL "none" AND NOT ACPP_COMPILER_FEATURE_PROFILE STREQUAL "minimal")
    set(BUILD_CLANG_PLUGIN true)
    set(WITH_SSCP_COMPILER true)
  endif()
endif()

if(BUILD_CLANG_PLUGIN)
  set(LLVM_DIR_OLD ${LLVM_DIR})
  find_package(LLVM CONFIG)
  if(LLVM_DIR_OLD AND NOT ("${LLVM_DIR_OLD}" STREQUAL "${LLVM_DIR}"))
    message(WARNING "Could not find LLVM in the requested location LLVM_DIR=${LLVM_DIR_OLD}; using ${LLVM_DIR}.")
  endif()
  message(STATUS "Building hipSYCL against LLVM configured from ${LLVM_DIR}")
  #find_package(Clang REQUIRED)

  find_program(CLANG_EXECUTABLE_PATH NAMES clang++-${LLVM_VERSION_MAJOR} clang++-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} clang++ HINTS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH CACHE STRING)
  if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
    find_program(CLANG_EXECUTABLE_PATH NAMES clang++-${LLVM_VERSION_MAJOR} clang++-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} clang++ CACHE STRING)
    if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
      message(SEND_ERROR "Could not find clang executable")
    endif()
  endif()
  if(WIN32)
    # plugins link agains clang.exe and not clang++.exe, thus using clang++ would result
    # in symbol clashes and therefore the plugin not loading.
    # on Windows it seems to be fine, as long as clang is able to deduce the language
    # from the file extension, as the msvcrt is linked explicitly anyways.
    string(REPLACE "clang++.exe" "clang.exe" CLANG_EXECUTABLE_PATH ${CLANG_EXECUTABLE_PATH})
  endif()
  message(STATUS "Selecting clang: ${CLANG_EXECUTABLE_PATH}")

  get_filename_component(LLVM_BIN_DIR "${CLANG_EXECUTABLE_PATH}" DIRECTORY)
  get_filename_component(LLVM_PREFIX_DIR "${LLVM_BIN_DIR}" DIRECTORY)
  # The path to the internal clang includes is currently required on ROCm
  # to let acpp fix a wrong order of system includes (clang's internal
  # includes are not of high enough priority in the include path search order).
  # We identify this path as the one containing __clang_cuda_runtime_wrapper.h,
  # which is a clang-specific header file.
  find_path(FOUND_CLANG_INCLUDE_PATH __clang_cuda_runtime_wrapper.h HINTS
    ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
    ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
    ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}/include
    ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
    ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
    ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include
    ${LLVM_PREFIX_DIR}/lib64/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
    ${LLVM_PREFIX_DIR}/lib64/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
    ${LLVM_PREFIX_DIR}/lib64/clang/${LLVM_VERSION_MAJOR}/include
    DOC "Path to internal clang headers. Typically, $LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include")

  if(EXISTS ${FOUND_CLANG_INCLUDE_PATH})
    # Required for newer ROCm versions
    set(CLANG_INCLUDE_PATH ${FOUND_CLANG_INCLUDE_PATH}/..)
  endif()

  if(NOT EXISTS ${CLANG_INCLUDE_PATH})
    message(SEND_ERROR "CLANG_INCLUDE_PATH ${CLANG_INCLUDE_PATH} does not exist. Please provide clang's internal include path manually: Find the directory where __clang_cuda_runtime_wrapper.h is. Provide this directory for older ROCm versions and the parent directory for newer ones.")
  endif()
  if(WITH_ROCM_BACKEND)
    execute_process(COMMAND ${CLANG_EXECUTABLE_PATH} "--version"
      OUTPUT_VARIABLE CLANG_VERBOSE_OUTPUT)
    string(REGEX MATCH "clang version [0-9]+.[0-9]+.[0-9]+ \\(.+ roc-([0-9]+).([0-9]+).([0-9]+)" AMD_CLANG_VERSION ${CLANG_VERBOSE_OUTPUT})

    if(AMD_CLANG_VERSION)
      set(ROCM_VERSION_MAJOR ${CMAKE_MATCH_1})
      set(ROCM_VERSION_MINOR ${CMAKE_MATCH_2})
      set(ROCM_VERSION_PATCH ${CMAKE_MATCH_3})
      message(STATUS "AMD clang version: ${ROCM_VERSION_MAJOR}.${ROCM_VERSION_MINOR}.${ROCM_VERSION_PATCH}")
    endif()
  endif()

  if(${LLVM_VERSION_MAJOR} LESS 14)
    if(${WITH_ACCELERATED_CPU} OR ${WITH_SSCP_COMPILER} OR ${WITH_STDPAR_COMPILER})
      message(WARNING "clang version too old (${LLVM_VERSION_MAJOR} < 14) to be used with advanced AdaptiveCpp compiler features, disabling WITH_STDPAR_COMPILER, WITH_ACCELERATED_CPU, WITH_SSCP_COMPILER")
      set(WITH_REFLECTION_BUILTINS OFF)
      set(WITH_STDPAR_COMPILER false)
      set(WITH_SSCP_COMPILER false)
      set(WITH_ACCELERATED_CPU false)
    endif()
  endif()
  message(STATUS "Using clang include directory: ${CLANG_INCLUDE_PATH}")

# Check if building on Windows and LLVM_LIBS is set, if not, use LLVM_AVAILABLE_LIBS
  if(WIN32 AND NOT LLVM_LIBS AND LLVM_AVAILABLE_LIBS)
    llvm_map_components_to_libnames(LLVM_LIBS analysis core support passes)
  endif()
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ACPP_CONFIG_FILE_PATH "${PROJECT_BINARY_DIR}")
set(ACPP_CONFIG_FILE_GLOBAL_INSTALLATION false CACHE BOOL
  "Whether to install the AdaptiveCpp configuration files into a global directory (typically, /etc/AdaptiveCpp). This is generally not recommended.")

if(ACPP_CONFIG_FILE_GLOBAL_INSTALLATION)
  set(ACPP_CONFIG_FILE_INSTALL_DIR /etc/AdaptiveCpp/)
else()
  set(ACPP_CONFIG_FILE_INSTALL_DIR etc/AdaptiveCpp)
endif()

if(Boost_FIBER_LIBRARY_DEBUG)
  get_filename_component(Boost_LIBRARY_DIR "${Boost_FIBER_LIBRARY_DEBUG}" DIRECTORY)
elseif(Boost_FIBER_LIBRARY_RELEASE)
  get_filename_component(Boost_LIBRARY_DIR "${Boost_FIBER_LIBRARY_RELEASE}" DIRECTORY)
else()
  get_target_property(FIBER_IMPORT_LIBRARY Boost::fiber IMPORTED_LOCATION_DEBUG)
  if(NOT FIBER_IMPORT_LIBRARY)
    get_target_property(FIBER_IMPORT_LIBRARY Boost::fiber IMPORTED_LOCATION_RELEASE)
  endif()
  get_filename_component(Boost_LIBRARY_DIR "${FIBER_IMPORT_LIBRARY}" DIRECTORY)
  get_target_property(Boost_INCLUDE_DIR Boost::fiber INTERFACE_INCLUDE_DIRECTORIES)
endif()

if(APPLE)
  set(DEFAULT_OMP_FLAG "-Xclang -fopenmp")

  if(Boost_FIBER_LIBRARY_DEBUG)
    set(DEFAULT_BOOST_LIBRARIES "${Boost_CONTEXT_LIBRARY_DEBUG} ${Boost_FIBER_LIBRARY_DEBUG} -Wl,-rpath ${Boost_LIBRARY_DIR}")
  else()
    set(DEFAULT_BOOST_LIBRARIES "${Boost_CONTEXT_LIBRARY_RELEASE} ${Boost_FIBER_LIBRARY_RELEASE} -Wl,-rpath ${Boost_LIBRARY_DIR}")
  endif()
else()
  set(DEFAULT_OMP_FLAG "-fopenmp")
endif()


set(ROCM_LIBS "-L${ROCM_PATH}/lib -L${ROCM_PATH}/hip/lib -lamdhip64" CACHE STRING "Necessary libraries for ROCm")
set(ACPP_FILESYSTEM_SEARCH_OPTIONS "Final;Experimental;Boost" CACHE STRING "Which filesystem libraries should be searched for")

##########################################################
## Do not define any variables that might be used by build
## targets below this line.
##########################################################
add_subdirectory(src)

set(DEFAULT_WIN32_CUDA_LINK_LINE "-L$HIPSYCL_CUDA_LIB_PATH -lcudart")
set(DEFAULT_CUDA_LINK_LINE "-Wl,-rpath=$HIPSYCL_CUDA_LIB_PATH -L$HIPSYCL_CUDA_LIB_PATH -lcudart")

set(DEFAULT_WIN32_HIP_LINK_LINE "-L$HIPSYCL_ROCM_PATH/lib -lamdhip64")

if(MINGW)
  set(DEFAULT_LINK_PREFIX "-l:")
else()
  set(DEFAULT_LINK_PREFIX "-l")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  get_filename_component(Boost_CONTEXT_LIBRARY_NAME "${Boost_CONTEXT_LIBRARY_DEBUG}" NAME)
  get_filename_component(Boost_FIBER_LIBRARY_NAME "${Boost_FIBER_LIBRARY_DEBUG}" NAME)
else()
  get_filename_component(Boost_CONTEXT_LIBRARY_NAME "${Boost_CONTEXT_LIBRARY_RELEASE}" NAME)
  get_filename_component(Boost_FIBER_LIBRARY_NAME "${Boost_FIBER_LIBRARY_RELEASE}" NAME)
endif()
set(DEFAULT_WIN32_OMP_LINK_LINE "-L${Boost_LIBRARY_DIR} ${DEFAULT_LINK_PREFIX}${Boost_CONTEXT_LIBRARY_NAME} ${DEFAULT_LINK_PREFIX}${Boost_FIBER_LIBRARY_NAME} ${DEFAULT_OMP_FLAG}")
set(DEFAULT_WIN32_SEQUENTIAL_LINK_LINE "-L${Boost_LIBRARY_DIR} ${DEFAULT_LINK_PREFIX}${Boost_CONTEXT_LIBRARY_NAME} ${DEFAULT_LINK_PREFIX}${Boost_FIBER_LIBRARY_NAME} -llibomp")

# need add_subdirectory(src) before this!
set(DEFAULT_APPLE_OMP_LINK_LINE "${DEFAULT_BOOST_LIBRARIES} ${DEFAULT_OMP_FLAG} ${hipSYCL_OpenMP_CXX_LIBRARIES}")
set(DEFAULT_APPLE_SEQUENTIAL_LINK_LINE "${DEFAULT_BOOST_LIBRARIES} ${hipSYCL_OpenMP_CXX_LIBRARIES}")
set(DEFAULT_OMP_LINK_LINE "-L${Boost_LIBRARY_DIR} -lboost_context -lboost_fiber -Wl,-rpath=${Boost_LIBRARY_DIR} ${DEFAULT_OMP_FLAG}")
set(DEFAULT_SEQUENTIAL_LINK_LINE "-L${Boost_LIBRARY_DIR} -lboost_context -lboost_fiber -Wl,-rpath=${Boost_LIBRARY_DIR}")

# If no link lines given, set to default.
if(WIN32)
  if(NOT ROCM_LINK_LINE)
    set(ROCM_LINK_LINE ${DEFAULT_WIN32_HIP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link ROCm libraries to SYCL applications")
  endif()
else()
  if(NOT ROCM_LINK_LINE)
    set(ROCM_LINK_LINE "-Wl,-rpath=$HIPSYCL_ROCM_PATH/lib -Wl,-rpath=$HIPSYCL_ROCM_PATH/hip/lib ${ROCM_LIBS}" CACHE STRING "Arguments passed to compiler to link ROCm libraries to SYCL applications")
  endif()
endif()

if(WIN32)
  if(NOT CUDA_LINK_LINE)
    set(CUDA_LINK_LINE ${DEFAULT_WIN32_CUDA_LINK_LINE} CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
  endif()
  if(NOT OMP_LINK_LINE)
    set(OMP_LINK_LINE ${DEFAULT_WIN32_OMP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link OpenMP libraries to SYCL applications")
  endif()
  if(NOT SEQUENTIAL_LINK_LINE)
    set(SEQUENTIAL_LINK_LINE ${DEFAULT_WIN32_SEQUENTIAL_LINK_LINE} CACHE STRING "Arguments passed to compiler to link host libraries to SYCL applications")
  endif()
elseif(APPLE)
  if(NOT CUDA_LINK_LINE) # no CUDA support here.. but make CMake happy.
    set(CUDA_LINK_LINE "" CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
  endif()
  if(NOT OMP_LINK_LINE)
    set(OMP_LINK_LINE ${DEFAULT_APPLE_OMP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link OpenMP libraries to SYCL applications")
  endif()
  if(NOT SEQUENTIAL_LINK_LINE)
    set(SEQUENTIAL_LINK_LINE ${DEFAULT_APPLE_SEQUENTIAL_LINK_LINE} CACHE STRING "Arguments passed to compiler to link host libraries to SYCL applications")
  endif()
else()
  if(NOT CUDA_LINK_LINE)
    set(CUDA_LINK_LINE ${DEFAULT_CUDA_LINK_LINE} CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
  endif()
  if(NOT OMP_LINK_LINE)
    set(OMP_LINK_LINE ${DEFAULT_OMP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link OpenMP libraries to SYCL applications")
  endif()
  if(NOT SEQUENTIAL_LINK_LINE)
    set(SEQUENTIAL_LINK_LINE ${DEFAULT_SEQUENTIAL_LINK_LINE} CACHE STRING "Arguments passed to compiler to link host libraries to SYCL applications")
  endif()
endif()

# If no compile flags given, set to default.
if(NOT ROCM_CXX_FLAGS)
  # clang erroneously sets feature detection flags for
  # __float128 even though it is not supported for CUDA / HIP,
  # see https://bugs.llvm.org/show_bug.cgi?id=47559.

  set(ROCM_CXX_FLAGS "-isystem $HIPSYCL_PATH/include/AdaptiveCpp/hipSYCL/std/hiplike -isystem ${CLANG_INCLUDE_PATH} -U__FLOAT128__ -U__SIZEOF_FLOAT128__ -I$HIPSYCL_ROCM_PATH/include -I$HIPSYCL_ROCM_PATH/include --rocm-device-lib-path=$HIPSYCL_ROCM_PATH/amdgcn/bitcode --rocm-path=$HIPSYCL_ROCM_PATH -fhip-new-launch-api -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -D__HIP_ROCclr__" CACHE STRING "Arguments passed to compiler to compile SYCL applications with ROCm")
endif()

if(NOT CUDA_CXX_FLAGS)
  # clang erroneously sets feature detection flags for
  # __float128 even though it is not supported for CUDA / HIP,
  # see https://bugs.llvm.org/show_bug.cgi?id=47559.
  set(CUDA_CXX_FLAGS "-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -isystem $HIPSYCL_PATH/include/AdaptiveCpp/hipSYCL/std/hiplike" CACHE STRING "Arguments passed to compiler to compile SYCL applications with CUDA")
endif()

# always need -D_ENABLE_EXTENDED_ALIGNED_STORAGE to allow correctly aligned local memory on CPU
if(NOT OMP_CXX_FLAGS)
  set(OMP_CXX_FLAGS "-I${Boost_INCLUDE_DIR} ${DEFAULT_OMP_FLAG} -D_ENABLE_EXTENDED_ALIGNED_STORAGE" CACHE STRING "Arguments passed to compiler to compile SYCL applications with OpenMP")
endif()

if(NOT SEQUENTIAL_CXX_FLAGS)
  set(SEQUENTIAL_CXX_FLAGS "-I${Boost_INCLUDE_DIR} -D_ENABLE_EXTENDED_ALIGNED_STORAGE" CACHE STRING "Arguments passed to compiler to compile SYCL applications on host")
endif()

if(BUILD_CLANG_PLUGIN)
  set(PLUGIN_LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
else()
  set(PLUGIN_LLVM_VERSION_MAJOR 0)
endif()

set(DEFAULT_GPU_ARCH "" CACHE STRING "(Deprecated, use DEFAULT_TARGETS instead) Optional: Default GPU architecture to compile for when targeting GPUs (e.g.: sm_60 or gfx900)")
set(DEFAULT_TARGETS "" CACHE STRING "Default targets to compile for")

if(NOT DEFAULT_TARGETS)
  if(DEFAULT_PLATFORM)
    message(DEPRECATION "DEFAULT_PLATFORM is deprecated; use DEFAULT_TARGETS instead.")

    if(DEFAULT_PLATFORM STREQUAL "cpu")
      set(DEFAULT_TARGETS "omp")
    endif()

    if(DEFAULT_GPU_ARCH)
      message(DEPRECATION "DEFAULT_GPU_ARCH is deprecated; use DEFAULT_TARGETS instead.")

      if(DEFAULT_PLATFORM STREQUAL "cuda")
        set(DEFAULT_TARGETS "cuda:${DEFAULT_GPU_ARCH}")
      elseif(DEFAULT_PLATFORM STREQUAL "rocm")
        set(DEFAULT_TARGETS "hip:${DEFAULT_GPU_ARCH}")
      else()
        message(SEND_ERROR "Invalid value for DEFAULT_PLATFORM: \"${DEFAULT_PLATFORM}\". When DEFAULT_GPU_ARCH is specified, only \"cuda\" and \"rocm\" are supported.")
      endif()

    endif()
  elseif(DEFAULT_GPU_ARCH)
    message(DEPRECATION "DEFAULT_GPU_ARCH is deprecated; use DEFAULT_TARGETS instead.")
    message(SEND_ERROR "DEFAULT_GPU_ARCH cannot be used without specifying DEFAULT_PLATFORM.")
  endif()
endif()

if(NOT DEFAULT_TARGETS)
  if(WITH_SSCP_COMPILER)
    set(DEFAULT_TARGETS "generic")
  else()
    message(WARNING "The generic SSCP compiler is disabled; without it, offloading code to accelerators requires explicitly specifying --acpp-targets when invoking acpp."
" A binary generated by invoking acpp without --acpp-targets will only be able to run on the host. Set -DDEFAULT_TARGETS to your desired default targets during cmake "
" configuration to avoid this warning. In the future, configurations without the SSCP compiler enabled might potentially be only supported if -DDEFAULT_TARGETS is provided explicitly.")
    set(DEFAULT_TARGETS "omp")
  endif()
endif()

message(STATUS "Default compilation target(s): ${DEFAULT_TARGETS}")


set(ACPP_CORE_CONFIG_FILE "{
  \"version-major\" : \"${ACPP_VERSION_MAJOR}\",
  \"version-minor\" : \"${ACPP_VERSION_MINOR}\",
  \"version-patch\" : \"${ACPP_VERSION_PATCH}\",
  \"version-suffix\" : \"${ACPP_VERSION_SUFFIX}\",
  \"plugin-llvm-version-major\" : \"${PLUGIN_LLVM_VERSION_MAJOR}\",
  \"plugin-with-cpu-acceleration\" : \"${WITH_ACCELERATED_CPU}\",
  \"default-clang\"     : \"${CLANG_EXECUTABLE_PATH}\",
  \"default-targets\"  : \"${DEFAULT_TARGETS}\",
  \"default-cpu-cxx\"   : \"${CMAKE_CXX_COMPILER}\",
  \"default-rocm-path\" : \"${ROCM_PATH}\",
  \"default-use-bootstrap-mode\" : \"false\",
  \"default-is-dryrun\" : \"false\",
  \"default-use-accelerated-cpu\" : \"${WITH_ACCELERATED_CPU}\",
  \"default-clang-include-path\" : \"${CLANG_INCLUDE_PATH}\",
  \"default-sequential-link-line\" : \"${SEQUENTIAL_LINK_LINE}\",
  \"default-sequential-cxx-flags\" : \"${SEQUENTIAL_CXX_FLAGS}\",
  \"default-omp-link-line\" : \"${OMP_LINK_LINE}\",
  \"default-omp-cxx-flags\" : \"${OMP_CXX_FLAGS}\",
  \"default-is-explicit-multipass\" : \"false\",
  \"default-save-temps\" : \"false\"
}
")


set(ACPP_CUDA_CONFIG_FILE "{
  \"default-nvcxx\"     : \"${NVCXX_COMPILER}\",
  \"default-cuda-path\" : \"${CUDA_TOOLKIT_ROOT_DIR}\",
  \"default-cuda-link-line\" : \"${CUDA_LINK_LINE}\",
  \"default-cuda-cxx-flags\" : \"${CUDA_CXX_FLAGS}\"
}
")


set(ACPP_ROCM_CONFIG_FILE "{
  \"default-rocm-path\" : \"${ROCM_PATH}\",
  \"default-rocm-link-line\" : \"${ROCM_LINK_LINE}\",
  \"default-rocm-cxx-flags\" : \"${ROCM_CXX_FLAGS}\"
}
")

file(WRITE ${ACPP_CONFIG_FILE_PATH}/acpp-core.json ${ACPP_CORE_CONFIG_FILE})
if(WITH_CUDA_BACKEND)
  file(WRITE ${ACPP_CONFIG_FILE_PATH}/acpp-cuda.json ${ACPP_CUDA_CONFIG_FILE})
endif()
if(WITH_ROCM_BACKEND)
  file(WRITE ${ACPP_CONFIG_FILE_PATH}/acpp-rocm.json ${ACPP_ROCM_CONFIG_FILE})
endif()


# must stay after add_subdirectory(src) as HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME is declared there
configure_file(
  ${PROJECT_SOURCE_DIR}/include/hipSYCL/common/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/hipSYCL/common/config.hpp)

install(DIRECTORY include/CL DESTINATION include/AdaptiveCpp/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/SYCL DESTINATION include/AdaptiveCpp/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/hipSYCL DESTINATION include/AdaptiveCpp/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/hipSYCL DESTINATION include/AdaptiveCpp/ FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/hipSYCL/std DESTINATION include/AdaptiveCpp/hipSYCL/ )
install(FILES ${PROJECT_BINARY_DIR}/include/hipSYCL/common/config.hpp DESTINATION include/AdaptiveCpp/hipSYCL/common/)

if(NOT WIN32)
# Windows is case-insensitive, so don't copy to sycl/sycl.hpp as
# we already have SYCL/sycl.hpp
install(FILES include/SYCL/sycl.hpp DESTINATION include/AdaptiveCpp/sycl/)
endif()
install(PROGRAMS bin/acpp DESTINATION bin)
install(PROGRAMS bin/acpp DESTINATION bin RENAME syclcc)
install(PROGRAMS bin/acpp DESTINATION bin RENAME syclcc-clang)


install(FILES ${ACPP_CONFIG_FILE_PATH}/acpp-core.json DESTINATION ${ACPP_CONFIG_FILE_INSTALL_DIR})
if(WITH_CUDA_BACKEND)
  install(FILES ${ACPP_CONFIG_FILE_PATH}/acpp-cuda.json DESTINATION ${ACPP_CONFIG_FILE_INSTALL_DIR})
endif()
if(WITH_ROCM_BACKEND)
  install(FILES ${ACPP_CONFIG_FILE_PATH}/acpp-rocm.json DESTINATION ${ACPP_CONFIG_FILE_INSTALL_DIR})
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)



# Legacy module (hipSYCL)


set(HIPSYCL_INSTALL_CMAKE_DIR
  "lib/cmake/hipSYCL" CACHE PATH "Install path for legacy CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is
# the future install root
set(HIPSYCL_INSTALL_COMPILER_DIR bin)
set(HIPSYCL_INSTALL_LAUNCHER_DIR ${HIPSYCL_INSTALL_CMAKE_DIR})
set(HIPSYCL_INSTALL_LAUNCHER_RULE_DIR ${HIPSYCL_INSTALL_CMAKE_DIR})

install(PROGRAMS cmake/syclcc-launcher DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR})
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR})

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/hipsycl-config-version.cmake"
  VERSION ${hipSYCL_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/hipsycl-config.cmake.in
    ${PROJECT_BINARY_DIR}/hipsycl-config.cmake
    INSTALL_DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR}
    PATH_VARS
    HIPSYCL_INSTALL_COMPILER_DIR
    HIPSYCL_INSTALL_LAUNCHER_DIR
    HIPSYCL_INSTALL_LAUNCHER_RULE_DIR
)

install(FILES
  ${PROJECT_BINARY_DIR}/hipsycl-config.cmake
  ${PROJECT_BINARY_DIR}/hipsycl-config-version.cmake
  DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR}
)

install(EXPORT install_exports
  FILE "hipsycl-targets.cmake"
  NAMESPACE hipSYCL::
  DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR}
)

mark_as_advanced(
  HIPSYCL_INSTALL_CMAKE_DIR
)


# Legacy module (Open SYCL)

set(OPENSYCL_INSTALL_CMAKE_DIR
  "lib/cmake/OpenSYCL" CACHE PATH "Install path for CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is
# the future install root
set(OPENSYCL_INSTALL_COMPILER_DIR bin)
set(OPENSYCL_INSTALL_LAUNCHER_DIR ${OPENSYCL_INSTALL_CMAKE_DIR})
set(OPENSYCL_INSTALL_LAUNCHER_RULE_DIR ${OPENSYCL_INSTALL_CMAKE_DIR})

install(PROGRAMS cmake/syclcc-launcher DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR})
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR})

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/opensycl-config-version.cmake"
  VERSION ${hipSYCL_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/opensycl-config.cmake.in
    ${PROJECT_BINARY_DIR}/opensycl-config.cmake
    INSTALL_DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR}
    PATH_VARS
    OPENSYCL_INSTALL_COMPILER_DIR
    OPENSYCL_INSTALL_LAUNCHER_DIR
    OPENSYCL_INSTALL_LAUNCHER_RULE_DIR
)

install(FILES
  ${PROJECT_BINARY_DIR}/opensycl-config.cmake
  ${PROJECT_BINARY_DIR}/opensycl-config-version.cmake
  DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR}
)

install(EXPORT install_exports
  FILE "opensycl-targets.cmake"
  NAMESPACE OpenSYCL::
  DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR}
)

mark_as_advanced(
  OPENSYCL_INSTALL_CMAKE_DIR
)

# New module


set(ADAPTIVECPP_INSTALL_CMAKE_DIR
  "lib/cmake/AdaptiveCpp" CACHE PATH "Install path for CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is
# the future install root
set(ADAPTIVECPP_INSTALL_COMPILER_DIR bin)
set(ADAPTIVECPP_INSTALL_LAUNCHER_DIR ${ADAPTIVECPP_INSTALL_CMAKE_DIR})
set(ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR ${ADAPTIVECPP_INSTALL_CMAKE_DIR})

install(PROGRAMS cmake/syclcc-launcher DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR})
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR})

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/adaptivecpp-config-version.cmake"
  VERSION ${hipSYCL_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/adaptivecpp-config.cmake.in
    ${PROJECT_BINARY_DIR}/adaptivecpp-config.cmake
    INSTALL_DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
    PATH_VARS
    ADAPTIVECPP_INSTALL_COMPILER_DIR
    ADAPTIVECPP_INSTALL_LAUNCHER_DIR
    ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR
)

install(FILES
  ${PROJECT_BINARY_DIR}/adaptivecpp-config.cmake
  ${PROJECT_BINARY_DIR}/adaptivecpp-config-version.cmake
  DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
)

install(EXPORT install_exports
  FILE "adaptivecpp-targets.cmake"
  NAMESPACE AdaptiveCpp::
  DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
)

mark_as_advanced(
  ADAPTIVECPP_INSTALL_CMAKE_DIR
)

