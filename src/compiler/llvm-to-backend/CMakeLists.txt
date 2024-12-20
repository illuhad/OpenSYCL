
set(LLVM_TO_BACKEND_INCLUDE_DIRS 
  ../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ACPP_BINARY_ROOT}/include)

set(CMAKE_INSTALL_RPATH ${base} ${base}/../../)

function(create_llvm_based_library)
  set(options STATIC)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(CREATE_LLVM_LIBRARY
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CREATE_LLVM_LIBRARY_TARGET})
  set(sources ${CREATE_LLVM_LIBRARY_SOURCES})

  if(CREATE_LLVM_LIBRARY_STATIC)
    add_library(${target} STATIC
      ${sources})
  else()
    add_library(${target} SHARED
      ${sources})
  endif()

  target_include_directories(${target} PRIVATE
    ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(${target} PRIVATE
    ${LLVM_DEFINITIONS_LIST})
  target_compile_definitions(${target} PRIVATE -DHIPSYCL_COMPILER_COMPONENT)

  if(NOT ACPP_LLVM_COMPONENT)
    find_library(LLVM_LIBRARY NAMES LLVM LLVM-${LLVM_VERSION_MAJOR} HINTS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    if(NOT LLVM_LIBRARY)
      message(FATAL_ERROR "LLVM at ${LLVM_DIR} does not have libLLVM.so. Please disable SSCP and related backends (-DWITH_SSCP_COMPILER=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF) or choose another LLVM installation")
    endif()
  elseif(NOT LLVM_BUILD_LLVM_DYLIB AND NOT WIN32)
    message(FATAL_ERROR "LLVM is not configured to build libLLVM.so. Please disable reconfigure with LLVM_BUILD_LLVM_DYLIB and LLVM_LINK_LLVM_DYLIB.")
  endif()
  if(WIN32)
    llvm_config(${target} core support irreader passes)
  else()
    llvm_config(${target} USE_SHARED core support irreader passes)
  endif()
  # We need symbolic functions for stdpar
  target_link_libraries(${target} PRIVATE ${HIPSYCL_STDPAR_RT_LINKER_FLAGS})
  
  set(install_subdir hipSYCL/llvm-to-backend)
  install(TARGETS ${target} 
    RUNTIME DESTINATION bin/${install_subdir}
    LIBRARY DESTINATION lib/${install_subdir}
    ARCHIVE DESTINATION lib/${install_subdir})
endfunction()

function(create_llvm_to_backend_library)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(CREATE_LLVM_LIBRARY
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CREATE_LLVM_LIBRARY_TARGET})
  set(sources ${CREATE_LLVM_LIBRARY_SOURCES})

  if(WIN32)
  set(MODE STATIC)
  endif()
  create_llvm_based_library(TARGET ${target} ${MODE} SOURCES ${sources})
  target_link_libraries(${target} PUBLIC llvm-to-backend)
endfunction()

function(create_llvm_to_backend_tool)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(CREATE_LLVM_TOOL
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CREATE_LLVM_TOOL_TARGET})
  set(sources ${CREATE_LLVM_TOOL_SOURCES})
 
  add_executable(${target}-tool ${sources})

  target_include_directories(${target}-tool PRIVATE
    ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  target_include_directories(${target}-tool SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(${target}-tool PRIVATE
    ${LLVM_DEFINITIONS_LIST})
  target_compile_definitions(${target}-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
  target_link_libraries(${target}-tool PRIVATE ${target})

  set(install_subdir hipSYCL/llvm-to-backend)
  install(TARGETS ${target}-tool
    RUNTIME DESTINATION bin/${install_subdir}
    LIBRARY DESTINATION lib/${install_subdir}
    ARCHIVE DESTINATION lib/${install_subdir})
endfunction()

function(add_hipsycl_llvm_backend)
  set(options)
  set(one_value_keywords BACKEND)
  set(multi_value_keywords LIBRARY TOOL)
  cmake_parse_arguments(ADD_BACKEND
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(backend_name ${ADD_BACKEND_BACKEND})
  set(tool_sources ${ADD_BACKEND_TOOL})
  set(library_sources ${ADD_BACKEND_LIBRARY})

  create_llvm_to_backend_library(TARGET llvm-to-${backend_name} SOURCES ${library_sources})
  create_llvm_to_backend_tool(TARGET llvm-to-${backend_name} SOURCES ${tool_sources})
endfunction()

include(ExternalProject)

if(WITH_SSCP_COMPILER)
  if(WIN32)
    set(MODE STATIC)
  endif()
  create_llvm_based_library(
    ${MODE}
    TARGET llvm-to-backend
    SOURCES 
      LLVMToBackend.cpp 
      AddressSpaceInferencePass.cpp
      KnownGroupSizeOptPass.cpp
      KnownPtrParamAlignmentOptPass.cpp
      GlobalSizesFitInI32OptPass.cpp
      GlobalInliningAttributorPass.cpp
      DeadArgumentEliminationPass.cpp
      ProcessS2ReflectionPass.cpp
      ../sscp/KernelOutliningPass.cpp
      Utils.cpp
      )

  target_compile_definitions(llvm-to-backend PRIVATE
    -DACPP_CLANG_PATH="${CLANG_INSTALLED_PATH}" 
  )
  target_link_libraries(llvm-to-backend PUBLIC
    acpp-common
  )

  if(WITH_LLVM_TO_SPIRV)
    add_hipsycl_llvm_backend(
      BACKEND spirv 
      LIBRARY spirv/LLVMToSpirv.cpp 
      TOOL spirv/LLVMToSpirvTool.cpp)

    # Install LLVM-SPIRV translator for llvm-to-spirv
    if(NOT LLVMSPIRV_BRANCH)
      set(LLVMSPIRV_BRANCH llvm_release_${LLVM_VERSION_MAJOR}0)
    endif()
    set(LLVMSPIRV_RELATIVE_INSTALLDIR lib/hipSYCL/ext/llvm-spirv)
    set(LLVMSPIRV_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/${LLVMSPIRV_RELATIVE_INSTALLDIR})
    set(LLVMSPIRV_PATH ${LLVMSPIRV_INSTALLDIR}/bin/llvm-spirv)
    set(LLVMSPIRV_RELATIVE_PATH ${LLVMSPIRV_RELATIVE_INSTALLDIR}/bin/llvm-spirv)

    include(FetchContent)
    FetchContent_Declare(LLVMSpirvTranslator
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-LLVM-Translator
      GIT_TAG origin/${LLVMSPIRV_BRANCH}
      GIT_SHALLOW ON
    )

    FetchContent_GetProperties(LLVMSpirvTranslator)
    if(NOT LLVMSpirvTranslator_POPULATED)
      FetchContent_Populate(LLVMSpirvTranslator)
      execute_process(COMMAND patch -N -p0 -c --fuzz=4 --ignore-whitespace -i llvm-spirv.patch ${llvmspirvtranslator_SOURCE_DIR}/lib/SPIRV/SPIRVInternal.h ${CMAKE_CURRENT_SOURCE_DIR}/spirv/llvm-spirv.patch)
      execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DLLVM_SPIRV_BUILD_EXTERNAL=ON -DLLVM_DIR=${LLVM_DIR} -DCMAKE_INSTALL_PREFIX=${LLVMSPIRV_INSTALLDIR} -DCMAKE_BUILD_TYPE=Release -S ${llvmspirvtranslator_SOURCE_DIR} -B ${llvmspirvtranslator_BINARY_DIR})
    endif()
    
    add_custom_target(llvm-spirv-translator ALL COMMAND ${CMAKE_COMMAND} --build ${llvmspirvtranslator_BINARY_DIR} --config Release)
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --install ${llvmspirvtranslator_BINARY_DIR})")

    target_compile_definitions(llvm-to-spirv PRIVATE
      -DHIPSYCL_RELATIVE_LLVMSPIRV_PATH="${LLVMSPIRV_RELATIVE_PATH}")
    
    
  endif()

  if(WITH_LLVM_TO_PTX)
    add_hipsycl_llvm_backend(
      BACKEND ptx 
      LIBRARY ptx/LLVMToPtx.cpp 
      TOOL ptx/LLVMToPtxTool.cpp)

    target_compile_definitions(llvm-to-ptx PRIVATE
      -DHIPSYCL_CUDA_PATH="${CUDA_TOOLKIT_ROOT_DIR}")
  endif()

  if(WITH_LLVM_TO_AMDGPU_AMDHSA)
    add_hipsycl_llvm_backend(
      BACKEND amdgpu 
      LIBRARY amdgpu/LLVMToAmdgpu.cpp 
      TOOL amdgpu/LLVMToAmdgpuTool.cpp)

    target_compile_definitions(llvm-to-amdgpu PRIVATE
      -DACPP_ROCM_PATH="${ROCM_PATH}")
    
    find_program(HIPCC_PATH hipcc HINTS ${ROCM_PATH}/bin)
    if(HIPCC_PATH)
      target_compile_definitions(llvm-to-amdgpu PRIVATE
        -DACPP_HIPCC_PATH="${HIPCC_PATH}")
    endif()

    find_library(HIPRTC_LIBRARY hiprtc HINTS ${ROCM_PATH}/lib)
   
    if(HIPRTC_LIBRARY)
      message(STATUS "Found hipRTC: ${HIPRTC_LIBRARY}")

      target_include_directories(llvm-to-amdgpu PRIVATE ${ROCM_PATH}/include)
      target_link_libraries(llvm-to-amdgpu PRIVATE ${HIPRTC_LIBRARY})
      target_compile_definitions(llvm-to-amdgpu PRIVATE -DACPP_HIPRTC_LINK)
    endif()

  endif()


  if(WITH_LLVM_TO_HOST)
    include(CheckCXXCompilerFlag)

    check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
    check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)

    if(HAS_MARCH_NATIVE)
      set(HOST_CPU_FLAG -march=native)
    elseif(HAS_MCPU_NATIVE)
      set(HOST_CPU_FLAG -mcpu=native)
    else()
      set(HOST_CPU_FLAG "")
      message(WARNING "Could not find -mcpu=native or -march=native. Host code generation may be suboptimal.")
    endif()

    add_hipsycl_llvm_backend(
      BACKEND host
      LIBRARY host/LLVMToHost.cpp host/HostKernelWrapperPass.cpp
      TOOL host/LLVMToHostTool.cpp)

    target_compile_definitions(llvm-to-host PRIVATE
      -DHIPSYCL_HOST_CPU_FLAG="${HOST_CPU_FLAG}")
    target_link_libraries(llvm-to-host PRIVATE acpp-clang-cbs)
  endif()

endif()
