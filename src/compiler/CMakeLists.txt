cmake_minimum_required(VERSION 3.10)
project(acpp-clang)

get_filename_component(CLANG_BINARY_PREFIX ${CLANG_EXECUTABLE_PATH} DIRECTORY)

if(NOT HIPSYCL_DEBUG_LEVEL)
  set(HIPSYCL_DEBUG_LEVEL 2 CACHE INTEGER)
endif()

add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})
if(WITH_ACCELERATED_CPU)
  add_definitions(-DHIPSYCL_WITH_ACCELERATED_CPU)
endif()

if(WITH_ACCELERATED_CPU OR WITH_SSCP_COMPILER)
  set(CBS_PLUGIN cbs/LoopSplitterInlining.cpp
    cbs/SplitterAnnotationAnalysis.cpp
    cbs/IRUtils.cpp
    cbs/KernelFlattening.cpp
    cbs/LoopsParallelMarker.cpp
    cbs/PHIsToAllocas.cpp
    cbs/RemoveBarrierCalls.cpp
    cbs/CanonicalizeBarriers.cpp
    cbs/SimplifyKernel.cpp
    cbs/LoopSimplify.cpp
    cbs/PipelineBuilder.cpp
    cbs/SubCfgFormation.cpp
    cbs/UniformityAnalysis.cpp
    cbs/VectorShape.cpp
    cbs/VectorizationInfo.cpp
    cbs/AllocaSSA.cpp
    cbs/VectorShapeTransformer.cpp
    cbs/Region.cpp
    cbs/SyncDependenceAnalysis.cpp)

  add_library(acpp-clang-cbs OBJECT
    ${CBS_PLUGIN}
  )

  set_property(TARGET acpp-clang-cbs PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(WITH_SSCP_COMPILER)
  set(WITH_REFLECTION_BUILTINS ON)
  set(SSCP_COMPILER
    sscp/KernelOutliningPass.cpp
    sscp/IRConstantReplacer.cpp
    sscp/DynamicFunctionSupport.cpp
    sscp/TargetSeparationPass.cpp
    sscp/HostKernelNameExtractionPass.cpp
    sscp/AggregateArgumentExpansionPass.cpp
    sscp/StdBuiltinRemapperPass.cpp
    sscp/StdAtomicRemapperPass.cpp)
  add_definitions(-DHIPSYCL_WITH_SSCP_COMPILER)
else()
  set(SSCP_COMPILER "")
endif()

if(WITH_STDPAR_COMPILER)
  set(WITH_REFLECTION_BUILTINS ON)
  set(STDPAR_COMPILER
    stdpar/MallocToUSM.cpp
    stdpar/SyncElision.cpp)
  add_definitions(-DHIPSYCL_WITH_STDPAR_COMPILER)
else()
  set(STDPAR_COMPILER "")
endif()

if(WITH_REFLECTION_BUILTINS)
  set(REFLECTION_BUILTINS
    reflection/IntrospectStructPass.cpp
    reflection/FunctionNameExtractionPass.cpp)
  add_definitions(-DHIPSYCL_WITH_REFLECTION_BUILTINS)
else()
  set(REFLECTION_BUILTINS "")
endif()

add_library(acpp-clang MODULE
  AdaptiveCppClangPlugin.cpp
  GlobalsPruningPass.cpp
  SMCPCompatPass.cpp
  ${SSCP_COMPILER}
  ${STDPAR_COMPILER}
  ${REFLECTION_BUILTINS}
)

function(configure_target)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords)
  cmake_parse_arguments(CONFIGURE_LLVM_TARGET
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CONFIGURE_LLVM_TARGET_TARGET})

  target_include_directories(${target} PRIVATE
    ../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include)
  target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(${target} PRIVATE
    ${LLVM_DEFINITIONS_LIST})
  target_compile_definitions(${target} PRIVATE
    -DHIPSYCL_COMPILER_COMPONENT)

  if(ROCM_VERSION_MAJOR)
    target_compile_definitions(${target} PRIVATE -DROCM_CLANG_VERSION_MAJOR=${ROCM_VERSION_MAJOR} -DROCM_CLANG_VERSION_MINOR=${ROCM_VERSION_MINOR} -DROCM_CLANG_VERSION_PATCH=${ROCM_VERSION_PATCH})
  endif()

  if(NOT ${LLVM_ENABLE_EH})
    target_compile_options(${target} PRIVATE -fno-exceptions)
  endif()

  if(NOT ${LLVM_ENABLE_RTTI})
    target_compile_options(${target} PRIVATE -fno-rtti)
  endif()

  target_link_libraries(${target}
    ${LLVM_LIBS})

  if(WIN32)
    set(CLANG_LIBRARY_DIR ${LLVM_PREFIX_DIR}/lib/)
    target_link_directories(${target} PRIVATE ${CLANG_LIBRARY_DIR})
    target_link_libraries(${target} clang LLVMSupport)
  endif()

  install(TARGETS ${target}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endfunction(configure_target)

configure_target(TARGET acpp-clang)

if (WITH_ACCELERATED_CPU OR WITH_SSCP_COMPILER)
  configure_target(TARGET acpp-clang-cbs)

  target_link_libraries(acpp-clang acpp-clang-cbs)
endif()


if(APPLE)
  target_link_libraries(acpp-clang -Wl,-undefined -Wl,dynamic_lookup)
elseif(WIN32)
  set_target_properties(acpp-clang PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS On)
else()
  target_link_libraries(acpp-clang -Wl,-znodelete)
endif()


add_subdirectory(llvm-to-backend)
