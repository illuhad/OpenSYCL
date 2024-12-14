
add_library(acpp-common SHARED
    filesystem.cpp
    appdb.cpp)

target_include_directories(acpp-common
  PUBLIC
    $<BUILD_INTERFACE:${HIPSYCL_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${HIPSYCL_SOURCE_DIR}
    ${ACPP_BINARY_ROOT}/include
)

target_link_libraries(acpp-common PRIVATE ${HIPSYCL_STDPAR_RT_LINKER_FLAGS})

list(INSERT CMAKE_MODULE_PATH 0 "${ACPP_SOURCE_ROOT}/cmake/")
find_package(Filesystem REQUIRED "${ACPP_FILESYSTEM_SEARCH_OPTIONS}")
set(CXX_FILESYSTEM_HEADER "${CXX_FILESYSTEM_HEADER}" PARENT_SCOPE)
set(CXX_FILESYSTEM_NAMESPACE "${CXX_FILESYSTEM_NAMESPACE}" PARENT_SCOPE)
list(REMOVE_AT CMAKE_MODULE_PATH 0)

if(WIN32)
  target_compile_options(acpp-common PRIVATE /EHs)
endif()

# to get the search path we need to know the output name, so set it explicitly.
set(HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME "acpp-common")
if(WIN32)
  set_target_properties(acpp-common PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS On)

  # ensure actual output and expected are equivalent
  set_target_properties(acpp-common PROPERTIES RUNTIME_OUTPUT_NAME ${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME})
  if(MinGW)
    set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME "lib${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.dll" PARENT_SCOPE)
  else()
    set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME "${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.dll" PARENT_SCOPE)
  endif()
else()
  target_link_libraries(acpp-common PRIVATE dl std::filesystem)

  # ensure actual output and expected are equivalent
  set_target_properties(acpp-common PROPERTIES LIBRARY_OUTPUT_NAME ${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME})
  if(APPLE)
    set(HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME "lib${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.dylib" PARENT_SCOPE)
  else()
    set(HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME "lib${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.so" PARENT_SCOPE)
  endif()
endif()

install(TARGETS acpp-common
        EXPORT install_exports
        INCLUDES DESTINATION include/AdaptiveCpp
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
