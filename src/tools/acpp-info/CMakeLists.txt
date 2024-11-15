add_executable(acpp-info acpp-info.cpp)

target_compile_definitions(acpp-info PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(acpp-info PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${ACPP_BINARY_ROOT}/include)

target_link_libraries(acpp-info PRIVATE acpp-rt)

# Make sure that acpp-info uses compatible sanitizer flags for sanitized runtime builds
target_link_libraries(acpp-info PRIVATE ${ACPP_RT_SANITIZE_FLAGS})
target_compile_options(acpp-info PRIVATE ${ACPP_RT_SANITIZE_FLAGS})
set_target_properties(acpp-info PROPERTIES INSTALL_RPATH ${base}/../lib/)

install(TARGETS acpp-info DESTINATION bin)
