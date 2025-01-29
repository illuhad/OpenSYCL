add_executable(acpp-appdb-tool acpp-appdb-tool.cpp)
target_compile_definitions(acpp-appdb-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(acpp-appdb-tool PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)


target_link_libraries(acpp-appdb-tool PRIVATE acpp-common)

# Make sure that acpp-info uses compatible sanitizer flags for sanitized runtime builds
target_link_libraries(acpp-appdb-tool PRIVATE ${ACPP_RT_SANITIZE_FLAGS})
target_compile_options(acpp-appdb-tool PRIVATE ${ACPP_RT_SANITIZE_FLAGS})
set_target_properties(acpp-appdb-tool PROPERTIES INSTALL_RPATH ${base}/../lib/)

install(TARGETS acpp-appdb-tool DESTINATION ${CMAKE_INSTALL_BINDIR})
