add_executable(acpp-hcf-tool acpp-hcf-tool.cpp)
target_compile_definitions(acpp-hcf-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
target_include_directories(acpp-hcf-tool PRIVATE 
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)

install(TARGETS acpp-hcf-tool DESTINATION ${CMAKE_INSTALL_BINDIR})
