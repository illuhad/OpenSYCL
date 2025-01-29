
function(libkernel_generate_bitcode_library)
  set(options)
  set(one_value_keywords OUTPUT TRIPLE SOURCE)
  set(multi_value_keywords ADDITIONAL_ARGS)
  cmake_parse_arguments(GENERATE_BITCODE_LIBRARY
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(output ${GENERATE_BITCODE_LIBRARY_OUTPUT})
  set(source ${GENERATE_BITCODE_LIBRARY_SOURCE})
  set(triple ${GENERATE_BITCODE_LIBRARY_TRIPLE})
  set(additional ${GENERATE_BITCODE_LIBRARY_ADDITIONAL_ARGS})

  set(target_output_file ${output})
  add_custom_command(
      OUTPUT ${target_output_file}
      COMMAND ${CLANG_EXECUTABLE_PATH} -target ${triple} -fno-exceptions -fno-rtti -O3 -emit-llvm -std=c++17 -DHIPSYCL_SSCP_LIBKERNEL_LIBRARY
                    -I ${HIPSYCL_SOURCE_DIR}/include ${additional} -o ${target_output_file} -c ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      VERBATIM)

  install(FILES ${target_output_file} DESTINATION ${CMAKE_INSTALL_LIBDIR}/hipSYCL/bitcode)
endfunction()

function(libkernel_generate_bitcode_target)
  set(options)
  set(one_value_keywords TARGETNAME TRIPLE)
  set(multi_value_keywords SOURCES ADDITIONAL_ARGS)

  cmake_parse_arguments(GENERATE_BITCODE_TARGET
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${GENERATE_BITCODE_TARGET_TARGETNAME})
  set(sources ${GENERATE_BITCODE_TARGET_SOURCES})
  set(triple ${GENERATE_BITCODE_TARGET_TRIPLE})
  set(additional ${GENERATE_BITCODE_TARGET_ADDITIONAL_ARGS})

  set(output_files )
  foreach(source ${sources})
    get_filename_component(target_source ${source} NAME_WE)
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/libkernel-sscp-${target}-${target_source}.bc)
    list(APPEND output_files ${output_file})
    
    libkernel_generate_bitcode_library(
      OUTPUT ${output_file}
      TRIPLE ${triple} 
      SOURCE ${source}
      ADDITIONAL_ARGS ${additional})
  endforeach()

  set(linked_output ${CMAKE_CURRENT_BINARY_DIR}/libkernel-sscp-${target}-full.bc)
  add_custom_command(
    OUTPUT ${linked_output}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-link -o=${linked_output} ${output_files}
    DEPENDS ${output_files}
  )

  install(FILES ${linked_output} DESTINATION ${CMAKE_INSTALL_LIBDIR}/hipSYCL/bitcode)

  add_custom_target(libkernel-sscp-${target} ALL
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${linked_output}
  )

endfunction()

add_subdirectory(spirv)
add_subdirectory(ptx)
add_subdirectory(amdgpu)
add_subdirectory(host)
