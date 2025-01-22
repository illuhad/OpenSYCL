# This module serves as a compatibility layer to support cmake versions < 3.17
# (which do not support the replacement for the deprecated FindCUDA,
# FindCUDAToolkit) and cmake versions >= 3.17 (which do have FindCUDAToolkit).

# Remove our own modules from cmake's module search path to prevent
# it from recursively calling this module when using find_package(CUDA) below.
set(OLD_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(REMOVE_ITEM CMAKE_MODULE_PATH "${ACPP_SOURCE_ROOT}/cmake/")

if (CMAKE_VERSION VERSION_LESS 3.17)
   find_package(CUDA QUIET)

   if (CUDA_FOUND)
      find_library(CUDA_DRIVER_LIBRARY
                   cuda
                   HINTS
                   ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
                   ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 REQUIRED)

      message(STATUS "Found CUDA version ${CUDA_VERSION} in ${CUDA_TOOLKIT_ROOT_DIR}")

      set(CUDA_LIBS "")
      list(APPEND CUDA_LIBS ${CUDA_LIBRARIES})
      list(APPEND CUDA_LIBS ${CUDA_DRIVER_LIBRARY})
   endif()
else()
   if (DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDAToolkit_ROOT)
      set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
   endif()

   find_package(CUDAToolkit QUIET)

   if (CUDAToolkit_FOUND)
      # CMake version < 3.18 does not define CUDAToolkit_LIBRARY_ROOT, and in some
      # cases CMake does not define this variable even for CMake version >= 3.18.
      # In these cases we define it ourselves by taking the parent directory of
      # the CUDAToolkit_LIBRARY_DIR directory and check if this is the right directory.
      if (CMAKE_VERSION VERSION_LESS 3.18 OR NOT CUDAToolkit_LIBRARY_ROOT)
         get_filename_component(CUDAToolkit_LIBRARY_ROOT ${CUDAToolkit_LIBRARY_DIR} DIRECTORY)
         if(WIN32)
            # Nested in architecture specific folder.
            get_filename_component(CUDAToolkit_LIBRARY_ROOT ${CUDAToolkit_LIBRARY_ROOT} DIRECTORY)
         endif()

         if (NOT EXISTS "${CUDAToolkit_LIBRARY_ROOT}/nvvm")
            message(WARNING "CUDAToolkit_LIBRARY_ROOT ${CUDAToolkit_LIBRARY_ROOT} does not point to the correct directory, try setting it manually. Detected CUDA installation cannot be used.")
            set(CUDAToolkit_FOUND FALSE)
         endif()
      endif()
   endif()

   if (CUDAToolkit_FOUND)
      message(STATUS "Found CUDA version ${CUDAToolkit_VERSION} in ${CUDAToolkit_LIBRARY_ROOT}")

      set(CUDA_FOUND TRUE)
      set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_LIBRARY_ROOT})
      set(CUDA_LIBS CUDA::cudart CUDA::cuda_driver)
   endif()
endif()

# Restore the cmake module path that contains our modules
set(CMAKE_MODULE_PATH OLD_CMAKE_MODULE_PATH)
