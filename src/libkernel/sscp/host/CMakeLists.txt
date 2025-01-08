if(WITH_LLVM_TO_HOST)
  if(NOT DEFINED LLVM_TARGET_TRIPLE)
    set(LLVM_TARGET_TRIPLE ${TARGET_TRIPLE})
  endif()

  set(HOST_LIBKERNEL_BITCODE_SOURCES
    atomic.cpp
    barrier.cpp
    core.cpp
    integer.cpp
    half.cpp
    math.cpp
    native.cpp
    shuffle.cpp
    print.cpp
    relational.cpp
    localmem.cpp
    subgroup.cpp
    reduction.cpp
    broadcast.cpp
    scan_inclusive.cpp
    scan_exclusive.cpp
    collpredicate.cpp)

  libkernel_generate_bitcode_target(
      TARGETNAME host 
      TRIPLE ${LLVM_TARGET_TRIPLE}
      SOURCES ${HOST_LIBKERNEL_BITCODE_SOURCES})

  libkernel_generate_bitcode_target(
      TARGETNAME host-fast
      TRIPLE ${LLVM_TARGET_TRIPLE}
      SOURCES ${HOST_LIBKERNEL_BITCODE_SOURCES}
      ADDITIONAL_ARGS -ffast-math)
endif()
