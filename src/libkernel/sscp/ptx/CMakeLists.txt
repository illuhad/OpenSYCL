
if(WITH_LLVM_TO_PTX)
  if(WIN32)
    set(EXTRA_FLAGS -fshort-wchar)
  endif()
  libkernel_generate_bitcode_target(
      TARGETNAME ptx 
      TRIPLE nvptx64-nvidia-cuda
      SOURCES 
      atomic.cpp
      barrier.cpp
      core.cpp
      half.cpp
      integer.cpp
      print.cpp
      relational.cpp
      math.cpp
      native.cpp
      localmem.cpp
      subgroup.cpp
      shuffle.cpp
      reduction.cpp
      broadcast.cpp
      scan_inclusive.cpp
      scan_exclusive.cpp
      collpredicate.cpp
      ADDITIONAL_ARGS -Xclang -target-feature -Xclang +sm_60 ${EXTRA_FLAGS})
endif()
