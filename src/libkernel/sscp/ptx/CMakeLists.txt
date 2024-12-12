
if(WITH_LLVM_TO_PTX)
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
      reduce.cpp
      broadcast.cpp
      scan_inclusive.cpp
      scan_exclusive.cpp
      collpredicate.cpp
      ADDITIONAL_ARGS -Xclang -target-feature -Xclang +sm_60)
endif()
