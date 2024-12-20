add_executable(bruteforce_nbody bruteforce_nbody.cpp)
add_sycl_to_target(TARGET bruteforce_nbody SOURCES bruteforce_nbody.cpp)
install(TARGETS bruteforce_nbody
        RUNTIME DESTINATION share/AdaptiveCpp/examples/)
