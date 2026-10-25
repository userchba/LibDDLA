# Distributed and serial LU factorization and the triangular solves built on
# it.  Only pgetf2_kernels.cpp is GPU-only; the rest compiles on every backend.
target_sources(ddla_lib PRIVATE
    getrf.cpp
    laswp.cpp
    pgetf2.cpp
    lu_panel.cpp
    pgetrf.cpp
    pgetrf_bpiv.cpp
    pgetrf_nopiv.cpp
    getrf_nopiv.cpp
    pgetrs.cpp
    pgetrs_nopiv.cpp
    pgetrs_bpiv.cpp
    pgesv.cpp
    pgesv_nopiv.cpp
    pgesv_bpiv.cpp
)
if(NOT ddla_cpu_only_build)
    target_sources(ddla_lib PRIVATE pgetf2_kernels.cpp)
endif()
# getrf_nopiv.cpp keeps a GPU kernel next to its CPU arm: plain C++ in a
# CPU-only build, device language under CUDA/HIP.  Source properties are
# looked up in the directory of ddla_lib (src/), so set them there explicitly
# (TARGET_DIRECTORY, CMake >= 3.18).
if(DDLA_USE_CUDA)
    set_source_files_properties(getrf_nopiv.cpp pgetf2_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(getrf_nopiv.cpp pgetf2_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP)
endif()
