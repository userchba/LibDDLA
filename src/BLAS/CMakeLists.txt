# BLAS wrappers (the unified ddla::gemm/herk/... front ends).  The source set
# is identical on every backend; only trtrm.cpp, gemm_mixed.cpp and
# gemm_vbatched.cpp carry device code and switch LANGUAGE under CUDA/HIP.
target_sources(ddla_lib PRIVATE
    gemm.cpp
    gemm_mixed.cpp
    herk.cpp
    her2k.cpp
    hemv.cpp
    hemm.cpp
    her2.cpp
    gemv.cpp
    trmv.cpp
    nrm2.cpp
    dot.cpp
    trsm.cpp
    syrk.cpp
    trmm.cpp
    trtrm.cpp
    gemmBatched.cpp
    scal.cpp
    geam.cpp
    swap.cpp
    axpy.cpp
    iamax.cpp
    geru.cpp
    gerc.cpp
)

# Source properties are looked up in the directory of the target that
# compiles the source, so every property below must carry
# TARGET_DIRECTORY ddla_lib (CMake >= 3.18) to reach src/CMakeLists.txt.
#
# BLAS/geam.cpp (the OpenMP CPU arm of ddla::geam) and BLAS/gemmBatched.cpp (its CPU arm runs one BLAS gemm per OpenMP thread) are plain host C++
# rounded with -ffp-contract=off, like the eigensolver host kernels
# (src/CMakeLists.txt computes ${_ddla_cpu_kernel_opts}).
set_source_files_properties(geam.cpp gemmBatched.cpp TARGET_DIRECTORY ddla_lib
    PROPERTIES COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")

if(DDLA_USE_CUDA)
    target_sources(ddla_lib PRIVATE gemm_vbatched.cpp)
    set_source_files_properties(trtrm.cpp gemm_mixed.cpp gemm_vbatched.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    target_sources(ddla_lib PRIVATE gemm_vbatched.cpp)
    set_source_files_properties(trtrm.cpp gemm_mixed.cpp gemm_vbatched.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP)
endif()
