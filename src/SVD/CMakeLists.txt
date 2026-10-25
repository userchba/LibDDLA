# Distributed SVD (p?gesvd: bidiagonalization + divide-and-conquer on the
# bidiagonal).  Only pgebrd_kernels.cpp is GPU-only; the rest compiles on
# every backend.
target_sources(ddla_lib PRIVATE
    pbdsdc_dc.cpp
    pgebrd.cpp
    punmbr.cpp
    pgesvd.cpp
)
if(NOT ddla_cpu_only_build)
    target_sources(ddla_lib PRIVATE pgebrd_kernels.cpp)
endif()
# SVD/pgebrd_kernels.cpp carries the exact LAPACK scalar ports (larfg/dladiv)
# and the robust norms; only a contraction-free compile keeps them
# reassociation-free (see QR/CMakeLists.txt for the same flag on
# pgeqrf_kernels.cpp).  The kernels are memory-bound, so the cost is not
# measurable.
# Source properties are looked up in the directory of ddla_lib (src/), so set
# them there explicitly (TARGET_DIRECTORY, CMake >= 3.18).  pbdsdc_dc.cpp
# carries device code but keeps the normal contraction (only the kernels file
# is rounded reassociation-free).
if(DDLA_USE_CUDA)
    set_source_files_properties(pbdsdc_dc.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(pgebrd_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA
        COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-fmad=false>")
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(pbdsdc_dc.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP)
    set_source_files_properties(pgebrd_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP
        COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:HIP,Clang>:-ffp-contract=off>")
endif()
