# Distributed QR (p?geqrf) and its host LAPACK arms.  Only pgeqrf_kernels.cpp
# is GPU-only; the rest compiles on every backend.
target_sources(ddla_lib PRIVATE
    geqrf.cpp
    larft.cpp
    pgeqrf.cpp
    pgeqr2_panel.cpp
    qr_larfb.cpp
)
if(NOT ddla_cpu_only_build)
    target_sources(ddla_lib PRIVATE pgeqrf_kernels.cpp)
endif()
# pgeqrf_kernels.cpp (index maps and the T recurrence of the QR panel) and
# SVD/pgebrd_kernels.cpp carry the exact LAPACK scalar ports (larfg/dladiv) and
# the robust norms, and only a contraction-free compile keeps them
# reassociation-free (the grid tests also compare the GPU scalar path against
# the host LAPACK port).  The kernels are memory-bound, so the cost is not
# measurable.
# Source properties are looked up in the directory of ddla_lib (src/), so set
# them there explicitly (TARGET_DIRECTORY, CMake >= 3.18).
if(DDLA_USE_CUDA)
    set_source_files_properties(pgeqrf_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA
        COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-fmad=false>")
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(pgeqrf_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP
        COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:HIP,Clang>:-ffp-contract=off>")
endif()
