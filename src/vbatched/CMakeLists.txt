# Variable-batch GEMM device cores (ddla::gemm_vbatched).  They exist only
# with a GPU backend and are always compiled by the device compiler; the BLAS
# front end (BLAS/gemm_vbatched.cpp) is listed in BLAS/CMakeLists.txt.
# Source properties are looked up in the directory of ddla_lib (src/), so set
# them there explicitly (TARGET_DIRECTORY, CMake >= 3.18).
if(DDLA_USE_CUDA)
    target_sources(ddla_lib PRIVATE
        sgemm_vbatched_core.cpp
        dgemm_vbatched_core.cpp
        cgemm_vbatched_core.cpp
        zgemm_vbatched_core.cpp
    )
    set_source_files_properties(
        sgemm_vbatched_core.cpp
        dgemm_vbatched_core.cpp
        cgemm_vbatched_core.cpp
        zgemm_vbatched_core.cpp
        TARGET_DIRECTORY ddla_lib
        PROPERTIES
        LANGUAGE CUDA
    )
endif()
if(DDLA_USE_HIP)
    target_sources(ddla_lib PRIVATE
        sgemm_vbatched_core.cpp
        dgemm_vbatched_core.cpp
        cgemm_vbatched_core.cpp
        zgemm_vbatched_core.cpp
    )
    set_source_files_properties(
        sgemm_vbatched_core.cpp
        dgemm_vbatched_core.cpp
        cgemm_vbatched_core.cpp
        zgemm_vbatched_core.cpp
        TARGET_DIRECTORY ddla_lib
        PROPERTIES
        LANGUAGE HIP
    )
endif()
