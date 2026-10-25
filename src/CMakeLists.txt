add_library(ddla_lib SHARED)


set_target_properties(ddla_lib PROPERTIES
  OUTPUT_NAME ${LIB_NAME}
  VERSION ${CMAKE_PROJECT_VERSION}
  SOVERSION ${CMAKE_PROJECT_SOVERSION_MAJOR}
)
message(STATUS "Building ddla library version ${CMAKE_PROJECT_VERSION}")

# The library's sources are registered one CMakeLists.txt per feature
# subdirectory (BLAS/, Cholesky/, Eigensolver/, LU/, PBLAS/, QR/, SVD/,
# vbatched/); each of those files selects its own sources and their per-file
# compile properties (LANGUAGE, -fmad=false, ...).  This file keeps the target
# itself, the sources that sit directly in src/, and the per-target compile
# definitions, include paths, link libraries and install rules.
#
# Source-file properties are looked up in the directory of the TARGET, so the
# subdirectory files must set theirs with TARGET_DIRECTORY ddla_lib
# (CMake >= 3.18); a property set without it in a subdirectory is silently
# ignored for this target.

# A CPU-only build (DDLA_USE_CPU without a GPU backend) leaves the GPU-kernel
# translation units out; the subdirectories read ddla_cpu_only_build for that.
# The files that keep a GPU kernel next to their CPU arm (LU/getrf_nopiv.cpp,
# PBLAS/pdam.cpp, PBLAS/pgeadd.cpp, Cholesky/potrf_bottom_right_transform.cpp)
# are always listed and compile as plain C++ in a CPU-only build: all kernel
# code is behind DDLA_USE_CUDA / DDLA_USE_HIP.  transport_block.cpp is here
# because the CPU pgemm and the triangular solves call the unified
# transport_block<DdlaBackend::CPU,T>.
if(DDLA_USE_CPU AND NOT DDLA_DUAL_BUILD)
    set(ddla_cpu_only_build ON)
else()
    set(ddla_cpu_only_build OFF)
endif()

# ---------------------------------------------------------------------------
# Host OpenMP kernels of the CPU eigensolver path (src/Eigensolver/eig_cpu_kernels.cpp).  That ONE translation unit
# is compiled by the plain host compiler in every build (never by nvcc/hipcc) with -ffp-contract=off, so
# its AVX2/AVX-512 function clones round exactly like the scalar reference.  With OpenMP found it gets
# -fopenmp and libgomp is linked PRIVATE (the shared library records the NEEDED entry); without OpenMP
# the kernels run serially with identical results.  Pair a threaded run with an OpenMP-built BLAS
# (OpenBLAS USE_OPENMP=1): a pthreads BLAS inside OMP_NUM_THREADS ranks oversubscribes.
# ---------------------------------------------------------------------------
option(DDLA_USE_OPENMP "OpenMP threads for the host kernels of the CPU eigensolver" ON)
set(_ddla_cpu_kernel_opts "-ffp-contract=off")
if(DDLA_USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        list(APPEND _ddla_cpu_kernel_opts ${OpenMP_CXX_FLAGS})
        target_link_libraries(ddla_lib PRIVATE ${OpenMP_CXX_LIBRARIES})
        message(STATUS "ddla: OpenMP host kernels enabled (${OpenMP_CXX_FLAGS})")
    else()
        message(WARNING "ddla: OpenMP not found; the CPU eigensolver host kernels run serially")
    endif()
endif()
# Eigensolver/hb2st_sink_host.cpp (the CPU accumulation sink: tile kernels with the GPU's explicit fma roundings, planner, finish) is plain host
# C++ for the same reason: -ffp-contract=off keeps the products rounded, and the target_clones attributes would not survive nvcc.
# BLAS/geam.cpp (the OpenMP CPU arm of ddla::geam) and BLAS/gemmBatched.cpp (its CPU arm runs one BLAS gemm per OpenMP thread) are plain host C++
# for the same reason and take the same flags; the Eigensolver and BLAS CMakeLists attach them to their files.

# Sources that sit directly in src/.
target_sources(ddla_lib PRIVATE
    write_matrix.cpp
    random_generate.cpp
    matrix_transfer.cpp
    transport_block.cpp
    ddla_handle_t.cpp
    ddla_desc.cpp
)
if(NOT ddla_cpu_only_build)
    # GPU copy kernels of ddla::matrix_transfer (matrix_transfer.cpp keeps the
    # distributed driver and the CPU arm).
    target_sources(ddla_lib PRIVATE matrix_transfer_kernels.cpp)
endif()
if(DDLA_USE_CUDA)
    set_source_files_properties(matrix_transfer_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(matrix_transfer_kernels.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP)
endif()

add_subdirectory(BLAS)
add_subdirectory(Cholesky)
add_subdirectory(Eigensolver)
add_subdirectory(LU)
add_subdirectory(PBLAS)
add_subdirectory(QR)
add_subdirectory(SVD)
add_subdirectory(vbatched)

# ---------------------------------------------------------------------------
# Per-target compile definitions (moved from directory-global)
# ---------------------------------------------------------------------------
if(DDLA_USE_CUDA)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_CUDA)
endif()
if(DDLA_USE_HIP)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_HIP __HIP_PLATFORM_AMD__ HIPBLAS_V1)
endif()
if(DDLA_USE_CPU AND NOT DDLA_DUAL_BUILD)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_CPU)
endif()
if(DDLA_USE_CCL)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_CCL)
endif()
if(DDLA_USE_GPU_CPU_TUNNEL)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_GPU_CPU_TUNNEL)
endif()
if(DDLA_USE_DEBUG)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_DEBUG)
endif()

if(DDLA_USE_CUDA)
    # Mirror of the HIP block below: CXX-language TUs (BLAS/gemm.cpp, BLAS/trmv.cpp, the
    # tests, ...) include vendor CUDA headers through ddla_connector.h
    # (nccl.h -> cuda_runtime.h).  Only LANGUAGE CUDA sources get the toolkit
    # include path from nvcc itself.  PUBLIC: test targets also include
    # ddla_connector.h from their own sources.
    target_include_directories(ddla_lib PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()

if(DDLA_USE_HIP)
    # CXX-language TUs (BLAS/gemm.cpp, BLAS/scal.cpp, the tests, ...) include vendor HIP
    # headers through ddla_connector.h. Only the LANGUAGE HIP sources get ROCm
    # include paths from hipcc itself, so point every other compile at
    # ${ROCM_PATH}/include explicitly (the df_dcu module env does this via
    # CPATH; the GitHub rocm container does not). PUBLIC: test targets also
    # include ddla_connector.h from their own sources.
    target_include_directories(ddla_lib PUBLIC
        "${ROCM_PATH}/include"
    )
    target_link_directories(ddla_lib PUBLIC
        "${ROCM_PATH}/lib"
        "${ROCM_PATH}/lib64"
        "${ROCM_PATH}/dcc/lib"
    )
    set_target_properties(ddla_lib PROPERTIES
        BUILD_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib64;${ROCM_PATH}/dcc/lib"
        INSTALL_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib64;${ROCM_PATH}/dcc/lib"
    )
endif()
target_link_libraries(ddla_lib PUBLIC ${math_libs})

install(TARGETS ddla_lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*.h.in" EXCLUDE
)
