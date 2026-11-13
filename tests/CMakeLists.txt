# ---------------------------------------------------------------------------
# Apply per-target backend connector definitions.
# In CPU-only mode only DDLA_USE_CPU+DEBUG are relevant.
# In GPU/dual mode the selected GPU vendor + CCL/DEBUG are required
# because test sources include ddla_connector.h.
# ---------------------------------------------------------------------------
function(apply_test_connector_defs target)
    if(DDLA_USE_CUDA)
        target_compile_definitions(${target} PRIVATE DDLA_USE_CUDA)
    elseif(DDLA_USE_HIP)
        target_compile_definitions(${target}
            PRIVATE DDLA_USE_HIP __HIP_PLATFORM_AMD__)
    else()
        target_compile_definitions(${target} PRIVATE DDLA_USE_CPU)
    endif()
    if(DDLA_USE_CCL)
        target_compile_definitions(${target} PRIVATE DDLA_USE_CCL)
    endif()
    if(DDLA_USE_DEBUG)
        target_compile_definitions(${target} PRIVATE DDLA_USE_DEBUG)
    endif()
endfunction()

# Default number of MPI ranks for ctest-registered MPI tests.  Override with
# -DDDLA_TEST_NP=N at configure time (AGENTS.md documents `mpirun -np 4`).
set(DDLA_TEST_NP 4 CACHE STRING "MPI ranks used by ctest-registered tests")

# add_test_executable with automatic connector definitions.  Tests are MPI
# programs, so each is registered under mpirun (never run the binary bare);
# GPU tests get the "gpu" label, everything else "cpu", so CI can select
# with `ctest -L cpu`.
function(add_test_executable test_name)
    # Optional second argument overrides the MPI rank count for tests with a
    # fixed rank requirement (e.g. test_potrf_bottom_right needs 1).
    # Defaults to DDLA_TEST_NP.
    set(_np "${DDLA_TEST_NP}")
    if(ARGC GREATER 1)
        set(_np "${ARGV1}")
    endif()
    set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp")
    # A third argument reuses a fixture under a different compile-time mode.
    if(ARGC GREATER 2)
        set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/${ARGV2}")
    endif()
    if(EXISTS "${test_source}")
        add_executable(${test_name} "${test_source}")
        target_link_libraries(${test_name} PRIVATE ddla_lib)
        apply_test_connector_defs(${test_name})
        add_test(NAME ${test_name}_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}>
                    ${MPIEXEC_POSTFLAGS})
        set_tests_properties(${test_name}_mpi PROPERTIES PROCESSORS ${_np})
        if(DDLA_HAS_GPU)
            # GPU jobs grant ~6 CPUs per GPU; without these caps every rank
            # spawns node-wide OpenMP/OpenBLAS pools (4 ranks x 16 threads on
            # 16 CPUs) and the barrier-heavy host stages of the eigensolver
            # tests collapse ~40x (measured 2026-10-24: phb2st 837 s -> 35 s,
            # pheevd >2.5 h timeout -> 27 s).
            set_tests_properties(${test_name}_mpi PROPERTIES LABELS gpu
                ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        else()
            set_tests_properties(${test_name}_mpi PROPERTIES LABELS cpu)
        endif()
    else()
        message(STATUS "Skipping missing test source: ${test_name}.cpp")
    endif()
endfunction()

# add_bench_executable: same as add_test_executable but does not register
# with ctest -- heavy benchmarks should not run as part of the regression
# gate (F1: ctest should be a pass/fail signal, not a benchmark run).
function(add_bench_executable test_name)
    set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp")
    if(EXISTS "${test_source}")
        add_executable(${test_name} ${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE ddla_lib)
        apply_test_connector_defs(${test_name})
    else()
        message(STATUS "Skipping missing test source: ${test_name}.cpp")
    endif()
endfunction()

enable_testing()
# This client must compile without private connector/vendor definitions.
add_executable(test_public_handle_dispatch test_public_handle_dispatch.cpp)
target_link_libraries(test_public_handle_dispatch PRIVATE ddla_lib)
add_test(NAME test_public_handle_dispatch_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_public_handle_dispatch>
            ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_public_handle_dispatch_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP})
if(DDLA_HAS_GPU)
    set_tests_properties(test_public_handle_dispatch_mpi PROPERTIES LABELS gpu
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
else()
    set_tests_properties(test_public_handle_dispatch_mpi PROPERTIES LABELS cpu)
endif()
set(DDLA_PUBLIC_CHECK_INCLUDES "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
include(${CMAKE_CURRENT_SOURCE_DIR}/public_api_compile_checks.cmake)
ddla_check_public_scalar_api()
add_test_executable(test_api_grid_punmtr)
if(TARGET test_api_grid_punmtr)
    target_link_libraries(test_api_grid_punmtr PRIVATE ${CMAKE_DL_LIBS})
endif()

# Check/query/borrowed-execution tests use an optional executable-local
# diagnostic preload; the numerical library has no tracing hooks.
add_test_executable(test_comm)
add_test_executable(test_runtime_communication)
target_link_libraries(test_comm PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(test_runtime_communication PRIVATE ${CMAKE_DL_LIBS})
add_test_executable(test_workspace_core 1)
add_test_executable(test_workspace_write_matrix 1)
add_test_executable(test_workspace_transfer)
add_test_executable(test_workspace_pblas)
add_test_executable(test_workspace_factorizations)
add_test_executable(test_workspace_trace 1)
if(TARGET test_workspace_trace)
    set_tests_properties(test_workspace_trace_mpi PROPERTIES SKIP_RETURN_CODE 77)
endif()
add_test_executable(test_workspace_spectral)
add_test_executable(test_workspace_solvers)
add_test_executable(test_workspace_pgesvd)
add_test_executable(test_pstedc)
target_link_libraries(test_pstedc PRIVATE ${CMAKE_DL_LIBS})
add_test_executable(test_workspace_pstedc)
add_test_executable(test_api_grid_phetrd)
target_link_libraries(test_api_grid_phetrd PRIVATE ${CMAKE_DL_LIBS})
add_test_executable(test_workspace_one_stage)
add_test_executable(test_workspace_history)
add_test_executable(test_workspace_alias_storage)
add_test_executable(test_workspace_pheevd)
add_test_executable(test_workspace_householder)
add_test_executable(test_workspace_host_accum)
add_test_executable(test_workspace_cpu_replay)
if(DDLA_HAS_GPU)
    add_test_executable(test_workspace_gpu_replay)
endif()
foreach(_workspace_test test_workspace_write_matrix test_workspace_core test_workspace_transfer test_workspace_spectral test_workspace_trace
                        test_workspace_pblas test_workspace_factorizations test_workspace_solvers test_workspace_pgesvd
                        test_workspace_pstedc test_workspace_one_stage test_workspace_history
                        test_workspace_alias_storage test_workspace_householder test_workspace_host_accum
                        test_workspace_cpu_replay test_workspace_gpu_replay test_workspace_pheevd)
    if(TARGET ${_workspace_test})
        target_link_libraries(${_workspace_test} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endforeach()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/workspace_trace.cpp")
    add_library(workspace_trace SHARED EXCLUDE_FROM_ALL workspace_trace.cpp)
    apply_test_connector_defs(workspace_trace)
    target_compile_definitions(workspace_trace PRIVATE DDLA_USE_CPU)
    target_include_directories(workspace_trace PRIVATE $<TARGET_PROPERTY:ddla_lib,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(workspace_trace PRIVATE MPI::MPI_CXX ${CMAKE_DL_LIBS})
endif()
# Without the preload workspace_test::Scope is a silent no-op, so every
# "no allocation in run" assertion would pass vacuously; REQUIRED makes a
# missing preload fail instead.
if(TARGET workspace_trace)
    foreach(_traced_test test_workspace_core test_workspace_write_matrix test_workspace_transfer
                         test_workspace_pblas test_workspace_factorizations test_workspace_spectral
                         test_workspace_solvers test_workspace_trace test_workspace_pgesvd
                         test_pstedc test_workspace_pstedc test_workspace_one_stage
                         test_workspace_history test_workspace_alias_storage test_workspace_householder
                         test_workspace_host_accum test_workspace_cpu_replay test_workspace_gpu_replay
                         test_workspace_pheevd)
        if(TARGET ${_traced_test})
            add_dependencies(${_traced_test} workspace_trace)
            set_tests_properties(${_traced_test}_mpi PROPERTIES ENVIRONMENT
                "LD_PRELOAD=$<TARGET_FILE:workspace_trace>;DDLA_WORKSPACE_TRACE_REQUIRED=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
        endif()
    endforeach()
endif()
if(DDLA_USE_CUDA)
    if(TARGET test_workspace_trace)
        target_link_libraries(test_workspace_trace PRIVATE CUDA::cudart)
    endif()
    add_test_executable(test_vendor_workspace_query 1)
    if(TARGET test_vendor_workspace_query)
        target_link_libraries(test_vendor_workspace_query PRIVATE ${CUSOLVER_LIBRARY} ${CUBLAS_LIBRARY} CUDA::cudart ${CMAKE_DL_LIBS})
    endif()
endif()

add_test_executable(test_backend_gemm)
add_test_executable(test_backend_gemm_mixed 1)
add_bench_executable(benchmark_gemm_mixed)
add_test_executable(test_backend_vector)
if(TARGET test_backend_vector)
    target_link_libraries(test_backend_vector PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(test_backend_vector PROPERTIES
        CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_options(test_backend_vector PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-pedantic-errors>)
endif()
add_test_executable(test_pgemm_branches)
add_test_executable(test_pgemm_mixed)
add_test_executable(test_gemmt)
add_test_executable(test_pgeadd_indexed)
add_test_executable(test_pgeadd_direct)
add_bench_executable(benchmark_pgemm_branches)
add_bench_executable(benchmark_phemm_pgemm)
add_test_executable(test_backend_level1)
add_test_executable(test_backend_dotc 1)
# BLAS-3 families that had no numerical coverage before the wrapper
# unification (trsm, syrk, trmm, gemmBatched), plus per-family handle and
# backend-mismatch rejection.
add_test_executable(test_backend_blas3)
add_test_executable(test_backend_trtrm 1)
add_bench_executable(benchmark_trtrm)
add_bench_executable(benchmark_products)
add_test_executable(test_backend_geam)
# LAPACK-level wrappers (getrf, potrf, laswp) and swap: CPU arm = host LAPACK, GPU arm = cuSOLVER / hipSOLVER.
add_test_executable(test_backend_lapack)
# Distributed operations dispatch from the supplied handle. The same scalar-only
# calls exercise CPU and GPU handles in dual builds; explicitly private tests
# retain implementation mismatch checks through backend_api.h.
add_test_executable(test_backend_solvers)
add_test_executable(test_api_grid_pgeadd_panels)
add_test_executable(test_api_grid_ptrtrs)
add_test_executable(test_api_grid_ptrmm)
add_test_executable(test_api_grid_phemm)
add_test_executable(test_api_grid_pblas_structured)
add_bench_executable(benchmark_pblas_structured)
add_test_executable(test_api_grid_ptrtrm)
add_test_executable(test_api_grid_plapiv)
add_test_executable(test_api_grid_pswap)
add_test_executable(test_api_grid_pgetf2)
add_test_executable(test_lu_gather)
add_test_executable(test_matrix_transfer)
foreach(_parallel_transfer_test test_matrix_transfer test_workspace_transfer)
    if(TARGET ${_parallel_transfer_test})
        set_property(TEST ${_parallel_transfer_test}_mpi APPEND PROPERTY ENVIRONMENT
            "DDLA_EIG_THREADS=2")
    endif()
endforeach()
add_test_executable(test_api_grid_pgetrf)
add_test_executable(test_api_grid_pgetrs)
add_test_executable(test_api_grid_pgesv)
add_test_executable(test_api_grid_pgetrf_nopiv)
add_test_executable(test_api_grid_pgetrs_nopiv)
add_test_executable(test_api_grid_pgesv_nopiv)
add_test_executable(test_api_grid_pgetrs_bpiv)
add_test_executable(test_api_grid_pgesv_bpiv)
add_test_executable(test_api_grid_pgetrf_bpiv)
add_test_executable(test_api_grid_subcomm_lu)
add_test_executable(test_api_grid_getrf_nopiv)
add_test_executable(test_api_grid_ppotrf)
add_test_executable(test_ppotrf_grids)
add_test_executable(test_api_grid_ppotrs)
add_test_executable(test_api_grid_pposv)
add_test_executable(test_api_grid_ppotrf_bottom_right)
add_test_executable(test_api_grid_submatrix)
add_test_executable(test_api_grid_pgeadd)
add_test_executable(test_api_grid_pdam)
add_test_executable(test_api_grid_scalapack_desc)
add_test_executable(test_random_generate)
add_test_executable(test_api_grid_pvector_redist)
add_test_executable(test_api_grid_phe2hb)
# phe2hb must ignore the pgeqrf tuning knob DDLA_QR_LOOKAHEAD: output bitwise equal with it set
add_test_executable(test_api_grid_phe2hb_knobs)
add_test_executable(test_api_grid_phb2st)
add_test_executable(test_api_grid_pheevd)
add_test_executable(test_pheevd_one_stage)
add_test_executable(test_pheevd_nev)
add_test_executable(test_pheevd_method)
add_test_executable(test_heevd)
add_test_executable(test_pheevd_dispatch)
if(TARGET test_pheevd_dispatch)
    set_target_properties(test_pheevd_dispatch PROPERTIES ENABLE_EXPORTS ON)
endif()
add_test_executable(test_pheevd_workspace)
add_test_executable(test_pheevd_finish_panels)
if(NOT DDLA_HAS_GPU)
    add_test_executable(test_phb2st_no_records)
endif()
# the streaming wave schedule against the offline one (host only; needs the internal header)
add_test_executable(test_hb2st_wave_gen 1)
# pheevd again with the chase-window knobs that force the record RING / ELASTIC ring to wrap and to be left early (tiny rings, tiny
# stager chunks): eigenvalue-only runs record into a ring of 1 step, eigenvector runs on GPU use an elastic ring of 2 chunks.  Same
# executable, same checks; the knobs are ignored where they do not apply.
if(TARGET test_api_grid_pheevd)
    add_test(NAME test_api_grid_pheevd_ring_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_api_grid_pheevd> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_api_grid_pheevd_ring_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP}
        ENVIRONMENT "DDLA_EIG_CHASE_RING_RECORDS=1;DDLA_EIG_CHASE_ELASTIC=2;DDLA_EIG_ACCUM_CHUNK=2048;DDLA_EIG_ACCUM_RING_SLOTS=4")
    if(DDLA_HAS_GPU)
        set_tests_properties(test_api_grid_pheevd_ring_mpi PROPERTIES LABELS gpu
            ENVIRONMENT "DDLA_EIG_CHASE_RING_RECORDS=1;DDLA_EIG_CHASE_ELASTIC=2;DDLA_EIG_ACCUM_CHUNK=2048;DDLA_EIG_ACCUM_RING_SLOTS=4;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    else()
        set_tests_properties(test_api_grid_pheevd_ring_mpi PROPERTIES LABELS cpu)
    endif()
endif()
# CPU handles (CPU-only builds): stage 4 has two engines -- the accumulation of Q_band during the chase (hb2st_sink_host.cpp) and the classic
# replay (the node-shared wave-ordered stream).  The default 'auto' picks the replay on the small test grids, so both are forced here:
# the accumulation with tiny blocks and panels (the block ring, the block slots and the k-panels wrap; many block boundaries) and the replay.
if(TARGET test_api_grid_pheevd AND NOT DDLA_HAS_GPU)
    add_test(NAME test_api_grid_pheevd_cpuaccum_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_api_grid_pheevd> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_api_grid_pheevd_cpuaccum_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP} LABELS cpu
        ENVIRONMENT "DDLA_EIG_CPU_ACCUM=1;DDLA_EIG_HOST_ACCUM_BLOCK=4096;DDLA_EIG_HOST_ACCUM_SLOTS=2;DDLA_EIG_HOST_FINISH_PANEL=64")
    add_test(NAME test_api_grid_pheevd_cpureplay_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_api_grid_pheevd> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_api_grid_pheevd_cpureplay_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP} LABELS cpu
        ENVIRONMENT "DDLA_EIG_CPU_ACCUM=0")
endif()
# Stage 0 of pheevd, in isolation: full Hermitian matrix from the stored triangle (poisoned opposite triangle).
add_test_executable(test_api_grid_pherm_complete)
add_test_executable(test_api_grid_pgeqrf)
add_test_executable(test_api_grid_pgeqrs)
# Host unit probe for the QR robust-norm fold/scale math (no GPU, no grid
# work); the math is host code compiled into every backend, so one cpu label
# registration is enough and the gpu label keeps gpu ctest fast.
add_test_executable(test_qr_norm_fold)
if(TARGET test_qr_norm_fold)
    set_tests_properties(test_qr_norm_fold_mpi PROPERTIES LABELS cpu)
endif()

# The pgeqrf grid test's bitwise-repeat poisoning needs glibc's malloc
# perturbation to reach tcache-sized allocations: the per-thread tcache
# serves them WITHOUT perturbation unless the tunable disables it (glibc
# 2.28 on this cluster).
if(TARGET test_api_grid_pgeqrf)
    set_tests_properties(test_api_grid_pgeqrf_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0")
endif()
# QL factorization (pgeqlf) and the distinct QR/QL public entries, same
# bitwise-repeat poisoning requirement as the pgeqrf grid test.
add_test_executable(test_api_grid_pgeqrf_ql)
if(TARGET test_api_grid_pgeqrf_ql)
    set_tests_properties(test_api_grid_pgeqrf_ql_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0")
endif()
# the GLIBC_TUNABLES overrides replace the whole ENVIRONMENT list:
# re-append the GPU lane thread caps so the host-side panel work does not
# oversubscribe (measured 2026-10-24: pgeqrf ~16-thread pools, GPUs idle).
if(DDLA_HAS_GPU AND TARGET test_api_grid_pgeqrf)
    set_tests_properties(test_api_grid_pgeqrf_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()
if(DDLA_HAS_GPU AND TARGET test_api_grid_pgeqrf_ql)
    set_tests_properties(test_api_grid_pgeqrf_ql_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()
add_test_executable(test_api_grid_pgebrd)
add_test_executable(test_api_grid_pgesvd)
add_test_executable(test_api_grid_pbdsdc)

if(DDLA_HAS_GPU)
    add_test_executable(test_pzgemm)
    add_test_executable(test_pdam)
    add_test_executable(test_getrf_nopiv)
    add_test_executable(test_pgetrf_nopiv)
    add_test_executable(test_pgesv_nopiv)
    add_test_executable(test_sv_gemm_hermitian_positive)
    add_test_executable(test_gemm_vbatched)
    add_test_executable(test_api_grid_device_memory)
    add_test_executable(test_api_grid_pgemm)
    add_test_executable(test_potrf_bottom_right_block 1)
    add_test_executable(test_potrf_bottom_right 1)
    add_test_executable(test_ppotrf_real)
    add_test_executable(test_pgemm_debug)
    add_test_executable(test_pgemm_debug2)
    add_test_executable(benchmark_pgemm)
    add_bench_executable(bench_pzgemm)
    add_bench_executable(benchmark_pgetrf)
    add_bench_executable(benchmark_pgetf2)
    add_bench_executable(benchmark_ptrtrs)
    add_bench_executable(benchmark_solve_side)
    add_bench_executable(benchmark_solvers)
    add_bench_executable(benchmark_ppotrf_bottom_right)
    add_bench_executable(benchmark_plapiv)
endif()

# Eigensolver stage benchmark — backend-templated (both the GPU kernels and the
# OpenBLAS CPU arms exist), so it is registered for every configuration.  Heavy:
# never run by ctest, see add_bench_executable.
add_bench_executable(benchmark_eig_stages)

# Stage 3 alone (the distributed tridiagonal solve): timing, per-rank device memory under
# DDLA_EIG_PROFILE=1, residual/orthogonality at the timed size.  GPU handles only.  Heavy: never run by ctest.
add_bench_executable(benchmark_pstedc)
add_bench_executable(benchmark_pstedc_compare)
if(TARGET benchmark_pstedc_compare)
    find_package(Threads REQUIRED)
    target_link_libraries(benchmark_pstedc_compare PRIVATE Threads::Threads)
endif()

# The full eigensolver at scale: time, DDLA_EIG_PROFILE per-stage memory, residual/orthogonality at the timed
# size (--check).  Either backend.  Heavy: never run by ctest.
add_bench_executable(benchmark_pheevd)

# Distributed Householder QR benchmark — backend-templated (runs on the
# explicitly selected fixture backend).  Heavy: never run by ctest.
add_bench_executable(benchmark_pgeqrf)

# Distributed SVD benchmark (pgebrd / punmbr / pgesvd stage times, checked at
# the timed size) -- backend-neutral like benchmark_pgeqrf.  Heavy: never run
# by ctest.
add_bench_executable(benchmark_pgesvd)
add_bench_executable(benchmark_svd_compare)

# Backend handle test — works for all backends
add_test_executable(test_memory_tri)

add_executable(test_backend_handle test_backend_handle.cpp)
target_link_libraries(test_backend_handle PRIVATE ddla_lib)
apply_test_connector_defs(test_backend_handle)
add_test(NAME test_backend_handle_mpi4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_backend_handle>
            ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_backend_handle_mpi4 PROPERTIES PROCESSORS 4 LABELS cpu)
if(DDLA_HAS_GPU)
    set_tests_properties(test_backend_handle_mpi4 PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

add_executable(test_cpu_pgemm test_cpu_pgemm.cpp)
target_link_libraries(test_cpu_pgemm PRIVATE ddla_lib)
# CPU handles in every build; private headers use the library's vendor selection.
apply_test_connector_defs(test_cpu_pgemm)
add_test(NAME test_cpu_pgemm_mpi4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_cpu_pgemm> 2 2
            ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_cpu_pgemm_mpi4 PROPERTIES PROCESSORS 4 LABELS cpu)

set(DDLA_SCALAPACK_LIBRARY "" CACHE FILEPATH "Path to ScaLAPACK library for CPU benchmark")
if(DDLA_SCALAPACK_LIBRARY)
    if(NOT EXISTS "${DDLA_SCALAPACK_LIBRARY}")
        message(FATAL_ERROR "DDLA_SCALAPACK_LIBRARY set to nonexistent file: ${DDLA_SCALAPACK_LIBRARY}")
    endif()
    message(STATUS "Using ScaLAPACK: ${DDLA_SCALAPACK_LIBRARY}")
    set(_scalapack_lib "${DDLA_SCALAPACK_LIBRARY}")
else()
    find_library(SCALAPACK_LIBRARY NAMES scalapack)
    if(SCALAPACK_LIBRARY)
        message(STATUS "Found ScaLAPACK: ${SCALAPACK_LIBRARY}")
        set(_scalapack_lib "${SCALAPACK_LIBRARY}")
    else()
        message(STATUS "ScaLAPACK not found; skipping benchmark_cpu_pgemm_scalapack")
    endif()
endif()
if(_scalapack_lib)
    if(TARGET benchmark_pblas_structured)
        target_link_libraries(benchmark_pblas_structured PRIVATE "${_scalapack_lib}")
        target_compile_definitions(benchmark_pblas_structured PRIVATE DDLA_BENCH_SCALAPACK)
    endif()
    if(TARGET benchmark_svd_compare)
        target_link_libraries(benchmark_svd_compare PRIVATE "${_scalapack_lib}")
        target_compile_definitions(benchmark_svd_compare PRIVATE DDLA_BENCH_SCALAPACK)
    endif()
    set(bench_src "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_cpu_pgemm_scalapack.cpp")
    if(EXISTS "${bench_src}")
        add_executable(benchmark_cpu_pgemm_scalapack
            benchmark_cpu_pgemm_scalapack.cpp)
        target_link_libraries(benchmark_cpu_pgemm_scalapack PRIVATE
            ddla_lib "${_scalapack_lib}")
        apply_test_connector_defs(benchmark_cpu_pgemm_scalapack)
        # Heavy benchmark -- not added to CTest
    else()
        message(STATUS "Skipping missing benchmark source: benchmark_cpu_pgemm_scalapack.cpp")
    endif()

    set(bench_getrf_src "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_cpu_pgetrf_scalapack.cpp")
    if(EXISTS "${bench_getrf_src}")
        add_executable(benchmark_cpu_pgetrf_scalapack
            benchmark_cpu_pgetrf_scalapack.cpp)
        target_link_libraries(benchmark_cpu_pgetrf_scalapack PRIVATE
            ddla_lib "${_scalapack_lib}")
        apply_test_connector_defs(benchmark_cpu_pgetrf_scalapack)
        # Heavy benchmark -- not added to CTest
    else()
        message(STATUS "Skipping missing benchmark source: benchmark_cpu_pgetrf_scalapack.cpp")
    endif()
endif()

# Dual-build tests
if(DDLA_HAS_GPU)
    add_executable(test_dual_pgemm test_dual_pgemm.cpp)
    target_link_libraries(test_dual_pgemm PRIVATE ddla_lib)
    apply_test_connector_defs(test_dual_pgemm)
    add_test(NAME test_dual_pgemm_mpi4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_dual_pgemm> 2 2
                ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_dual_pgemm_mpi4 PROPERTIES PROCESSORS 4)
endif()

add_test_executable(test_phb2st_distributed)
if(DDLA_HAS_GPU)
    add_test_executable(test_pheevd_gpu_replay)
    if(TARGET test_pheevd_gpu_replay)
        target_link_libraries(test_pheevd_gpu_replay PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

add_test_executable(test_pstedc_scratch)
add_test_executable(test_hb2st_cpu_failures)
if(TARGET test_hb2st_cpu_failures)
    set_tests_properties(test_hb2st_cpu_failures_mpi PROPERTIES TIMEOUT 180 LABELS cpu)
    add_test(NAME test_hb2st_cpu_failures_mpi1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_hb2st_cpu_failures> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_hb2st_cpu_failures_mpi1 PROPERTIES TIMEOUT 180 PROCESSORS 1 LABELS cpu)
endif()

add_test_executable(test_pheevd_cpu_replay)
if(TARGET test_pheevd_cpu_replay)
    set_tests_properties(test_pheevd_cpu_replay_mpi PROPERTIES TIMEOUT 600 LABELS cpu)
endif()

# CPU handle even in dual builds; executable LAPACK interposition needs ELF/dl.
if(UNIX AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pstedc_threads.cpp")
    add_executable(test_pstedc_threads test_pstedc_threads.cpp)
    target_link_libraries(test_pstedc_threads PRIVATE ddla_lib ${CMAKE_DL_LIBS})
    target_compile_definitions(test_pstedc_threads PRIVATE DDLA_USE_CPU)
    set_target_properties(test_pstedc_threads PROPERTIES ENABLE_EXPORTS ON)
    foreach(_np 1 2)
        set(_name test_pstedc_threads)
        if(_np EQUAL 2)
            set(_name test_pstedc_threads_mpi2)
        endif()
        add_test(NAME ${_name}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_pstedc_threads> --threads 4 --blas-threads 4 ${MPIEXEC_POSTFLAGS})
        math(EXPR _processors "4 * ${_np}")
        set_tests_properties(${_name} PROPERTIES PROCESSORS ${_processors} LABELS cpu SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=4;OMP_DYNAMIC=false;OMP_PROC_BIND=false;OMP_WAIT_POLICY=PASSIVE;OPENBLAS_MAIN_FREE=1")
    endforeach()
endif()

add_test_executable(test_hb2st_householder_cpu)
add_test_executable(test_hb2st_householder_cpu_rows)
if(TARGET test_hb2st_householder_cpu_rows)
    target_link_libraries(test_hb2st_householder_cpu_rows PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(test_hb2st_householder_cpu_rows PROPERTIES ENABLE_EXPORTS ON)
endif()

add_test_executable(test_hb2st_householder_blas)
if(TARGET test_hb2st_householder_blas AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_hb2st_householder_blas PRIVATE -ffp-contract=off)
endif()

add_test_executable(test_hb2st_householder_paired_state)
if(TARGET test_hb2st_householder_paired_state AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_hb2st_householder_paired_state PRIVATE -ffp-contract=off)
    target_compile_definitions(test_hb2st_householder_paired_state PRIVATE DDLA_HH_PAIR_ENABLE_CLONES=1)
endif()

add_test_executable(test_phe2hb_upper_cpu)
if(TARGET test_phe2hb_upper_cpu AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_phe2hb_upper_cpu PRIVATE -ffp-contract=off)
endif()

# ---------------------------------------------------------------------------
# Fortran bindings (built only with DDLA_ENABLE_FORTRAN=ON): one MPI program
# exercising handle lifecycle, descriptors, memory copies and the pgemm /
# pgesv / ppotrf(+pdam,ppotrs) / pgeqrf paths of the ddla module for real and
# complex types, on the explicitly selected fixture backend exactly like the C++ grid
# tests (device buffers + ddla_memcpy on a GPU handle).
# ---------------------------------------------------------------------------
if(DDLA_ENABLE_FORTRAN)
    find_package(MPI REQUIRED COMPONENTS Fortran)
    add_executable(test_fortran_api test_fortran_api.f90)
    target_link_libraries(test_fortran_api PRIVATE ddla_fortran MPI::MPI_Fortran)
    add_executable(test_fortran_trmm test_fortran_trmm.f90)
    target_link_libraries(test_fortran_trmm PRIVATE ddla_fortran MPI::MPI_Fortran)
    add_test(NAME test_fortran_trmm_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_fortran_trmm> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_fortran_trmm_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP})
    add_executable(test_fortran_trtrm test_fortran_trtrm.f90)
    target_link_libraries(test_fortran_trtrm PRIVATE ddla_fortran MPI::MPI_Fortran)
    add_test(NAME test_fortran_trtrm_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_fortran_trtrm> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_fortran_trtrm_mpi PROPERTIES PROCESSORS 1)
    # mpif.h has no explicit interface: gfortran >= 10 flags the (benign) type
    # variety of mpi_allreduce buffers across call sites without this flag.
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        target_compile_options(test_fortran_api PRIVATE -fallow-argument-mismatch)
    endif()
    add_test(NAME test_fortran_api_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_fortran_api>
                ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_fortran_api_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP})
    if(DDLA_HAS_GPU)
        set_tests_properties(test_fortran_api_mpi PROPERTIES LABELS gpu
            ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    else()
        set_tests_properties(test_fortran_api_mpi PROPERTIES LABELS cpu)
    endif()
endif()

foreach(_hh_private_test test_hb2st_householder_kernel_state test_hb2st_householder_kernel_rows test_pheevd_householder_stage4)
    add_test_executable(${_hh_private_test})
    if(TARGET ${_hh_private_test})
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${_hh_private_test} PRIVATE -ffp-contract=off)
            target_compile_definitions(${_hh_private_test} PRIVATE DDLA_HH_PAIR_ENABLE_CLONES=1)
        endif()
    endif()
endforeach()

foreach(_allocation_probe_test test_pheevd_cpu_replay test_hb2st_householder_kernel_rows)
    if(TARGET ${_allocation_probe_test})
        target_link_libraries(${_allocation_probe_test} PRIVATE ${CMAKE_DL_LIBS})
        set_target_properties(${_allocation_probe_test} PROPERTIES ENABLE_EXPORTS ON)
    endif()
endforeach()

foreach(_stage5_test test_punmtr_he2hb_grouped test_punmtr_he2hb_grouped_integration)
    add_test_executable(${_stage5_test})
    if(TARGET ${_stage5_test})
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${_stage5_test} PRIVATE -ffp-contract=off)
        endif()
    endif()
endforeach()

# Public CPU eigensolver routing regressions; all calls use explicit CPU handles.
if(DDLA_USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX QUIET)
endif()
foreach(_public_cpu_test test_public_routing_query test_public_pheevd_routing)
    add_bench_executable(${_public_cpu_test})
    if(TARGET ${_public_cpu_test})
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${_public_cpu_test} PRIVATE -ffp-contract=off)
        endif()
        if(DDLA_USE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(${_public_cpu_test} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endforeach()
if(TARGET test_public_routing_query)
    add_test(NAME test_public_routing_query_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_public_routing_query>
            --grid 2x2 --mode cold --n 4096 --nb 64 --nbw 64 --require-nodes 1
            ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_public_routing_query_mpi
        PROPERTIES PROCESSORS 4 LABELS cpu
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_EIG_THREADS=1")
endif()
if(TARGET test_public_pheevd_routing)
    add_test(NAME test_public_pheevd_routing_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_public_pheevd_routing>
            --grid 2x2 --n 129 --nb 64 --type d --uplo U --alias 0 --shared-parent 0
            ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_public_pheevd_routing_mpi
        PROPERTIES PROCESSORS 4 LABELS cpu
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_EIG_THREADS=1")
endif()

add_test_executable(test_api_grid_punmbr)
if(TARGET test_api_grid_punmbr)
    target_link_libraries(test_api_grid_punmbr PRIVATE ${CMAKE_DL_LIBS})
endif()

# The communication test checks cold/warm host-cache allocations explicitly.
if(TARGET workspace_trace)
    foreach(_communication_test test_comm test_runtime_communication)
        add_dependencies(${_communication_test} workspace_trace)
        set_tests_properties(${_communication_test}_mpi PROPERTIES ENVIRONMENT
            "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_WORKSPACE_TRACE_REQUIRED=1;LD_PRELOAD=$<TARGET_FILE:workspace_trace>")
    endforeach()
endif()

# Reused workspace and selected-triangle contracts of batched GEMMT.
add_test_executable(test_gemmt_batched 1)

# Matrix-region scaling and allocation-free distributed ownership checks.
add_test_executable(test_scal_matrix 1)
target_link_libraries(test_scal_matrix PRIVATE ${CMAKE_DL_LIBS})
if(TARGET workspace_trace)
    add_dependencies(test_scal_matrix workspace_trace)
    set_tests_properties(test_scal_matrix_mpi PROPERTIES ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_WORKSPACE_TRACE_REQUIRED=1;LD_PRELOAD=$<TARGET_FILE:workspace_trace>")
endif()

# Mixed-domain local and distributed structured products.
add_test_executable(test_mixed_products 1)
target_link_libraries(test_mixed_products PRIVATE ${CMAKE_DL_LIBS})
if(TARGET workspace_trace)
    add_dependencies(test_mixed_products workspace_trace)
    set_tests_properties(test_mixed_products_mpi PROPERTIES ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_WORKSPACE_TRACE_REQUIRED=1;LD_PRELOAD=$<TARGET_FILE:workspace_trace>")
endif()
add_bench_executable(benchmark_mixed_products)

# Mixed real/complex fixed-size pointer-array GEMM batches.
add_test_executable(test_gemm_mixed_batched 1)
target_link_libraries(test_gemm_mixed_batched PRIVATE ${CMAKE_DL_LIBS})
if(TARGET workspace_trace)
    add_dependencies(test_gemm_mixed_batched workspace_trace)
    set_tests_properties(test_gemm_mixed_batched_mpi PROPERTIES ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:workspace_trace>;DDLA_WORKSPACE_TRACE_REQUIRED=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

add_test_executable(test_dgmm 1)

add_test_executable(test_ldlt)

add_test_executable(test_qr_apply)

add_test_executable(test_hemmt 1)
target_link_libraries(test_hemmt PRIVATE ${CMAKE_DL_LIBS})

# Symmetric mode shares the triangle-product fixture, including workspace guards.
add_test_executable(test_symmt 1 test_hemmt.cpp)
target_compile_definitions(test_symmt PRIVATE DDLA_TEST_SYMMT)
target_link_libraries(test_symmt PRIVATE ${CMAKE_DL_LIBS})
if(TARGET workspace_trace)
    foreach(_triangle_test test_hemmt test_symmt)
        add_dependencies(${_triangle_test} workspace_trace)
        set_tests_properties(${_triangle_test}_mpi PROPERTIES ENVIRONMENT
            "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_WORKSPACE_TRACE_REQUIRED=1;LD_PRELOAD=$<TARGET_FILE:workspace_trace>")
    endforeach()
endif()

# Local complex-double HETRF must keep its matrix arithmetic on the GPU.
if(DDLA_HAS_GPU AND UNIX AND NOT APPLE)
    add_library(zhetrf_cpu_trace SHARED zhetrf_cpu_trace.cpp)
    target_link_libraries(zhetrf_cpu_trace PRIVATE ${CMAKE_DL_LIBS})
    add_test_executable(test_zhetrf_gpu 1)
    target_link_libraries(test_zhetrf_gpu PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(test_zhetrf_gpu zhetrf_cpu_trace)
    set_property(TEST test_zhetrf_gpu_mpi APPEND PROPERTY ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:zhetrf_cpu_trace>" "DDLA_EXPECT_NATIVE=1")
endif()

# GPU HETRF_NOPIV and distributed HETRF must not fall back to CPU arithmetic.
if(DDLA_HAS_GPU AND UNIX AND NOT APPLE)
    add_library(ldlt_cpu_trace SHARED ldlt_cpu_trace.cpp)
    target_link_libraries(ldlt_cpu_trace PRIVATE ${CMAKE_DL_LIBS})
    add_test_executable(test_ldlt_gpu_native 1)
    target_link_libraries(test_ldlt_gpu_native PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(test_ldlt_gpu_native ldlt_cpu_trace)
    set_property(TEST test_ldlt_gpu_native_mpi APPEND PROPERTY ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:ldlt_cpu_trace>" "DDLA_EXPECT_NATIVE=1")
    add_library(ldlt_root_failure SHARED ldlt_root_failure.cpp)
    target_compile_definitions(ldlt_root_failure PRIVATE OMPI_SKIP_MPICXX=1)
    target_link_libraries(ldlt_root_failure PRIVATE MPI::MPI_CXX)
    add_test_executable(test_ldlt_root_failure 2)
    add_dependencies(test_ldlt_root_failure ldlt_root_failure)
    set_property(TEST test_ldlt_root_failure_mpi APPEND PROPERTY ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:ldlt_root_failure>")
    set_tests_properties(test_ldlt_root_failure_mpi PROPERTIES SKIP_RETURN_CODE 77)
endif()

# CPU/GPU distributed LDLT factors and immutable distributed solves.
add_test_executable(test_ldlt_distributed)
target_link_libraries(test_ldlt_distributed PRIVATE ${CMAKE_DL_LIBS})
set_property(TEST test_ldlt_distributed_mpi APPEND PROPERTY ENVIRONMENT "DDLA_LDLT_PANEL=gather")
add_test(NAME test_ldlt_distributed_inplace_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_ldlt_distributed> ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_ldlt_distributed_inplace_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP}
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;DDLA_LDLT_PANEL=inplace")
if(DDLA_HAS_GPU)
    set_tests_properties(test_ldlt_distributed_inplace_mpi PROPERTIES LABELS gpu)
    if(TARGET ldlt_cpu_trace)
        add_dependencies(test_ldlt_distributed ldlt_cpu_trace)
        foreach(_ldlt_test test_ldlt_distributed_mpi test_ldlt_distributed_inplace_mpi)
            set_property(TEST ${_ldlt_test} APPEND PROPERTY ENVIRONMENT
                "LD_PRELOAD=$<TARGET_FILE:ldlt_cpu_trace>" "DDLA_EXPECT_NATIVE=1")
        endforeach()
    endif()
else()
    set_tests_properties(test_ldlt_distributed_inplace_mpi PROPERTIES LABELS cpu)
endif()

if(DDLA_HAS_GPU)
    add_test_executable(test_pheevd_householder_gpu)
endif()

if(DDLA_HAS_GPU)
    add_bench_executable(test_hb2st_device_stage2)
    add_bench_executable(test_pheevd_device_stage2)
endif()

if(DDLA_HAS_GPU)
    add_bench_executable(test_hb2st_givens_device)
    add_bench_executable(test_pheevd_givens_device_stage2)
endif()

foreach(_method_test test_pheevd_methods test_pheevd_policy)
    add_test_executable(${_method_test})
endforeach()
