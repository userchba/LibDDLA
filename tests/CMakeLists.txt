# ---------------------------------------------------------------------------
# Apply per-target backend connector definitions.
# In CPU-only mode only DDLA_USE_CPU+DEBUG are relevant.
# In GPU/dual mode the selected GPU vendor + CCL/TUNNEL/DEBUG are required
# because test sources include ddla_connector.h.
# ---------------------------------------------------------------------------
function(apply_test_connector_defs target)
    if(DDLA_USE_CUDA)
        target_compile_definitions(${target} PRIVATE DDLA_USE_CUDA)
    elseif(DDLA_USE_HIP)
        target_compile_definitions(${target}
            PRIVATE DDLA_USE_HIP __HIP_PLATFORM_AMD__)
    elseif(DDLA_USE_CPU)
        target_compile_definitions(${target} PRIVATE DDLA_USE_CPU)
    endif()
    if(DDLA_USE_CCL)
        target_compile_definitions(${target} PRIVATE DDLA_USE_CCL)
    endif()
    if(DDLA_USE_GPU_CPU_TUNNEL)
        target_compile_definitions(${target} PRIVATE DDLA_USE_GPU_CPU_TUNNEL)
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
    if(EXISTS "${test_source}")
        add_executable(${test_name} ${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE ddla_lib)
        apply_test_connector_defs(${test_name})
        add_test(NAME ${test_name}_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}>
                    ${MPIEXEC_POSTFLAGS})
        set_tests_properties(${test_name}_mpi PROPERTIES PROCESSORS ${_np})
        if(DDLA_USE_CUDA OR DDLA_USE_HIP)
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

add_test_executable(test_backend_gemm)
add_test_executable(test_backend_gemm_mixed 1)
add_bench_executable(benchmark_gemm_mixed)
add_test_executable(test_backend_vector)
add_test_executable(test_pgemm_branches)
add_test_executable(test_pgeadd_indexed)
add_test_executable(test_pgeadd_direct)
add_bench_executable(benchmark_pgemm_branches)
add_test_executable(test_backend_level1)
# BLAS-3 families that had no numerical coverage before the wrapper
# unification (trsm, syrk, trmm, gemmBatched), plus per-family handle and
# backend-mismatch rejection.
add_test_executable(test_backend_blas3)
add_test_executable(test_backend_trtrm 1)
add_bench_executable(benchmark_trtrm)
add_test_executable(test_backend_geam)
# LAPACK-level wrappers (getrf, potrf, laswp) and swap: CPU arm = host LAPACK, GPU arm = cuSOLVER / hipSOLVER.
add_test_executable(test_backend_lapack)
# The distributed triangular solves, LU and Cholesky factorizations/solvers and their helpers are backend-templated like pgemm.
# The api_grid tests call them bare (the build's default backend: a CPU handle in a CPU-only build, a GPU handle otherwise);
# test_backend_solvers calls them with an explicit backend on every backend the build contains, which is what exercises the CPU
# arms in a dual build.
add_test_executable(test_backend_solvers)
add_test_executable(test_api_grid_transport_block)
add_test_executable(test_api_grid_ptrtrs)
add_test_executable(test_api_grid_ptrmm)
add_test_executable(test_api_grid_plapiv)
add_test_executable(test_api_grid_pswap)
add_test_executable(test_api_grid_pgetf2)
add_test_executable(test_lu_gather)
add_test_executable(test_matrix_transfer)
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
add_test_executable(test_api_grid_ppotrf_head)
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
add_test_executable(test_pheevd_workspace)
add_test_executable(test_pheevd_finish_panels)
if(TARGET test_pheevd_finish_panels)
    target_include_directories(test_pheevd_finish_panels PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()
if(DDLA_USE_CPU AND NOT (DDLA_USE_CUDA OR DDLA_USE_HIP))
    add_test_executable(test_phb2st_no_records)
endif()
# the streaming wave schedule against the offline one (host only; needs the internal header)
add_test_executable(test_hb2st_wave_gen 1)
if(TARGET test_hb2st_wave_gen)
    target_include_directories(test_hb2st_wave_gen PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()
# pheevd again with the chase-window knobs that force the record RING / ELASTIC ring to wrap and to be left early (tiny rings, tiny
# stager chunks): eigenvalue-only runs record into a ring of 1 step, eigenvector runs on GPU use an elastic ring of 2 chunks.  Same
# executable, same checks; the knobs are ignored where they do not apply.
if(TARGET test_api_grid_pheevd)
    add_test(NAME test_api_grid_pheevd_ring_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_api_grid_pheevd> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_api_grid_pheevd_ring_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP}
        ENVIRONMENT "DDLA_EIG_CHASE_RING_RECORDS=1;DDLA_EIG_CHASE_ELASTIC=2;DDLA_EIG_ACCUM_CHUNK=2048;DDLA_EIG_ACCUM_RING_SLOTS=4")
    if(DDLA_USE_CUDA OR DDLA_USE_HIP)
        set_tests_properties(test_api_grid_pheevd_ring_mpi PROPERTIES LABELS gpu
            ENVIRONMENT "DDLA_EIG_CHASE_RING_RECORDS=1;DDLA_EIG_CHASE_ELASTIC=2;DDLA_EIG_ACCUM_CHUNK=2048;DDLA_EIG_ACCUM_RING_SLOTS=4;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    else()
        set_tests_properties(test_api_grid_pheevd_ring_mpi PROPERTIES LABELS cpu)
    endif()
endif()
# CPU handles (CPU-only builds): stage 4 has two engines -- the accumulation of Q_band during the chase (hb2st_sink_host.cpp) and the classic
# replay (the node-shared wave-ordered stream).  The default 'auto' picks the replay on the small test grids, so both are forced here:
# the accumulation with tiny blocks and panels (the block ring, the block slots and the k-panels wrap; many block boundaries) and the replay.
if(TARGET test_api_grid_pheevd AND DDLA_USE_CPU AND NOT (DDLA_USE_CUDA OR DDLA_USE_HIP))
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
# QL factorization (pgeqrf with uplo = 'L') and the uplo dispatch, same
# bitwise-repeat poisoning requirement as the pgeqrf grid test.
add_test_executable(test_api_grid_pgeqrf_ql)
if(TARGET test_api_grid_pgeqrf_ql)
    set_tests_properties(test_api_grid_pgeqrf_ql_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0")
endif()
# the GLIBC_TUNABLES overrides replace the whole ENVIRONMENT list:
# re-append the GPU lane thread caps so the host-side panel work does not
# oversubscribe (measured 2026-10-24: pgeqrf ~16-thread pools, GPUs idle).
if((DDLA_USE_CUDA OR DDLA_USE_HIP) AND TARGET test_api_grid_pgeqrf)
    set_tests_properties(test_api_grid_pgeqrf_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()
if((DDLA_USE_CUDA OR DDLA_USE_HIP) AND TARGET test_api_grid_pgeqrf_ql)
    set_tests_properties(test_api_grid_pgeqrf_ql_mpi PROPERTIES
        ENVIRONMENT "GLIBC_TUNABLES=glibc.malloc.tcache_count=0;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()
add_test_executable(test_api_grid_pgebrd)
add_test_executable(test_api_grid_pgesvd)
add_test_executable(test_api_grid_pbdsdc)

if(DDLA_USE_CUDA OR DDLA_USE_HIP)
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

# The full eigensolver at scale: time, DDLA_EIG_PROFILE per-stage memory, residual/orthogonality at the timed
# size (--check).  Either backend.  Heavy: never run by ctest.
add_bench_executable(benchmark_pheevd)

# Distributed Householder QR benchmark — backend-templated (runs on the
# build's default backend).  Heavy: never run by ctest.
add_bench_executable(benchmark_pgeqrf)

# Distributed SVD benchmark (pgebrd / punmbr / pgesvd stage times, checked at
# the timed size) -- backend-neutral like benchmark_pgeqrf.  Heavy: never run
# by ctest.
add_bench_executable(benchmark_pgesvd)

# Backend handle test — works for all backends
add_executable(test_backend_handle test_backend_handle.cpp)
target_link_libraries(test_backend_handle PRIVATE ddla_lib)
apply_test_connector_defs(test_backend_handle)
add_test(NAME test_backend_handle_mpi4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_backend_handle>
            ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_backend_handle_mpi4 PROPERTIES PROCESSORS 4 LABELS cpu)
if(DDLA_USE_CUDA OR DDLA_USE_HIP)
    set_tests_properties(test_backend_handle_mpi4 PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

if(DDLA_USE_CPU OR DDLA_DUAL_BUILD)
    add_executable(test_cpu_pgemm test_cpu_pgemm.cpp)
    target_link_libraries(test_cpu_pgemm PRIVATE ddla_lib)
    # CPU-only test: needs DDLA_USE_CPU, never GPU backend macros
    target_compile_definitions(test_cpu_pgemm PRIVATE DDLA_USE_CPU)
    if(DDLA_USE_DEBUG)
        target_compile_definitions(test_cpu_pgemm PRIVATE DDLA_USE_DEBUG)
    endif()
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
        set(bench_src "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_cpu_pgemm_scalapack.cpp")
        if(EXISTS "${bench_src}")
            add_executable(benchmark_cpu_pgemm_scalapack
                benchmark_cpu_pgemm_scalapack.cpp)
            target_link_libraries(benchmark_cpu_pgemm_scalapack PRIVATE
                ddla_lib "${_scalapack_lib}")
            target_compile_definitions(benchmark_cpu_pgemm_scalapack PRIVATE DDLA_USE_CPU)
            if(DDLA_USE_DEBUG)
                target_compile_definitions(benchmark_cpu_pgemm_scalapack PRIVATE DDLA_USE_DEBUG)
            endif()
            # Heavy benchmark -- not added to CTest
        else()
            message(STATUS "Skipping missing benchmark source: benchmark_cpu_pgemm_scalapack.cpp")
        endif()
    endif()
endif()

# Dual-build tests
if(DDLA_DUAL_BUILD)
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
if(TARGET test_phb2st_distributed)
    target_include_directories(test_phb2st_distributed PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()
if(DDLA_USE_CUDA OR DDLA_USE_HIP)
    add_test_executable(test_pheevd_gpu_replay)
    if(TARGET test_pheevd_gpu_replay)
        target_include_directories(test_pheevd_gpu_replay PRIVATE ${CMAKE_SOURCE_DIR}/src)
    endif()
endif()

add_test_executable(test_pstedc_scratch)
if(TARGET test_pstedc_scratch)
    target_include_directories(test_pstedc_scratch PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()

if(DDLA_USE_CPU)
    add_test_executable(test_hb2st_cpu_failures)
endif()
if(TARGET test_hb2st_cpu_failures)
    target_include_directories(test_hb2st_cpu_failures PRIVATE ${CMAKE_SOURCE_DIR}/src)
    set_tests_properties(test_hb2st_cpu_failures_mpi PROPERTIES TIMEOUT 180 LABELS cpu)
    add_test(NAME test_hb2st_cpu_failures_mpi1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_hb2st_cpu_failures> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_hb2st_cpu_failures_mpi1 PROPERTIES TIMEOUT 180 PROCESSORS 1 LABELS cpu)
endif()

if(DDLA_USE_CPU)
    add_test_executable(test_pheevd_cpu_replay)
endif()
if(TARGET test_pheevd_cpu_replay)
    target_include_directories(test_pheevd_cpu_replay PRIVATE ${CMAKE_SOURCE_DIR}/src)
    set_tests_properties(test_pheevd_cpu_replay_mpi PROPERTIES TIMEOUT 600 LABELS cpu)
endif()

# CPU handle even in dual builds; executable LAPACK interposition needs ELF/dl.
if((DDLA_USE_CPU OR DDLA_DUAL_BUILD) AND UNIX
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pstedc_threads.cpp")
    add_executable(test_pstedc_threads test_pstedc_threads.cpp)
    target_link_libraries(test_pstedc_threads PRIVATE ddla_lib ${CMAKE_DL_LIBS})
    target_include_directories(test_pstedc_threads PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_pstedc_threads PRIVATE DDLA_USE_CPU)
    set_target_properties(test_pstedc_threads PROPERTIES ENABLE_EXPORTS ON)
    add_test(NAME test_pstedc_threads
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_pstedc_threads> --threads 4 --blas-threads 4 ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_pstedc_threads PROPERTIES PROCESSORS 4 LABELS cpu SKIP_RETURN_CODE 77 TIMEOUT 180
        ENVIRONMENT "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=4;OMP_DYNAMIC=false;OMP_PROC_BIND=false;OMP_WAIT_POLICY=PASSIVE;OPENBLAS_MAIN_FREE=1")
endif()

# Exercise the in-place TRMM interface through all typed Fortran bindings.
if(DDLA_ENABLE_FORTRAN)
    find_package(MPI REQUIRED COMPONENTS Fortran)
    add_executable(test_fortran_trmm test_fortran_trmm.f90)
    target_link_libraries(test_fortran_trmm PRIVATE ddla_fortran MPI::MPI_Fortran)
    add_test(NAME test_fortran_trmm_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${DDLA_TEST_NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_fortran_trmm> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_fortran_trmm_mpi PROPERTIES PROCESSORS ${DDLA_TEST_NP})
endif()
