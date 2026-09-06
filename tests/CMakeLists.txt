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
    # fixed rank requirement (e.g. test_ptran needs 6, test_potrf_bottom_right
    # needs 1).  Defaults to DDLA_TEST_NP.
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
            set_tests_properties(${test_name}_mpi PROPERTIES LABELS gpu)
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
add_test_executable(test_backend_level1)
add_test_executable(test_random_generate)

if(DDLA_USE_CUDA OR DDLA_USE_HIP)
    add_test_executable(test_pzgemm)
    add_test_executable(test_ptran 6)
    add_test_executable(test_pdam)
    add_test_executable(test_getrf_nopiv)
    add_test_executable(test_pgetrf_nopiv)
    add_test_executable(test_pgesv_nopiv)
    add_test_executable(test_sv_gemm_hermitian_positive)
    add_test_executable(test_gemm_vbatched)
    add_test_executable(test_api_grid_device_memory)
    add_test_executable(test_api_grid_pgemm)
    add_test_executable(test_api_grid_scalapack_desc)
    add_test_executable(test_api_grid_pgeadd)
    add_test_executable(test_api_grid_ptran)
    add_test_executable(test_api_grid_transport_block)
    add_test_executable(test_api_grid_pdam)
    add_test_executable(test_api_grid_pswap)
    add_test_executable(test_api_grid_plapiv)
    add_test_executable(test_api_grid_ptrtrs)
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
    add_test_executable(test_api_grid_pgetf2)
    add_test_executable(test_api_grid_pgetf2_panel)
    add_test_executable(test_api_grid_getrf_nopiv)
    add_test_executable(test_api_grid_ppotrf)
    add_test_executable(test_api_grid_ppotrf_head)
    add_test_executable(test_api_grid_ppotrs)
    add_test_executable(test_api_grid_pposv)
    add_test_executable(test_potrf_bottom_right_block 1)
    add_test_executable(test_potrf_bottom_right 1)
    add_test_executable(test_api_grid_ppotrf_bottom_right)
    add_test_executable(test_api_grid_submatrix)
    add_test_executable(test_ppotrf_real)
    add_test_executable(test_pgemm_debug)
    add_test_executable(test_pgemm_debug2)
    add_bench_executable(bench_pzgemm)
    add_bench_executable(benchmark_pgetrf)
    add_bench_executable(benchmark_pgetf2)
    add_bench_executable(benchmark_ptrtrs)
    add_bench_executable(benchmark_solve_side)
    add_bench_executable(benchmark_solvers)
    add_bench_executable(benchmark_ppotrf_bottom_right)
    add_bench_executable(benchmark_plapiv)
endif()

# Backend handle test — works for all backends
add_executable(test_backend_handle test_backend_handle.cpp)
target_link_libraries(test_backend_handle PRIVATE ddla_lib)
apply_test_connector_defs(test_backend_handle)
add_test(NAME test_backend_handle_mpi4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_backend_handle>
            ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_backend_handle_mpi4 PROPERTIES PROCESSORS 4 LABELS cpu)

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
