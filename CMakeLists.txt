cmake_minimum_required(VERSION 3.13)

project(LibDDLA NONE)

file(READ ${PROJECT_SOURCE_DIR}/src/version.h version_file)
string(REGEX MATCH "LIBDDLA_MAJOR_VERSION ([0-9]+)" libddla_major_version ${version_file})
set(libddla_major_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBDDLA_MINOR_VERSION ([0-9]+)" libddla_minor_version ${version_file})
set(libddla_minor_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBDDLA_MICRO_VERSION ([0-9]+)" libddla_micro_version ${version_file})
set(libddla_micro_version ${CMAKE_MATCH_1})
set(version "${libddla_major_version}.${libddla_minor_version}.${libddla_micro_version}")

message(STATUS "LibDDLA version: ${version}")

set(CMAKE_PROJECT_SOVERSION_MAJOR "${libddla_major_version}")
set(CMAKE_PROJECT_VERSION "${version}")
set(LIB_NAME            "ddla")
set(math_libs "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_TESTS "Build test programs" OFF) 
option(DDLA_USE_CPU "Build with CPU-only backend" OFF)
option(DDLA_USE_CUDA "Build with CUDA" OFF)
option(DDLA_USE_HIP "Build with HIP" OFF)
option(DDLA_USE_DEBUG "DEBUG for program" OFF)
option(DDLA_USE_CCL "ENABLE NCCL or RCCL" OFF)
option(DDLA_USE_GPU_CPU_TUNNEL "use gpu cpu tunnel for communication" OFF)
option(DDLA_ENABLE_FORTRAN "Build the Fortran bindings (ddla_fortran library + module)" OFF)
option(DDLA_ENABLE_PYTHON "Build the Python bindings and C-ABI library" OFF)
if(DDLA_USE_CUDA AND DDLA_USE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP")
endif()
if ((NOT DDLA_USE_CPU) AND (NOT DDLA_USE_CUDA) AND (NOT DDLA_USE_HIP))
    message(FATAL_ERROR "Must enable at least one backend (CPU, CUDA, or HIP)")
endif()
# Dual-build mode: CPU + (CUDA or HIP).
# Each library source is compiled once with the selected GPU vendor active;
# backend templates provide CPU and GPU implementations in the same target.
set(DDLA_DUAL_BUILD OFF)
if(DDLA_USE_CPU AND (DDLA_USE_CUDA OR DDLA_USE_HIP))
    set(DDLA_DUAL_BUILD ON)
endif()

# CUDA/HIP target definitions are set in src/CMakeLists.txt on ddla_lib

enable_language(CXX)
include(GNUInstallDirs)
include_directories(src)
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# ---------------------------------------------------------------------------
# Generated capability header ddla_config.h
# ---------------------------------------------------------------------------
if(DDLA_USE_CPU OR DDLA_DUAL_BUILD)
    set(DDLA_HAS_CPU 1)
else()
    set(DDLA_HAS_CPU 0)
endif()
if(DDLA_USE_CUDA OR DDLA_USE_HIP)
    set(DDLA_HAS_GPU 1)
else()
    set(DDLA_HAS_GPU 0)
endif()
configure_file(
    ${PROJECT_SOURCE_DIR}/include/ddla/ddla_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/ddla/ddla_config.h
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/ddla/ddla_config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ddla)

find_package(MPI REQUIRED)
list(APPEND math_libs MPI::MPI_CXX)



message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "DDLA_USE_CCL: ${DDLA_USE_CCL}")
message(STATUS "DDLA_USE_GPU_CPU_TUNNEL: ${DDLA_USE_GPU_CPU_TUNNEL}")
if (DDLA_USE_CCL AND DDLA_USE_GPU_CPU_TUNNEL)
    message(WARNING "DDLA_USE_CCL and DDLA_USE_GPU_CPU_TUNNEL are both enabled, the library will need ccl library but it will not be used, if you want use ccl communication, please set DDLA_USE_GPU_CPU_TUNNEL to OFF")
endif()
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")


if(DDLA_USE_HIP)
    enable_language(HIP)
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
            set(ROCM_PATH "$ENV{ROCM_PATH}")
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
    message(STATUS "ROCM_PATH: ${ROCM_PATH}")
    set(HIP_LIBRARIES "")
    foreach(_hip_lib IN ITEMS hipblas rccl hipsolver amdhip64 hiprand omp)
        if(EXISTS "${ROCM_PATH}/lib/lib${_hip_lib}.so" OR EXISTS "${ROCM_PATH}/lib64/lib${_hip_lib}.so")
            list(APPEND HIP_LIBRARIES "-l${_hip_lib}")
        endif()
    endforeach()
    # Hygon/GALAXY DCU-specific HIP runtime: link only on DCU platforms.
    if(EXISTS "${ROCM_PATH}/dcc/lib/libgalaxyhip.so" OR
       EXISTS "${ROCM_PATH}/lib/libgalaxyhip.so" OR
       EXISTS "${ROCM_PATH}/lib64/libgalaxyhip.so")
        list(APPEND HIP_LIBRARIES "-lgalaxyhip")
    endif()
    if(HIP_LIBRARIES)
        message(STATUS "HIP_LIBRARIES: ${HIP_LIBRARIES}")
        list(APPEND math_libs ${HIP_LIBRARIES})
    endif()
endif()
# cpu setting
if(DDLA_USE_CPU)
    # DDLA_USE_CPU definition is set on target in src/CMakeLists.txt
endif()
if(DDLA_USE_CPU OR DDLA_DUAL_BUILD)
    if(DDLA_USE_CCL AND NOT DDLA_DUAL_BUILD)
        message(FATAL_ERROR "DDLA_USE_CCL is not supported with CPU-only backend")
    endif()
    if(DDLA_USE_GPU_CPU_TUNNEL AND NOT DDLA_DUAL_BUILD)
        message(FATAL_ERROR "DDLA_USE_GPU_CPU_TUNNEL is not supported with CPU-only backend")
    endif()
    set(DDLA_CPU_BLAS_LIBRARY "" CACHE FILEPATH "Path to BLAS library for CPU backend")
    if(DDLA_CPU_BLAS_LIBRARY)
        if(NOT EXISTS "${DDLA_CPU_BLAS_LIBRARY}")
            message(FATAL_ERROR "DDLA_CPU_BLAS_LIBRARY set to nonexistent file: ${DDLA_CPU_BLAS_LIBRARY}")
        endif()
        message(STATUS "Using CPU BLAS: ${DDLA_CPU_BLAS_LIBRARY}")
        list(APPEND math_libs "${DDLA_CPU_BLAS_LIBRARY}")
    else()
        find_package(BLAS REQUIRED)
        message(STATUS "Using BLAS: ${BLAS_LIBRARIES}")
        list(APPEND math_libs ${BLAS_LIBRARIES})
    endif()
endif()
# ---------------------------------------------------------------------------
# Host LAPACK (Fortran ABI) -- required whenever the LIBRARY is built from
# source: the QR panel's CPU arm (?GEQRF / ?GEQLF / ?LARFT) and the host references of
# the QR tests call it on the CPU and GPU backends alike.  CPU-only and dual builds already carry it
# (OpenBLAS ships LAPACK in the same library the CPU backend links);
# GPU-only builds must be pointed at one explicitly.  Link-only consumers
# of an installed LibDDLA (BUILD_TESTS=OFF does not change this; the
# library target is still built here) see the same requirement.
# ---------------------------------------------------------------------------
set(DDLA_LAPACK_LIBRARY "" CACHE FILEPATH "Path to host LAPACK library (pgeqrf host arms)")
if(NOT DDLA_LAPACK_LIBRARY AND NOT DDLA_CPU_BLAS_LIBRARY AND DEFINED ENV{OPENBLAS} AND EXISTS "$ENV{OPENBLAS}")
    set(DDLA_LAPACK_LIBRARY "$ENV{OPENBLAS}")
endif()
if(DDLA_LAPACK_LIBRARY)
    if(NOT EXISTS "${DDLA_LAPACK_LIBRARY}")
        message(FATAL_ERROR "DDLA_LAPACK_LIBRARY set to nonexistent file: ${DDLA_LAPACK_LIBRARY}")
    endif()
    message(STATUS "Using host LAPACK: ${DDLA_LAPACK_LIBRARY}")
    list(APPEND math_libs "${DDLA_LAPACK_LIBRARY}")
elseif(DDLA_CPU_BLAS_LIBRARY)
    # OpenBLAS and the other common CPU BLAS builds ship LAPACK in the same
    # library.  In CPU-only and dual builds that library is already on
    # math_libs (the CPU-BLAS block above runs whenever DDLA_USE_CPU, and a
    # dual build sets it); only a GPU-only build must add it here, or the
    # CPU-arm template instantiations link undefined.
    if(NOT DDLA_USE_CPU)
        message(STATUS "Using host LAPACK from DDLA_CPU_BLAS_LIBRARY: ${DDLA_CPU_BLAS_LIBRARY}")
        list(APPEND math_libs "${DDLA_CPU_BLAS_LIBRARY}")
    endif()
else()
    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
        message(FATAL_ERROR "LibDDLA needs a host LAPACK (pgeqrf host arms): pass "
                            "-DDDLA_LAPACK_LIBRARY=<path> (OpenBLAS works, e.g. "
                            "-DDDLA_LAPACK_LIBRARY=/path/to/libopenblas.so) or, for "
                            "CPU/dual builds, -DDDLA_CPU_BLAS_LIBRARY=<path> if that "
                            "library ships LAPACK (OpenBLAS does).")
    endif()
    message(STATUS "Using LAPACK: ${LAPACK_LIBRARIES}")
    list(APPEND math_libs ${LAPACK_LIBRARIES})
endif()

# cuda setting
if(DDLA_USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    # Use CUDAToolkit for the runtime and to locate the toolkit, then pick up
    # the math libraries (cublas, curand, cusolver) from the HPC SDK math_libs tree.
    find_package(CUDAToolkit REQUIRED)
    get_filename_component(_cuda_lib_parent "${CUDAToolkit_LIBRARY_DIR}" DIRECTORY)
    get_filename_component(_cuda_version "${_cuda_lib_parent}" NAME)
    get_filename_component(_cuda_root "${_cuda_lib_parent}" DIRECTORY)
    get_filename_component(_sdk_root "${_cuda_root}" DIRECTORY)
    set(_math_lib_dir "${_sdk_root}/math_libs/${_cuda_version}/lib64")
    message(STATUS "Looking for CUDA math libraries in: ${_math_lib_dir}")

    find_library(CUBLAS_LIBRARY NAMES cublas PATHS ${_math_lib_dir} NO_DEFAULT_PATH REQUIRED)
    find_library(CURAND_LIBRARY NAMES curand PATHS ${_math_lib_dir} NO_DEFAULT_PATH REQUIRED)
    find_library(CUSOLVER_LIBRARY NAMES cusolver PATHS ${_math_lib_dir} NO_DEFAULT_PATH REQUIRED)

    list(APPEND math_libs
        ${CUBLAS_LIBRARY}
        ${CURAND_LIBRARY}
        ${CUSOLVER_LIBRARY}
        ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    )

    if(DDLA_USE_CCL)
        # NCCL is not shipped with the CUDA toolkit in this HPC SDK layout;
        # try the CMake config first, otherwise fall back to the plain library.
        find_package(NCCL QUIET)
        if(NCCL_FOUND)
            list(APPEND math_libs NCCL::nccl)
        else()
            message(STATUS "NCCL CMake package not found, linking -lnccl directly")
            list(APPEND math_libs nccl)
        endif()
    endif()
endif(DDLA_USE_CUDA)
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Fortran & Python bindings (optional)
# ---------------------------------------------------------------------------
if(DDLA_ENABLE_FORTRAN)
    include(CheckLanguage)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
    else()
        message(FATAL_ERROR "DDLA_ENABLE_FORTRAN is ON but no Fortran compiler was found")
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
endif()

if(DDLA_ENABLE_FORTRAN OR DDLA_ENABLE_PYTHON)
    add_subdirectory(fortran)
endif()

if(DDLA_ENABLE_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
    add_subdirectory(python)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
# ---------------------------------------------------------------------------
# Installable CMake package config (find_package(LibDDLA))
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
set(LIBDDLA_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/LibDDLA")
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/LibDDLAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LibDDLAConfig.cmake
    INSTALL_DESTINATION ${LIBDDLA_CMAKE_INSTALL_DIR}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/LibDDLAConfigVersion.cmake
    VERSION ${version}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LibDDLAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LibDDLAConfigVersion.cmake
    DESTINATION ${LIBDDLA_CMAKE_INSTALL_DIR}
)

