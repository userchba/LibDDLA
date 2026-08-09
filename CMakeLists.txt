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

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_TESTS "Build test programs" OFF) 
option(DDLA_USE_CPU "Build with CPU-only backend" OFF)
option(DDLA_USE_CUDA "Build with CUDA" OFF)
option(DDLA_USE_HIP "Build with HIP" OFF)
option(DDLA_USE_DEBUG "DEBUG for program" OFF)
option(DDLA_USE_CCL "ENABLE NCCL or RCCL" OFF)
option(DDLA_USE_GPU_CPU_TUNNEL "use gpu cpu tunnel for communication" OFF)
if(DDLA_USE_CUDA AND DDLA_USE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP")
endif()
if ((NOT DDLA_USE_CPU) AND (NOT DDLA_USE_CUDA) AND (NOT DDLA_USE_HIP))
    message(FATAL_ERROR "Must enable at least one backend (CPU, CUDA, or HIP)")
endif()
# Dual-build mode: CPU + (CUDA or HIP).
# The main library target compiles with the selected GPU backend macros;
# CPU pgemm compiles in a separate object target with DDLA_USE_CPU only.
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
    set(HIP_LIBRARIES "-lhipblas -lrccl -lhipsolver -lamdhip64 -lhiprand -lgalaxyhip -lomp")
    # -lamdhip64 -lhsa-runtime64
    # find_package(HIP REQUIRED)
    list(APPEND math_libs
        ${HIP_LIBRARIES}
    )
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
# cuda setting
if(DDLA_USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()
    set(CMAKE_CUDA_STANDARD 11)
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

if(BUILD_TESTS)
    enable_testing()
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

