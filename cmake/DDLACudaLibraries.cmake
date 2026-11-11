# Called after FindCUDAToolkit. Keep standard toolkit layouts independent of
# NVHPC, while permitting that SDK's separate, versioned math_libs tree.
set(DDLA_CUDA_MATH_LIBRARY_DIR "" CACHE PATH
    "Fallback directory for CUDA math libraries (for example NVHPC math_libs/<version>/lib64)")
set(_ddla_cuda_hints "${CUDAToolkit_LIBRARY_DIR}")
if(DDLA_CUDA_MATH_LIBRARY_DIR)
    list(APPEND _ddla_cuda_hints "${DDLA_CUDA_MATH_LIBRARY_DIR}")
else()
    get_filename_component(_ddla_cuda_root "${CUDAToolkit_BIN_DIR}" DIRECTORY)
    get_filename_component(_ddla_cuda_version "${_ddla_cuda_root}" NAME)
    get_filename_component(_ddla_cuda_parent "${_ddla_cuda_root}" DIRECTORY)
    get_filename_component(_ddla_sdk_root "${_ddla_cuda_parent}" DIRECTORY)
    set(_ddla_nvhpc_math "${_ddla_sdk_root}/math_libs/${_ddla_cuda_version}/lib64")
    if(EXISTS "${_ddla_nvhpc_math}")
        list(APPEND _ddla_cuda_hints "${_ddla_nvhpc_math}")
    endif()
endif()

set(_ddla_cuda_names cublas cublasLt curand cusolver cudart)
set(_ddla_cuda_overrides CUBLAS_LIBRARY CUBLASLT_LIBRARY CURAND_LIBRARY CUSOLVER_LIBRARY CUDART_LIBRARY)
set(_ddla_cuda_providers "")
foreach(_ddla_index RANGE 0 4)
    list(GET _ddla_cuda_names ${_ddla_index} _ddla_name)
    list(GET _ddla_cuda_overrides ${_ddla_index} _ddla_override)
    set(${_ddla_override} "" CACHE FILEPATH "Explicit ${_ddla_name} library override")
    if(${_ddla_override})
        if(NOT EXISTS "${${_ddla_override}}")
            message(FATAL_ERROR "${_ddla_override} set to nonexistent file: ${${_ddla_override}}")
        endif()
        set(_ddla_provider "${${_ddla_override}}")
    elseif(TARGET CUDA::${_ddla_name})
        set(_ddla_provider CUDA::${_ddla_name})
        set(DDLA_PACKAGE_NEEDS_CUDATOOLKIT ON)
    else()
        # An empty cache entry would suppress find_library. Search the normal
        # locations as well as the controlled split-layout hints.
        unset(${_ddla_override} CACHE)
        find_library(${_ddla_override} NAMES ${_ddla_name}
            HINTS ${_ddla_cuda_hints} REQUIRED)
        set(_ddla_provider "${${_ddla_override}}")
    endif()
    list(APPEND math_libs "${_ddla_provider}")
    list(APPEND _ddla_cuda_providers "${_ddla_name}=${_ddla_provider}")
    message(STATUS "LibDDLA CUDA ${_ddla_name} provider: ${_ddla_provider}")
endforeach()
# Keep the discovery result available for build/installation audits.
set(DDLA_CUDA_LIBRARY_PROVIDERS "${_ddla_cuda_providers}" CACHE INTERNAL
    "Resolved CUDA link providers" FORCE)
