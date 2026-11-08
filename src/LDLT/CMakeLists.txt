target_sources(ddla_lib PRIVATE
    sytrf.cpp
    sytrs.cpp
    hetrf.cpp
    hetrs.cpp
    psytrf.cpp
    psytrs.cpp
    phetrf.cpp
    phetrs.cpp
)

if(NOT ddla_cpu_only_build)
    target_sources(ddla_lib PRIVATE hetrf_gpu.cpp hetrf_kernels.cpp hetrf_nopiv_gpu.cpp phetrf_gpu.cpp ldlt_native_kernels.cpp)
endif()
if(DDLA_USE_CUDA)
    set_source_files_properties(hetrf_kernels.cpp ldlt_native_kernels.cpp TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(hetrf_kernels.cpp ldlt_native_kernels.cpp TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE HIP)
endif()
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/MAGMA-LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/LibDDLA RENAME MAGMA-BSD-3-Clause.txt)
