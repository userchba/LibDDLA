# Shared eigensolver operations and the public method dispatcher.
target_sources(ddla_lib PRIVATE
    eig_params.cpp
    eig_cpu_kernels.cpp
    pvector_redist.cpp
    pstedc.cpp
    pstedc_check.cpp
    pstedc_workspace_size.cpp
    pstedc_run.cpp
    stedc.cpp
    selected_tridiagonal_core.cpp
    pheevd.cpp
    heevd.cpp
    hegvd.cpp
    phegvd.cpp
    phegst.cpp
    syevd.cpp
    pheevd_adapter.cpp
    pherm_complete.cpp
    chebyshev.cpp
)

# Selected inverse iteration operates on host scratch even for GPU matrices.
set_source_files_properties(selected_tridiagonal_core.cpp TARGET_DIRECTORY ddla_lib
    PROPERTIES LANGUAGE CXX)

# Keep shared CPU kernels on the host compiler in every backend.
set_source_files_properties(eig_cpu_kernels.cpp TARGET_DIRECTORY ddla_lib
    PROPERTIES COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")

if(DDLA_USE_CUDA)
    set_source_files_properties(
        pvector_redist.cpp pstedc_run.cpp stedc.cpp pheevd.cpp
        heevd.cpp hegvd.cpp phegvd.cpp phegst.cpp syevd.cpp pheevd_adapter.cpp pherm_complete.cpp chebyshev.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(
        pvector_redist.cpp pstedc_run.cpp stedc.cpp pheevd.cpp
        heevd.cpp hegvd.cpp phegvd.cpp phegst.cpp syevd.cpp pheevd_adapter.cpp pherm_complete.cpp chebyshev.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE HIP)
endif()

add_subdirectory(one_stage)
add_subdirectory(two_stage)

install(FILES ${PROJECT_SOURCE_DIR}/LICENSES/MAGMA-BSD-3-Clause.txt
    DESTINATION ${CMAKE_INSTALL_DATADIR}/LibDDLA/licenses)
