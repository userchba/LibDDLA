# Full-to-band reduction, band-to-tridiagonal chase, back-transforms and driver.
target_sources(ddla_lib PRIVATE
    hb2st_sink_host.cpp
    phe2hb.cpp
    phe2hb_panel.cpp
    punmtr_he2hb.cpp
    punmtr_he2hb_grouped.cpp
    phb2st.cpp
    pheevd_two_stage.cpp
    hb2st_replay_storage.cpp
)

set_source_files_properties(hb2st_sink_host.cpp TARGET_DIRECTORY ddla_lib
    PROPERTIES COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")

if(DDLA_USE_CUDA)
    set_source_files_properties(
        phe2hb.cpp phe2hb_panel.cpp punmtr_he2hb.cpp punmtr_he2hb_grouped.cpp phb2st.cpp
        pheevd_two_stage.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(
        phe2hb_panel.cpp punmtr_he2hb_grouped.cpp phb2st.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE HIP)
    # Preserve the existing plain-CXX mode for phe2hb.cpp, punmtr_he2hb.cpp
    # and pheevd_two_stage.cpp. HIP is not available on HPC3 for validation.
endif()

if(DDLA_USE_CUDA OR DDLA_USE_HIP)
    target_sources(ddla_lib PRIVATE hb2st_replay_gpu.cpp hb2st_householder_gpu.cpp hb2st_givens_gpu.cpp)
    if(DDLA_USE_CUDA)
        set_source_files_properties(hb2st_replay_gpu.cpp hb2st_householder_gpu.cpp hb2st_givens_gpu.cpp TARGET_DIRECTORY ddla_lib
            PROPERTIES LANGUAGE CUDA)
    else()
        set_source_files_properties(hb2st_replay_gpu.cpp hb2st_householder_gpu.cpp hb2st_givens_gpu.cpp TARGET_DIRECTORY ddla_lib
            PROPERTIES LANGUAGE HIP)
    endif()
endif()

target_sources(ddla_lib PRIVATE hb2st_householder_cpu.cpp hb2st_householder_cpu_rows.cpp)
set_source_files_properties(hb2st_householder_cpu.cpp hb2st_householder_cpu_rows.cpp
    TARGET_DIRECTORY ddla_lib
    PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")
set_property(SOURCE hb2st_householder_cpu.cpp hb2st_householder_cpu_rows.cpp
    TARGET_DIRECTORY ddla_lib APPEND PROPERTY COMPILE_DEFINITIONS DDLA_HH_PAIR_ENABLE_CLONES=1)
