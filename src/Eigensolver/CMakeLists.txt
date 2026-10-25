# Distributed Hermitian eigensolver (p?heevd: reduction to band/tridiagonal,
# divide-and-conquer, back-transform), its host kernels and the GPU replay of
# the reduction history.
target_sources(ddla_lib PRIVATE
    eig_params.cpp
    eig_cpu_kernels.cpp
    hb2st_sink_host.cpp
    pvector_redist.cpp
    phe2hb.cpp
    phe2hb_panel.cpp
    punmtr_he2hb.cpp
    phb2st.cpp
    hb2st_accum.cpp
    pstedc.cpp
    punmtr_hb2st.cpp
    pheevd.cpp
    pherm_complete.cpp
)

# Source properties are looked up in the directory of ddla_lib (src/), so set
# them there explicitly (TARGET_DIRECTORY, CMake >= 3.18).
#
# Host kernels of the CPU eigensolver path: eig_cpu_kernels.cpp and
# hb2st_sink_host.cpp are compiled by the plain host compiler in every build
# (never by nvcc/hipcc) with -ffp-contract=off so their AVX2/AVX-512 function
# clones round exactly like the scalar reference (src/CMakeLists.txt computes
# ${_ddla_cpu_kernel_opts} and links OpenMP PRIVATE when found).
set_source_files_properties(eig_cpu_kernels.cpp hb2st_sink_host.cpp TARGET_DIRECTORY ddla_lib
    PROPERTIES COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")

if(DDLA_USE_CUDA)
    set_source_files_properties(
        pvector_redist.cpp
        phe2hb_panel.cpp
        phe2hb.cpp
        punmtr_he2hb.cpp
        phb2st.cpp
        hb2st_accum.cpp
        pstedc.cpp
        punmtr_hb2st.cpp
        pheevd.cpp
        pherm_complete.cpp
        TARGET_DIRECTORY ddla_lib
        PROPERTIES
        LANGUAGE CUDA
    )
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(
        pvector_redist.cpp
        phe2hb_panel.cpp
        phb2st.cpp
        hb2st_accum.cpp
        pstedc.cpp
        punmtr_hb2st.cpp
        pherm_complete.cpp
        TARGET_DIRECTORY ddla_lib
        PROPERTIES
        LANGUAGE HIP
    )
    # NOTE: phe2hb.cpp, punmtr_he2hb.cpp and pheevd.cpp are in the CUDA list
    # above but NOT here, so the HIP build would compile them as plain CXX.
    # Whether that breaks is unverified: hpc3 has no ROCm toolchain, so a HIP
    # build can only be tested on df_dcu (canonical tree
    # /public/home/hbchen/app/github/LibDDLA).  Add them here only together
    # with a HIP build+test run, not on the strength of the CUDA list alone.
    # Eigensolver/hb2st_accum.cpp and Eigensolver/pstedc.cpp ARE here (like Eigensolver/phb2st.cpp and
    # Eigensolver/punmtr_hb2st.cpp) because they contain __global__ kernels or <<<>>>
    # launches that plain CXX cannot compile (Eigensolver/pstedc.cpp since the distributed
    # divide and conquer moved into it).  Eigensolver/pherm_complete.cpp (the Hermitian
    # completion of stage 0, with __global__ kernels) is here for the same
    # reason; it is UNTESTED on HIP (no ROCm on hpc3).  Eigensolver/pheevd.cpp itself no
    # longer contains kernels (the old uplo='L' symmetrization moved out).
endif()

# Bounded direct GPU replay of the distributed eigensolver history.
if(DDLA_USE_CUDA OR DDLA_USE_HIP)
    target_sources(ddla_lib PRIVATE hb2st_replay_gpu.cpp)
    if(DDLA_USE_CUDA)
        set_source_files_properties(hb2st_replay_gpu.cpp TARGET_DIRECTORY ddla_lib
            PROPERTIES LANGUAGE CUDA)
    else()
        set_source_files_properties(hb2st_replay_gpu.cpp TARGET_DIRECTORY ddla_lib
            PROPERTIES LANGUAGE HIP)
    endif()
endif()

# GPU solver storage ownership; plain C++ on every backend.
target_sources(ddla_lib PRIVATE hb2st_replay_storage.cpp)

# Private CPU Householder band-reduction prototype; public pheevd keeps its
# existing selection until native correctness and performance gates pass.
if(DDLA_USE_CPU)
    target_sources(ddla_lib PRIVATE hb2st_householder_cpu.cpp hb2st_householder_cpu_rows.cpp)
    set_source_files_properties(hb2st_householder_cpu.cpp hb2st_householder_cpu_rows.cpp
        TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")
    set_property(SOURCE hb2st_householder_cpu.cpp hb2st_householder_cpu_rows.cpp
        TARGET_DIRECTORY ddla_lib APPEND PROPERTY COMPILE_DEFINITIONS DDLA_HH_PAIR_ENABLE_CLONES=1)
endif()
