# PBLAS-level distributed operations: the parallel GEMM family, triangular
# multiplies/solves, pivoting (plapiv) and row swaps (pswap).  The source set
# is identical on every backend; pdam.cpp and pgeadd.cpp keep a GPU kernel
# next to their CPU arm (plain C++ in a CPU-only build) and switch LANGUAGE
# under CUDA/HIP.
target_sources(ddla_lib PRIVATE
    pdgmm.cpp
    pgemm.cpp
    pgemmt.cpp
    pherk.cpp
    pher2k.cpp
    psyrk.cpp
    psyr2k.cpp
    prankk_impl.cpp
    pgemm_select.cpp
    pgemm_stationary.cpp
    ptrtrs.cpp
    ptrmm.cpp
    phemm.cpp
    phemmt.cpp
    psymm.cpp
    psymmt.cpp
    ptrtrm.cpp
    plapiv.cpp
    pswap.cpp
    pgeadd.cpp
    pdam.cpp
)
# Source properties are looked up in the directory of ddla_lib (src/), so set
# them there explicitly (TARGET_DIRECTORY, CMake >= 3.18).
if(DDLA_USE_CUDA)
    set_source_files_properties(pdam.cpp pgeadd.cpp phemm.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(pdam.cpp pgeadd.cpp phemm.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP)
endif()
