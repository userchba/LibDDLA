# Cholesky factorization and the triangular solves built on it.  The source
# set is identical on every backend; potrf_bottom_right_transform.cpp keeps a
# GPU kernel next to its CPU arm (plain C++ in a CPU-only build) and switches
# LANGUAGE under CUDA/HIP.
target_sources(ddla_lib PRIVATE
    potrf.cpp
    ppotrf.cpp
    potrf_bottom_right.cpp
    potrf_bottom_right_transform.cpp
    ppotrf_bottom_right.cpp
    ppotrs.cpp
    pposv.cpp
)
# Source properties are looked up in the directory of ddla_lib (src/), so set
# them there explicitly (TARGET_DIRECTORY, CMake >= 3.18).
if(DDLA_USE_CUDA)
    set_source_files_properties(potrf_bottom_right_transform.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(potrf_bottom_right_transform.cpp TARGET_DIRECTORY ddla_lib
        PROPERTIES LANGUAGE HIP)
endif()
