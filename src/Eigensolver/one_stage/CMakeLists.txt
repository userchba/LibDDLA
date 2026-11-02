# Direct Householder reduction, tridiagonal back-transform and one-stage driver.
target_sources(ddla_lib PRIVATE pheevd_one_stage.cpp phetrd.cpp punmtr_trd.cpp)

if(DDLA_USE_CUDA)
    set_source_files_properties(pheevd_one_stage.cpp phetrd.cpp punmtr_trd.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    set_source_files_properties(pheevd_one_stage.cpp phetrd.cpp punmtr_trd.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE HIP)
endif()
