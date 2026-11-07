# Full-to-band reduction over the shared distributed QL panel core.
target_sources(ddla_lib PRIVATE phe2hb.cpp phe2hb_panel.cpp)
if(DDLA_USE_CUDA)
    set_source_files_properties(phe2hb.cpp phe2hb_panel.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    # Preserve the current host-compiler mode for phe2hb.cpp.
    set_source_files_properties(phe2hb_panel.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE HIP)
endif()
