# Full-to-band reduction and the corresponding eigenvector back-transform.
target_sources(ddla_lib PRIVATE
    phe2hb.cpp phe2hb_panel.cpp
    punmtr_he2hb.cpp punmtr_he2hb_grouped.cpp
)
if(DDLA_USE_CUDA)
    set_source_files_properties(phe2hb.cpp phe2hb_panel.cpp punmtr_he2hb.cpp punmtr_he2hb_grouped.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE CUDA)
endif()
if(DDLA_USE_HIP)
    # Preserve the current host-compiler mode for phe2hb.cpp and punmtr_he2hb.cpp.
    set_source_files_properties(phe2hb_panel.cpp punmtr_he2hb_grouped.cpp
        TARGET_DIRECTORY ddla_lib PROPERTIES LANGUAGE HIP)
endif()
