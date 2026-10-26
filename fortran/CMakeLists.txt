# ---------------------------------------------------------------------------
# C-ABI and Fortran bindings for LibDDLA
# ddla_c: pure C++ / MPI shared library implementing the C ABI shim
# ddla_fortran: Fortran module wrapping ddla_c (requires Fortran compiler)
# ---------------------------------------------------------------------------

add_library(ddla_c SHARED
    ddla_c_api.cpp
)
set_target_properties(ddla_c PROPERTIES
    OUTPUT_NAME ddla_c
    VERSION ${CMAKE_PROJECT_VERSION}
    SOVERSION ${CMAKE_PROJECT_SOVERSION_MAJOR}
)
target_link_libraries(ddla_c PUBLIC ddla_lib MPI::MPI_CXX)
target_include_directories(ddla_c PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/BLAS
    ${PROJECT_SOURCE_DIR}/src/PBLAS
    ${PROJECT_SOURCE_DIR}/src/LU
    ${PROJECT_SOURCE_DIR}/src/QR
    ${PROJECT_SOURCE_DIR}/src/Cholesky
    ${PROJECT_BINARY_DIR}/include
)

install(TARGETS ddla_c
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(DDLA_ENABLE_FORTRAN)
    add_library(ddla_fortran SHARED
        ddla.f90
    )
    set_target_properties(ddla_fortran PROPERTIES
        OUTPUT_NAME ddla_fortran
        VERSION ${CMAKE_PROJECT_VERSION}
        SOVERSION ${CMAKE_PROJECT_SOVERSION_MAJOR}
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod
    )
    target_link_libraries(ddla_fortran PUBLIC ddla_c)
    target_include_directories(ddla_fortran INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/mod)
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(ddla_fortran PRIVATE stdc++)
    endif()
    install(TARGETS ddla_fortran
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mod/ddla.mod
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ddla)
endif()
