# ---------------------------------------------------------------------------
# Python bindings: ddla
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Found Python3 for LibDDLA bindings: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

# Stage ddla Python package in binary directory so tests can run without site-packages install
add_custom_target(ddla_python ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/ddla
            ${CMAKE_CURRENT_BINARY_DIR}/ddla
    DEPENDS ddla_c
    COMMENT "Staging ddla Python package"
)

if(BUILD_TESTS)
    set(_ddla_py_np 4)
    if(DEFINED DDLA_TEST_NP)
        set(_ddla_py_np ${DDLA_TEST_NP})
    endif()

    add_test(NAME test_python_api_serial
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
    )
    set_tests_properties(test_python_api_serial PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${CMAKE_CURRENT_BINARY_DIR};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:ddla_c>:$<TARGET_FILE_DIR:ddla_lib>:$ENV{LD_LIBRARY_PATH}"
        LABELS "python;cpu"
    )

    add_test(NAME test_python_api_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_ddla_py_np}
                ${MPIEXEC_PREFLAGS} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
                ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(test_python_api_mpi PROPERTIES
        PROCESSORS ${_ddla_py_np}
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${CMAKE_CURRENT_BINARY_DIR};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:ddla_c>:$<TARGET_FILE_DIR:ddla_lib>:$ENV{LD_LIBRARY_PATH}"
        LABELS "python;cpu"
    )
endif()
