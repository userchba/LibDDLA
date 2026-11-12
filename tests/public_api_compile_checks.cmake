# Compile-only checks use public include directories, with no backend_api.h.
# A positive control prevents missing headers/toolchain failures from making
# the expected negative checks appear to pass.
function(ddla_check_public_scalar_api)
    include(CheckCXXSourceCompiles)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_REQUIRED_INCLUDES ${DDLA_PUBLIC_CHECK_INCLUDES} ${MPI_CXX_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES "")
    set(CMAKE_REQUIRED_DEFINITIONS "")
    set(CMAKE_REQUIRED_FLAGS "")
    set(_prefix "#include <ddla/ddla.h>\n#include <complex>\n#include <type_traits>\n")
    set(_positive "${_prefix}
void client(ddla::DdlaHandle_t h) {
    double* p=nullptr; std::complex<double>* z=nullptr; int info=0;
    ddla::gemm<double>(h,'N','N',0,0,0,1.,p,1,p,1,0.,p,1);
    ddla::scal(h,0,1.,p,1);
    ddla::gemm<double>(h,'N','N',0,0,0,std::complex<double>(1),p,1,z,1,std::complex<double>(0),z,1);
    ddla::write_matrix<double>(h,p,0,0,\"unused\");
    static_assert(std::is_same<decltype(ddla::ppotrf<double>(h,'L',0,p,1,1,nullptr,info)),void>::value,\"Cholesky status API\");
}")
    unset(DDLA_PUBLIC_SCALAR_SYNTAX CACHE)
    check_cxx_source_compiles("${_positive}" DDLA_PUBLIC_SCALAR_SYNTAX)
    if(NOT DDLA_PUBLIC_SCALAR_SYNTAX)
        message(FATAL_ERROR "Public scalar-only API positive compile control failed")
    endif()
    foreach(_case GEMM SCAL MIXED WRITER POINTER)
        if(_case STREQUAL GEMM)
            set(_call "ddla::gemm<ddla::DdlaBackend::CPU,double>(h,'N','N',0,0,0,1.,p,1,p,1,0.,p,1);")
        elseif(_case STREQUAL SCAL)
            set(_call "ddla::scal<ddla::DdlaBackend::CPU>(h,0,1.,p,1);")
        elseif(_case STREQUAL MIXED)
            set(_call "ddla::gemm<ddla::DdlaBackend::GPU,double>(h,'N','N',0,0,0,std::complex<double>(1),p,1,z,1,std::complex<double>(0),z,1);")
        elseif(_case STREQUAL WRITER)
            set(_call "ddla::write_matrix<ddla::DdlaBackend::CPU,double>(p,0,0,\"unused\");")
        else()
            set(_call "using F=void(*)(const ddla::DdlaHandle_t&,int,const double&,double*,int); F f=&ddla::scal<ddla::DdlaBackend::GPU,double>; (void)f;")
        endif()
        set(_result "DDLA_OLD_BACKEND_${_case}_ACCEPTED")
        unset(${_result} CACHE)
        check_cxx_source_compiles("${_prefix}\nvoid client(ddla::DdlaHandle_t h){double* p=nullptr;std::complex<double>* z=nullptr;${_call}}" ${_result})
        if(${_result})
            message(FATAL_ERROR "Old public Backend template syntax remains visible: ${_case}")
        endif()
    endforeach()
    message(STATUS "Public scalar-only positive/old-Backend negative compile checks passed")
endfunction()
