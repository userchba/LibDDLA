# Compile-only checks use public include directories, with no backend_api.h.
# A positive control prevents missing headers/toolchain failures from making
# the expected negative checks appear to pass.
function(ddla_check_public_scalar_api)
    include(CheckCXXSourceCompiles)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_REQUIRED_INCLUDES ${DDLA_PUBLIC_CHECK_INCLUDES} ${MPI_CXX_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES "")
    set(CMAKE_REQUIRED_DEFINITIONS "")
    set(CMAKE_REQUIRED_FLAGS "")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_REQUIRED_FLAGS "-pedantic-errors")
    endif()
    set(_prefix "#include <ddla/ddla.h>\n#include <complex>\n#include <type_traits>\n")
    set(_positive "${_prefix}
void client(ddla::DdlaHandle_t h) {
    double* p=nullptr; std::complex<double>* z=nullptr; int info=0,iterations=0,converged=0;
    float* f=nullptr; std::complex<float>* c=nullptr;
    ddla::ddlaInit(h,ddla::DdlaBackend::CPU);
    ddla::gemm<double>(h,'N','N',0,0,0,1.,p,1,p,1,0.,p,1);
    ddla::scal(h,0,1.,p,1);
    ddla::rscl(h,0,1.,p,1);
    ddla::rscl<std::complex<double>>(h,0,2.,z,1);
    ddla::rscl(h,0,std::complex<float>(2,1),c,1);
    ddla::gelqf(h,0,0,p,1,p);
    ddla::gerqf<std::complex<double>>(h,0,0,z,1,z);
    ddla::pgelqf(h,0,0,p,nullptr,p,info);
    ddla::pgerqf<std::complex<double>>(h,0,0,z,nullptr,z,info);
    std::complex<double> logdet;
    ddla::diag_logdet(h,0,p,1,logdet);
    ddla::pdiag_logdet(h,0,z,nullptr,logdet);
    ddla::gemm<double>(h,'N','N',0,0,0,std::complex<double>(1),p,1,z,1,std::complex<double>(0),z,1);
    ddla::write_matrix<double>(h,p,0,0,\"unused\");
    ddla::phegst<double>(h,'L',0,p,nullptr,static_cast<const double*>(p),nullptr);
    ddla::phegst<std::complex<double>>(h,'U',0,z,nullptr,static_cast<const std::complex<double>*>(z),nullptr);
    ddla::pheevd_chebyshev<float>(h,'L',0,0,0,f,nullptr,f,f,nullptr,1e-5,20,12,false,iterations,converged,info);
    ddla::pheevd_chebyshev<double>(h,'L',0,0,0,p,nullptr,p,p,nullptr,1e-10,20,12,false,iterations,converged,info);
    ddla::pheevd_chebyshev<std::complex<float>>(h,'L',0,0,0,c,nullptr,f,c,nullptr,1e-5,20,12,false,iterations,converged,info);
    ddla::pheevd_chebyshev<std::complex<double>>(h,'L',0,0,0,z,nullptr,p,z,nullptr,1e-10,20,12,false,iterations,converged,info);
    static_assert(std::is_same<decltype(ddla::ppotrf<double>(h,'L',0,p,1,1,nullptr,info)),void>::value,\"Cholesky status API\");
}")
    unset(DDLA_PUBLIC_SCALAR_SYNTAX CACHE)
    check_cxx_source_compiles("${_positive}" DDLA_PUBLIC_SCALAR_SYNTAX)
    if(NOT DDLA_PUBLIC_SCALAR_SYNTAX)
        message(FATAL_ERROR "Public scalar-only API positive compile control failed")
    endif()
    foreach(_case GEMM SCAL MIXED WRITER POINTER INIT DEFAULT_BACKEND PHEGST CHEBYSHEV CHEBYSHEV_DEFAULTS)
        if(_case STREQUAL GEMM)
            set(_call "ddla::gemm<ddla::DdlaBackend::CPU,double>(h,'N','N',0,0,0,1.,p,1,p,1,0.,p,1);")
        elseif(_case STREQUAL SCAL)
            set(_call "ddla::scal<ddla::DdlaBackend::CPU>(h,0,1.,p,1);")
        elseif(_case STREQUAL MIXED)
            set(_call "ddla::gemm<ddla::DdlaBackend::GPU,double>(h,'N','N',0,0,0,std::complex<double>(1),p,1,z,1,std::complex<double>(0),z,1);")
        elseif(_case STREQUAL WRITER)
            set(_call "ddla::write_matrix<ddla::DdlaBackend::CPU,double>(p,0,0,\"unused\");")
        elseif(_case STREQUAL PHEGST)
            set(_call "ddla::phegst<ddla::DdlaBackend::CPU,double>(h,'L',0,p,nullptr,p,nullptr);")
        elseif(_case STREQUAL CHEBYSHEV)
            set(_call "ddla::pheevd_chebyshev<ddla::DdlaBackend::CPU,double>(h,'L',0,0,0,p,nullptr,p,p,nullptr,1e-10,20,12,false,iterations,converged,info);")
        elseif(_case STREQUAL CHEBYSHEV_DEFAULTS)
            set(_call "ddla::pheevd_chebyshev<double>(h,'L',0,0,0,p,nullptr,p,p,nullptr);")
        elseif(_case STREQUAL INIT)
            set(_call "ddla::ddlaInit(h);")
        elseif(_case STREQUAL DEFAULT_BACKEND)
            set(_call "auto backend=ddla::default_backend_v; (void)backend;")
        else()
            set(_call "using F=void(*)(const ddla::DdlaHandle_t&,int,const double&,double*,int); F f=&ddla::scal<ddla::DdlaBackend::GPU,double>; (void)f;")
        endif()
        set(_result "DDLA_OLD_BACKEND_${_case}_ACCEPTED")
        unset(${_result} CACHE)
        check_cxx_source_compiles("${_prefix}\nvoid client(ddla::DdlaHandle_t h){double* p=nullptr;std::complex<double>* z=nullptr;int iterations=0,converged=0,info=0;${_call}}" ${_result})
        if(${_result})
            if(_case STREQUAL CHEBYSHEV_DEFAULTS)
                message(FATAL_ERROR "Chebyshev controls unexpectedly have defaults")
            else()
                message(FATAL_ERROR "Old public Backend template syntax remains visible: ${_case}")
            endif()
        endif()
    endforeach()
    message(STATUS "Explicit backend initialization compile checks passed")
    message(STATUS "Public scalar-only positive/old-Backend negative compile checks passed")
endfunction()
