#ifndef DDLA_INSTANTIATE_H
#define DDLA_INSTANTIATE_H

// ---------------------------------------------------------------------------
// Internal helper: the explicit-instantiation lists every source file ends with.
//
//     #define INSTANTIATE(BACKEND, TYPE) template void f<BACKEND, TYPE>(...)
//     DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
//     #undef INSTANTIATE
//
// expands INSTANTIATE for the CPU backend, for the GPU backend when this build
// has one, and for float, double, std::complex<float> and std::complex<double>.
// The for-each macro writes the ';' after each expansion, so the macro handed
// to it must not end in one.
// ---------------------------------------------------------------------------

#include <ddla/ddla_config.h>
#include <ddla/ddla_handle_t.h>
#include "backend_api.h"
#include <complex>

#define DDLA_FOR_EACH_TYPE(M) \
    M(float); M(double); M(std::complex<float>); M(std::complex<double>);

#define DDLA_FOR_EACH_TYPE_OF(M, BACKEND) \
    M(BACKEND, float); M(BACKEND, double); \
    M(BACKEND, std::complex<float>); M(BACKEND, std::complex<double>);

#if DDLA_HAS_GPU
#define DDLA_FOR_EACH_BACKEND(M) M(::ddla::DdlaBackend::CPU); M(::ddla::DdlaBackend::GPU);
#define DDLA_FOR_EACH_BACKEND_TYPE(M) \
    DDLA_FOR_EACH_TYPE_OF(M, ::ddla::DdlaBackend::CPU) \
    DDLA_FOR_EACH_TYPE_OF(M, ::ddla::DdlaBackend::GPU)
#else
#define DDLA_FOR_EACH_BACKEND(M) M(::ddla::DdlaBackend::CPU);
#define DDLA_FOR_EACH_BACKEND_TYPE(M) DDLA_FOR_EACH_TYPE_OF(M, ::ddla::DdlaBackend::CPU)
#endif

#endif // DDLA_INSTANTIATE_H
