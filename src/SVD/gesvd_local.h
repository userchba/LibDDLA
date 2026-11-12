#ifndef DDLA_GESVD_LOCAL_H
#define DDLA_GESVD_LOCAL_H
#include <ddla/ddla.h>
#include <cstddef>
namespace ddla { namespace spectral_impl {
// GPU GESVD accepts m>=n. Wide GPU inputs keep the existing local blocked
// algorithm rather than allocating an extra global transpose.
template <class T>
void local_gesvd_workspace_size(const DdlaHandle_t&, char, char, int, int,
                                std::size_t*, std::size_t*);
template <DdlaBackend B, class T>
void local_gesvd_run(const DdlaHandle_t&, char, char, int, int, T*, int,
    typename real_type<T>::type*, T*, int, T*, int, int&,
    void*, std::size_t, void*, std::size_t);
} }
#endif
