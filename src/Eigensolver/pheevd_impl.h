#ifndef DDLA_PHEEVD_IMPL_H
#define DDLA_PHEEVD_IMPL_H
#include <ddla/ddla.h>

namespace ddla { namespace detail {
// The single-rank adapter preserves descriptor padding and host eigenvalues.
template <DdlaBackend B, class T>
void pheevd_local(const DdlaHandle_t&, char, char, int, T*, const int*,
                  typename real_type<T>::type*, T*, const int*, int&);

} }
#endif
