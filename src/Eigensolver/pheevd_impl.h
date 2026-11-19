#ifndef DDLA_PHEEVD_DISPATCH_IMPL_H
#define DDLA_PHEEVD_DISPATCH_IMPL_H
#include <ddla/ddla.h>

namespace ddla { namespace detail {
enum class PheevdRoute { Local, OneStage, TwoStage };
// The public selector, including automatic single-rank/vendor and tuned routes.
// Selection is collective and belongs outside sizing and execution.
template <DdlaBackend B, class T>
PheevdRoute pheevd_select(const DdlaHandle_t&, char, int, int, const int*, const int*);

// The single-rank adapter preserves descriptor padding and host eigenvalues.
template <DdlaBackend B, class T>
void pheevd_local(const DdlaHandle_t&, char, char, int, int, T*, const int*,
                  typename real_type<T>::type*, T*, const int*, int&);

} }
#endif
