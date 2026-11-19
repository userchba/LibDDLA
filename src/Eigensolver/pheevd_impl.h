#ifndef DDLA_PHEEVD_DISPATCH_IMPL_H
#define DDLA_PHEEVD_DISPATCH_IMPL_H
#include <ddla/ddla.h>
#include <cstddef>

namespace ddla { namespace detail {
enum class PheevdRoute { Local, OneStage, TwoStage };
// The public selector, including automatic single-rank/vendor and tuned routes.
// Selection is collective and belongs outside sizing and execution.
template <DdlaBackend B, class T>
PheevdRoute pheevd_select(const DdlaHandle_t&, char, int, int, const int*, const int*);

struct LocalPheevdPlan {
    DdlaBackend backend{};
    char job='N', uplo='U';
    int n=0, nev=0, lda=1, ldz=1;
    bool copy=false, overlap=false;
};
struct LocalPheevdWorkspace {
    int lwork=0, lrwork=0, liwork=0;
    std::size_t padding=0, backend_bytes=0, host_bytes=0;
};
template <DdlaBackend B, class T>
LocalPheevdPlan pheevd_local_check(const DdlaHandle_t&, char, char, int, int,
    const T*, const int*, const typename real_type<T>::type*, const T*, const int*);
// query_values is a valid backend pointer on GPU. The caller prepares its
// O(n) values/info resources before this allocation-free vendor query.
template <class T>
LocalPheevdWorkspace pheevd_local_workspace_size(const DdlaHandle_t&, const LocalPheevdPlan&,
    T*, typename real_type<T>::type* query_values);
// No allocation/query. Device values/info are separate explicitly prepared
// O(n) resources. Work/padding and a post-solve overlap copy reuse one region.
template <DdlaBackend B, class T>
void pheevd_local_run(const DdlaHandle_t&, const LocalPheevdPlan&, const LocalPheevdWorkspace&,
    T*, typename real_type<T>::type*, T*, int&, typename real_type<T>::type*, int*,
    void*, std::size_t, void*, std::size_t);
// Owning convenience adapter, preserving descriptor padding and host values.
template <DdlaBackend B, class T>
void pheevd_local(const DdlaHandle_t&, char, char, int, int, T*, const int*,
    typename real_type<T>::type*, T*, const int*, int&);
} }
#endif
