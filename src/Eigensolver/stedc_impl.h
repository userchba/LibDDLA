#ifndef DDLA_STEDC_IMPL_H
#define DDLA_STEDC_IMPL_H

#include "pstedc_impl.h"

namespace ddla { namespace spectral_impl {

// Local geometry: never derive communicator ranks or collective policy here.
// scratch is private, prevalidated PSTEDC factor storage; the public
// owning call always passes null.
struct StedcPlan {
    char compz = 'N';
    int n = 0, lwork = 0, lrwork = 0, liwork = 0;
    bool pack_vectors = false;
    std::size_t backend_bytes = 0, host_bytes = 0;
    std::size_t engine_backend = 0, engine_host = 0, multiply_bytes = 0;
    PstedcVectorPlan rows;
};

template<class T>
StedcPlan stedc_plan(const DdlaHandle_t& h, char compz, int n, int ldz);

template<DdlaBackend B, class T>
void stedc_run(const DdlaHandle_t& h, const StedcPlan& p, PstedcResources& resources,
               typename real_type<T>::type* d, typename real_type<T>::type* e,
               T* z, int ldz, int& info, void* backend_work, void* host_work,
               typename real_type<T>::type* scratch = nullptr);

} } // namespace ddla::spectral_impl
#endif
