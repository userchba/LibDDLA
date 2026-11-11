#include <ddla/ddla.h>
#include "check_agree.h"
#include "instantiate.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ddla {

template <DdlaBackend Backend, typename T>
void pheevd(const DdlaHandle_t& handle, char jobz, char uplo, int n,
            T* A, const int* descA, typename real_type<T>::type* w,
            T* Z, const int* descZ, int& info)
{
    info = 0;
    if (!handle)
        throw std::invalid_argument("pheevd: null handle");
    const bool one_stage = detail::agree_check(handle, "pheevd", [&](detail::Agree& agreement) {
        if (handle->comm == MPI_COMM_NULL || handle->nprocs < 1 ||
            handle->nprows_ < 1 || handle->npcols_ < 1 ||
            static_cast<long long>(handle->nprows_) * handle->npcols_ != handle->nprocs)
            throw std::invalid_argument("pheevd: initialized process grid required");
        const char* method = std::getenv("DDLA_EIGEN_METHOD");
        const bool one = method && std::strcmp(method, "one_stage") == 0;
        if (method && *method && !one && std::strcmp(method, "two_stage") != 0)
            throw std::invalid_argument("DDLA_EIGEN_METHOD must be one_stage or two_stage");
        // Agree before entering either algorithm's collective schedule.
        agreement.mode("DDLA_EIGEN_METHOD", one ? 1 : 2);
        return one;
    });
    if (one_stage)
        pheevd_one_stage<Backend, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
    else
        pheevd_two_stage<Backend, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
}

#define INSTANTIATE_PHEEVD(BACKEND, TYPE)                                        \
    template void pheevd<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, int,               \
        TYPE*, const int*, typename real_type<TYPE>::type*,                      \
        TYPE*, const int*, int&)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PHEEVD)

#undef INSTANTIATE_PHEEVD

} // namespace ddla
