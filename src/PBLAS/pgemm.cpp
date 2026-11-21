#include "pgemm_impl.h"
#include "pblas_workspace.h"
#include "check_agree.h"
#include "instantiate.h"

#include <ddla/ddla_vector.h>
#include <vector>

namespace ddla {
namespace pgemm_impl {

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void run(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* A, const int* descA,
    const TB* B, const int* descB,
    const T& beta,
    T* C, const int* descC, Algorithm algorithm)
{
    int panel_width = kDefaultPanelWidth;
    algorithm = pgemm_check<Backend>(handle, transa, transb, m, n, k, alpha == T(0),
                                    descA, descB, descC, algorithm, panel_width);
    if (m == 0 || n == 0) return;
    const bool scale_only = k == 0 || alpha == T(0);
    if constexpr (!std::is_same_v<T, TA> || !std::is_same_v<T, TB>) {
        detail::agree_check(handle, "pgemm(mixed)", [&](detail::Agree&) {
            int pr, pc, r, c;
            ddlaGetGridDims(handle, pr, pc); ddlaGetGridCoords(handle, r, c);
            auto owns = [&](int rows, int cols, const int* d) {
                return num_loc(rows,d[DDLA_MB_],r,d[DDLA_RSRC_],pr)>0 &&
                       num_loc(cols,d[DDLA_NB_],c,d[DDLA_CSRC_],pc)>0;
            };
            if (!(scale_only && beta == T(1)) && owns(m,n,descC) && C==nullptr)
                throw std::invalid_argument("pgemm(mixed): null local C");
            if (!scale_only &&
                ((owns(transa=='N'?m:k,transa=='N'?k:m,descA) && A==nullptr) ||
                 (owns(transb=='N'?k:n,transb=='N'?n:k,descB) && B==nullptr)))
                throw std::invalid_argument("pgemm(mixed): null local input");
        });
    }
    if (scale_only && beta == T(1)) return;
    std::size_t backend_bytes = 0, host_bytes = 0;
    vector<char> backend_work(handle);
    std::vector<char> host_work;
    auto prepare = [&] {
        pgemm_workspace_size<T, TA, TB>(handle, algorithm, panel_width, alpha == T(0), transa, transb, m, n, k, beta != T(0),
                                        descA, descB, descC, &backend_bytes, &host_bytes);
        backend_work.resize(backend_bytes);
        host_work.resize(host_bytes);
        if constexpr(Backend==DdlaBackend::GPU) {
            if(handle->communication()==DdlaCommunication::HostStagedMPI) {
                const auto staged=pgemm_staging_bytes(handle,algorithm,!scale_only,transa,transb,
                    m,n,k,beta!=T(0),descA,descB,descC,
                    sizeof(TA),sizeof(TB),sizeof(T));
                (void)handle->host_buffers<std::byte>(staged[0],staged[1]);
            }
        }
    };
    if (handle->nprocs > 1)
        detail::agree_local(handle, "pgemm workspace preparation", prepare);
    else
        prepare();
    pgemm_run<Backend, T, TA, TB>(handle, algorithm, panel_width, transa, transb, m, n, k, alpha,
                          A, descA, B, descB, beta, C, descC,
                          backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

} // namespace pgemm_impl

template <DdlaBackend Backend, typename T>
void pgemm(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC)
{
    pgemm_impl::run<Backend>(handle, transa, transb, m, n, k, alpha,
                            A, descA, B, descB, beta, C, descC, pgemm_impl::Algorithm::Auto);
}

template <DdlaBackend Backend, typename R>
void pgemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
           const std::complex<R>& alpha, const R* a, const int* da,
           const std::complex<R>* b, const int* db, const std::complex<R>& beta,
           std::complex<R>* c, const int* dc)
{
    pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a,da,b,db,beta,c,dc,pgemm_impl::Algorithm::Auto);
}

template <DdlaBackend Backend, typename R>
void pgemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
           const std::complex<R>& alpha, const std::complex<R>* a, const int* da,
           const R* b, const int* db, const std::complex<R>& beta,
           std::complex<R>* c, const int* dc)
{
    pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a,da,b,db,beta,c,dc,pgemm_impl::Algorithm::Auto);
}
// Each component owns the explicit instantiations of its implementation.
#define INSTANTIATE_PGEMM(BACKEND, TYPE) \
    template void pgemm<BACKEND, TYPE>(const DdlaHandle_t&, \
        char, char, int, int, int, \
        const TYPE&, const TYPE*, const int*, const TYPE*, const int*, \
        const TYPE&, TYPE*, const int*); \
    template void pgemm_impl::run<BACKEND, TYPE>(const DdlaHandle_t&, \
        char, char, int, int, int, \
        const TYPE&, const TYPE*, const int*, const TYPE*, const int*, \
        const TYPE&, TYPE*, const int*, pgemm_impl::Algorithm)

// Mixed-domain instantiations retain real input storage and complex output.
#define INSTANTIATE_PGEMM_MIXED(BACKEND, R, TA, TB) \
    template void pgemm<BACKEND, R>(const DdlaHandle_t&, char, char, int, int, int, \
        const std::complex<R>&, const TA*, const int*, const TB*, const int*, \
        const std::complex<R>&, std::complex<R>*, const int*); \
    template void pgemm_impl::run<BACKEND, std::complex<R>, TA, TB>(const DdlaHandle_t&, \
        char, char, int, int, int, const std::complex<R>&, const TA*, const int*, const TB*, const int*, \
        const std::complex<R>&, std::complex<R>*, const int*, pgemm_impl::Algorithm);
#define INSTANTIATE_PGEMM_MIXED_BACKEND(BACKEND) \
    INSTANTIATE_PGEMM_MIXED(BACKEND, float, float, std::complex<float>); \
    INSTANTIATE_PGEMM_MIXED(BACKEND, float, std::complex<float>, float); \
    INSTANTIATE_PGEMM_MIXED(BACKEND, double, double, std::complex<double>); \
    INSTANTIATE_PGEMM_MIXED(BACKEND, double, std::complex<double>, double)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGEMM)
DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGEMM_MIXED_BACKEND)

#undef INSTANTIATE_PGEMM_MIXED_BACKEND
#undef INSTANTIATE_PGEMM_MIXED
#undef INSTANTIATE_PGEMM

} // namespace ddla
