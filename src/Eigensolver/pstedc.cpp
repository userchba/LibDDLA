// Owning entry points for distributed tridiagonal diagonalization.
#include <ddla/ddla.h>

#include "check_agree.h"
#include "ddla_stream.h"
#include "eig_memprof.h"
#include "pstedc_impl.h"

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla {

template <typename rt>
void pstedc_allocate(const DdlaHandle_t& handle, const spectral_impl::PstedcPlan& p,
                     rt* diag_in, rt* e_in, rt* w, rt* z, rt* scratch = nullptr)
{
    const int n = p.rows.n;
    const bool vectors = p.mode != spectral_impl::PstedcMode::values;
    if (!n)
        return;
    std::optional<detail::EigMemProfile> profile;
    {
        std::size_t backend_bytes = 0, host_bytes = 0;
        vector<char> backend_work(handle);
        std::vector<char> host_work;
        std::optional<spectral_impl::PstedcResources> rows;
        // Queries, allocation, upload fences and the optional profiling thread
        // may fail on one rank. Prepare all of them before collective execution.
        detail::agree_local(handle, "pstedc: workspace preparation", [&] {
            if (vectors && handle->backend == DdlaBackend::GPU) {
                profile.emplace(handle, p.rows.profile, true);
                profile->phase("start");
            }
            spectral_impl::pstedc_workspace_size<rt>(handle, p, &backend_bytes, &host_bytes);
            backend_work.resize(backend_bytes);
            host_work.resize(host_bytes);
            if (vectors)
                rows.emplace(handle, p.rows);
        });
        auto run = [&](auto backend) {
            constexpr DdlaBackend B = decltype(backend)::value;
            if (vectors)
                spectral_impl::pstedc_run<B, rt>(handle, p.rows, *rows, diag_in, e_in, w, z, scratch,
                    backend_work.data(), backend_bytes, host_work.data(), host_bytes);
            else
                spectral_impl::pstedc_run<B, rt>(handle, p, diag_in, e_in, w, z, backend_work.data(),
                    backend_bytes, host_work.data(), host_bytes, nullptr);
        };
        switch (handle->backend) {
        case DdlaBackend::CPU:
            run(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
            break;
#if DDLA_HAS_GPU
        case DdlaBackend::GPU:
            run(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
            break;
#endif
        default:
            throw std::runtime_error("pstedc: backend not built");
        }
    } // Drain upload fences and release solve scratch before the final mark.
    if (profile) {
        profile->phase("pstedc");
        profile->report(handle->myid, n);
    }
}

template <typename T>
void pstedc(const DdlaHandle_t& handle, char jobz, int n, int nblk, typename real_type<T>::type* d,
            typename real_type<T>::type* e, typename real_type<T>::type* w,
            typename real_type<T>::type* z, int ldz, int& info)
{
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "pstedc operates on real scalar types");
    info = 0;
    const auto p = spectral_impl::pstedc_check<typename real_type<T>::type>(handle, jobz, n, nblk, d, e, w, z, ldz);
    pstedc_allocate(handle, p, d, e, w, z);
}

template void pstedc<float>(const DdlaHandle_t&, char, int, int, float*, float*, float*, float*, int,
                                       int&);
template void pstedc<double>(const DdlaHandle_t&, char, int, int, double*, double*, double*, double*,
                                        int, int&);

template <typename rt>
void pstedc(const DdlaHandle_t& handle, int n, int nblk, rt* d, rt* e, rt* w, rt* Y, int ldy)
{
    pstedc(handle, n, nblk, d, e, w, Y, ldy, static_cast<rt*>(nullptr), 0);
}

template <typename rt>
void pstedc(const DdlaHandle_t& handle, int n, int nblk, rt* d, rt* e, rt* w, rt* Y, int ldy, rt* scratch,
                 std::size_t scratch_elements)
{
    static_assert(std::is_same<rt, float>::value || std::is_same<rt, double>::value,
                  "pstedc operates on the real types (float, double)");
    if (handle == nullptr)
        throw std::runtime_error("pstedc: null handle");
    if (n < 0 || nblk < 1)
        throw std::runtime_error("pstedc: invalid n or nblk");
    if (n == 0)
        return;
    const auto backend = ddlaGetBackend(handle);
    if (!ddlaBackendAvailable(backend))
        throw std::runtime_error("pstedc: the handle's backend is not built into this library");
    spectral_impl::PstedcPlan p;
    p.mode = spectral_impl::PstedcMode::vectors;
    p.rows = spectral_impl::pstedc_check(handle, n, nblk, d, e, w, Y, ldy, scratch, scratch_elements);
    pstedc_allocate(handle, p, d, e, w, Y, scratch);
}

template void pstedc<float>(const DdlaHandle_t&, int, int, float*, float*, float*, float*, int);
template void pstedc<double>(const DdlaHandle_t&, int, int, double*, double*, double*, double*, int);
template void pstedc<float>(const DdlaHandle_t&, int, int, float*, float*, float*, float*, int, float*,
                                 std::size_t);
template void pstedc<double>(const DdlaHandle_t&, int, int, double*, double*, double*, double*, int, double*,
                                  std::size_t);

} // namespace ddla
