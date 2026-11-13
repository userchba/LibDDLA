#include <ddla/ddla.h>
#include "check_agree.h"
#include "instantiate.h"
#include "eig_params.h"
#include "pheevd_impl.h"
#include "host_threads.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ddla {
namespace {
// A nonempty two-stage option keeps its existing meaning under automatic
// dispatch. The common tridiagonal-solver and thread controls apply to both.
bool two_stage_options_requested()
{
    constexpr const char* options[] = {
        "DDLA_EIG_NBW", "DDLA_EIG_B2F_BLOCKING", "DDLA_EIG_STRIPE_WIDTH",
        "DDLA_EIG_CPU_ACCUM", "DDLA_EIG_CHASE_TEAM_THREADS", "DDLA_EIG_CHASE_TEAM_RANKS",
        "DDLA_EIG_CHASE_THREADS", "DDLA_EIG_CHASE_TEAM", "DDLA_EIG_CHASE_RING_RECORDS",
        "DDLA_EIG_CHASE_TEAM_DEBUG_MAPS", "DDLA_EIG_CHASE_TEAM_PREFAULT",
        "DDLA_EIG_HOST_ACCUM_BLOCK", "DDLA_EIG_HOST_ACCUM_SLOTS", "DDLA_EIG_HOST_ACCUM_RB",
        "DDLA_EIG_HOST_FINISH_PANEL", "DDLA_EIG_REPLAY_TILE", "DDLA_EIG_REPLAY_WAVE",
        "DDLA_EIG_REPLAY_RB", "DDLA_EIG_REPLAY_LAM", "DDLA_EIG_REPLAY_PROF"
    };
    for (const char* name : options)
        if (const char* value = std::getenv(name); value && *value) return true;
    return eig_profile_enabled();
}
} // namespace

template <DdlaBackend Backend, typename T>
void pheevd(const DdlaHandle_t& handle, char jobz, char uplo, int n,
            T* A, const int* descA, typename real_type<T>::type* w,
            T* Z, const int* descZ, int& info)
{
    info = 0;
    if (!handle)
        throw std::invalid_argument("pheevd: null handle");
    int prefer_one = 0;
    bool automatic_shape = false;
    const int mode = detail::agree_check(handle, "pheevd", [&](detail::Agree& agreement) {
        if (handle->comm == MPI_COMM_NULL || handle->nprocs < 1 ||
            handle->nprows_ < 1 || handle->npcols_ < 1 ||
            static_cast<long long>(handle->nprows_) * handle->npcols_ != handle->nprocs)
            throw std::invalid_argument("pheevd: initialized process grid required");
        if (ddlaGetBackend(handle) != Backend)
            throw std::invalid_argument("pheevd: backend mismatch");
        const char* method = std::getenv("DDLA_EIGEN_METHOD");
        int selected = 0;
        if (method && *method && std::strcmp(method, "auto") != 0) {
            if (std::strcmp(method, "one_stage") == 0) selected = 1;
            else if (std::strcmp(method, "two_stage") == 0) selected = 2;
            else throw std::invalid_argument("DDLA_EIGEN_METHOD must be auto, one_stage or two_stage");
        }
        agreement.mode("DDLA_EIGEN_METHOD", selected);
        agreement.mode("pheevd backend", int(Backend));
        automatic_shape = !selected && descA && descA[DDLA_MB_] == descA[DDLA_NB_] &&
            descA[DDLA_RSRC_] == 0 && descA[DDLA_CSRC_] == 0 &&
            detail::pheevd_one_stage_shape<Backend,T>(jobz,n,descA[DDLA_MB_],
                handle->nprows_,handle->npcols_,handle->major);
#if !defined(DDLA_USE_CUDA) || !defined(DDLA_USE_CCL)
        if constexpr (Backend == DdlaBackend::GPU) automatic_shape = false;
#else
        if constexpr (Backend == DdlaBackend::GPU)
            automatic_shape = automatic_shape && handle->communication == DdlaCommunication::CCL;
#endif
        // Shape metadata is global; worker budgets, tuning options and devices
        // may differ locally. An ineligible rank requests the conservative route.
        if (jobz == 'V')
            automatic_shape = automatic_shape && descZ && descZ[DDLA_MB_] == descA[DDLA_MB_] &&
                descZ[DDLA_NB_] == descA[DDLA_NB_] && descZ[DDLA_RSRC_] == 0 && descZ[DDLA_CSRC_] == 0;
        agreement.mode("pheevd automatic shape", automatic_shape ? 1 : 0);
        if (automatic_shape && handle->node_ranks == handle->nprocs &&
            !two_stage_options_requested()) {
            const int threads = detail::ScopedBlasThreads::current();
            if constexpr (Backend == DdlaBackend::CPU) {
                prefer_one = handle->host_threads == detail::pheevd_method_cpu_threads &&
                             threads == detail::pheevd_method_cpu_threads;
            } else {
                prefer_one = handle->host_threads == 1 && threads == 1;
#if defined(DDLA_USE_CUDA) && defined(DDLA_USE_CCL)
                if (prefer_one) {
                    int major = 0, minor = 0;
                    GPU_CHECK(gpuDeviceGetAttribute(&major, gpuDevAttrComputeCapabilityMajor, handle->local_device));
                    GPU_CHECK(gpuDeviceGetAttribute(&minor, gpuDevAttrComputeCapabilityMinor, handle->local_device));
                    prefer_one = major == 7 && minor == 0;
                }
#endif
            }
        }
        return selected;
    });
    if (!mode && handle->nprocs == 1) {
        detail::pheevd_local<Backend,T>(handle,jobz,uplo,n,A,descA,w,Z,descZ,info);
        return;
    }
    if (!mode && automatic_shape && handle->nprocs > 1)
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &prefer_one, 1, MPI_INT, MPI_MIN, handle->comm));
    const bool one_stage = mode == 1 || (!mode && prefer_one);
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
