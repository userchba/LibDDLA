#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_connector.h"
#include "pheevd_one_stage.h"
#include "one_stage_workspace.h"
#include "phetrd_impl.h"
#include "punmtr_trd_impl.h"
#include "check_agree.h"
#include "plan_util.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "Eigensolver/pheevd_alias.h"
#include "matrix_transfer.h"
#include "Eigensolver/pherm_complete_workspace.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <complex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla { namespace detail {
template <bool StoredOnly, DdlaBackend B, class T>
void one_stage_solve(const DdlaHandle_t& h, char job, char uplo, int n, T* a, const int* da,
                     typename real_type<T>::type* eval, T* z, const int* dz, int& info,
                     OneStageTimes* times) {
    info = 0;
    const auto p = spectral_impl::one_stage_check<T>(h, B, StoredOnly, job, uplo, n, a, da, eval, z, dz);
    if (times)
        *times = {};
    if (!n)
        return;
    // size, allocation and resources may fail on one rank only: agree before the collective run
    std::size_t backend_bytes = 0, host_bytes = 0;
    ddla::vector<char> backend_work(h);
    std::vector<char> host_work;
    std::optional<spectral_impl::OneStageResources> resources;
    agree_check(h,"pheevd_one_stage allocation",[&](Agree&) {
        spectral_impl::one_stage_workspace_size<T>(h, p, &backend_bytes, &host_bytes);
        backend_work.resize(backend_bytes);
        host_work.resize(host_bytes);
        resources.emplace(h, p);
    });
    spectral_impl::one_stage_run<B, T>(h, p, *resources, a, eval, z, info, times, backend_work.data(),
                                       backend_bytes, host_work.data(), host_bytes);
}

// Preserve the original private timing ABI as well as the public fast entry.
template <DdlaBackend B, class T>
void pheevd_one_stage_timed(const DdlaHandle_t& h, char job, char uplo, int n, T* a, const int* da,
                            typename real_type<T>::type* eval, T* z, const int* dz, int& info,
                            OneStageTimes* times) {
    one_stage_solve<false, B, T>(h, job, uplo, n, a, da, eval, z, dz, info, times);
}

template <DdlaBackend B, class T>
void pheevd_one_stage_uplo_timed(const DdlaHandle_t& h, char job, char uplo, int n, T* a, const int* da,
                                 typename real_type<T>::type* eval, T* z, const int* dz, int& info,
                                 OneStageTimes* times) {
    one_stage_solve<true, B, T>(h, job, uplo, n, a, da, eval, z, dz, info, times);
}
} // namespace detail

namespace spectral_impl {
template <class T>
OneStagePlan one_stage_check(const DdlaHandle_t& h, DdlaBackend backend, bool stored_only, char job,
                             char uplo, int n, const T* a, const int* da,
                             const typename real_type<T>::type* eval, const T* z, const int* dz) {
    using R = typename real_type<T>::type;
    auto p = detail::agree_check(h, "pheevd_one_stage", [&](detail::Agree&) {
        if (!h)
            throw std::invalid_argument("pheevd_one_stage: null handle");
        if (ddlaGetBackend(h) != backend || n < 0 || (job != 'N' && job != 'V') ||
            (uplo != 'U' && uplo != 'L'))
            throw std::invalid_argument("pheevd_one_stage: invalid arguments");
        check_desc(da, h);
        if (da[DDLA_MB_] != da[DDLA_NB_] || da[DDLA_RSRC_] != 0 || da[DDLA_CSRC_] != 0 || n > da[DDLA_M_] ||
            n > da[DDLA_N_] || n > INT_MAX / 2)
            throw std::invalid_argument(
                "pheevd_one_stage: requires square source-zero blocks and valid extent");
        if (n && !eval)
            throw std::invalid_argument("pheevd_one_stage: missing eigenvalues");
        OneStagePlan result;
        result.backend = backend;
        result.job = job;
        result.uplo = uplo;
        result.stored_only = stored_only;
        result.n = n;
        ddlaGetGridDims(h, result.pr, result.pc);
        ddlaGetGridCoords(h, result.mr, result.mc);
        std::copy(da, da + DDLA_DLEN_, result.a.begin());
        const int ar = num_loc(n, da[DDLA_MB_], result.mr, 0, result.pr),
                  ac = num_loc(n, da[DDLA_NB_], result.mc, 0, result.pc);
        if (ar > 0 && ac > 0 && !a)
            throw std::invalid_argument("pheevd_one_stage: missing active input");
        if (job == 'V') {
            check_desc(dz, h);
            if (dz[DDLA_MB_] != da[DDLA_MB_] || dz[DDLA_RSRC_] != 0 || n > dz[DDLA_M_] || n > dz[DDLA_N_])
                throw std::invalid_argument("pheevd_one_stage: incompatible output rows");
            const int zr = num_loc(n, dz[DDLA_MB_], result.mr, dz[DDLA_RSRC_], result.pr);
            const int zc = num_loc(n, dz[DDLA_NB_], result.mc, dz[DDLA_CSRC_], result.pc);
            if (zr > 0 && zc > 0 && !z)
                throw std::invalid_argument("pheevd_one_stage: missing active output");
            std::copy(dz, dz + DDLA_DLEN_, result.z.begin());
            const auto span = [](int rows, int cols, int ld) {
                return rows && cols ? (static_cast<std::size_t>(cols - 1) * ld + rows) * sizeof(T)
                                    : std::size_t(0);
            };
            const auto ab = span(ar, ac, da[DDLA_LLD_]), zb = span(zr, zc, dz[DDLA_LLD_]);
            const auto ap = reinterpret_cast<std::uintptr_t>(a), zp = reinterpret_cast<std::uintptr_t>(z);
            result.backup = ab && zb && (ap <= zp ? zp - ap < ab : ap - zp < zb);
            result.saved = result.a;
            result.saved[DDLA_M_] = result.saved[DDLA_N_] = n;
            result.saved[DDLA_LLD_] = std::max(1, ar);
            result.output_scratch = detail::output_scratch_elements(h, n, a, da, const_cast<T*>(z), dz);
            result.tridiagonal_vectors = result.z;
            if constexpr (!std::is_same<T, R>::value)
                result.tridiagonal_vectors[DDLA_M_] = result.tridiagonal_vectors[DDLA_N_] = n;
            result.tridiagonal_vectors[DDLA_LLD_] =
                std::is_same<T, R>::value ? dz[DDLA_LLD_] : std::max(1, zr);
        }
        return result;
    });
    int sig[21] = {n, job, uplo, int(backend), stored_only ? 1 : 0, h->nprows_, h->npcols_};
    detail::descriptor_signature(sig + 7, da);
    if (job == 'V')
        detail::descriptor_signature(sig + 14, dz);
    detail::agree_signature(h, "pheevd_one_stage", sig);
    if (!n)
        return p;
    if (!stored_only)
        p.complete = pherm_complete_check(h, backend, uplo, n, a, da);
    if (job == 'V') {
        int overlap = p.backup ? 1 : 0;
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &overlap, 1, MPI_INT, MPI_MAX, h->comm));
        p.backup = overlap != 0;
        p.vectors = pstedc_policy<R>(h, n);
        const detail::RowShare rows(n,h->nprocs);
        p.vectors.ldy = std::max(1,rows.count(h->myid));
        // The driver owns a separate, tightly packed row slab, disjoint from d/e/w.
        p.vectors.borrow_q2 = rows.count(h->myid)>0;

    } else
        p.values = pstedc_values_policy<R>(h, n);
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
    if(backend==DdlaBackend::GPU)detail::agree_check(h,"one-stage communication storage",[&](detail::Agree&){
        const int nb=da[DDLA_NB_];
        const auto nr=std::size_t(num_loc(n,nb,0,0,p.pr)),nc=std::size_t(num_loc(n,nb,0,0,p.pc));
        const auto width=std::size_t(job=='V'?std::min(nb,n-1):0);
        const auto zc=std::size_t(job=='V'?num_loc(n,dz[DDLA_NB_],0,0,p.pc):0);
        // Reduction vectors, reflector panels/Gram matrices, and their sums.
        const auto elements=std::max({nr+1,nc+1,std::max({nr,nc,zc})*width,width*width});
        std::size_t send=elements*sizeof(T),recv=send;
        if(!stored_only){std::size_t pack,local,unpack,diag,tx,rx;
            pherm_complete_geometry(p.complete,pack,local,unpack,diag,tx,rx);
            send=std::max(send,tx*sizeof(T));recv=std::max(recv,rx*sizeof(T));}
        if(job=='V'){std::size_t b=0,s=0;matrix_impl::scatter_rows_workspace_size<R>(h,n,&b,&s);
            send=std::max(send,(b+1)/2);}
        detail::tunnel_staging<char>(h,0,send);detail::tunnel_staging<char>(h,1,recv);
    });
#endif
    return p;
}

template <class T>
void one_stage_workspace_size(const DdlaHandle_t& h, const OneStagePlan& p, std::size_t* backend_bytes,
                              std::size_t* host_bytes) {
    using R = typename real_type<T>::type;
    *backend_bytes = *host_bytes = 0;
    if (!p.n)
        return;
    std::size_t phase = 0, phase_host = 0, bytes = 0, host = 0;
    const auto stage = [&](std::size_t backend, std::size_t host) {
        phase = std::max(phase, backend);
        phase_host = std::max(phase_host, host);
    };
    phetrd_workspace_size<T>(h, p, &bytes, &host);
    stage(bytes, host);
    if (!p.stored_only) {
        const std::size_t capacity =
            p.backend == DdlaBackend::GPU ? p.output_scratch / (sizeof(T) / sizeof(R)) : 0;
        pherm_complete_workspace_size<T>(h, p.complete, capacity, &bytes, &host);
        stage(bytes, host);
    }
    if (p.job == 'V') {
        const std::size_t y =
            std::is_same<T, R>::value
                ? 0
                : detail::padded_bytes<R>(static_cast<std::size_t>(p.tridiagonal_vectors[DDLA_LLD_]) *
                                          num_loc(p.n, p.z[DDLA_NB_], p.mc, p.z[DDLA_CSRC_], p.pc));
        const std::size_t rows=detail::padded_bytes<R>(std::size_t(p.vectors.ldy)*p.n);
        pstedc_workspace_size<R>(h, p.vectors, &bytes, &host);
        stage(rows + bytes, host);
        matrix_impl::scatter_rows_workspace_size<R>(h,p.n,&bytes,&host);
        stage(rows + y + bytes,host);
        const auto apply=punmtr_trd_geometry(h,p.backend,'L',p.stored_only?p.uplo:'U','N',p.n,p.n,p.a.data(),p.z.data());
        punmtr_trd_workspace_size<T>(h,apply,&bytes);host=0;
        stage(bytes, host);
    } else {
        pstedc_workspace_size<R>(h, p.values, &bytes, &host);
        stage(bytes, host);
    }
    const std::size_t backup = p.backup ? detail::padded_bytes<T>(std::max(
                                              std::size_t(1), static_cast<std::size_t>(p.saved[DDLA_LLD_]) *
                                                                  num_loc(p.n, p.a[DDLA_NB_], p.mc, 0, p.pc)))
                                        : 0;
    *backend_bytes = detail::padded_bytes<T>(std::max(1, p.n - 1)) + backup + phase;
    *host_bytes = detail::padded_bytes<R>(2 * static_cast<std::size_t>(p.n)) + phase_host;
}

template <DdlaBackend B, class T>
void one_stage_run(const DdlaHandle_t& h, const OneStagePlan& p, OneStageResources& resources, T* a,
                   typename real_type<T>::type* eval, T* z, int& info, detail::OneStageTimes* times,
                   void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes) {
    try {
        using R = typename real_type<T>::type;
        info = 0;
        if (times)
            *times = {};
        if (!p.n)
            return;
        const int n = p.n;
        const int* da = p.a.data();
        const int* dz = p.z.data();
        // tau, the reflector copy and d | e in the order one_stage_workspace_size counts them;
        // the phases take turns on the rest of each kind.
        char* const base = static_cast<char*>(backend_work);
        char* w = base;
        char* const host_base = static_cast<char*>(host_work);
        char* hw = host_base;
        T* const tau = detail::take<T>(w, std::max(1, n - 1));
        T* const saved =
            p.backup
                ? detail::take<T>(w, std::max(std::size_t(1), static_cast<std::size_t>(p.saved[DDLA_LLD_]) *
                                                                  num_loc(n, da[DDLA_NB_], p.mc, 0, p.pc)))
                : nullptr;
        R* const de = detail::take<R>(hw, std::size_t(2) * n);
        assert(static_cast<std::size_t>(w - base) <= backend_bytes &&
               static_cast<std::size_t>(hw - host_base) <= host_bytes);
        void* const phase = w;
        const std::size_t phase_bytes = backend_bytes - static_cast<std::size_t>(w - base);
        void* const phase_host = hw;
        const std::size_t phase_host_bytes = host_bytes - static_cast<std::size_t>(hw - host_base);
        double last = 0;
        auto mark = [&](double* value) {
            if (times) {
                DDLA_CHECK(ddlaSynchronize(h));
                const double now = MPI_Wtime();
                if (value)
                    *value = now - last;
                last = now;
            }
        };
        mark(nullptr);
        if (!p.stored_only) {
            const auto capacity = B == DdlaBackend::GPU ? p.output_scratch / (sizeof(T) / sizeof(R)) : 0;
            pherm_complete_run<B, T>(h, p.complete, a, capacity ? z : nullptr, capacity, phase, phase_bytes,
                                     phase_host, phase_host_bytes);
        }
        mark(times ? &times->prepare : nullptr);

        phetrd_run<B,T>(h,p,resources,a,tau,de,info,phase,phase_bytes,phase_host,phase_host_bytes);
        mark(times ? &times->reduction : nullptr);
        const T* reflectors = a;
        const int* reflector_desc = da;
        if (p.backup) {
            const int ar = num_loc(n, da[DDLA_MB_], p.mr, 0, p.pr),
                      ac = num_loc(n, da[DDLA_NB_], p.mc, 0, p.pc), ld = p.saved[DDLA_LLD_];
            detail::agree_check(h,"pheevd_one_stage reflector copy",[&](detail::Agree&) {
                if (p.stored_only)
                    for (int c = 0; c < ac; ++c) {
                        const int global = indxl2g(c, da[DDLA_NB_], p.mc, 0, p.pc);
                        const int first =
                            p.uplo == 'L' ? num_loc(global + 1, da[DDLA_MB_], p.mr, 0, p.pr) : 0;
                        const int end = p.uplo == 'U' ? num_loc(global, da[DDLA_MB_], p.mr, 0, p.pr) : ar;
                        if (end > first)
                            DDLA_CHECK(ddlaMemcpy(h, saved + first + std::size_t(c) * ld,
                                                  a + first + std::size_t(c) * da[DDLA_LLD_],
                                                  std::size_t(end - first) * sizeof(T)));
                    }
                else if (ar && ac)
                    DDLA_CHECK(ddlaMemcpy2D(h, saved, std::size_t(ld) * sizeof(T), a,
                                            std::size_t(da[DDLA_LLD_]) * sizeof(T),
                                            std::size_t(ar) * sizeof(T), ac));
                DDLA_CHECK(ddlaSynchronize(h));
            });
            reflectors = saved;
            reflector_desc = p.saved.data();
        }
        if (p.job == 'N') {
            pstedc_run<B, R>(h, p.values, de, de + n, eval, nullptr, phase, phase_bytes,
                             phase_host, phase_host_bytes);
            mark(times ? &times->solve : nullptr);
            return;
        }
        // Keep the row slab until the existing bounded transfer has read it.
        // Real panels are communicated before any local complex widening.
        char* solve = static_cast<char*>(phase);
        R* const rows=detail::take<R>(solve,std::size_t(p.vectors.ldy)*n);
        const std::size_t solve_bytes=phase_bytes-std::size_t(solve-static_cast<char*>(phase));
        assert(resources.vectors);
        pstedc_run<B,R>(h,p.vectors,*resources.vectors,de,de+n,eval,rows,nullptr,
            solve,solve_bytes,phase_host,phase_host_bytes);
        mark(times ? &times->solve : nullptr);
        const int zr = num_loc(n, dz[DDLA_MB_], p.mr, dz[DDLA_RSRC_], p.pr);
        const int zc = num_loc(n, dz[DDLA_NB_], p.mc, dz[DDLA_CSRC_], p.pc);
        const int ldq = p.tridiagonal_vectors[DDLA_LLD_];
        R* y = nullptr;
        if constexpr (std::is_same<T, R>::value)
            y = z;
        else
            y = detail::take<R>(solve, static_cast<std::size_t>(ldq) * zc);
        matrix_impl::scatter_rows_run<B,R>(h,n,rows,p.vectors.ldy,false,n,n,y,
            p.tridiagonal_vectors.data(),solve,
            phase_bytes-std::size_t(solve-static_cast<char*>(phase)),phase_host,phase_host_bytes);
        if constexpr (!std::is_same<T, R>::value) {
            matrix_impl::convert_rows<B,T>(h, y, ldq, z, dz[DDLA_LLD_], zr, zc, false);
            // The next stage reuses the real input storage.
            DDLA_CHECK(ddlaSynchronize(h));
        }
        mark(times ? &times->redistribution : nullptr);
        const auto apply=punmtr_trd_geometry(h,B,'L',p.stored_only?p.uplo:'U','N',n,n,reflector_desc,dz);
        punmtr_trd_run<B,T>(h,apply,reflectors,tau,z,phase,phase_bytes);
        mark(times ? &times->backtransform : nullptr);
        DDLA_CHECK(ddlaSynchronize(h));
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try {
                (void)ddlaSynchronize(h);
            } catch (...) {
            }
        throw;
    }
}

#define INST_ONE_STAGE_CHECK(T)                                                                              \
    template OneStagePlan one_stage_check<T>(const DdlaHandle_t&, DdlaBackend, bool, char, char, int,        \
                                             const T*, const int*, const typename real_type<T>::type*,       \
                                             const T*, const int*);                                          \
    template void one_stage_workspace_size<T>(const DdlaHandle_t&, const OneStagePlan&, std::size_t*,        \
                                              std::size_t*)
INST_ONE_STAGE_CHECK(float);
INST_ONE_STAGE_CHECK(double);
INST_ONE_STAGE_CHECK(std::complex<float>);
INST_ONE_STAGE_CHECK(std::complex<double>);
#undef INST_ONE_STAGE_CHECK
} // namespace spectral_impl

template <DdlaBackend B, class T>
void pheevd_one_stage(const DdlaHandle_t& h, char job, char uplo, int n, T* a, const int* da,
                      typename real_type<T>::type* eval, T* z, const int* dz, int& info) {
    detail::pheevd_one_stage_timed<B, T>(h, job, uplo, n, a, da, eval, z, dz, info, nullptr);
}

template <DdlaBackend B, class T>
void pheevd_one_stage_uplo(const DdlaHandle_t& h, char job, char uplo, int n, T* a, const int* da,
                           typename real_type<T>::type* eval, T* z, const int* dz, int& info) {
    detail::pheevd_one_stage_uplo_timed<B, T>(h, job, uplo, n, a, da, eval, z, dz, info, nullptr);
}

#define INST(B, T)                                                                                           \
    template void spectral_impl::one_stage_run<B, T>(                                                        \
        const DdlaHandle_t&, const spectral_impl::OneStagePlan&, spectral_impl::OneStageResources&, T*,      \
        typename real_type<T>::type*, T*, int&, detail::OneStageTimes*, void*, std::size_t, void*,           \
        std::size_t);                                                                                        \
    template void pheevd_one_stage<B, T>(const DdlaHandle_t&, char, char, int, T*, const int*,               \
                                         typename real_type<T>::type*, T*, const int*, int&);                \
    template void detail::pheevd_one_stage_timed<B, T>(const DdlaHandle_t&, char, char, int, T*, const int*, \
                                                       typename real_type<T>::type*, T*, const int*, int&,   \
                                                       detail::OneStageTimes*);                              \
    template void pheevd_one_stage_uplo<B, T>(const DdlaHandle_t&, char, char, int, T*, const int*,          \
                                              typename real_type<T>::type*, T*, const int*, int&);           \
    template void detail::pheevd_one_stage_uplo_timed<B, T>(const DdlaHandle_t&, char, char, int, T*,        \
                                                            const int*, typename real_type<T>::type*, T*,    \
                                                            const int*, int&, detail::OneStageTimes*);
#if DDLA_HAS_CPU
INST(DdlaBackend::CPU, float)
INST(DdlaBackend::CPU, double)
INST(DdlaBackend::CPU, std::complex<float>)
INST(DdlaBackend::CPU, std::complex<double>)
#endif
#if DDLA_HAS_GPU
INST(DdlaBackend::GPU, float)
INST(DdlaBackend::GPU, double)
INST(DdlaBackend::GPU, std::complex<float>)
INST(DdlaBackend::GPU, std::complex<double>)
#endif
#undef INST
} // namespace ddla
