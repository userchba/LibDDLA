// ---------------------------------------------------------------------------
// Stage 5 of the two-stage Hermitian eigensolver: apply the band-stage
// orthogonal transform to Z (ELPA2 trans_ev_band_to_full, compact WY within each panel).
//
// phe2hb produced A_out = Q_total A_pre Q_total^H, i.e. A_pre = Q_total^H
// A_out Q_total, with Q_total = Qp_1 Qp_2 ... Qp_nsteps and
// Qp_i = I - V_i T_i V_i^H (panel i's reflectors, panels combined in
// ascending istep order).  Eigenvectors therefore map as
// Y (of A_out) -> Q_total^H Y (of A_pre), and this routine offers both
// directions:
//
//   trans = 'N'  Z := Q_total   Z  -- loop istep = nsteps .. 1 (current
//                                       first-applied panel is the largest
//                                       istep), each Z := Z - V (T (V^H Z));
//   trans = 'C'  Z := Q_total^H Z  -- loop istep = 1 .. nsteps, each
//                                       Z := Z - V (T^H (V^H Z)).
//
// Both orderings are needed: the eigensolver driver back-transforms with 'C',
// while 'N' is the inverse direction (and the mode the isolation test pins by
// checking that Z = I maps to Q_total).
//
// V is gathered row-distributed from A's reflector-storage region and T is
// read from the replicated tmat slots.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include "Eigensolver/eig_workspace.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "plan_util.h"
#include "punmtr_he2hb_grouped.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ddla {

namespace spectral_impl {
UnmtrHe2hbPlan punmtr_he2hb_geometry(
    const DdlaHandle_t& h,
    DdlaBackend backend,
    char trans,
    int n,
    int nbw,
    const int* a,
    const int* z)
{
    UnmtrHe2hbPlan p;
    p.backend = backend;
    p.trans = trans;
    p.n = n;
    p.nbw = nbw;
    p.nb = a[DDLA_NB_];
    p.lldA = a[DDLA_LLD_];
    p.lldZ = z[DDLA_LLD_];
    p.nprows = h->nprows_;
    p.npcols = h->npcols_;
    p.myprow = h->myprow_;
    p.mypcol = h->mypcol_;
    p.nsteps = n ? (n - 1) / nbw : 0;
    p.zcols = num_loc(n, z[DDLA_NB_], p.mypcol, z[DDLA_CSRC_], p.npcols);
    if (p.nsteps) {
        const int last = p.nsteps * nbw;
        const int nc = std::min(n, last + nbw) - last;
        p.ldv = std::max(num_loc(last + nc - nbw, p.nb, p.myprow, 0, p.nprows), 1);
    }
    return p;
}

UnmtrHe2hbPlan punmtr_he2hb_check(
    const DdlaHandle_t& handle,
    DdlaBackend backend,
    char trans,
    int n,
    int nbw,
    const void* A,
    const int* descA,
    const void* tmat,
    const void* Z,
    const int* descZ)
{
    if (!handle)
        throw std::runtime_error("punmtr_he2hb: null handle");
    if (ddlaGetBackend(handle) != backend)
        throw std::runtime_error("punmtr_he2hb: template backend does not match handle backend");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        throw std::invalid_argument("punmtr_he2hb: invalid trans");
    check_desc(descA, handle);
    check_desc(descZ, handle);
    const int nb = descA[DDLA_NB_];
    if (descA[DDLA_MB_] != nb)
        throw std::runtime_error("punmtr_he2hb: requires square blocks for A");
    if (descA[DDLA_RSRC_] != 0 || descA[DDLA_CSRC_] != 0)
        throw std::runtime_error("punmtr_he2hb: requires RSRC == CSRC == 0 for A");
    if (descZ[DDLA_MB_] != nb || descZ[DDLA_RSRC_] != descA[DDLA_RSRC_])
        throw std::runtime_error("punmtr_he2hb: Z row block size and source must match A");
    if (nbw <= 0 || nbw > nb || nb % nbw != 0)
        throw std::runtime_error("punmtr_he2hb: nbw must divide the block size nb");
    if (n < 0 || n > descA[DDLA_M_] || n > descA[DDLA_N_] || n > descZ[DDLA_M_] ||
        n > descZ[DDLA_N_])
        throw std::invalid_argument("punmtr_he2hb: order exceeds descriptor");
    const auto p = punmtr_he2hb_geometry(handle, backend, trans, n, nbw, descA, descZ);
    if (p.nsteps) {
        const int rows = num_loc(n, nb, p.myprow, 0, p.nprows);
        const int cols = num_loc(n, nb, p.mypcol, 0, p.npcols);
        if (!tmat || (rows > 0 && cols > 0 && !A) || (rows > 0 && p.zcols > 0 && !Z))
            throw std::invalid_argument("punmtr_he2hb: missing active input/output");
    }
    return p;
}

template <class T>
void punmtr_he2hb_workspace_size(
    const DdlaHandle_t& h, const UnmtrHe2hbPlan& p, std::size_t* backend_bytes)
{
    const auto options = detail::grouped_stage5_default_options<T>(p.backend, p.n);
    const auto grouped = detail::grouped_stage5_geometry(p, options);
    detail::grouped_stage5_workspace_size<T>(h, grouped, backend_bytes);
}

template <DdlaBackend Backend, class T>
void punmtr_he2hb_run(
    const DdlaHandle_t& h,
    const UnmtrHe2hbPlan& p,
    const T* A,
    const int*,
    const T* tmat,
    T* Z,
    const int*,
    int& info,
    void* work,
    std::size_t bytes)
{
    const auto options = detail::grouped_stage5_default_options<T>(p.backend, p.n);
    const auto grouped = detail::grouped_stage5_geometry(p, options);
    try {
        detail::grouped_stage5_run<Backend, T>(h, grouped, A, tmat, Z, info, work, bytes);
        if constexpr (Backend == DdlaBackend::GPU)
            DDLA_CHECK(ddlaSynchronize(h));
    } catch (...) {
        if constexpr (Backend == DdlaBackend::GPU) {
            try {
                (void)ddlaSynchronize(h);
            } catch (...) {
            }
        }
        throw;
    }
}

} // namespace spectral_impl

template <DdlaBackend Backend, class T>
void punmtr_he2hb(
    const DdlaHandle_t& handle,
    char trans,
    int n,
    int nbw,
    const T* A,
    const int* descA,
    const T* tmat,
    T* Z,
    const int* descZ,
    int& info)
{
    info = 0;
    const auto p =
        spectral_impl::punmtr_he2hb_check(handle, Backend, trans, n, nbw, A, descA, tmat, Z, descZ);
    std::size_t backend_bytes = 0;
    spectral_impl::punmtr_he2hb_workspace_size<T>(handle, p, &backend_bytes);
    vector<char> backend_work(handle, backend_bytes);
    spectral_impl::punmtr_he2hb_run<Backend, T>(
        handle, p, A, descA, tmat, Z, descZ, info, backend_work.data(), backend_bytes);
}

#define INSTANTIATE_PUNMTR_HE2HB(BACKEND, TYPE)                                                    \
    template void punmtr_he2hb<BACKEND, TYPE>(const DdlaHandle_t&,                                 \
                                              char,                                                \
                                              int,                                                 \
                                              int,                                                 \
                                              const TYPE*,                                         \
                                              const int*,                                          \
                                              const TYPE*,                                         \
                                              TYPE*,                                               \
                                              const int*,                                          \
                                              int&);                                               \
    template void spectral_impl::punmtr_he2hb_run<BACKEND, TYPE>(                                  \
        const DdlaHandle_t&,                                                                       \
        const spectral_impl::UnmtrHe2hbPlan&,                                                      \
        const TYPE*,                                                                               \
        const int*,                                                                                \
        const TYPE*,                                                                               \
        TYPE*,                                                                                     \
        const int*,                                                                                \
        int&,                                                                                      \
        void*,                                                                                     \
        std::size_t)
#define INSTANTIATE_PUNMTR_HE2HB_SIZE(TYPE)                                                        \
    template void spectral_impl::punmtr_he2hb_workspace_size<TYPE>(                                \
        const DdlaHandle_t&, const spectral_impl::UnmtrHe2hbPlan&, std::size_t*)

INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, float);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, double);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, float);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, double);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PUNMTR_HE2HB_SIZE(float);
INSTANTIATE_PUNMTR_HE2HB_SIZE(double);
INSTANTIATE_PUNMTR_HE2HB_SIZE(std::complex<float>);
INSTANTIATE_PUNMTR_HE2HB_SIZE(std::complex<double>);

#undef INSTANTIATE_PUNMTR_HE2HB_SIZE
#undef INSTANTIATE_PUNMTR_HE2HB

} // namespace ddla
