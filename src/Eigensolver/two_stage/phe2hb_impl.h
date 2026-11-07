#ifndef DDLA_PHE2HB_IMPL_H
#define DDLA_PHE2HB_IMPL_H

#include <ddla/ddla.h>
#include <array>
#include <cstddef>

namespace ddla {
namespace spectral_impl {
struct Phe2hbPlan {
    DdlaBackend backend{};std::array<int,DDLA_DLEN_> descriptor{};
    int n=0,nbw=1,nb=1,lld=1,steps=0,extent=0,ldv=1,ldc=1,upper_tile=0;
    int nprows=1,npcols=1,myprow=0,mypcol=0;
    bool inplace=false;
};
Phe2hbPlan phe2hb_geometry(const DdlaHandle_t&,DdlaBackend,int,const int*,int,int upper_tile=0,bool inplace=false);
Phe2hbPlan phe2hb_check(const DdlaHandle_t&,DdlaBackend,int,const void*,const int*,int,const void*,
    int requested_upper_tile=0,bool automatic_cpu_qr=false);

// The scratch of phe2hb_run on this rank (none without a panel).  Backend, in carving
// order: [V|U] row-distributed (ldv x nbw, then U: ldv x nbw, or max(ldv, ldc) x nbw on
// the CPU upper path, whose temporary column result it also holds), [Uc|Vc]
// column-distributed (ldc x 2*nbw), the A V partials and their row sum (ldv x nbw
// each), Z = V^H Y, the all-reduce scratch that then holds W = T Z, and M = T Z T^H
// (nbw x nbw each); then one region that the panel's QL factorization
// (phe2hb_panel_workspace_size) and the vector transposes (the largest panel's) share.
// Host: the QL factorization's.  A and tau are what the panels' vendor queries are made
// with.
template<class T>void phe2hb_workspace_size(const DdlaHandle_t& h,const Phe2hbPlan& p,T* A,T* tau,
    std::size_t* backend_bytes,std::size_t* host_bytes);
// Stage 1 on the caller's scratch.  No validation, no allocation.
template<DdlaBackend B,class T>void phe2hb_run(const DdlaHandle_t&,const Phe2hbPlan&,T* A,T* tmat,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);

/// Factor one nbw-column panel of the leading block in place (stage 1 of the
/// two-stage eigensolver; ELPA2 bandred panel step).
///
/// @p panel_start is the 0-based global index of the panel's first column and
/// @p n_cols its width (n_cols <= nbw; the panel lies inside ONE block column, so nbw
/// divides the descriptor block size); @p nbw is the semi-bandwidth.  On
/// exit, for each panel column ncol = panel_start + lc (lc = 0..n_cols-1,
/// processed descending), the entries of column ncol at rows < ncol - nbw are
/// zeroed and replaced by the Householder vector (v(nrow) = 1 at the pivot
/// row nrow = ncol - nbw is implicit), the band entry A(nrow, ncol) holds the
/// transformed pivot value, and:
///   - @p vmr (row-distributed, ld_vmr x n_cols, backend memory) holds the
///     reflector vectors with explicit anchor: rows [0, nrow] of column lc,
///     v(nrow) = 1 on the pivot-row owner, zeros elsewhere;
///   - @p tslot (backend memory, nbw x nbw, leading dimension nbw, zeroed here) holds the
///     panel's upper-triangular block-reflector factor T of Qp = I - V T V^H (= H_1...H_m):
///     T = T_L^H of the QL factorization, its diagonal the raw Larfg taus tau = conj(tau
///     of LAPACK ?GEQL2).  It is written stream-ordered (no host synchronization) on every
///     rank.
///
/// The panel is a distributed QL factorization of A(0:lr_end, panel columns),
/// lr_end = panel_start + n_cols - nbw (see phe2hb_panel.cpp): gathered to one
/// root rank, factored through the QR panel core on the reversed window, and
/// scattered back (@p inplace: factored column by column on the distributed
/// window instead); T is Row-broadcast and stored adjoint into @p tslot on every rank.
///
/// The scratch is the caller's, sized by phe2hb_panel_workspace_size.  No
/// validation, no allocation.  Every rank must call collectively.  Requires
/// panel_start >= nbw.
template <DdlaBackend Backend, typename T>
void phe2hb_panel_run(const DdlaHandle_t& handle, T* A, const int* descA,
                      int panel_start, int n_cols, int nbw, bool inplace,
                      T* vmr, int ld_vmr, T* tslot,
                      void* backend_work, std::size_t backend_bytes,
                      void* host_work, std::size_t host_bytes);

/// The scratch of phe2hb_panel_run for every panel of the leading n x n block
/// at semi-bandwidth @p nbw on this rank: the QL block driver's
/// (qr_impl::qr_block_workspace_size) for each panel's window, the largest of
/// them.  @p A and @p tau are what the vendor queries are made with.  No panel
/// (n <= nbw) needs nothing.
template <typename T>
void phe2hb_panel_workspace_size(const DdlaHandle_t& handle, const int* descA, int n, int nbw,
                                 bool inplace, T* A, T* tau,
                                 std::size_t* backend_bytes, std::size_t* host_bytes);
} // namespace spectral_impl

namespace detail {
// Private CPU stage1 policy. The measured CPU cases used NBW64 with two to
// five ranks per owner column. Keep the whole call on one panel variant,
// including late/partial panels and ranks with no local window rows.
inline bool phe2hb_cpu_qr_inplace_eligible(int n,int nb,int nbw,int nprows,int npcols) noexcept
{
    return n>=2048 && nbw==64 && nb>=nbw && nb%nbw==0 &&
           nprows>=2 && nprows<=5 && npcols>=1;
}

// Requires collectively validated metadata. A recognized DDLA_QR_PANEL takes
// precedence; an invalid value keeps the existing warning/gather behavior.
// Agrees the resolved bool before any panel allocation or communication.
bool phe2hb_cpu_qr_inplace(const DdlaHandle_t&,int n,const int* descA,int nbw);

// Private counterpart of phe2hb_upper_cpu with automatic CPU panel selection.
// Same validation, default tile, upper update and dense fallback. Return value
// still reports whether the upper update was used, not the QR policy.
template<class T>
bool phe2hb_cpu_auto_qr(const DdlaHandle_t&,int n,T* A,const int* descA,
                      int nbw,T* tmat,int& info,int tile_columns=2560);

// Private candidate, source0 only. Coarse tiles are anchored at global0;
// their full diagonal and upper offdiagonal storage remain valid. Lower
// offdiagonal tiles are intentionally unused. Small extents use the public
// full path. The tile argument makes actual skipped tiles testable cheaply.
template<class T>
bool phe2hb_upper_cpu(const DdlaHandle_t&,int n,T* A,const int* descA,
                     int nbw,T* tmat,int& info,int tile_columns=2560);

// Prepared source0 AV primitive and its dense oracle. Wr/Wc partial/result pairs
// use ld_rows/ld_cols respectively. The result-column buffer may reuse the
// future U region only if it has ld_cols*nvec capacity. All buffers are
// disjoint while live; source vectors survive both reductions/transposition.
template<class T>
void phe2hb_upper_av_cpu(const DdlaHandle_t&,int extent,const T* A,const int* descA,
                         const T* Vr,int ld_rows,const T* Vc,int ld_cols,int nvec,
                         T* Wr_partial,T* Wr,T* Wc_partial,T* Wc,
                         T* transpose_work,std::size_t transpose_elements,int tile_columns);

template<class T>
void phe2hb_upper_update_cpu(const DdlaHandle_t&,int extent,T* A,const int* descA,
                            const T* packed_rows,int ld_rows,const T* packed_cols,
                            int ld_cols,int inner,int tile_columns);
} // namespace detail
} // namespace ddla
#endif
