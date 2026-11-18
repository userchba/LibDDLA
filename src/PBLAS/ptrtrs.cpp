#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "BLAS/blas_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "pblas_workspace.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "ptrtrs_impl.h"
#include "instantiate.h"
namespace ddla{

namespace ptrtrs_impl {

// Component 1.
template <DdlaBackend Backend>
void ptrtrs_check(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
                  int m, int n, const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "ptrtrs", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "ptrtrs");

        if (array_descA[DDLA_MB_] != array_descA[DDLA_NB_])
            throw std::invalid_argument("ptrtrs: A needs square blocks");
        if (array_descA[DDLA_MB_] != array_descB[DDLA_MB_])
            throw std::invalid_argument("ptrtrs: A and B need the same row block size");
        if (side != 'L' && side != 'R')
            throw std::invalid_argument("ptrtrs: side must be 'L' or 'R'");
        // The solve walks B with A's distribution: B's rows (side 'L') or
        // columns (side 'R') must be owned like the matching rows/columns of A.
        if (side == 'L' && array_descA[DDLA_RSRC_] != array_descB[DDLA_RSRC_])
            throw std::invalid_argument("ptrtrs: side 'L' needs B's rows distributed like A's (RSRC)");
        if (side == 'R' && (array_descA[DDLA_NB_] != array_descB[DDLA_NB_] ||
                            array_descA[DDLA_CSRC_] != array_descB[DDLA_CSRC_]))
            throw std::invalid_argument("ptrtrs: side 'R' needs B's columns distributed like A's (NB, CSRC)");
        if (uplo != 'L' && uplo != 'U')
            throw std::invalid_argument("ptrtrs: uplo must be 'L' or 'U'");
        if (diag != 'U' && diag != 'N')
            throw std::invalid_argument("ptrtrs: diag must be 'U' or 'N'");
        if (trans != 'N' && trans != 'T' && trans != 'C')
            throw std::invalid_argument("ptrtrs: trans must be 'N', 'T' or 'C'");
        // The descriptor may describe a matrix larger than the logical
        // sub-matrix (leading block anchored at global (0,0)).
        const int n_solve = (side == 'L') ? m : n;
        if (n_solve > array_descA[DDLA_M_] || n_solve > array_descA[DDLA_N_] ||
            m > array_descB[DDLA_M_] || n > array_descB[DDLA_N_])
            throw std::invalid_argument("ptrtrs: m x n exceeds the descriptors");
    });
}

namespace {
struct SolvePanel { int m,n,i,j; };

// Stored far-side panel of A. Preparation and execution share only descriptor
// geometry; solve order, pointer offsets and all BLAS calls remain unchanged.
SolvePanel solve_panel(char side,char uplo,char trans,int order,int start,int width)
{
    const bool left=side=='L';
    const bool backward=(left?uplo=='U':uplo=='L')==(trans=='N');
    const int origin=backward?0:start+width;
    const int length=backward?start:order-start-width;
    const bool column=left==(trans=='N');
    return column?SolvePanel{length,width,origin,start}:SolvePanel{width,length,start,origin};
}

// Element counts of ptrtrs's own buffers, in carving order: the diagonal
// block (nb x nb) | the panel of A (a block column of the local rows or a
// block row of the local columns) | the block row/column of B.
void scratch_counts(const DdlaHandle_t& handle, char side, int m, int n,
                    const int* array_descA, const int* array_descB, std::size_t (&count)[3])
{
    const int nb = array_descA[DDLA_MB_], order = side == 'L' ? m : n;
    const int ar = num_loc(order, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    const int ac = num_loc(order, array_descA[DDLA_NB_], handle->mypcol_, array_descA[DDLA_CSRC_], handle->npcols_);
    const int br = num_loc(m, array_descB[DDLA_MB_], handle->myprow_, array_descB[DDLA_RSRC_], handle->nprows_);
    const int bc = num_loc(n, array_descB[DDLA_NB_], handle->mypcol_, array_descB[DDLA_CSRC_], handle->npcols_);
    count[0] = std::size_t(nb) * nb;
    count[1] = std::size_t(std::max(ar, ac)) * nb;
    count[2] = std::size_t(nb) * std::max(br, bc);
}
} // namespace

// Component 2.  Backend [ diagonal block | panel of A | block of B
// (scratch_counts) | the panel transport's send staging ], host [ the panel
// transport's schedule ].  The A panel of a transposed solve is a row panel
// (side 'L', nb x n_solve) or a column panel (side 'R', n_solve x nb)
// gathered through pgeadd and row/column broadcasts with reusable redistribution scratch.
template <typename T>
void ptrtrs_workspace_size(const DdlaHandle_t& handle, char side, char trans, int m, int n,
                           const int* array_descA, const int* array_descB,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m == 0 || n == 0 || handle->nprocs == 1) return;
    const int nb = array_descA[DDLA_MB_], order = side == 'L' ? m : n;
    std::size_t count[3], transport_bytes = 0;
    scratch_counts(handle, side, m, n, array_descA, array_descB, count);
    const bool left = side=='L';
    const bool stored_column = left == (trans=='N');
    pblas::panel_workspace_size<T>(handle,left,trans,
        stored_column?order:nb,stored_column?nb:order,array_descA,array_descA,
        &transport_bytes,host_bytes);
    for (const std::size_t c : count) *backend_bytes += detail::padded_bytes<T>(c);
    *backend_bytes += transport_bytes;
}

std::array<std::size_t,2> ptrtrs_staging_bytes(const DdlaHandle_t& h,
    char side,char uplo,char trans,int m,int n,const int* da,const int* db,std::size_t scalar_bytes)
{
    std::array<std::size_t,2> bytes{};
    if(h->backend!=DdlaBackend::GPU || h->communication()!=DdlaCommunication::HostStagedMPI ||
       !m || !n || h->nprocs==1)return bytes;
    const bool left=side=='L';
    const int order=left?m:n,nb=da[DDLA_MB_];
    for(int start=0;start<order;) {
        const int width=std::min(nb,order-start);
        const int owner=indxg2p(start,nb,da[left?DDLA_RSRC_:DDLA_CSRC_],
            left?h->nprows_:h->npcols_);
        if((left?h->myprow_:h->mypcol_)==owner)
            bytes[0]=std::max(bytes[0],pblas::communication_bytes(std::size_t(width)*width,scalar_bytes));
        pblas::staging_max(bytes,pblas::broadcast_staging_bytes(h,!left,
            left?width:m,left?n:width,left?start:0,left?0:start,db,scalar_bytes));
        const auto panel=solve_panel(side,uplo,trans,order,start,width);
        pblas::staging_max(bytes,pblas::panel_staging_bytes(h,left,trans,
            panel.m,panel.n,panel.i,panel.j,da,da,scalar_bytes));
        start+=width;
    }
    return bytes;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void ptrtrs_run(const DdlaHandle_t& handle,
                char side, char uplo, char trans, char diag, int m, int n,
                const T* A, const int* array_descA, T* B, const int* array_descB,
                void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes)
{
    try {
        if(m==0 || n==0)return;
        if (handle->nprocs == 1) {
            // The existing TRSM adapter takes a mutable pointer but only reads A.
            blas_impl::trsm_run<Backend,T>(handle,side,uplo,trans,diag,m,n,T(1),
                const_cast<T*>(A),array_descA[DDLA_LLD_],B,array_descB[DDLA_LLD_]);
            return;
        }
        const bool left = side == 'L';
        const int order = left ? m : n;
        const int nb = array_descA[DDLA_MB_];
        const int lda = array_descA[DDLA_LLD_];
        const int ldb = array_descB[DDLA_LLD_];
        const int row = handle->myprow_, col = handle->mypcol_;
        const int pr = handle->nprows_, pc = handle->npcols_;
        const int coord = left ? row : col;
        const int procs = left ? pr : pc;
        const int source = array_descA[left ? DDLA_RSRC_ : DDLA_CSRC_];
        const int local_order = num_loc(order, nb, coord, source, procs);
        const int ml = num_loc(m, array_descB[DDLA_MB_], row, array_descB[DDLA_RSRC_], pr);
        const int nl = num_loc(n, array_descB[DDLA_NB_], col, array_descB[DDLA_CSRC_], pc);

        // The scratch layout is shared with ptrtrs_workspace_size. The remaining
        // space belongs to panel transport; no allocation occurs in the sweep.
        std::size_t count[3];
        scratch_counts(handle, side, m, n, array_descA, array_descB, count);
        char* w = static_cast<char*>(backend_work);
        T* const block_diag = detail::take<T>(w, count[0]);
        T* const block_A = detail::take<T>(w, count[1]);
        T* const block_B = detail::take<T>(w, count[2]);
        const std::size_t used = static_cast<std::size_t>(w - static_cast<char*>(backend_work));
        assert(used <= backend_bytes);
        void* const transport_work = w;
        const std::size_t transport_bytes = backend_bytes - used;

        // Left solves visit lower op(A) forward and upper op(A) backward.
        // Right solves reverse that order; both update the unsolved part of B.
        const bool lower = (uplo == 'L') == (trans == 'N');
        const bool backward = left ? !lower : lower;
        const int blocks = (order - 1) / nb + 1;
        for (int step = 0; step < blocks; ++step) {
            const int k = (backward ? blocks - 1 - step : step) * nb;
            const int width = std::min(nb, order - k);
            const int ar = num_loc(k, nb, row, array_descA[DDLA_RSRC_], pr);
            const int ac = num_loc(k, nb, col, array_descA[DDLA_CSRC_], pc);
            const int owner_row = indxg2p(k, nb, array_descA[DDLA_RSRC_], pr);
            const int owner_col = indxg2p(k, nb, array_descA[DDLA_CSRC_], pc);
            const int local_k = left ? ar : ac;
            const bool owns_block = coord == (left ? owner_row : owner_col);
            const int local_width = owns_block ? width : 0;

            if (row == owner_row && col == owner_col) {
                DDLA_CHECK(ddlaMemcpy2D(handle, block_diag, std::size_t(width) * sizeof(T),
                    A + ar + std::size_t(ac) * lda, std::size_t(lda) * sizeof(T),
                    std::size_t(width) * sizeof(T), width));
            }
            DDLA_CHECK(ddlaSynchronize(handle));
            if (owns_block) {
                commBcast<Backend>(handle, left ? CommScope::Row : CommScope::Col,
                    block_diag, std::size_t(width) * width, left ? owner_col : owner_row);
                blas_impl::trsm_run<Backend, T>(handle, side, uplo, trans, diag,
                    left ? width : ml, left ? nl : width, T(1), block_diag, width,
                    B + (left ? std::size_t(local_k) : std::size_t(local_k) * ldb), ldb);
            }

            // Share the solved B row/column, then pack the corresponding panel
            // of op(A). solve_panel also supplies the host-staging prediction.
            pblas::broadcast_panel<Backend, T>(handle, !left,
                left ? width : m, left ? n : width, B, left ? k : 0, left ? 0 : k,
                array_descB, block_B);
            const auto panel = solve_panel(side, uplo, trans, order, k, width);
            pblas::panel_run<Backend, T>(handle, left, trans, panel.m, panel.n,
                A, panel.i, panel.j, array_descA, array_descA, block_A,
                transport_work, transport_bytes, host_work, host_bytes);

            const int first = backward ? 0 : local_k + local_width;
            const int length = backward ? local_k : local_order - first;
            DDLA_CHECK(ddlaSynchronize(handle));
            if (left) {
                if (length > 0)
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'N', length, nl, width,
                        T(-1), block_A, length, block_B, width, T(1), B + first, ldb);
            } else {
                if (length > 0 && ml > 0)
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'N', ml, length, width,
                        T(-1), block_B, ml, block_A, width, T(1),
                        B + std::size_t(first) * ldb, ldb);
            }
            DDLA_CHECK(ddlaSynchronize(handle));
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace ptrtrs_impl

template <DdlaBackend Backend, typename T>
void ptrtrs(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
    int m, int n,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    ptrtrs_impl::ptrtrs_check<Backend>(handle, side, uplo, trans, diag, m, n, array_descA, array_descB);
    if (m == 0 || n == 0) return;
    std::size_t backend_bytes = 0, host_bytes = 0;
    vector<char> backend_work(handle);
    std::vector<char> host_work;
    auto prepare = [&] {
        ptrtrs_impl::ptrtrs_workspace_size<T>(handle, side, trans, m, n, array_descA, array_descB,
                                              &backend_bytes, &host_bytes);
        backend_work.resize(backend_bytes);
        host_work.resize(host_bytes);
        const auto staged=ptrtrs_impl::ptrtrs_staging_bytes(handle,side,uplo,trans,m,n,
            array_descA,array_descB,sizeof(T));
        if(staged[0] || staged[1])handle->host_buffers<std::byte>(staged[0],staged[1]);
    };
    if (handle->nprocs > 1)
        detail::agree_local(handle, "ptrtrs workspace preparation", prepare);
    else
        prepare();
    ptrtrs_impl::ptrtrs_run<Backend, T>(handle, side, uplo, trans, diag, m, n, A, array_descA, B, array_descB,
                                        backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}


#define INSTANTIATE_PTRTRS(BACKEND, TYPE)                                   \
    template void ptrtrs<BACKEND, TYPE>(const DdlaHandle_t&, char side, char uplo, char trans, char diag, \
    int m, int n, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB); \
    template void ptrtrs_impl::ptrtrs_run<BACKEND, TYPE>(const DdlaHandle_t&, \
    char side, char uplo, char trans, char diag, int m, int n, \
    const TYPE* A, const int* array_descA, TYPE* B, const int* array_descB, \
    void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PTRTRS_CHECK(BACKEND) \
    template void ptrtrs_impl::ptrtrs_check<BACKEND>(const DdlaHandle_t&, char, char, char, char, \
    int, int, const int*, const int*)

#define INSTANTIATE_PTRTRS_SIZE(TYPE) \
    template void ptrtrs_impl::ptrtrs_workspace_size<TYPE>(const DdlaHandle_t&, char, char, \
    int, int, const int*, const int*, std::size_t*, std::size_t*)

DDLA_FOR_EACH_BACKEND(INSTANTIATE_PTRTRS_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PTRTRS)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PTRTRS_SIZE)

#undef INSTANTIATE_PTRTRS_SIZE
#undef INSTANTIATE_PTRTRS_CHECK
#undef INSTANTIATE_PTRTRS



}
