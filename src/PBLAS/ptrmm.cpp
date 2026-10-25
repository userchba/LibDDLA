#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "comm_traits.h"
#include "require_backend.h"
#include "transport_block.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace ddla {
namespace {

// Extract only the off-diagonal panel, in op(A)'s orientation and B's
// distribution along the long dimension. The short dimension is replicated.
template <DdlaBackend Backend, typename T>
void triangular_panel(const DdlaHandle_t& h, bool left, char trans,
                      int k, int width, int first, int length,
                      const T* A, const int* da, const int* db,
                      vector<T>& panel, vector<T>& work)
{
    const int nb = da[DDLA_MB_];
    const int coord = left ? h->myprow_ : h->mypcol_;
    const int procs = left ? h->nprows_ : h->npcols_;
    const int target = db[left ? DDLA_RSRC_ : DDLA_CSRC_];
    if (trans == 'N') {
        transport_block<Backend>(h, left ? 'C' : 'R', 'N',
            left ? length : width, left ? width : length,
            A, left ? first : k, left ? k : first, da, panel.data());
        const int source = da[left ? DDLA_RSRC_ : DDLA_CSRC_];
        if (source != target) {
            const int send_length = num_loc(first + length, nb, coord, source, procs)
                                  - num_loc(first, nb, coord, source, procs);
            const int recv_length = num_loc(first + length, nb, coord, target, procs)
                                  - num_loc(first, nb, coord, target, procs);
            const int shift = (target - source + procs) % procs;
            commSendrecv<Backend>(h, left ? CommScope::Col : CommScope::Row,
                panel.data(), std::size_t(send_length) * width, (coord + shift) % procs,
                work.data(), std::size_t(recv_length) * width, (coord - shift + procs) % procs);
            panel.swap(work);
        }
        return;
    }

    // pgeadd handles rectangular grids and independent source coordinates.
    // A compact descriptor preserves the block phase of the cropped panel;
    // the origin is block-aligned and its short dimension is one block.
    int d[DDLA_DLEN_];
    const int owner = indxg2p(first, nb, target, procs);
    DDLA_CHECK(ddlaDescInit(d, h, left ? length : width, left ? width : length,
                           nb, nb, left ? owner : 0, left ? 0 : owner));
    pgeadd<Backend>(h, trans, 'N', d[DDLA_M_], d[DDLA_N_],
        T(1), A, (left ? k : first) + 1, (left ? first : k) + 1, da,
        T(0), static_cast<const T*>(nullptr), 1, 1, d, work.data(), 1, 1, d);
    transport_block<Backend>(h, left ? 'C' : 'R', 'N',
                            d[DDLA_M_], d[DDLA_N_], work.data(), 0, 0, d, panel.data());
}

} // namespace

template <DdlaBackend Backend, typename T>
void ptrmm(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
           int m, int n, const T& alpha,
           const T* A, const int* descA, T* B, const int* descB)
{
    detail::require_backend<Backend>(handle, "ptrmm");
    check_desc(descA, handle);
    check_desc(descB, handle);
    side = static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uplo = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    trans = static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
    diag = static_cast<char>(std::toupper(static_cast<unsigned char>(diag)));
    if ((side != 'L' && side != 'R') || (uplo != 'L' && uplo != 'U') ||
        (trans != 'N' && trans != 'T' && trans != 'C') || (diag != 'N' && diag != 'U'))
        throw std::invalid_argument("ptrmm: invalid side, uplo, trans or diag");
    const bool left = side == 'L';
    const int order = left ? m : n;
    if (m < 0 || n < 0 || order > descA[DDLA_M_] || order > descA[DDLA_N_] ||
        m > descB[DDLA_M_] || n > descB[DDLA_N_])
        throw std::invalid_argument("ptrmm: dimensions outside descriptors");
    const int nb = descA[DDLA_MB_];
    if (nb != descA[DDLA_NB_] || nb != descB[left ? DDLA_MB_ : DDLA_NB_])
        throw std::invalid_argument("ptrmm: square A blocks must match the multiplied dimension of B");
    if (m == 0 || n == 0) return;

    const int pr = handle->nprows_, pc = handle->npcols_;
    const int row = handle->myprow_, col = handle->mypcol_;
    const int ml = num_loc(m, descB[DDLA_MB_], row, descB[DDLA_RSRC_], pr);
    const int nl = num_loc(n, descB[DDLA_NB_], col, descB[DDLA_CSRC_], pc);
    const int ldb = descB[DDLA_LLD_];
    if (alpha == T(0)) {
        // BLAS zero-alpha semantics: neither A nor the old B is read.
        if (ml > 0) for (int j = 0; j < nl; ++j)
            DDLA_CHECK(ddlaMemset(handle, B + std::size_t(j) * ldb, 0, std::size_t(ml) * sizeof(T)));
        return;
    }

    const int along = left ? ml : nl;
    const int source_along = num_loc(order, nb, left ? row : col,
                                     descA[left ? DDLA_RSRC_ : DDLA_CSRC_], left ? pr : pc);
    // Allocated once; every operation, collective and release uses handle's stream.
    vector<T> diagonal(handle, std::size_t(nb) * nb);
    vector<T> block_B(handle, std::size_t(nb) * std::max(1, left ? nl : ml));
    vector<T> panel(handle, std::size_t(nb) * std::max({1, along, source_along}));
    vector<T> work(handle, panel.size());

    const bool lower = (uplo == 'L') == (trans == 'N');
    const bool forward = left ? !lower : lower;
    const int blocks = (order - 1) / nb + 1;
    for (int step = 0; step < blocks; ++step) {
        const int k = (forward ? step : blocks - 1 - step) * nb;
        const int width = std::min(nb, order - k);
        const int ar = indxg2p(k, nb, descA[DDLA_RSRC_], pr);
        const int ac = indxg2p(k, nb, descA[DDLA_CSRC_], pc);
        if (row == ar && col == ac) {
            const int i = num_loc(k, nb, row, descA[DDLA_RSRC_], pr);
            const int j = num_loc(k, nb, col, descA[DDLA_CSRC_], pc);
            DDLA_CHECK(ddlaMemcpy2D(handle, diagonal.data(), std::size_t(width) * sizeof(T),
                A + i + std::size_t(j) * descA[DDLA_LLD_], std::size_t(descA[DDLA_LLD_]) * sizeof(T),
                std::size_t(width) * sizeof(T), width));
        }
        commBcast<Backend>(handle, CommScope::Grid, diagonal.data(),
                          std::size_t(width) * width, handle->rc_to_rank(ar, ac));

        // Save this original B panel before applying its diagonal block. The
        // sweep updates only panels already visited, so every input is used
        // before it is overwritten and alpha is applied exactly once.
        transport_block<Backend>(handle, left ? 'R' : 'C', 'N',
            left ? width : m, left ? n : width,
            B, left ? k : 0, left ? 0 : k, descB, block_B.data());
        const int coord = left ? row : col, procs = left ? pr : pc;
        const int source = descB[left ? DDLA_RSRC_ : DDLA_CSRC_];
        const int local_k = num_loc(k, nb, coord, source, procs);
        if (coord == indxg2p(k, nb, source, procs) && (left ? nl : ml) > 0) {
            trmm<Backend>(handle, side, uplo, trans, diag, left ? width : ml, left ? nl : width,
                alpha, diagonal.data(), width, block_B.data(), left ? width : ml,
                B + (left ? std::size_t(local_k) : std::size_t(local_k) * ldb), ldb);
        }

        const int first = forward ? 0 : k + width;
        const int length = forward ? k : order - first;
        if (length == 0) continue;
        triangular_panel<Backend>(handle, left, trans, k, width, first, length,
                                  A, descA, descB, panel, work);
        const int start = num_loc(first, nb, coord, source, procs);
        const int count = num_loc(first + length, nb, coord, source, procs) - start;
        if (count > 0 && (left ? nl : ml) > 0) {
            if (left)
                gemm<Backend>(handle, 'N', 'N', count, nl, width, alpha,
                    panel.data(), count, block_B.data(), width, T(1), B + start, ldb);
            else
                gemm<Backend>(handle, 'N', 'N', ml, count, width, alpha,
                    block_B.data(), ml, panel.data(), width, T(1), B + std::size_t(start) * ldb, ldb);
        }
    }
    work.reset();
    panel.reset();
    block_B.reset();
    diagonal.reset();
}

#define INSTANTIATE_PTRMM(BACKEND, TYPE) \
    template void ptrmm<BACKEND, TYPE>(const DdlaHandle_t&, char, char, char, char, \
        int, int, const TYPE&, const TYPE*, const int*, TYPE*, const int*)
#if DDLA_HAS_CPU
INSTANTIATE_PTRMM(DdlaBackend::CPU, float);
INSTANTIATE_PTRMM(DdlaBackend::CPU, double);
INSTANTIATE_PTRMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PTRMM(DdlaBackend::CPU, std::complex<double>);
#endif
#if DDLA_HAS_GPU
INSTANTIATE_PTRMM(DdlaBackend::GPU, float);
INSTANTIATE_PTRMM(DdlaBackend::GPU, double);
INSTANTIATE_PTRMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PTRMM(DdlaBackend::GPU, std::complex<double>);
#endif
#undef INSTANTIATE_PTRMM

} // namespace ddla
