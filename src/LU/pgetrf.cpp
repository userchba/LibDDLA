#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <algorithm>
#include <vector>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "lu_panel.h"

namespace ddla{

template <DdlaBackend Backend, typename T>
void pgetrf(const DdlaHandle_t& handle, int m, int n, T* A, int ia, int ja,
             const int* desc, int* ipiv, int& info)
{
    detail::require_backend<Backend>(handle, "pgetrf");
    lu_impl::validate(handle, m, n, A, ia, ja, desc, ipiv);
    lu_impl::Workspace<T> workspace(handle);
    info = 0;
    if (m == 0 || n == 0) return;

    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int nb = desc[DDLA_MB_], ld = desc[DDLA_LLD_];
    const int row0 = ia - 1, col0 = ja - 1;
    const int row_end = num_loc(row0 + m, nb, myprow, desc[DDLA_RSRC_], nprows);
    const int col_end = num_loc(col0 + n, nb, mypcol, desc[DDLA_CSRC_], npcols);
    const int local_rows = row_end - num_loc(row0, nb, myprow, desc[DDLA_RSRC_], nprows);
    const int local_cols = col_end - num_loc(col0, nb, mypcol, desc[DDLA_CSRC_], npcols);

    T *temp_block = nullptr, *temp_L = nullptr, *temp_U = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, &temp_block, sizeof(T)*std::size_t(nb)*nb));
    DDLA_CHECK(ddlaMalloc(handle, &temp_L, sizeof(T)*std::size_t(local_rows)*nb));
    DDLA_CHECK(ddlaMalloc(handle, &temp_U, sizeof(T)*std::size_t(nb)*local_cols));

    for (int k = 0; k < std::min(m,n); ) {
        const int r = row0 + k, c = col0 + k;
        // Independent offsets can reach row and column block boundaries at
        // different steps. Each diagonal block must fit both distributions.
        const int b = std::min({nb-r%nb, nb-c%nb, m-k, n-k});
        const int il = indx_g2l_r(desc, handle, r), jl = indx_g2l_c(desc, handle, c);
        const int pr = indxg2p(r, nb, desc[DDLA_RSRC_], nprows);
        const int pc = indxg2p(c, nb, desc[DDLA_CSRC_], npcols);

        pgetf2<Backend,T>(handle, m-k, n-k, b, A, ia+k, ja+k, desc,
                          ipiv, info, workspace, ja);
        if (info != 0) { if (info > 0) info += k; break; }

        const int row_after = num_loc(r+b, nb, myprow, desc[DDLA_RSRC_], nprows);
        const int col_after = num_loc(c+b, nb, mypcol, desc[DDLA_CSRC_], npcols);
        if (jl >= 0) {
            if (il >= 0)
                DDLA_CHECK(ddlaMemcpy2D(handle, temp_block, std::size_t(b)*sizeof(T),
                    A+il+std::size_t(jl)*ld, std::size_t(ld)*sizeof(T), std::size_t(b)*sizeof(T), b));
            if (row_after < row_end)
                DDLA_CHECK(ddlaMemcpy2D(handle, temp_L, std::size_t(row_end-row_after)*sizeof(T),
                    A+row_after+std::size_t(jl)*ld, std::size_t(ld)*sizeof(T),
                    std::size_t(row_end-row_after)*sizeof(T), b));
        }
        if (row_after < row_end)
            commBcast<Backend>(handle, CommScope::Row, temp_L, std::size_t(row_end-row_after)*b, pc);
        if (il >= 0) {
            commBcast<Backend>(handle, CommScope::Row, temp_block, std::size_t(b)*b, pc);
            if (col_after < col_end) {
                trsm<Backend,T>(handle, 'L', 'L', 'N', 'U', b, col_end-col_after, T(1),
                    temp_block, b, A+il+std::size_t(col_after)*ld, ld);
                DDLA_CHECK(ddlaMemcpy2D(handle, temp_U, std::size_t(b)*sizeof(T),
                    A+il+std::size_t(col_after)*ld, std::size_t(ld)*sizeof(T),
                    std::size_t(b)*sizeof(T), col_end-col_after));
            }
        }
        if (col_after < col_end)
            commBcast<Backend>(handle, CommScope::Col, temp_U, std::size_t(b)*(col_end-col_after), pr);
        if (row_after < row_end && col_after < col_end)
            gemm<Backend,T>(handle, 'N', 'N', row_end-row_after, col_end-col_after, b,
                -1.0, temp_L, row_end-row_after, temp_U, b, 1.0,
                A+row_after+std::size_t(col_after)*ld, ld);
        k += b;
    }
    DDLA_CHECK(ddlaFree(handle, temp_block));
    DDLA_CHECK(ddlaFree(handle, temp_L));
    DDLA_CHECK(ddlaFree(handle, temp_U));
    DDLA_CHECK(ddlaSynchronize(handle));
}

#define INSTANTIATE_PGETRF(BACKEND, TYPE)                                   \
    template void pgetrf<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, int ia, int ja, const int* array_descA, \
    int* ipiv, \
    int& info)

#if DDLA_HAS_CPU
INSTANTIATE_PGETRF(DdlaBackend::CPU, float);
INSTANTIATE_PGETRF(DdlaBackend::CPU, double);
INSTANTIATE_PGETRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRF(DdlaBackend::GPU, float);
INSTANTIATE_PGETRF(DdlaBackend::GPU, double);
INSTANTIATE_PGETRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETRF


}
