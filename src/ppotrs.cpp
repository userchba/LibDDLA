#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include "ddla_stream_impl.h"
#include "require_gpu.h"
namespace ddla{

template <typename T>
void ppotrs(
    const DdlaHandle_t& handle, const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB,
    bool is_nega, int location
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "ppotrs");
    // A is Hermitian, so op(A) == A for both trans='N' and trans='C'; the
    // solve path is identical.  trans='T' would require conjugating B and is
    // not supported.
    assert(trans == 'N' || trans == 'C');
    assert(side == 'L' || side == 'R');
    assert(uplo == 'L' || uplo == 'U');
    // Leading-block sub-matrix: descriptors may be larger than the logical
    // sub-matrix (anchored at global (0,0)).
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    if(side == 'L')
        assert(n <= array_descB[DDLA_M_]);
    else
        assert(n <= array_descB[DDLA_N_]);

    // Head correction: when ppotrf was called with is_head=true and a
    // location != -1 (and != n), it relocated the head element to the last
    // global index by an in-place symmetric permutation of A.  The same
    // permutation must be applied to B on the side of the solve -- rows of B
    // for side='L' (B is n x nrhs), columns of B for side='R' (B is
    // nrhs x n) -- before the solve and again after (self-inverse), so X
    // comes back in the caller's original ordering.  This keeps direct
    // ppotrf+ppotrs head-correction users correct without external pswap
    // bookkeeping; pposv relies on it too.
    const bool needs_permute = (location != -1 && location != n);
    if(needs_permute){
        if(side == 'L')
            pswap(handle, nrhs, d_B, location, 1, array_descB, array_descB[DDLA_M_],
                        d_B, n,        1, array_descB, array_descB[DDLA_M_]);
        else
            pswap(handle, array_descB[DDLA_M_], d_B, 1, location, array_descB, 1,
                        d_B, 1, n,        array_descB, 1);
    }

    // Solve op(A)*X=B (side='L') or X*op(A)=B (side='R') with the Cholesky
    // factor: for uplo='L', A = L*L^H.  Left solve applies L then L^H; right
    // solve applies L^H then L (the product order of the solves reverses).
    const bool left = (side == 'L');
    const char first_trans = (left == (uplo == 'L')) ? 'N' : 'C';
    const char second_trans = (left == (uplo == 'L')) ? 'C' : 'N';
    const int b_rows = left ? n : nrhs;
    const int b_cols = left ? nrhs : n;

    ptrtrs(handle, 
        side, uplo, first_trans, 'N', b_rows, b_cols,
        d_A, array_descA,
        d_B, array_descB
    );
    if(is_nega){
        int i_loc = indx_g2l_r(array_descA, handle, n - 1);
        int j_loc = indx_g2l_c(array_descA, handle, n - 1);
        if(i_loc >= 0 && j_loc >= 0){
            RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
            T correction;
            RUNTIME_CHECK(runtimeMemcpy(&correction, d_A + i_loc + j_loc * array_descA[DDLA_LLD_], sizeof(T), runtimeMemcpyDeviceToHost));
            correction = -correction;
            RUNTIME_CHECK(runtimeMemcpy(d_A + i_loc + j_loc * array_descA[DDLA_LLD_], &correction, sizeof(T), runtimeMemcpyHostToDevice));
        }
    }
    ptrtrs(handle, 
        side, uplo, second_trans, 'N', b_rows, b_cols,
        d_A, array_descA,
        d_B, array_descB
    );

    if(needs_permute){
        if(side == 'L')
            pswap(handle, nrhs, d_B, location, 1, array_descB, array_descB[DDLA_M_],
                        d_B, n,        1, array_descB, array_descB[DDLA_M_]);
        else
            pswap(handle, array_descB[DDLA_M_], d_B, 1, location, array_descB, 1,
                        d_B, 1, n,        array_descB, 1);
    }
}

template void ppotrs<float>(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    float* d_A, const int* array_descA,
    float* d_B, const int* array_descB,
    bool is_nega, int location
);

template void ppotrs<double>(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    double* d_A, const int* array_descA,
    double* d_B, const int* array_descB,
    bool is_nega, int location
);

template void ppotrs<std::complex<double>>(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const int* array_descA,
    std::complex<double>* d_B, const int* array_descB,
    bool is_nega, int location
);

template void ppotrs<std::complex<float>>(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    std::complex<float>* d_A, const int* array_descA,
    std::complex<float>* d_B, const int* array_descB,
    bool is_nega, int location
);


}
