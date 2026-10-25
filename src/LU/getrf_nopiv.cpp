#include <ddla/ddla.h>
#include "lu_internal.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "helpers.h"
#include "require_backend.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

// Maximum panel width for the register-based getf2_nopiv kernel.
// Each thread stores one row of the panel in registers, so this
// must be small enough to avoid register spilling.
// MAGMA uses n <= 32 for their register-cached nopiv kernel.
#define GETF2_MAX_N 32

namespace ddla {

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

// --------------------------------------------------------------------------
// Zero-pivot check helpers (device)
// --------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ bool is_zero_nopiv(T val) {
    return val == T(0);
}

template <typename T>
__device__ __forceinline__ bool is_zero_nopiv(thrust::complex<T> val) {
    return thrust::abs(val) == T(0);
}

// --------------------------------------------------------------------------
// Device helper: unblocked LU factorization without pivoting
//
// Follows MAGMA's zgetf2_nopiv_device pattern:
//   - Each thread processes one row (coalesced load/store)
//   - Matrix data stays in registers during factorization
//   - Shared memory used only for broadcasting pivot column
//   - Zero-pivot protection: divisor = 1 when pivot == 0
//
// This is called from the kernel with n <= GETF2_MAX_N.
// --------------------------------------------------------------------------

template <typename T, int MAX_N>
__device__ __forceinline__ void
getf2_nopiv_device(int m, int n, T* dA, int ldda, int* info,
                   int tx, T* sx, int gbstep)
{
    // Register array for one row.  n is a runtime value <= MAX_N,
    // but the compiler can partially unroll when n is known at launch.
    T rA[MAX_N];
    // Preserve the first non-zero info reported by earlier panels.
    if (gbstep > 0 && *info != 0) {
        return;
    }


    // Load one row from global memory (coalesced: adjacent threads
    // access adjacent addresses in the same column)
#ifdef DDLA_USE_CUDA
    #pragma unroll
#endif
    for (int i = 0; i < MAX_N; ++i) {
        if (i < n)
            rA[i] = dA[i * ldda + tx];
    }

    int linfo = 0;

    for (int col = 0; col < n; ++col) {
        // Broadcast pivot column via shared memory:
        // thread col copies its row [col..n-1] to shared mem.
        if (tx == col) {
#ifdef DDLA_USE_CUDA
            #pragma unroll
#endif
            for (int j = col; j < MAX_N; ++j) {
                if (j < n)
                    sx[j - col] = rA[j];
            }
        }
        __syncthreads();

        // Check for zero pivot (info is 1-based, global index)
        if (is_zero_nopiv(sx[0]) && linfo == 0)
            linfo = gbstep + col + 1;

        // Zero-pivot protection: use 1 instead of 0 to avoid NaN/Inf.
        // When pivot is zero, the arithmetic has no effect (multiplying by 1).
        T divisor = is_zero_nopiv(sx[0]) ? T(1) : sx[0];
        T reg_inv = T(1) / divisor;

        // Rank-1 update: for rows strictly below the pivot row
        if (tx > col && tx < m) {
            // Scale column col: store L factor (below diagonal)
            rA[col] *= reg_inv;
            T factor = rA[col];

            // GER: update trailing part of this row
            //      rA[j] -= factor * sx[j-col]  for j = col+1 .. n-1
#ifdef DDLA_USE_CUDA
            #pragma unroll
#endif
            for (int j = col + 1; j < MAX_N; ++j) {
                if (j < n)
                    rA[j] -= factor * sx[j - col];
            }
        }
        __syncthreads();
    }

    // Store result back to global memory (coalesced)
#ifdef DDLA_USE_CUDA
    #pragma unroll
#endif
    for (int i = 0; i < MAX_N; ++i) {
        if (i < n)
            dA[i * ldda + tx] = rA[i];
    }

    // Write info back (only thread 0)
    if (tx == 0) {
        (*info) = linfo;
    }
}

// --------------------------------------------------------------------------
// Unblocked panel kernel (launched with 1 block)
//
// Each thread handles one row.  Shared memory size = n * sizeof(T)
// is allocated dynamically at kernel launch.
// --------------------------------------------------------------------------

template <typename T>
__global__ void
getf2_nopiv_kernel(int m, int n, T* dA, int ldda, int* info, int gbstep)
{
    extern __shared__ char sdata[];
    T* sx = reinterpret_cast<T*>(sdata);

    const int tx = threadIdx.x;

    getf2_nopiv_device<T, GETF2_MAX_N>(m, n, dA, ldda, info, tx, sx, gbstep);
}

// Factors the jb x jb diagonal block: 1 block, jb threads, jb * sizeof(T) shared memory, one row per thread.
template <typename T>
void launch_getf2_nopiv_kernel(const DdlaHandle_t& ddla_handle, int jb, T* d_block, int lda, int* d_info, int gbstep)
{
    const gpuStream_t stream = ddla_handle->stream;
    const int threads = std::min(jb, 512);
    const int shmem   = jb * sizeof(T);

    using DeviceT = detail::DeviceScalarT<T>;
    getf2_nopiv_kernel<DeviceT><<<1, threads, shmem, stream>>>(
        jb, jb, reinterpret_cast<DeviceT*>(d_block), lda, d_info, gbstep);
    DDLA_CHECK(ddlaGetLastError(ddla_handle));
}

#endif // DDLA_USE_CUDA || DDLA_USE_HIP

// --------------------------------------------------------------------------
// CPU arm of the diagonal-block factorization: the same column loop as the kernel (zero pivot -> divisor 1, first zero
// recorded as a 1-based global index, an earlier panel's info is kept), with the scale and the rank-1 update as
// host scal / geru calls.  A block is at most GETF2_MAX_N wide, so this stays a few small BLAS-1/2 calls.
// --------------------------------------------------------------------------
template <typename T>
void getf2_nopiv_host(const DdlaHandle_t& ddla_handle, int n, T* A, int lda, int* info, int gbstep)
{
    if (gbstep > 0 && *info != 0) {
        return;
    }
    int linfo = 0;
    for (int col = 0; col < n; ++col) {
        T* column = A + col + static_cast<std::size_t>(col) * lda;   // the pivot, then the rows below it
        const bool zero_pivot = std::abs(column[0]) == 0;
        if (zero_pivot && linfo == 0) {
            linfo = gbstep + col + 1;
        }
        const int rows = n - col - 1;
        if (rows > 0) {
            scal<DdlaBackend::CPU, T>(ddla_handle, rows, T(1) / (zero_pivot ? T(1) : column[0]), column + 1, 1);
            geru<DdlaBackend::CPU, T>(ddla_handle, rows, rows, T(-1),
                                      column + 1, 1, column + lda, lda, column + 1 + lda, lda);
        }
    }
    *info = linfo;
}

// --------------------------------------------------------------------------
// Blocked getrf_nopiv (pure GPU, right-looking Level-3 BLAS)
//
// Algorithm (matches MAGMA magma_zgetrf_nopiv):
//   for j = 0 to min(m,n) step nb:
//      jb = min(nb, min(m,n)-j)
//
//      1) Factor diagonal block: A(j:j+jb, j:j+jb) = L*U  (getf2)
//      2) If m > j+jb:
//         Solve L panel: A(j+jb:m, j:j+jb) = A(j+jb:m, j:j+jb) * U^{-1}  (trsm)
//      3) If n > j+jb:
//         Solve U panel: A(j:j+jb, j+jb:n) = L^{-1} * A(j:j+jb, j+jb:n)  (trsm)
//      4) If m > j+jb and n > j+jb:
//         Update trailing: A(j+jb:m, j+jb:n) -= L_panel * U_panel  (gemm)
//
// The panel factorization (step 1) is done by the getf2_nopiv kernel,
// which keeps data in registers and uses shared memory for pivot broadcast.
// The trailing update (step 4) is done by the highly optimized cuBLAS/hipBLAS
// gemm, which dominates the runtime for large matrices.
// --------------------------------------------------------------------------

template <DdlaBackend Backend, typename T>
void getrf_nopiv(int m, int n, T* A, int lda, int* info_buffer, const DdlaHandle_t& ddla_handle)
{
    detail::require_backend<Backend>(ddla_handle, "getrf_nopiv");

    // Block size for the blocked algorithm.  Must be <= GETF2_MAX_N
    // because the panel kernel stores one row in registers.
    // GETF2_MAX_N = 32 is chosen to balance register pressure
    // (32 * sizeof(T) per thread) with arithmetic intensity.
    const int nb = GETF2_MAX_N;

    // Initialize info_buffer to 0 in backend memory (caller owns info_buffer memory)
    DDLA_CHECK(ddlaMemset(ddla_handle, info_buffer, 0, sizeof(int)));

    for (int j = 0; j < std::min(m, n); j += nb)
    {
        int jb = std::min(nb, std::min(m, n) - j);
        int mm = m - j;          // rows remaining in the panel
        int nn = n - j - jb;     // cols in the trailing submatrix

        // ---------------------------------------------------------------
        // 1) Factor the diagonal block A(j:j+jb, j:j+jb): the register kernel on a GPU handle, the host loop on a CPU one.
        // ---------------------------------------------------------------
        if constexpr (Backend == DdlaBackend::CPU)
        {
            getf2_nopiv_host<T>(ddla_handle, jb, A + j * lda + j, lda, info_buffer, j);
        }
        else
        {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
            launch_getf2_nopiv_kernel<T>(ddla_handle, jb, A + j * lda + j, lda, info_buffer, j);
#endif
        }

        // ---------------------------------------------------------------
        // 2) Solve L panel: A(j+jb:m-1, j:j+jb) = A(j+jb:m-1, j:j+jb) * U^{-1}
        //
        // The subdiagonal block A(j+jb:m, j:j+jb) now contains the
        // raw matrix values.  We need to compute L21 = A21 * U^{-1}
        // where U is the upper triangular factor from step 1.
        //
        // trsm(Right, Upper, NoTrans, NonUnit, m-jb, jb, 1, U, A21)
        // ---------------------------------------------------------------
        if (j + jb < m)
        {
            T one = T(1);
            trsm<Backend, T>(
                ddla_handle,
                'R', 'U',                   // op(A)*X = B  =>  X = B*inv(A)
                'N', 'N',                   // no transpose; U has non-unit diagonal
                mm - jb, jb,                // (m-jb) rows, jb cols
                one,
                A + j * lda + j,          // U factor (jb x jb)
                lda,
                A + j * lda + (j + jb),   // A21 input, L21 output
                lda
            );
        }

        // ---------------------------------------------------------------
        // 3) Solve U panel: A(j:j+jb, j+jb:n-1) = L^{-1} * A(j:j+jb, j+jb:n-1)
        //
        // The block row A(j:j+jb, j+jb:n) now contains the raw values.
        // We need U12 = inv(L) * A12 where L is the lower triangular
        // factor (with unit diagonal) from step 1.
        //
        // trsm(Left, Lower, NoTrans, Unit, jb, n-jb, 1, L, A12)
        // ---------------------------------------------------------------
        if (j + jb < n)
        {
            T one = T(1);
            trsm<Backend, T>(
                ddla_handle,
                'L', 'L',                   // op(A)*X = B  =>  X = inv(A)*B
                'N', 'U',                   // no transpose; L has unit diagonal
                jb, nn,                     // jb rows, (n-jb) cols
                one,
                A + j * lda + j,          // L factor (jb x jb, implicit unit diag)
                lda,
                A + (j + jb) * lda + j,   // A12 input, U12 output
                lda
            );

            // -----------------------------------------------------------
            // 4) Update trailing submatrix
            //
            // A(j+jb:m, j+jb:n) -= L_panel(j+jb:m, j:j+jb) * U_panel(j:j+jb, j+jb:n)
            //
            // gemm(NoTrans, NoTrans, m-jb, n-jb, jb, -1, L_panel, U_panel, 1, A22)
            // -----------------------------------------------------------
            if (j + jb < m)
            {
                T neg_one = T(-1);
                gemm<Backend, T>(
                    ddla_handle,
                    'N', 'N',
                    mm - jb, nn, jb,
                    neg_one,
                    A + j * lda + (j + jb),      // L_panel (m-jb x jb)
                    lda,
                    A + (j + jb) * lda + j,      // U_panel (jb x n-jb)
                    lda,
                    one,
                    A + (j + jb) * lda + (j + jb), // A22 trailing (m-jb x n-jb)
                    lda
                );
            }
        }
    }
    // info_buffer stays in backend memory; caller owns the memory and is responsible
    // for reading it back after stream synchronization.
}

// --------------------------------------------------------------------------
// Explicit instantiations
// --------------------------------------------------------------------------

#define INSTANTIATE_GETRF_NOPIV(BACKEND, TYPE) \
    template void getrf_nopiv<BACKEND, TYPE>(int, int, TYPE*, int, int*, const DdlaHandle_t&)

#if DDLA_HAS_CPU
INSTANTIATE_GETRF_NOPIV(DdlaBackend::CPU, float);
INSTANTIATE_GETRF_NOPIV(DdlaBackend::CPU, double);
INSTANTIATE_GETRF_NOPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GETRF_NOPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GETRF_NOPIV(DdlaBackend::GPU, float);
INSTANTIATE_GETRF_NOPIV(DdlaBackend::GPU, double);
INSTANTIATE_GETRF_NOPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GETRF_NOPIV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GETRF_NOPIV

} // namespace ddla
