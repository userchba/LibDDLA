#include "BLAS/blas_impl.h"
#include "potrf_bottom_right_internal.h"
#include "potrf_bottom_right_transform.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "plan_util.h"
#include "require_backend.h"

// Single-process bottom-right Cholesky, split into the three components of src/lapack_impl.h:
//   potrf_bottom_right_check / _workspace_size / _run   the public potrf_bottom_right,
//   potrf_bottom_right_block_check + the two            detail::potrf_bottom_right_block (one diagonal block;
//   potrf_bottom_right_block overloads                   the overload with the ?potrf scratch supplied is its run).
// The public routine is check -> (empty return) -> workspace size -> allocate -> run: the backend scratch holds the
// reverse-packed block, its info word and the ?potrf scratch of that block.

namespace ddla{

namespace detail {

namespace {

constexpr int kBlockSize = 128;

template <DdlaBackend Backend, typename T>
void update_leading(
    const DdlaHandle_t& handle,
    const char uplo,
    int leading_n, int block_n,
    const T* panel, int lda,
    T* leading)
{
    const bool upper = uplo == 'U';
    // leading -= panel * panel^H (or its transpose, for the lower case).
    // The unified syrk/herk take a LAPACK-style character flag, so the
    // transpose choice is the same 'N'/'T'/'C' string the CPU arm hands to
    // Fortran -- one convention across both backends.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        const T minus_one = T(-1);
        const T one = T(1);
        blas_impl::syrk_run<Backend, T>(handle, uplo, upper ? 'N' : 'T',
                                  leading_n, block_n,
                                  minus_one, panel, lda,
                                  one, leading, lda);
    } else {
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        blas_impl::herk_run<Backend, T>(handle, uplo, upper ? 'N' : 'C',
                                  leading_n, block_n,
                                  minus_one, panel, lda,
                                  one, leading, lda);
    }
}

} // namespace

// Run of one diagonal block: the selected triangle is reverse-packed into work, factored by the ?potrf of the
// opposite triangle on the supplied scratch, and mapped back.  No validation, no allocation.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_block(
    char uplo, int n, T* A, int lda,
    int global_offset,
    T* work, int* info_buffer, int& info,
    const DdlaHandle_t& handle,
    void* potrf_work, std::size_t potrf_bytes)
{
    info = 0;
    if(n == 0){
        return;
    }

    DDLA_CHECK(ddlaMemset(handle, info_buffer, 0, sizeof(int)));
    if(uplo == 'U'){
        reverse_upper_to_lower<Backend>(handle, n, A, lda, work, n);
    }else{
        reverse_lower_to_upper<Backend>(handle, n, A, lda, work, n);
    }
    // The selected triangle was reverse-packed into the opposite one of work: factor that triangle.
    lapack_impl::potrf_run<Backend, T>(handle, uplo == 'U' ? 'L' : 'U', n, work, n, info_buffer,
                                       potrf_work, potrf_bytes);

    int solver_info = 0;
    DDLA_CHECK(ddlaMemcpy(handle, &solver_info, info_buffer, sizeof(int)));
    DDLA_CHECK(ddlaSynchronize(handle));
    if(solver_info != 0){
        info = global_offset + n - solver_info + 1;
        return;
    }

    if(uplo == 'U'){
        reverse_lower_to_upper<Backend>(handle, n, work, n, A, lda);
    }else{
        reverse_upper_to_lower<Backend>(handle, n, work, n, A, lda);
    }
}

// The existing signature: check, allocate the ?potrf scratch of this one block, run.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_block(
    char uplo, int n, T* A, int lda,
    int global_offset,
    T* work, int* info_buffer, int& info,
    const DdlaHandle_t& handle)
{
    lapack_impl::potrf_bottom_right_block_check<Backend>(
        uplo, n, A, lda, global_offset, work, info_buffer, handle);
    info = 0;
    if(n == 0){
        return;
    }
    std::size_t potrf_bytes = 0;
    lapack_impl::potrf_workspace_size<T>(handle, uplo == 'U' ? 'L' : 'U', n, work, n, &potrf_bytes);
    vector<char> potrf_work(handle, potrf_bytes);
    potrf_bottom_right_block<Backend, T>(
        uplo, n, A, lda, global_offset, work, info_buffer, info, handle, potrf_work.data(), potrf_bytes);
}

#define INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(BACKEND, TYPE)                                   \
    template void potrf_bottom_right_block<BACKEND, TYPE>(char, int, TYPE*, int, int, TYPE*, int*, int&, \
    const DdlaHandle_t&);                                                                      \
    template void potrf_bottom_right_block<BACKEND, TYPE>(char, int, TYPE*, int, int, TYPE*, int*, int&, \
    const DdlaHandle_t&, void*, std::size_t)

INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK


} // namespace detail

namespace lapack_impl {

// Component 1 of detail::potrf_bottom_right_block.  The asserts of the former body are real throws now (the family of
// require_backend): every precondition of the block routine is rejected before anything runs.
template <DdlaBackend Backend>
void potrf_bottom_right_block_check(
    char uplo, int n, const void* A, int lda, int global_offset,
    const void* work, const void* info_buffer, const DdlaHandle_t& handle)
{
    detail::require_backend<Backend>(handle, "potrf_bottom_right_block");
    if(uplo != 'U' && uplo != 'L'){
        throw std::runtime_error("potrf_bottom_right_block: uplo must be 'U' or 'L'");
    }
    if(n < 0){
        throw std::runtime_error("potrf_bottom_right_block: negative matrix size");
    }
    if(global_offset < 0){
        throw std::runtime_error("potrf_bottom_right_block: negative global offset");
    }
    if(n == 0){
        return;
    }
    if(A == nullptr) throw std::runtime_error("potrf_bottom_right_block: null A");
    if(work == nullptr) throw std::runtime_error("potrf_bottom_right_block: null work");
    if(info_buffer == nullptr) throw std::runtime_error("potrf_bottom_right_block: null info_buffer");
    if(lda < n) throw std::runtime_error("potrf_bottom_right_block: lda too small");
}

// Component 1 of the public potrf_bottom_right.  The asserts of the former body are real throws now.
template <DdlaBackend Backend>
void potrf_bottom_right_check(
    char uplo, int n, const void* A, int lda, const DdlaHandle_t& handle)
{
    detail::require_backend<Backend>(handle, "potrf_bottom_right");
    if(uplo != 'U' && uplo != 'L'){
        throw std::runtime_error("potrf_bottom_right: uplo must be 'U' or 'L'");
    }
    if(n < 0){
        throw std::runtime_error("potrf_bottom_right: negative matrix size");
    }
    if(n == 0){
        return;
    }
    if(A == nullptr) throw std::runtime_error("potrf_bottom_right: null A");
    if(lda < n) throw std::runtime_error("potrf_bottom_right: lda too small");
}

// Component 2: [ the reverse-packed block (nb x nb, nb = min(n, kBlockSize)) | its info word | the ?potrf scratch of
// the opposite triangle for both block orders that occur, nb and n % kBlockSize ].  A is what the vendor query is made
// with.
template <typename T>
void potrf_bottom_right_workspace_size(const DdlaHandle_t& handle, char uplo, int n, T* A, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (n <= 0) return;
    const int nb = std::min(detail::kBlockSize, n);
    std::size_t potrf_bytes = 0;
    for (int order : {nb, n % detail::kBlockSize}) {   // vendor workspace need not be monotone in the order
        std::size_t bytes = 0;
        potrf_workspace_size<T>(handle, uplo == 'U' ? 'L' : 'U', order, A, std::max(1, order), &bytes);
        potrf_bytes = std::max(potrf_bytes, bytes);
    }
    *backend_bytes = detail::padded_bytes<T>(static_cast<std::size_t>(nb) * nb) + detail::padded_bytes<int>(1)
                   + potrf_bytes;
}

// Component 3: the blocks from the bottom-right corner up.  Ends with a synchronize, like the public routine.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_run(
    const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int& info,
    void* backend_work, std::size_t backend_bytes)
{
    info = 0;
    if(n <= 0){
        return;
    }

    const int nb = std::min(detail::kBlockSize, n);
    char* w = static_cast<char*>(backend_work);
    T* const work = detail::take<T>(w, static_cast<std::size_t>(nb) * nb);
    int* const info_buffer = detail::take<int>(w, 1);
    const std::size_t used = static_cast<std::size_t>(w - static_cast<char*>(backend_work));
    assert(used <= backend_bytes);
    const int last_block_start =
        ((n - 1) / detail::kBlockSize) * detail::kBlockSize;
    for(int block_start = last_block_start;
        block_start >= 0;
        block_start -= detail::kBlockSize){
        const int block_n = std::min(detail::kBlockSize, n - block_start);
        T* const diag = A + block_start
                        + static_cast<std::size_t>(block_start) * lda;
        detail::potrf_bottom_right_block<Backend, T>(
            uplo, block_n, diag, lda, block_start,
            work, info_buffer, info, handle, w, backend_bytes - used);
        if(info != 0){
            break;
        }
        if(block_start == 0){
            continue;
        }

        T* const panel = uplo == 'U'
            ? A + static_cast<std::size_t>(block_start) * lda
            : A + block_start;
        const T one = T(1);
        // For real T a conjugate transpose is a plain transpose; for the
        // complex types 'C' is required.  One flag, both backends.
        const char trans = (std::is_same_v<T, float> || std::is_same_v<T, double>)
            ? 'T' : 'C';
        if(uplo == 'U'){
            blas_impl::trsm_run<Backend, T>(handle, 'R', 'U', 'C', 'N',
                                      block_start, block_n, one,
                                      diag, lda, panel, lda);
        }else{
            blas_impl::trsm_run<Backend, T>(handle, 'L', 'L', trans, 'N',
                                      block_n, block_start, one,
                                      diag, lda, panel, lda);
        }
        detail::update_leading<Backend>(
            handle,
            uplo,
            block_start, block_n,
            panel, lda,
            A);
    }

    DDLA_CHECK(ddlaSynchronize(handle));
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void potrf_bottom_right(
    char uplo, int n, T* A, int lda,
    int& info, const DdlaHandle_t& handle)
{
    // 1. parameter check.
    lapack_impl::potrf_bottom_right_check<Backend>(uplo, n, A, lda, handle);
    // No-op: nothing to factor.
    info = 0;
    if(n == 0){
        return;
    }
    // 2. the reverse-packed block, its info word and the ?potrf scratch.
    std::size_t backend_bytes = 0;
    lapack_impl::potrf_bottom_right_workspace_size<T>(handle, uplo, n, A, &backend_bytes);
    vector<char> work(handle, backend_bytes);
    // 3. the factorization.
    lapack_impl::potrf_bottom_right_run<Backend, T>(handle, uplo, n, A, lda, info, work.data(), backend_bytes);
}

#define INSTANTIATE_POTRF_BOTTOM_RIGHT(BACKEND, TYPE)                                   \
    template void potrf_bottom_right<BACKEND, TYPE>(char, int, TYPE*, int, int&, const DdlaHandle_t&); \
    template void lapack_impl::potrf_bottom_right_run<BACKEND, TYPE>(const DdlaHandle_t&, char, int, TYPE*, int, int&, \
        void*, std::size_t)
#define INSTANTIATE_POTRF_BOTTOM_RIGHT_SIZE(TYPE)                                       \
    template void lapack_impl::potrf_bottom_right_workspace_size<TYPE>(const DdlaHandle_t&, char, int, TYPE*, \
        std::size_t*)

template void lapack_impl::potrf_bottom_right_check<DdlaBackend::CPU>(
    char, int, const void*, int, const DdlaHandle_t&);
template void lapack_impl::potrf_bottom_right_block_check<DdlaBackend::CPU>(
    char, int, const void*, int, int, const void*, const void*, const DdlaHandle_t&);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
template void lapack_impl::potrf_bottom_right_check<DdlaBackend::GPU>(
    char, int, const void*, int, const DdlaHandle_t&);
template void lapack_impl::potrf_bottom_right_block_check<DdlaBackend::GPU>(
    char, int, const void*, int, int, const void*, const void*, const DdlaHandle_t&);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_POTRF_BOTTOM_RIGHT_SIZE(float);
INSTANTIATE_POTRF_BOTTOM_RIGHT_SIZE(double);
INSTANTIATE_POTRF_BOTTOM_RIGHT_SIZE(std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT_SIZE(std::complex<double>);

#undef INSTANTIATE_POTRF_BOTTOM_RIGHT_SIZE
#undef INSTANTIATE_POTRF_BOTTOM_RIGHT


} // namespace ddla
