#include "potrf_bottom_right_internal.h"
#include "potrf_bottom_right_transform.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <type_traits>

#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"

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
        syrk<Backend, T>(handle, uplo, upper ? 'N' : 'T',
                                  leading_n, block_n,
                                  minus_one, panel, lda,
                                  one, leading, lda);
    } else {
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        herk<Backend, T>(handle, uplo, upper ? 'N' : 'C',
                                  leading_n, block_n,
                                  minus_one, panel, lda,
                                  one, leading, lda);
    }
}

} // namespace

template <DdlaBackend Backend, typename T>
void potrf_bottom_right_block(
    char uplo, int n, T* A, int lda,
    int global_offset,
    T* work, int* info_buffer, int& info,
    const DdlaHandle_t& handle)
{
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(global_offset >= 0);
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    assert(A != nullptr);
    assert(work != nullptr);
    assert(info_buffer != nullptr);
    assert(lda >= n);

    DDLA_CHECK(ddlaMemset(handle, info_buffer, 0, sizeof(int)));
    if(uplo == 'U'){
        reverse_upper_to_lower<Backend>(handle, n, A, lda, work, n);
    }else{
        reverse_lower_to_upper<Backend>(handle, n, A, lda, work, n);
    }
    // The selected triangle was reverse-packed into the opposite one of work: factor that triangle.
    potrf<Backend, T>(handle, uplo == 'U' ? 'L' : 'U', n, work, n, info_buffer);

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

#define INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(BACKEND, TYPE)                                   \
    template void potrf_bottom_right_block<BACKEND, TYPE>(char, int, TYPE*, int, int, TYPE*, int*, int&, \
    const DdlaHandle_t&)

#if DDLA_HAS_CPU
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_POTRF_BOTTOM_RIGHT_BLOCK


} // namespace detail

template <DdlaBackend Backend, typename T>
void potrf_bottom_right(
    char uplo, int n, T* A, int lda,
    int& info, const DdlaHandle_t& handle)
{
    detail::require_backend<Backend>(handle, "potrf_bottom_right");
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    assert(A != nullptr);
    assert(lda >= n);

    T* work = nullptr;
    int* info_buffer = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&work),
                          static_cast<std::size_t>(detail::kBlockSize) * detail::kBlockSize * sizeof(T)));
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&info_buffer), sizeof(int)));

    const int last_block_start =
        ((n - 1) / detail::kBlockSize) * detail::kBlockSize;
    for(int block_start = last_block_start;
        block_start >= 0;
        block_start -= detail::kBlockSize){
        const int block_n = std::min(detail::kBlockSize, n - block_start);
        T* const diag = A + block_start
                        + static_cast<std::size_t>(block_start) * lda;
        detail::potrf_bottom_right_block<Backend>(
            uplo, block_n, diag, lda, block_start,
            work, info_buffer, info, handle);
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
            trsm<Backend, T>(handle, 'R', 'U', 'C', 'N',
                                      block_start, block_n, one,
                                      diag, lda, panel, lda);
        }else{
            trsm<Backend, T>(handle, 'L', 'L', trans, 'N',
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

    DDLA_CHECK(ddlaFree(handle, work));
    DDLA_CHECK(ddlaFree(handle, info_buffer));
    DDLA_CHECK(ddlaSynchronize(handle));
}

#define INSTANTIATE_POTRF_BOTTOM_RIGHT(BACKEND, TYPE)                                   \
    template void potrf_bottom_right<BACKEND, TYPE>(char, int, TYPE*, int, int&, const DdlaHandle_t&)

#if DDLA_HAS_CPU
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, float);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, double);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_POTRF_BOTTOM_RIGHT(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_POTRF_BOTTOM_RIGHT


} // namespace ddla
