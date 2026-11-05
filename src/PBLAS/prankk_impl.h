#ifndef DDLA_PRANKK_IMPL_H
#define DDLA_PRANKK_IMPL_H

#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "BLAS/blas_impl.h"
#include "ddla_stream.h"

#include <cctype>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace ddla { namespace prankk_impl {

template <DdlaBackend Backend, typename T>
bool check(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, const T* C, const int* dc,
    bool rank2 = false);

// Backend: the k panels of the product (op(A) in C's rows, op(B) in C's
// columns) and their exchange buffer; for the batched rank update the second
// term's panels and, on a GPU handle, the device copies of the gemmBatched
// pointer lists; then the scratch of the transposed-panel pgeadd calls.
// Host: the batched rank update's tile/pointer lists and host transfer scratch.  Nothing for n == 0 or a
// one-process product (a local BLAS call).
template <typename T>
void workspace_size(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb, bool product, bool rank2, int n, int k,
    const int* da, const int* db, const int* dc,
    std::size_t* backend_bytes, std::size_t* host_bytes);

template <DdlaBackend Backend, typename T>
void run(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb, bool product, bool rank2,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, T* C, const int* dc,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);



// Both rank-2k terms share the same panel allocations and traverse
// each k panel before advancing; beta is applied exactly once.
template <bool Hermitian, bool Rank2, DdlaBackend Backend, typename T>
void execute(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, T* C, const int* dc)
{
    trans=static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
    if constexpr (!Hermitian && std::is_arithmetic_v<T>)
        if (trans=='C') trans='T';
    constexpr char reflected=Hermitian ? 'C' : 'T';
    const char tb=trans=='N' ? reflected : 'N';
    const bool product=prankk_impl::check<Backend,T>(h,Hermitian ? 'H' : 'S',uplo,
        trans,tb,n,k,alpha,A,da,B,db,beta,C,dc,Rank2);
    uplo=blas_impl::upper(uplo);
    if (n==0 || (!Hermitian && !product && beta==T(1))) return;
    if (h->nprocs==1 && product) {
        // Reuse the no-check local path, including its precision safeguards.
        prankk_impl::run<Backend,T>(h,Hermitian ? 'H' : 'S',uplo,trans,tb,product,Rank2,
            n,k,alpha,A,da,B,db,beta,C,dc,nullptr,0,nullptr,0);
        return;
    }
    std::size_t backend_bytes=0,host_bytes=0;
    prankk_impl::workspace_size<T>(h,Hermitian ? 'H' : 'S',uplo,trans,tb,product,Rank2,n,k,da,db,dc,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    prankk_impl::run<Backend,T>(h,Hermitian ? 'H' : 'S',uplo,trans,tb,product,Rank2,n,k,alpha,A,da,B,db,beta,C,dc,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

}} // namespace ddla::prankk_impl
#endif
