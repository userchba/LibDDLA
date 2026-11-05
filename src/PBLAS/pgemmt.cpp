#include "helpers.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "pgemmt_impl.h"
#include "pgemmt_panel_impl.h"
#include "pgeadd_impl.h"
#include "pblas_workspace.h"
#include "BLAS/blas_impl.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "require_backend.h"
#include "plan_util.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <vector>

namespace ddla { namespace pgemmt_impl { namespace {

template <typename T>
void scratch(const DdlaHandle_t& h, char ta, char tb, bool product, int n, int k,
    const int* da, const int* db, const int* dc, std::size_t (&count)[4],
    std::size_t (&add)[2][2], std::size_t& triangle_bytes)
{
    pblas::product_workspace_size<T>(h,ta,tb,product,false,n,k,da,db,dc,count,add);
    triangle_bytes=0;
    if (n==0 || !product) return;
    if (h->nprocs>1 && h->backend==DdlaBackend::GPU) {
        std::size_t diagonals=0;
        const std::size_t leaf=std::min(dc[DDLA_MB_],128);
        panel::diagonal_tiles(h,n,dc,[&](int,int,int width) {diagonals+=(width+leaf-1)/leaf;});
        triangle_bytes=detail::padded_bytes<T>(diagonals*leaf*leaf)
            +detail::padded_bytes<T*>(diagonals)+detail::padded_bytes<int>(diagonals);
    } else {
        const int local_n=h->nprocs==1 ? n : std::min(n,dc[DDLA_MB_]);
        const int local_k=h->nprocs==1 ? k : std::min(k,pblas::contraction_width(h,dc[DDLA_MB_],k));
        blas_impl::gemmt_workspace_size<T>(h,ta,tb,local_n,local_k,&triangle_bytes);
    }
}

}} // namespace pgemmt_impl::<anonymous>

namespace pgemmt_impl {

template <DdlaBackend Backend, typename T>
bool check(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, const T* C, const int* dc)
{
    return detail::agree_check(h,"pgemmt",[&](detail::Agree& agree) {
        return pblas::product_check<Backend,T>(h,agree,uplo,ta,tb,n,k,alpha,A,da,B,db,beta,C,dc);
    });
}

template <typename T>
void workspace_size(const DdlaHandle_t& h, char uplo, char ta, char tb, bool product, int n, int k,
    const int* da, const int* db, const int* dc,
    std::size_t* backend_bytes, std::size_t* host_bytes)
{
    std::size_t count[4],add[2][2],triangle_bytes=0;
    scratch<T>(h,ta,tb,product,n,k,da,db,dc,count,add,triangle_bytes);
    *backend_bytes=triangle_bytes;
    *host_bytes=0;
    for (const std::size_t c : count) *backend_bytes+=detail::padded_bytes<T>(c);
    for (const auto& bytes : add) {
        *backend_bytes+=detail::padded_bytes<char>(bytes[0]);
        *host_bytes+=detail::padded_bytes<char>(bytes[1]);
    }
    if (n>0 && product && h->nprocs>1 && h->backend==DdlaBackend::GPU) {
        std::size_t backend=0,host=0;
        panel::batch_bytes<T>(h,uplo,n,k,pblas::contraction_width(h,dc[DDLA_MB_],k),dc,backend,host);
        *backend_bytes+=backend;
        *host_bytes+=host;
    }
}

template <DdlaBackend Backend, typename T>
void run(const DdlaHandle_t& h, char uplo, char ta, char tb, bool product,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, T* C, const int* dc,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    std::size_t count[4],add[2][2],triangle_bytes=0;
    scratch<T>(h,ta,tb,product,n,k,da,db,dc,count,add,triangle_bytes);
    void* triangle_work=triangle_bytes ? static_cast<char*>(backend_work)+backend_bytes-triangle_bytes : nullptr;
    backend_bytes-=triangle_bytes;
    try {
        if(h->nprocs==1 && product) {
            blas_impl::gemmt_run<Backend,T>(h,uplo,ta,tb,n,k,alpha,A,da[DDLA_LLD_],B,db[DDLA_LLD_],beta,C,dc[DDLA_LLD_],triangle_work,triangle_bytes);
            return;
        }

        if (n==0 || (!product && beta==T(1))) return;

        // The scratch, in the order workspace_size counts it.
        char* w=static_cast<char*>(backend_work);
        char* hw=static_cast<char*>(host_work);
        T* const a=detail::take<T>(w,count[0]);
        T* const b=detail::take<T>(w,count[1]);
        T* const work=detail::take<T>(w,count[2]);
        T* const b_tile=detail::take<T>(w,count[3]);
        char* add_work[2];
        char* add_host[2];
        for (int i=0;i<2;++i) {
            add_work[i]=detail::take<char>(w,add[i][0]);
            add_host[i]=detail::take<char>(hw,add[i][1]);
        }
        const std::size_t used=static_cast<std::size_t>(w-static_cast<char*>(backend_work));
        const std::size_t host_used=static_cast<std::size_t>(hw-static_cast<char*>(host_work));
        assert(used<=backend_bytes && host_used<=host_bytes);
        // One k panel of op(A) (left) or op(B) into output, through the exchange
        // buffer and the panel's pgeadd scratch.
        const auto load=[&](bool left,char trans,int g,int kw,const T* input,const int* di,T* output) {
            const int i=left ? 0 : 1;
            pblas::product_panel_run<Backend,T>(h,left,trans,n,g,kw,input,di,dc,output,work,b_tile,
                add_work[i],add[i][0],add_host[i],add[i][1]);
        };
        if (product) {
            const auto prepare=[&](int g,int kw) {
                load(true,ta,g,kw,A,da,a);
                load(false,tb,g,kw,B,db,b);
            };
            panel::run<Backend,T>(h,uplo,n,k,pblas::contraction_width(h,dc[DDLA_MB_],k),
                alpha,beta,C,dc,prepare,a,b,w,backend_bytes-used,hw,host_bytes-host_used,
                triangle_work,triangle_bytes,/*cpu_batch=*/false);
        } else
            pblas::triangle_panel_run<Backend,T>(h,uplo,false,n,0,alpha,a,b,beta,
                C,dc,triangle_work,triangle_bytes);

    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

} // namespace pgemmt_impl

template <DdlaBackend Backend, typename T>
void pgemmt(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, T* C, const int* dc)
{
    const bool product=pgemmt_impl::check<Backend,T>(h,uplo,ta,tb,n,k,alpha,A,da,B,db,beta,C,dc);
    uplo=blas_impl::upper(uplo);ta=blas_impl::upper(ta);tb=blas_impl::upper(tb);
    if (n==0 || (!product && beta==T(1))) return;
    std::size_t backend_bytes=0,host_bytes=0;
    pgemmt_impl::workspace_size<T>(h,uplo,ta,tb,product,n,k,da,db,dc,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    pgemmt_impl::run<Backend,T>(h,uplo,ta,tb,product,n,k,alpha,A,da,B,db,beta,C,dc,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE(B,T) \
 template void pgemmt<B,T>(const DdlaHandle_t&,char,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*); \
 template bool pgemmt_impl::check<B,T>(const DdlaHandle_t&,char,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,const T*,const int*); \
 template void pgemmt_impl::run<B,T>(const DdlaHandle_t&,char,char,char,bool,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*,void*,std::size_t,void*,std::size_t)
#define INSTANTIATE_TYPES(B) \
 INSTANTIATE(B,float); INSTANTIATE(B,double); \
 INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
INSTANTIATE_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_TYPES(DdlaBackend::GPU);
#endif
#define INSTANTIATE_SIZE(T) template void pgemmt_impl::workspace_size<T>(const DdlaHandle_t&,char,char,char,bool,int,int,const int*,const int*,const int*,std::size_t*,std::size_t*)
INSTANTIATE_SIZE(float); INSTANTIATE_SIZE(double);
INSTANTIATE_SIZE(std::complex<float>); INSTANTIATE_SIZE(std::complex<double>);
#undef INSTANTIATE_SIZE
#undef INSTANTIATE_TYPES
#undef INSTANTIATE
} // namespace ddla
