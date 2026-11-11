#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "punmtr_impl.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "plan_util.h"
#include "instantiate.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <stdexcept>
#include <type_traits>

namespace ddla {
namespace detail {

template<class T>void punmtr_check(const DdlaHandle_t& h,DdlaBackend backend,
    char side,char uplo,char trans,int m,int n,int k,const T* V,int ldv,
    const T* tmat,int ldt,T* C,const int* descC)
{
    agree_check(h,"punmtr",[&](Agree& agreement) {
        agreement.mode("punmtr interface",0);
        if(!h||ddlaGetBackend(h)!=backend)
            throw std::invalid_argument("punmtr: null handle or backend mismatch");
        const char s=static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
        const char u=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
        const char t=static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
        constexpr bool real=std::is_same<T,typename real_type<T>::type>::value;
        if((s!='L'&&s!='R')||(u!='U'&&u!='L')||(t!='N'&&t!='C'&&!(real&&t=='T')))
            throw std::invalid_argument("punmtr: invalid side, uplo or trans");
        check_desc(descC,h);
        if(m<0||n<0||k<0||m>descC[DDLA_M_]||n>descC[DDLA_N_])
            throw std::invalid_argument("punmtr: invalid matrix extent or reflector count");
        const int rows=num_loc(m,descC[DDLA_MB_],h->myprow_,descC[DDLA_RSRC_],h->nprows_);
        const int cols=num_loc(n,descC[DDLA_NB_],h->mypcol_,descC[DDLA_CSRC_],h->npcols_);
        if(!m||!n||!k)return;
        const bool empty=s=='L'?cols==0:rows==0;
        const int ldw=s=='L'?std::max(k,1):std::max(rows,1);
        const std::size_t elements=empty?0:static_cast<std::size_t>(ldw)*(s=='L'?cols:k);
        if(ldv<std::max(1,s=='L'?rows:cols)||ldt<std::max(1,k))
            throw std::invalid_argument("punmtr: invalid V or T leading dimension");
        if(elements>static_cast<std::size_t>(INT_MAX/2))
            throw std::overflow_error("punmtr: workspace collective count exceeds int");
        if(!empty) {
            if(!tmat||((s=='L'?rows:cols)>0&&!V)||(rows>0&&cols>0&&!C))
                throw std::invalid_argument("punmtr: missing active input or output");
        }
    });
    int signature[17]={int(backend),int(blas_impl::upper(side)),int(blas_impl::upper(uplo)),int(blas_impl::upper(trans)),m,n,k,
                       h->nprows_,h->npcols_,int(h->major)};
    descriptor_signature(signature+10,descC);
    agree_signature(h,"punmtr",signature);
}

template<class T>void punmtr_workspace_size(const DdlaHandle_t&,char side,int rows,int cols,int k,int ldw,
    bool reduce,std::size_t* bytes)
{
    const bool empty=k==0||(side=='L'?cols==0:rows==0);
    const std::size_t elements=empty?0:static_cast<std::size_t>(ldw)*(side=='L'?cols:k);
    *bytes=padded_bytes<T>(elements)*(reduce?2:1);
}

template<DdlaBackend B,class T>void punmtr_project_run(const DdlaHandle_t& h,char side,
    int rows,int cols,int k,int ldv,int ldc,int ldw,const T* V,const T* C,T* work)
{
    const bool left=side=='L';
    if(k==0||(left?cols==0:rows==0))return;
    constexpr char adj=std::is_same<T,typename real_type<T>::type>::value?'T':'C';
    const int active_rows=left?k:rows;
    const std::size_t elements=static_cast<std::size_t>(ldw)*(left?cols:k);
    // Include pitch gaps and zero contributions on empty contraction ranks.
    if(!rows||!cols||ldw>active_rows)
        DDLA_CHECK(ddlaMemset(h,work,0,elements*sizeof(T)));
    if(rows>0&&cols>0) {
        if(left)blas_impl::gemm_run<B,T>(h,adj,'N',k,cols,rows,T(1),V,ldv,C,ldc,T(0),work,ldw);
        else blas_impl::gemm_run<B,T>(h,'N','N',rows,k,cols,T(1),C,ldc,V,ldv,T(0),work,ldw);
    }
}

template<DdlaBackend B,class T>void punmtr_update_run(const DdlaHandle_t& h,char side,
    int rows,int cols,int k,int ldv,int ldc,int ldw,const T* V,const T* product,T* C)
{
    if(k==0||!rows||!cols)return;
    constexpr char adj=std::is_same<T,typename real_type<T>::type>::value?'T':'C';
    if(side=='L')blas_impl::gemm_run<B,T>(h,'N','N',rows,cols,k,T(-1),V,ldv,product,ldw,T(1),C,ldc);
    else blas_impl::gemm_run<B,T>(h,'N',adj,rows,cols,k,T(-1),product,ldw,V,ldv,T(1),C,ldc);
}

template<DdlaBackend B,class T>void punmtr_run(const DdlaHandle_t& h,char side,char uplo,char trans,
    int rows,int cols,int k,int ldv,int ldt,int ldc,int ldw,bool reduce,
    const T* V,const T* tmat,T* C,T* work,T* reduced_work)
{
    if(k==0||(side=='L'?cols==0:rows==0))return;
    const std::size_t elements=static_cast<std::size_t>(ldw)*(side=='L'?cols:k);
    punmtr_project_run<B,T>(h,side,rows,cols,k,ldv,ldc,ldw,V,C,work);
    if(reduce)commAllReduce<B>(h,side=='L'?CommScope::Col:CommScope::Row,work,reduced_work,int(elements),cclSum);
    T* const product=reduce?reduced_work:work;
    trans=trans=='N'?'N':(std::is_same<T,typename real_type<T>::type>::value?'T':'C');
    blas_impl::trmm_run<B,T>(h,side,uplo,trans,'N',side=='L'?k:rows,side=='L'?cols:k,T(1),tmat,ldt,product,ldw);
    punmtr_update_run<B,T>(h,side,rows,cols,k,ldv,ldc,ldw,V,product,C);
}
} // namespace detail

template<DdlaBackend B,class T>void punmtr(const DdlaHandle_t& h,char side,char uplo,char trans,
    int m,int n,int k,const T* V,int ldv,const T* tmat,int ldt,T* C,const int* descC)
{
    detail::punmtr_check<T>(h,B,side,uplo,trans,m,n,k,V,ldv,tmat,ldt,C,descC);
    if(!m||!n||!k)return;
    side=blas_impl::upper(side);uplo=blas_impl::upper(uplo);trans=blas_impl::upper(trans);
    const int rows=num_loc(m,descC[DDLA_MB_],h->myprow_,descC[DDLA_RSRC_],h->nprows_);
    const int cols=num_loc(n,descC[DDLA_NB_],h->mypcol_,descC[DDLA_CSRC_],h->npcols_);
    const int ldw=side=='L'?std::max(k,1):std::max(rows,1);
    const bool reduce=(side=='L'?h->nprows_:h->npcols_)>1;
    const bool empty=side=='L'?cols==0:rows==0;
    const std::size_t elements=empty?0:static_cast<std::size_t>(ldw)*(side=='L'?cols:k);
    std::size_t bytes=0;detail::punmtr_workspace_size<T>(h,side,rows,cols,k,ldw,reduce,&bytes);
    vector<char> storage(h);
    detail::agree_check(h,"punmtr allocation",[&](detail::Agree&){storage.resize(bytes);});
    if(empty)return;
    char* next=storage.data();T* work=detail::take<T>(next,elements);
    T* reduced=reduce?detail::take<T>(next,elements):nullptr;
    detail::punmtr_run<B,T>(h,side,uplo,trans,rows,cols,k,ldv,ldt,descC[DDLA_LLD_],ldw,reduce,V,tmat,C,work,reduced);
}

#define DDLA_UNMTR_INSTANTIATE(B,T) \
    template void punmtr<B,T>(const DdlaHandle_t&,char,char,char,int,int,int,const T*,int,const T*,int,T*,const int*); \
    template void detail::punmtr_run<B,T>(const DdlaHandle_t&,char,char,char,int,int,int,int,int,int,int,bool,const T*,const T*,T*,T*,T*); \
    template void detail::punmtr_project_run<B,T>(const DdlaHandle_t&,char,int,int,int,int,int,int,const T*,const T*,T*); \
    template void detail::punmtr_update_run<B,T>(const DdlaHandle_t&,char,int,int,int,int,int,int,const T*,const T*,T*)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_UNMTR_INSTANTIATE)
#define DDLA_UNMTR_SIZE(T) \
    template void detail::punmtr_check<T>(const DdlaHandle_t&,DdlaBackend,char,char,char,int,int,int,const T*,int,const T*,int,T*,const int*); \
    template void detail::punmtr_workspace_size<T>(const DdlaHandle_t&,char,int,int,int,int,bool,std::size_t*)
DDLA_FOR_EACH_TYPE(DDLA_UNMTR_SIZE)
#undef DDLA_UNMTR_SIZE
#undef DDLA_UNMTR_INSTANTIATE
} // namespace ddla
