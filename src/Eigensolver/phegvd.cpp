#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "Cholesky/cholesky_workspace.h"
#include "PBLAS/ptrtrs_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "instantiate.h"
#include "require_backend.h"
#include <algorithm>
#include <cctype>
#include <vector>

namespace ddla {
template<DdlaBackend B,class T>
void phegvd(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,
            T* b,const int* db,typename real_type<T>::type* w,T* z,const int* dz,int& info)
{
    job=static_cast<char>(std::toupper(static_cast<unsigned char>(job)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    const bool vectors=job=='V' && nev>0;
    // Validate all outputs and factor/input aliasing before destroying B.
    detail::agree_check(h,"phegvd",[&](detail::Agree&) {
        detail::require_backend<B>(h,"phegvd");
        if(n<0 || nev<0 || nev>n || (job!='N' && job!='V') || (uplo!='U' && uplo!='L'))
            throw std::invalid_argument("phegvd: invalid order/count/options");
        check_desc(da,h);check_desc(db,h);
        const int nb=da[DDLA_MB_];
        for(const int* d:{da,db}) {
            if(n>d[DDLA_M_] || n>d[DDLA_N_] || d[DDLA_MB_]!=nb || d[DDLA_NB_]!=nb ||
               d[DDLA_RSRC_]!=0 || d[DDLA_CSRC_]!=0)
                throw std::invalid_argument("phegvd: A/B need matching square source-zero blocks and n-by-n extents");
        }
        const int ar=num_loc(n,nb,h->myprow_,0,h->nprows_);
        const int ac=num_loc(n,nb,h->mypcol_,0,h->npcols_);
        if((n && !w) || (ar && ac && (!a || !b)))
            throw std::invalid_argument("phegvd: missing active input/eigenvalues");
        if(detail::matrix_overlaps(a,da[DDLA_LLD_],ar,ac,b,db[DDLA_LLD_],ar,ac))
            throw std::invalid_argument("phegvd: A and B must not overlap");
        if(vectors) {
            check_desc(dz,h);
            if(n>dz[DDLA_M_] || nev>dz[DDLA_N_] || dz[DDLA_MB_]!=nb || dz[DDLA_NB_]!=nb || dz[DDLA_RSRC_]!=0)
                throw std::invalid_argument("phegvd: incompatible eigenvector descriptor");
            const int zc=num_loc(nev,nb,h->mypcol_,dz[DDLA_CSRC_],h->npcols_);
            if(ar && zc && !z)throw std::invalid_argument("phegvd: missing active output");
            if(detail::matrix_overlaps(b,db[DDLA_LLD_],ar,ac,z,dz[DDLA_LLD_],ar,zc))
                throw std::invalid_argument("phegvd: B and Z must not overlap");
        }
    });
    int signature[26]={n,nev,job,uplo,int(B)};
    detail::descriptor_signature(signature+5,da);
    detail::descriptor_signature(signature+12,db);
    if(vectors)detail::descriptor_signature(signature+19,dz);
    detail::agree_signature(h,"phegvd",signature);
    info=0;
    if(!n)return;
    const char left=uplo=='U'?'C':'N',right=uplo=='U'?'N':'C';
    std::size_t bytes=0,host_bytes=0;
    ddla::vector<char> scratch(h);
    std::vector<char> host;
    detail::agree_local(h,"phegvd workspace",[&] {
        cholesky_impl::ppotrf_workspace_size<T>(h,uplo,n,b,db,&bytes,&host_bytes);
        auto include=[&](char side,char trans,int columns,const int* out) {
            std::size_t device=0,cpu=0;
            ptrtrs_impl::ptrtrs_workspace_size<T>(h,side,trans,n,columns,db,out,&device,&cpu);
            bytes=std::max(bytes,device);host_bytes=std::max(host_bytes,cpu);
        };
        include('L',left,n,da);include('R',right,n,da);
        if(vectors)include('L',uplo=='U'?'N':'C',nev,dz);
        scratch.resize(bytes);host.resize(host_bytes);
    });
    cholesky_impl::ppotrf_run<B,T>(h,uplo,n,b,1,1,db,info,scratch.data(),bytes,host.data(),host_bytes);
    if(info) {if(info>0)info+=n;return;}
    // RHS is dense during both solves; the caller need only store UPLO of A.
    pherm_complete<B,T>(h,uplo,n,a,da);
    ptrtrs_impl::ptrtrs_run<B,T>(h,'L',uplo,left,'N',n,n,b,db,a,da,
        scratch.data(),bytes,host.data(),host_bytes);
    ptrtrs_impl::ptrtrs_run<B,T>(h,'R',uplo,right,'N',n,n,b,db,a,da,
        scratch.data(),bytes,host.data(),host_bytes);
    pheevd<B,T>(h,vectors?'V':'N',uplo,n,nev,a,da,w,z,dz,info);
    if(!info && vectors)
        ptrtrs_impl::ptrtrs_run<B,T>(h,'L',uplo,uplo=='U'?'N':'C','N',n,nev,b,db,z,dz,
            scratch.data(),bytes,host.data(),host_bytes);
    DDLA_CHECK(ddlaSynchronize(h));
}
#define GENERALIZED(B,T) template void phegvd<B,T>(const DdlaHandle_t&,char,char,int,int,T*,const int*,T*,const int*,typename real_type<T>::type*,T*,const int*,int&)
DDLA_FOR_EACH_BACKEND_TYPE(GENERALIZED)
#undef GENERALIZED
} // namespace ddla
