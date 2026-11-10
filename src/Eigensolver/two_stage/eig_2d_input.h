#ifndef DDLA_EIG_2D_INPUT_H
#define DDLA_EIG_2D_INPUT_H

#include "matrix_transfer.h"
#include "plan_util.h"
#include "check_agree.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include <algorithm>
#include <array>
#include <cassert>

namespace ddla { namespace detail {
// Stage 4 consumes complete eigenvector columns in its established local
// strip/tile format. Move bounded runs straight from the 2D factor into that
// storage; no intermediate matrix of contiguous row slabs is formed.
struct Eig2dInputPlan {
    int n=0,width=1;
    std::array<int,DDLA_DLEN_> descriptor{};
    bool enabled=false;
    std::size_t backend_bytes=0,host_bytes=0,cpu_host_offset=0;
};
template<class R>Eig2dInputPlan eig_2d_input_geometry(const DdlaHandle_t& h,int n,const int* desc)
{
    Eig2dInputPlan p;p.n=n;p.enabled=true;
    std::copy_n(desc,DDLA_DLEN_,p.descriptor.data());
    const auto maxcols=(static_cast<long long>(n)+h->nprocs-1)/h->nprocs;
    p.width=std::max(1,int(std::min({64LL,maxcols,
        (32LL<<20)/std::max(1LL,static_cast<long long>(n)*static_cast<long long>(sizeof(R)))})));
    if(n){const auto panel=std::size_t(n)*p.width;
        p.backend_bytes=padded_bytes<R>(panel)+panel*sizeof(R);
        p.host_bytes=padded_bytes<matrix_impl::Block<R>>(std::max(0,h->nprocs-1));
        p.cpu_host_offset=padded_bytes<char>(p.backend_bytes);}
    return p;
}
template<class R>void eig_2d_input_workspace_size(const DdlaHandle_t& h,const Eig2dInputPlan& p,
    std::size_t* backend_bytes,std::size_t* host_bytes)
{
    (void)h;*backend_bytes=p.backend_bytes;*host_bytes=p.host_bytes;
}
template<class R>std::size_t eig_2d_input_cpu_bytes(const DdlaHandle_t& h,const Eig2dInputPlan& p)
{
    (void)h;return p.cpu_host_offset+p.host_bytes;
}
// count(q) and column(q,k) describe the existing stage-4 column ownership.
// consume(first,count,panel,ld) runs only on its owner, before the panel is
// reused. Every rank follows identical transfers, including empty owners.
template<DdlaBackend B,class R,class Count,class Column,class Consume>
void eig_2d_input_run(const DdlaHandle_t& h,const Eig2dInputPlan& p,const R* Q,
    Count count,Column column,Consume consume,void* backend_work,std::size_t backend_bytes,
    void* host_work,std::size_t host_bytes)
{
    if(!p.n)return;
    char* work=static_cast<char*>(backend_work);
    R* panel=take<R>(work,std::size_t(p.n)*p.width);
    const auto used=std::size_t(work-static_cast<char*>(backend_work));
    assert(used<=backend_bytes);
    for(int root=0;root<h->nprocs;++root)for(int first=0;first<count(root);) {
        const int global=column(root,first);int columns=1;
        while(columns<p.width&&first+columns<count(root)&&column(root,first+columns)==global+columns)++columns;
        matrix_impl::gather_run<B,R>(h,p.n,columns,Q,1,global+1,p.descriptor.data(),root,panel,p.n,
            work,backend_bytes-used,host_work,host_bytes);
        agree_check(h,"2D eigenvector replay input",[&](Agree&){
            if(h->myid==root)consume(first,columns,panel,p.n);
            // Transfers and consume use the handle stream, so the following
            // panel reuse is ordered there. The enclosing GPU replay fences
            // once after all panels, before its output may overwrite Q.
        });
        first+=columns;
    }
}
template<class R,class Count,class Column,class Consume>
void eig_2d_input_cpu_run(const DdlaHandle_t& h,const Eig2dInputPlan& p,const R* Q,
    Count count,Column column,Consume consume,void* work,std::size_t bytes)
{
    assert(p.cpu_host_offset+p.host_bytes<=bytes);
    eig_2d_input_run<DdlaBackend::CPU,R>(h,p,Q,count,column,consume,work,p.backend_bytes,
        static_cast<char*>(work)+p.cpu_host_offset,p.host_bytes);
}
} }
#endif
