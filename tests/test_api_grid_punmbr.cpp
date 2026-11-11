#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"

// Apply distributed bidiagonal reflectors against independent host LAPACK operators.
#include "svd_test_common.h"

namespace {

template<class T> void one(const DdlaHandle_t& h,int m,int n,int mb,int nb) {
    using R=typename real_type<T>::type;
    const int k=std::min(m,n);
    std::vector<T> a(std::size_t(m)*n);
    for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[i+std::size_t(j)*m]=hash_value<T>(i,j,5);
    std::vector<R> d(k),e(std::max(1,k-1));std::vector<T> tq(k),tp(k);
    host_gebrd(a,m,n,d,e,tq,tp);
    std::vector<T> q(std::size_t(m)*m,T(0)),ph(std::size_t(n)*n,T(0));
    for(int j=0;j<k;++j)for(int i=0;i<m;++i)q[i+std::size_t(j)*m]=a[i+std::size_t(j)*m];
    for(int j=0;j<n;++j)for(int i=0;i<k;++i)ph[i+std::size_t(j)*n]=a[i+std::size_t(j)*m];
    host_ungbr('Q',m,n,q,tq);host_ungbr('P',n,m,ph,tp);
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    CaseGeom g{m,n,mb,nb,pr>1?1:0,pc>1?1:0,m+2,n+3};
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,g.dm,g.dn,mb,nb,g.rsrc,g.csrc));
    auto local=make_local<T>(h,desc,[&](int i,int j){return i<m&&j<n?a[i+std::size_t(j)*m]:T(7);});
    DeviceBuffer<T> storage(h,local.size());upload(h,storage.ptr,local);
    check_punmbr(h,g,storage.ptr,desc,tq,tp,q,ph,std::string(type_tag<T>())+" "+std::to_string(m)+"x"+std::to_string(n));
    const auto after=download<T>(h,storage.ptr,local.size());
    require_close(h,"punmbr preserves stored reflectors",std::memcmp(after.data(),local.data(),local.size()*sizeof(T))==0?0:1,0);
}
void cases(const DdlaHandle_t& h) {
    for(auto shape : {std::pair<int,int>{1,1},{7,5},{5,7},{17,17},{31,19},{19,31}}) {
        one<float>(h,shape.first,shape.second,3,5);
        one<double>(h,shape.first,shape.second,3,5);
        one<std::complex<float>>(h,shape.first,shape.second,3,5);
        one<std::complex<double>>(h,shape.first,shape.second,3,5);
    }
    // Large, nondivisible descriptor blocks split into bounded panels without
    // crossing process owners. Both reflector families and orders are checked.
    for (auto shape : {std::pair<int,int>{521,263},{263,521}}) {
        one<float>(h,shape.first,shape.second,257,259);
        one<double>(h,shape.first,shape.second,257,259);
        one<std::complex<float>>(h,shape.first,shape.second,257,259);
        one<std::complex<double>>(h,shape.first,shape.second,257,259);
    }
}
void check_body(const DdlaHandle_t& h,const Shape&) {
    cases(h);
#if DDLA_HAS_GPU
    if(ddlaGetBackend(h)==DdlaBackend::GPU) {
        int pr,pc;ddlaGetGridDims(h,pr,pc);DdlaHandle_t cpu=nullptr;
        DDLA_CHECK(ddlaInit(cpu,DdlaBackend::CPU));DDLA_CHECK(ddlaSet(cpu,ddlaGetCommunicator(h),pr,pc));
        cases(cpu);DDLA_CHECK(ddlaDestroy(cpu));
    }
#endif
}
} // namespace
int main(int argc,char** argv) {
    return run_grid_test(argc,argv,"test_api_grid_punmbr",check_body);
}
