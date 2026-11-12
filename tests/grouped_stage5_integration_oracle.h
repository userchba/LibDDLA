#ifndef DDLA_TEST_GROUPED_STAGE5_INTEGRATION_ORACLE_H
#define DDLA_TEST_GROUPED_STAGE5_INTEGRATION_ORACLE_H
#include "grouped_stage5_oracle.h"
#include "Eigensolver/two_stage/phe2hb_impl.h"
namespace grouped_stage5_oracle {
template<class T> MPI_Datatype datatype() {
    if constexpr(std::is_same<T,float>::value)return MPI_FLOAT;
    else if constexpr(std::is_same<T,double>::value)return MPI_DOUBLE;
    else if constexpr(std::is_same<T,std::complex<float>>::value)return MPI_C_FLOAT_COMPLEX;
    else return MPI_C_DOUBLE_COMPLEX;
}
struct IntegrationResult {Result apply;double reconstruction=0,guards=0;bool used_upper=false;};
template<class T,ddla::DdlaBackend Backend=ddla::DdlaBackend::CPU> IntegrationResult integration(const ddla::DdlaHandle_t&h,int n,int nb,int b,bool upper) {
    const auto g=grid(h);const int steps=(n-1)/b,tile=2*nb,parent=5,pad=3;
    int da[ddla::DDLA_DLEN_];ddla::ddlaDescInit(da,h,n+parent,n+parent,nb,nb,0,0);da[ddla::DDLA_LLD_]+=pad;
    auto a=local_matrix<T>(h,da,[&](int i,int j){
        if(i>=n||j>=n)return T(23);
        if(upper&&i/tile>j/tile)return scalar<T>(std::numeric_limits<Real<T>>::quiet_NaN());
        return input_a<T>(i,j);
    });
    const auto before=a;std::vector<T>t(std::max<std::size_t>(1,std::size_t(steps)*b*b),T(7));int info=0;bool used=false;
    if constexpr(Backend==ddla::DdlaBackend::CPU) {
        if(upper)used=ddla::detail::phe2hb_upper_cpu<T>(h,n,a.data(),da,b,t.data(),info,tile);
        else ddla::phe2hb<T>(h,n,a.data(),da,b,t.data(),info);
    } else {
#if DDLA_HAS_GPU
        ddla::vector<T> device_a(h,a.size()),device_t(h,t.size());
        DDLA_CHECK(ddlaMemcpy(h,device_a.data(),a.data(),a.size()*sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h,device_t.data(),t.data(),t.size()*sizeof(T)));
        ddla::phe2hb<T>(h,n,device_a.data(),da,b,device_t.data(),info);
        DDLA_CHECK(ddlaMemcpy(h,a.data(),device_a.data(),a.size()*sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h,t.data(),device_t.data(),t.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#endif
    }
    if(info)throw std::runtime_error("stage1 producer info");
    IntegrationResult out;out.used_upper=used;
    if(upper&&used!=(steps*b>tile))throw std::runtime_error("upper stage1 producer did not exercise expected branch");
    const int rows=ddla::num_loc(n,nb,g.mr,0,g.pr),cols=ddla::num_loc(n,nb,g.mc,0,g.pc);
    const int parent_rows=ddla::num_loc(n+parent,nb,g.mr,0,g.pr),parent_cols=ddla::num_loc(n+parent,nb,g.mc,0,g.pc);
    std::vector<T>part(std::size_t(n)*n,T(0)),full(part.size(),T(0));
    for(int c=0;c<cols;++c)for(int r=0;r<rows;++r){const int i=global(r,nb,g.mr,0,g.pr),j=global(c,nb,g.mc,0,g.pc);if(i<=j)part[i+std::size_t(j)*n]=a[r+std::size_t(c)*da[ddla::DDLA_LLD_]];}
    const int rc=::MPI_Allreduce(part.data(),full.data(),int(full.size()),datatype<T>(),MPI_SUM,ddlaGetCommunicator(h));
    if(rc!=MPI_SUCCESS)throw std::runtime_error("integration gather");
    Fixture<T>f{n,b,std::move(full),t,std::vector<T>(std::max(1,steps*b),T(0))};
    for(int slot=0;slot<steps;++slot){const int k=std::min(b,n-(slot+1)*b);for(int c=0;c<k;++c)f.tau[slot*b+c]=t[std::size_t(slot)*b*b+c+std::size_t(c)*b];}
    if constexpr(Backend==ddla::DdlaBackend::CPU)
        out.apply=prepared(h,f,a.data(),da,parent,pad,2*nb,g.pc>1?1:0);
#if DDLA_HAS_GPU
    else out.apply=prepared_gpu(h,f,a.data(),da,parent,pad,2*nb,g.pc>1?1:0);
#endif
    for(int c=0;c<parent_cols;++c)for(int r=0;r<da[ddla::DDLA_LLD_];++r){const int i=r<parent_rows?global(r,nb,g.mr,0,g.pr):n+parent,j=global(c,nb,g.mc,0,g.pc);const auto at=r+std::size_t(c)*da[ddla::DDLA_LLD_];if(i>=n||j>=n)out.guards+=std::memcmp(&a[at],&before[at],sizeof(T))!=0;}
    // Independent elementary-reflector Q and band reconstruction A=Q^H B Q.
    const auto q=q_from_tau(f);std::vector<Wide>band(std::size_t(n)*n),bq(band.size()),rec(band.size());
    for(int c=0;c<n;++c)for(int r=std::max(0,c-b);r<=c;++r){band[r+std::size_t(c)*n]=wide(f.a[r+std::size_t(c)*n]);if(r<c)band[c+std::size_t(r)*n]=std::conj(band[r+std::size_t(c)*n]);}
    for(int c=0;c<n;++c)for(int r=0;r<n;++r)for(int l=0;l<n;++l)bq[r+std::size_t(c)*n]+=band[r+std::size_t(l)*n]*q[l+std::size_t(c)*n];
    for(int c=0;c<n;++c)for(int r=0;r<n;++r)for(int l=0;l<n;++l)rec[r+std::size_t(c)*n]+=std::conj(q[l+std::size_t(r)*n])*bq[l+std::size_t(c)*n];
    long double scale=1,error=0;for(int c=0;c<n;++c)for(int r=0;r<n;++r){const auto v=wide(input_a<T>(r,c));scale=std::max(scale,std::abs(v));const auto d=std::abs(rec[r+std::size_t(c)*n]-v);if(!std::isfinite(d))error=std::numeric_limits<long double>::infinity();else error=std::max(error,d);}
    out.reconstruction=double(error/(scale*n*std::numeric_limits<Real<T>>::epsilon()));return out;
}
} // namespace grouped_stage5_oracle
#endif
