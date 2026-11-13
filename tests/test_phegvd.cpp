#include "generalized_eigen_test_common.h"
#include "mpi_datatype.h"
#include <cstring>
using namespace api_grid_test;
namespace {
template<class T>void solve(const DdlaHandle_t& h,int n,int nev,char uplo,bool alias,bool repeated,char job='V') {
    using R=typename real_type<T>::type;
    constexpr int nb=4;
    const bool vectors=job=='V'&&nev>0;
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    int da[DDLA_DLEN_],db[DDLA_DLEN_],dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,nb,nb,0,0));da[DDLA_LLD_]+=3;
    DDLA_CHECK(ddlaDescInit(db,h,n+3,n+2,nb,nb,0,0));db[DDLA_LLD_]+=2;
    DDLA_CHECK(ddlaDescInit(dz,h,n+2,nev+3,nb,nb,0,pc-1));dz[DDLA_LLD_]+=4;
    if(alias)std::copy_n(da,DDLA_DLEN_,dz);
    const T sentinel=T(9876),nan=T(std::numeric_limits<R>::quiet_NaN());
    auto aa=make_local<T>(h,da,[&](int i,int j){return i>=n||j>=n?sentinel:(uplo=='U'?i<=j:i>=j)?generalized_test::a<T>(i,j,n,repeated):nan;});
    auto bb=make_local<T>(h,db,[&](int i,int j){return i>=n||j>=n?sentinel:(uplo=='U'?i<=j:i>=j)?generalized_test::b<T>(i,j,n):nan;});
    for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=ddla_test::m_loc(h,da);i<da[DDLA_LLD_];++i)aa[i+std::size_t(j)*da[DDLA_LLD_]]=sentinel;
    for(int j=0;j<ddla_test::n_loc(h,db);++j)for(int i=ddla_test::m_loc(h,db);i<db[DDLA_LLD_];++i)bb[i+std::size_t(j)*db[DDLA_LLD_]]=sentinel;
    DeviceBuffer<T> a(h,std::max(std::size_t(1),aa.size())),b(h,std::max(std::size_t(1),bb.size())),z(h,std::max(std::size_t(1),local_size(h,dz)));
    upload(h,a.ptr,aa);upload(h,b.ptr,bb);
    upload(h,z.ptr,std::vector<T>(local_size(h,dz),sentinel));
    std::vector<R> w(n+4,R(-9876));int info=-1;
    phegvd<T>(h,job,uplo,n,nev,a.ptr,da,b.ptr,db,w.data()+2,vectors?(alias?a.ptr:z.ptr):nullptr,vectors?dz:nullptr,info);
    require_close(h,"distributed generalized info",info,0);
    bool guards=w[0]==R(-9876)&&w[1]==R(-9876)&&w[n+2]==R(-9876)&&w[n+3]==R(-9876);
    const auto after_a=download(h,a.ptr,aa.size()),after_b=download(h,b.ptr,bb.size());
    for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=0;i<da[DDLA_LLD_];++i) {
        const int gi=indx_l2g_r(da,h,i),gj=indx_l2g_c(da,h,j);const auto at=i+std::size_t(j)*da[DDLA_LLD_];
        if(i>=ddla_test::m_loc(h,da)||gi>=n||gj>=n)guards=guards&&!std::memcmp(&aa[at],&after_a[at],sizeof(T));
    }
    for(int j=0;j<ddla_test::n_loc(h,db);++j)for(int i=0;i<db[DDLA_LLD_];++i) {
        const int gi=indx_l2g_r(db,h,i),gj=indx_l2g_c(db,h,j);const auto at=i+std::size_t(j)*db[DDLA_LLD_];
        if(i>=ddla_test::m_loc(h,db)||gi>=n||gj>=n)guards=guards&&!std::memcmp(&bb[at],&after_b[at],sizeof(T));
    }
    std::vector<T> global(std::size_t(n)*(vectors?nev:0),T(0));
    if(vectors) {
        const auto out=download(h,alias?a.ptr:z.ptr,local_size(h,dz));
        for(int j=0;j<ddla_test::n_loc(h,dz);++j)for(int i=0;i<dz[DDLA_LLD_];++i) {
            const int gi=indx_l2g_r(dz,h,i),gj=indx_l2g_c(dz,h,j);const auto at=i+std::size_t(j)*dz[DDLA_LLD_];
            if(i<ddla_test::m_loc(h,dz)&&gi<n&&gj<nev)global[gi+std::size_t(gj)*n]=out[at];
            else if(!alias)guards=guards&&out[at]==sentinel;
        }
        MPI_Allreduce(MPI_IN_PLACE,global.data(),int(global.size()),detail::mpi_datatype<T>(),MPI_SUM,ddlaGetCommunicator(h));
    }
    require_close(h,"generalized parent/LLD/eigenvalue guards",guards?0:1,0);
    generalized_test::verify<T>(h,n,vectors?nev:0,w.data()+2,global,repeated);
}
template<class T>void failures(const DdlaHandle_t& h) {
    using R=typename real_type<T>::type;
    constexpr int n=7,nb=4;
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,n,n,nb,nb,0,0));
    auto aa=make_local<T>(h,d,[](int i,int j){return generalized_test::a<T>(i,j,n);});
    auto bb=make_local<T>(h,d,[](int i,int j){return T(i==j?(i==2?-1:1):0);});
    DeviceBuffer<T> a(h,std::max(std::size_t(1),aa.size())),b(h,std::max(std::size_t(1),bb.size())),z(h,std::max(std::size_t(1),aa.size()));
    upload(h,a.ptr,aa);upload(h,b.ptr,bb);
    std::vector<R> w(n,R(-9876));int info=0;
    phegvd<T>(h,'V','U',n,n,a.ptr,d,b.ptr,d,w.data(),z.ptr,d,info);
    const auto after=download(h,a.ptr,aa.size());
    require_close(h,"distributed non-positive B info=n+i and A unchanged",info==n+3&&!std::memcmp(after.data(),aa.data(),aa.size()*sizeof(T))&&w==std::vector<R>(n,R(-9876))?0:1,0);
    for(int mode=0;mode<5;++mode) {
        upload(h,a.ptr,aa);upload(h,b.ptr,bb);
        bool rejected=false;int count=n;
        if(mode==0)count=-1;
        if(mode==1&&ddlaGetRank(h)==0)count=n+1;
        if(mode==2&&ddlaGetSize(h)>1)count=ddlaGetRank(h)==0?1:n;
        if(mode==2&&ddlaGetSize(h)==1)continue;
        try{phegvd<T>(h,'V','U',n,count,a.ptr,d,mode==3?a.ptr:b.ptr,d,w.data(),mode==4?b.ptr:z.ptr,d,info);}
        catch(const std::exception&){rejected=true;}
        require_close(h,"generalized count/alias errors rejected collectively",rejected?0:1,0);
        const auto aout=download(h,a.ptr,aa.size()),bout=download(h,b.ptr,bb.size());
        require_close(h,"rejected generalized inputs unchanged",!std::memcmp(aout.data(),aa.data(),aa.size()*sizeof(T))&&!std::memcmp(bout.data(),bb.data(),bb.size()*sizeof(T))?0:1,0);
    }
}
template<class T>void cases(const DdlaHandle_t& h) {
    for(int n:{0,1,3,17})for(char uplo:{'U','L'}) {
        solve<T>(h,n,0,uplo,false,false);
        if(n){solve<T>(h,n,1,uplo,false,false);solve<T>(h,n,n,uplo,false,false);}
    }
    for(char uplo:{'U','L'}) {
        solve<T>(h,17,5,uplo,false,true);solve<T>(h,17,5,uplo,true,false);
        solve<T>(h,17,17,uplo,false,false,'N');
    }
    failures<T>(h);
}
void body(const DdlaHandle_t& h,const Shape&) {
    cases<float>(h);cases<double>(h);cases<std::complex<float>>(h);cases<std::complex<double>>(h);
}
}
int main(int argc,char** argv){return run_grid_test(argc,argv,"test_phegvd",body);}
