#include "api_grid_test_common.h"
#include "mpi_datatype.h"
#include "Eigensolver/one_stage/one_stage_workspace.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include <cstring>
#include <limits>
#include <type_traits>

using namespace api_grid_test;
namespace {
template<class T> T conjugate(T x) {
    if constexpr (std::is_arithmetic_v<T>) return x;
    else return std::conj(x);
}
template<class T> T phase(int i) {
    if constexpr (std::is_arithmetic_v<T>) return T(1);
    else return T(std::cos(.19*i),std::sin(.19*i));
}
// A = D Q diag(lambda) Q^T D^H, Q = I - 2 v v^T. This supplies a
// dense independent oracle and an exactly known ordered spectrum.
template<class T> T entry(int i,int j,int n,bool repeated) {
    double norm=0,weighted=0;
    for(int k=0;k<n;++k) {
        const double x=k+1,lambda=repeated?k/3+1:k+1;
        norm+=x*x;weighted+=x*x*lambda;
    }
    const double li=repeated?i/3+1:i+1,lj=repeated?j/3+1:j+1;
    const double a=(i==j?li:0)+(i+1.)*(j+1.)/norm*(4*weighted/norm-2*li-2*lj);
    return T(a)*phase<T>(i)*conjugate(phase<T>(j));
}
template<class T>
void solve(int method,const DdlaHandle_t& h,char job,char uplo,int n,int nev,
           T* a,const int* da,typename real_type<T>::type* w,T* z,const int* dz,int& info) {
    if(method>=3) {
        constexpr auto B=ddla_test::execution_backend;
        auto topology=spectral_impl::phb2st_topology_prepare(h);
        setenv("DDLA_EIG_CPU_ACCUM",method==4?"1":"0",1);
        const auto plan=spectral_impl::pheevd_check<T>(h,B,job,uplo,n,a,da,w,z,dz,*topology,
            method==3?'H':'G',nullptr,DdlaBackend::CPU,nev);
        std::size_t backend_bytes=0,host_bytes=0;
        spectral_impl::pheevd_workspace_size<T>(h,plan,a,&backend_bytes,&host_bytes);
        ddla::vector<char> backend(h,backend_bytes);
        std::vector<char> host(host_bytes);
        spectral_impl::PheevdResources<T> resources(h,plan);
        spectral_impl::pheevd_run<B,T>(h,plan,resources,a,w,z,info,backend.data(),backend_bytes,
                                     host.data(),host_bytes);
        resources.release();topology->release();unsetenv("DDLA_EIG_CPU_ACCUM");
    }
    else if(method==1) pheevd_one_stage<T>(h,job,uplo,n,nev,a,da,w,z,dz,info);
    else if(method==2) pheevd_two_stage<T>(h,job,uplo,n,nev,a,da,w,z,dz,info);
    else pheevd<T>(h,job,uplo,n,nev,a,da,w,z,dz,info);
}

template<class T>
void run(const DdlaHandle_t& h,int method,int n,int nev,char uplo,bool parent,bool alias,bool repeated=false) {
    using R=typename real_type<T>::type;
    constexpr int nb=4;
    int pr0,pc0;ddlaGetGridDims(h,pr0,pc0);
    if(ddlaGetRank(h)==0)std::cout<<"nev case method="<<method<<" n="<<n<<" nev="<<nev
        <<" uplo="<<uplo<<" parent="<<parent<<" alias="<<alias<<" bytes="<<sizeof(T)<<std::endl;
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];
    const int outcols=parent?n+3:nev;
    DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,nb,nb,0,0));
    DDLA_CHECK(ddlaDescInit(dz,h,n+(parent?2:0),outcols,nb,nb,0,parent?pc0-1:0));
    da[DDLA_LLD_]+=3;dz[DDLA_LLD_]+=2;
    if(alias)std::copy_n(da,DDLA_DLEN_,dz);
    DeviceBuffer<T> a(h,std::max(std::size_t(1),local_size(h,da)));
    DeviceBuffer<T> z(h,std::max(std::size_t(1),local_size(h,dz)));
    const T sentinel=T(9876);
    auto input=make_local<T>(h,da,[&](int i,int j) {
        if(i>=n||j>=n)return sentinel;
        if(uplo=='U'?i>j:i<j)return T(std::numeric_limits<R>::quiet_NaN());
        return entry<T>(i,j,n,repeated);
    });
    std::vector<T> before(local_size(h,dz),sentinel);
    upload(h,a.ptr,input);upload(h,z.ptr,before);
    std::vector<R> w(std::max(n,1)+4,R(-9876));int info=-1;
    T* output=nev?(alias?a.ptr:z.ptr):nullptr;
    if(!alias && local_size(h,dz)==0)output=nullptr;
    solve<T>(method,h,'V',uplo,n,nev,a.ptr,da,w.data()+2,output,nev?dz:nullptr,info);
    check_ddla_sync(h);require_close(h,"nev info",info,0);
    const double eps=std::numeric_limits<R>::epsilon();
    double ev_error=0;
    for(int j=0;j<n;++j) {
        const double expected=repeated?j/3+1:j+1;
        if(!std::isfinite(w[j+2]))ev_error=1e30;
        ev_error=std::max(ev_error,std::abs(double(w[j+2])-expected)/(eps*std::max(1,n)));
    }
    require_close(h,"all n eigenvalues match known spectrum",ev_error,100);
    require_close(h,"eigenvalue guards",w[0]==R(-9876)&&w[1]==R(-9876)&&w[n+2]==R(-9876)&&w[n+3]==R(-9876)?0:1,0);
    if(!nev)return;
    const auto result=download<T>(h,output,local_size(h,dz));
    std::vector<T> global(std::size_t(n)*nev,T(0));
    double padding_error=0;
    int pr,pc,mr,mc;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,mr,mc);
    const int nr=ddla_test::m_loc(h,dz),nc=ddla_test::n_loc(h,dz),ld=dz[DDLA_LLD_];
    for(int j=0;j<nc;++j)for(int i=0;i<ld;++i) {
        const int gi=indxl2g(i,nb,mr,0,pr),gj=indxl2g(j,nb,mc,dz[DDLA_CSRC_],pc);
        const auto at=i+std::size_t(j)*ld;
        if(i<nr&&gi<n&&gj<nev)global[gi+std::size_t(gj)*n]=result[at];
        else if(!alias || i>=nr || gi>=n || gj>=n) {
            const T expected=alias?input[at]:before[at];
            if(std::memcmp(&result[at],&expected,sizeof(T)))padding_error=1;
        }
    }
    require_close(h,"nev output and LLD padding preserved",padding_error,0);
    // Test-only global copies are deliberately small (n <= 65).
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,global.data(),int(global.size()),
        ddla::detail::mpi_datatype<T>(),MPI_SUM,ddlaGetCommunicator(h)));
    double residual=0,orthogonal=0;
    for(int j=0;j<nev;++j)for(int i=0;i<n;++i) {
        T r=-T(w[j+2])*global[i+std::size_t(j)*n];
        for(int k=0;k<n;++k)r+=entry<T>(i,k,n,repeated)*global[k+std::size_t(j)*n];
        if(!std::isfinite(std::abs(r)))residual=1e30;
        residual=std::max(residual,double(std::abs(r))/(eps*std::max(1,n)*std::max(1,n)));
    }
    for(int j=0;j<nev;++j)for(int k=0;k<nev;++k) {
        T dot=0;for(int i=0;i<n;++i)dot+=conjugate(global[i+std::size_t(j)*n])*global[i+std::size_t(k)*n];
        const double error=std::abs(dot-T(j==k));
        if(!std::isfinite(error))orthogonal=1e30;
        orthogonal=std::max(orthogonal,error/(eps*std::max(1,n)));
    }
    require_close(h,"partial eigenvectors residual",residual,100);
    require_close(h,"partial eigenvectors orthogonality",orthogonal,100);
}

template<class T> void cases(const DdlaHandle_t& h) {
    for(int method:{0,1,2})for(char uplo:{'U','L'}) {
        for(int nev:{0,1,5,17})run<T>(h,method,17,nev,uplo,false,false);
        run<T>(h,method,17,5,uplo,true,false);
        run<T>(h,method,17,5,uplo,true,true);
        run<T>(h,method,17,5,uplo,false,false,true);
        run<T>(h,method,2,1,uplo,false,false);
        run<T>(h,method,0,0,uplo,false,false);
    }
    setenv("DDLA_EIG_DC_LEAF","8",1);
    for(int method:{1,2})run<T>(h,method,65,5,'U',false,false);
    unsetenv("DDLA_EIG_DC_LEAF");
    for(char uplo:{'U','L'}) {
        run<T>(h,3,65,5,uplo,false,false);
        run<T>(h,3,17,1,uplo,true,true);
        if constexpr(ddla_test::execution_backend==DdlaBackend::CPU)
            run<T>(h,4,17,5,uplo,false,false);
    }
}
void errors(const DdlaHandle_t& h) {
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,17,17,4,4,0,0));
    DDLA_CHECK(ddlaDescInit(dz,h,17,4,4,4,0,0));
    DeviceBuffer<double> a(h,std::max(std::size_t(1),local_size(h,da)));
    DeviceBuffer<double> z(h,std::max(std::size_t(1),local_size(h,dz)));
    double w[17];int info=0;
    for(int method:{0,1,2})for(int nev:{-1,18,5}) {
        bool caught=false;
        try{solve<double>(method,h,'V','U',17,nev,a.ptr,da,w,z.ptr,dz,info);}
        catch(const std::invalid_argument&){caught=true;}
        require_close(h,"invalid nev/output rejected",caught?0:1,0);
    }
    if(ddlaGetSize(h)>1)for(int method:{0,1,2}) {
        bool caught=false;
        try{solve<double>(method,h,'V','U',17,ddlaGetRank(h)?2:1,a.ptr,da,w,z.ptr,dz,info);}
        catch(const std::exception& error) {
            caught=std::string(error.what()).find("differ")!=std::string::npos;
        }
        require_close(h,"inconsistent nev rejected collectively",caught?0:1,0);
    }
}
void body(const DdlaHandle_t& h,const Shape&) {
    cases<float>(h);cases<double>(h);cases<std::complex<float>>(h);cases<std::complex<double>>(h);
    errors(h);
}
}
int main(int argc,char** argv){return run_grid_test(argc,argv,"test_pheevd_nev",body);}
