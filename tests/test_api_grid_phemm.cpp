#include "backend_api.h" // Internal-only wrong-backend implementation checks.
#include "api_grid_test_common.h"
#include "PBLAS/phemm_impl.h"
#include "comm_traits.h"
#include <ddla/ddla_vector.h>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace ddla;
namespace {

template <typename T> T scalar(double r,double i=0) {
    if constexpr(std::is_arithmetic_v<T>) return T(r);
    else return T(r,i);
}
template <typename T> T value(int i,int j,int salt) {
    return scalar<T>(((i*17+j*11+salt*7)%31-15)/16.,((i*7+j*19+salt*13)%29-14)/16.);
}
template <typename T> std::complex<double> wide(T x) {
    if constexpr(std::is_arithmetic_v<T>) return {double(x),0};
    else return {double(x.real()),double(x.imag())};
}
template <typename T> std::complex<double> herm(char u,int i,int j) {
    if(i==j) return {wide(value<T>(i,j,1)).real(),0};
    return (u=='U' ? i<j : i>j) ? wide(value<T>(i,j,1)) : std::conj(wide(value<T>(j,i,1)));
}

template <typename T> struct Matrix {
    DdlaHandle_t h; int d[DDLA_DLEN_],nr,nc,pr,pc,r,c;
    std::vector<T> host; ddla::vector<T> data;
    Matrix(DdlaHandle_t handle,int m,int n,int mb,int nb,int rs,int cs) : h(handle),data(h) {
        ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
        nr=num_loc(m+3,mb,r,rs,pr);nc=num_loc(n+5,nb,c,cs,pc);
        DDLA_CHECK(ddlaDescInit(d,h,m+3,n+5,mb,nb,rs,cs,nr+3));
        host.assign(std::max(std::size_t(1),std::size_t(d[DDLA_LLD_])*nc),scalar<T>(-987.25,654.5));
        data.resize(host.size());
    }
    int gi(int i) const { return indxl2g(i,d[DDLA_MB_],r,d[DDLA_RSRC_],pr); }
    int gj(int j) const { return indxl2g(j,d[DDLA_NB_],c,d[DDLA_CSRC_],pc); }
    void upload() { DDLA_CHECK(ddlaMemcpy(h,data.data(),host.data(),host.size()*sizeof(T))); }
    std::vector<T> download() {
        std::vector<T> out(host.size());DDLA_CHECK(ddlaMemcpy(h,out.data(),data.data(),out.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));return out;
    }
    void stored_only(int order,char u) {
        for(int j=0;j<nc;++j) for(int i=0;i<nr;++i) {
            const int x=gi(i),y=gj(j);if(x>=order || y>=order || (u=='U' ? x>y : x<y)) continue;
            const auto p=i+std::size_t(j)*d[DDLA_LLD_];
            if constexpr(!std::is_arithmetic_v<T>) {
                if(x==y) {
                    using R=typename ddla::real_type<T>::type;
                    DDLA_CHECK(ddlaMemcpy(h,reinterpret_cast<R*>(data.data()+p),
                        reinterpret_cast<const R*>(host.data()+p),sizeof(R)));
                    continue;
                }
            }
            DDLA_CHECK(ddlaMemcpy(h,data.data()+p,host.data()+p,sizeof(T)));
        }
    }
    void padding_only(int m,int n) {
        for(int j=0;j<nc;++j) for(int i=0;i<d[DDLA_LLD_];++i)
            if(i>=nr || gi(i)>=m || gj(j)>=n) {
                const auto p=i+std::size_t(j)*d[DDLA_LLD_];
                DDLA_CHECK(ddlaMemcpy(h,data.data()+p,host.data()+p,sizeof(T)));
            }
        if(nc==0) DDLA_CHECK(ddlaMemcpy(h,data.data(),host.data(),sizeof(T)));
    }
};

template <DdlaBackend Backend,typename T>
void check_case(DdlaHandle_t h,char side,char u,int m,int n,int nb,int mode,bool shifted,
                bool uninitialized,int& cases,double& worst)
{
    int pr,pc;ddlaGetGridDims(h,pr,pc);const int q=side=='L' ? m : n;
    Matrix<T> a(h,q,q,nb,nb,shifted ? pr-1 : 0,shifted ? pc-1 : 0);
    Matrix<T> b(h,m,n,side=='L' ? nb : 3,side=='L' ? 3 : nb,0,shifted ? pc-1 : 0);
    Matrix<T> c(h,m,n,side=='L' ? nb : 3,side=='L' ? 3 : nb,
                shifted ? (pr>2 ? 1 : 0) : 0,shifted ? (pc>2 ? 1 : 0) : 0);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const T alpha=(mode>=2 && mode<=4) ? T(0) : mode==5 ? T(1) : scalar<T>(-.75,.25);
    const T beta=(mode==1 || mode==3 || mode==7) ? T(0) : (mode==4 || mode==5) ? T(1) : scalar<T>(.5,-.25);
    for(int j=0;j<a.nc;++j) for(int i=0;i<a.nr;++i) {
        int x=a.gi(i),y=a.gj(j);
        if(x<q && y<q) {
            const bool stored=u=='U' ? x<=y : x>=y;
            a.host[i+std::size_t(j)*a.d[DDLA_LLD_]]=stored ? value<T>(x,y,1) : scalar<T>(nan,nan);
            if(x==y) a.host[i+std::size_t(j)*a.d[DDLA_LLD_]]=scalar<T>(wide(value<T>(x,y,1)).real(),nan);
        }
    }
    for(int j=0;j<b.nc;++j) for(int i=0;i<b.nr;++i)
        if(b.gi(i)<m && b.gj(j)<n) b.host[i+std::size_t(j)*b.d[DDLA_LLD_]]=value<T>(b.gi(i),b.gj(j),2);
    for(int j=0;j<c.nc;++j) for(int i=0;i<c.nr;++i)
        if(c.gi(i)<m && c.gj(j)<n) c.host[i+std::size_t(j)*c.d[DDLA_LLD_]]=beta==T(0) ? scalar<T>(nan,nan) : value<T>(c.gi(i),c.gj(j),3);
    if(uninitialized) {
        if(alpha!=T(0)) { a.stored_only(q,u);b.upload(); }
        if(beta==T(0)) c.padding_only(m,n); else c.upload();
    } else { a.upload();b.upload();c.upload(); }
    const bool empty=m==0 || n==0,noop=empty || (alpha==T(0) && beta==T(1));
    const T* ap=empty || alpha==T(0) ? nullptr : a.data.data();
    const T* bp=empty || alpha==T(0) ? nullptr : b.data.data();
    if(mode==7 && !noop) {
        // One scratch serves both runs.
        std::size_t backend_bytes=0,host_bytes=0;
        phemm_impl::phemm_workspace_size<T>(h,side,m,n,false,a.d,b.d,c.d,true,&backend_bytes,&host_bytes);
        vector<char> backend_work(h,backend_bytes);
        std::vector<char> host_work(host_bytes);
        for(int pass=0;pass<2;++pass)
            phemm_impl::phemm_run<Backend,T>(h,side,u,m,n,alpha,ap,a.d,bp,b.d,beta,c.data.data(),c.d,true,
                backend_work.data(),backend_bytes,host_work.data(),host_bytes);
    } else {
        phemm<T>(h,mode==5 ? char(std::tolower(side)) : side,mode==5 ? char(std::tolower(u)) : u,
            m,n,alpha,ap,a.d,bp,b.d,beta,noop ? nullptr : c.data.data(),c.d);
    }
    const auto out=c.download();int bad=0;double error=0;
    if(!uninitialized) {
        const auto aa=a.download(),bb=b.download();
        bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T)) || std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T));
    }
    for(std::size_t p=0;p<out.size();++p) {
        const int i=p%c.d[DDLA_LLD_],j=p/c.d[DDLA_LLD_];
        if(i<c.nr && j<c.nc && c.gi(i)<m && c.gj(j)<n) {
            const int x=c.gi(i),y=c.gj(j);std::complex<double> ref=0;double bound=0;
            if(alpha!=T(0)) for(int k=0;k<q;++k) {
                const auto av=side=='L' ? herm<T>(u,x,k) : herm<T>(u,k,y);
                const auto bv=wide(side=='L' ? value<T>(k,y,2) : value<T>(x,k,2));
                ref+=av*bv;bound+=std::abs(av)*std::abs(bv);
            }
            ref*=wide(alpha);bound*=std::abs(wide(alpha));
            if(beta!=T(0)) {const auto cv=wide(value<T>(x,y,3));ref+=wide(beta)*cv;bound+=std::abs(wide(beta)*cv);}
            const double e=std::abs(wide(out[p])-ref)/(1+bound);
            if(!std::isfinite(e)) bad=1;else error=std::max(error,e);
        } else if(std::memcmp(&out[p],&c.host[p],sizeof(T))) bad=1;
    }
    int any;double global;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&error,&global,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<typename ddla::real_type<T>::type,float>;
    if(any || global>(single ? 3e-6 : 5e-14)) {
        if(ddlaGetRank(h)==0) std::cerr<<"PHEMM_FAIL "<<side<<u<<" m="<<m<<" n="<<n<<" nb="<<nb<<" mode="<<mode<<" shifted="<<shifted<<" bytes="<<sizeof(T)<<" error="<<global<<" guard="<<any<<std::endl;
        throw std::runtime_error("phemm numerical/padding/input check failed");
    }
    worst=std::max(worst,global);++cases;
}

template <DdlaBackend Backend,typename T>
void scalar_cases(DdlaHandle_t h,bool quick,bool uninitialized,bool stress,int& cases,double& worst) {
    for(char s:{'L','R'}) for(char u:{'U','L'}) {
        check_case<Backend,T>(h,s,u,13,9,4,0,true,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,7,5,4,1,true,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,1,2,4,0,true,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,7,5,32,1,false,uninitialized,cases,worst);
        if(quick) continue;
        check_case<Backend,T>(h,s,u,13,9,4,0,false,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,8,12,4,5,true,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,7,5,1,7,true,uninitialized,cases,worst);
        for(int mode:{2,3,4}) check_case<Backend,T>(h,s,u,13,9,4,mode,true,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,0,9,4,0,true,uninitialized,cases,worst);
        check_case<Backend,T>(h,s,u,13,0,4,0,true,uninitialized,cases,worst);
        if(stress) check_case<Backend,T>(h,s,u,137,119,32,0,true,uninitialized,cases,worst);
    }
}

template <DdlaBackend Backend>
void invalid_cases(DdlaHandle_t h) {
    Matrix<double> a(h,8,8,4,4,0,0),b(h,8,8,4,4,0,0),c(h,8,8,4,4,0,0);
    auto reject=[&](auto fn) {int caught=0;try {fn();}catch(const std::exception&) {caught=1;}
        int all;MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all) throw std::runtime_error("phemm invalid argument not rejected on every rank");};
    auto call=[&](char side,char u,int m,int n,const double* ap,const double* bp,double* cp,double alpha=0.,double beta=1.) {
        phemm<double>(h,side,u,m,n,alpha,ap,a.d,bp,b.d,beta,cp,c.d);};
    reject([&]{call('?','U',2,2,nullptr,nullptr,nullptr);});
    reject([&]{call('L','?',2,2,nullptr,nullptr,nullptr);});
    reject([&]{call('L','U',-1,2,nullptr,nullptr,nullptr);});
    reject([&]{call('R','U',2,-1,nullptr,nullptr,nullptr);});
    reject([&]{call('L','U',a.d[DDLA_M_]+1,2,nullptr,nullptr,nullptr);});
    const int old=c.d[DDLA_N_];c.d[DDLA_N_]=1;
    reject([&]{call('R','U',2,2,nullptr,nullptr,nullptr);});c.d[DDLA_N_]=old;
    b.d[DDLA_NB_]=3;reject([&]{call('L','U',2,2,nullptr,nullptr,nullptr);});b.d[DDLA_NB_]=4;
    a.d[DDLA_NB_]=3;reject([&]{call('L','U',2,2,nullptr,nullptr,nullptr);});a.d[DDLA_NB_]=4;
    const bool root=ddlaGetRank(h)==0;
    reject([&]{call('L','U',8,8,root ? nullptr : a.data.data(),b.data.data(),c.data.data(),1.,0.);});
    reject([&]{call('L','U',8,8,a.data.data(),root ? nullptr : b.data.data(),c.data.data(),1.,0.);});
    reject([&]{call('L','U',8,8,a.data.data(),b.data.data(),root ? nullptr : c.data.data(),0.,0.);});
    reject([&]{call('L','U',8,8,a.data.data(),b.data.data(),root ? b.data.data() : c.data.data(),1.,0.);});
    reject([&]{call('L','U',8,8,a.data.data(),b.data.data(),root ? a.data.data() : c.data.data(),1.,0.);});
    int size; MPI_Comm_size(ddlaGetCommunicator(h),&size);
    if(size>1) {
        reject([&]{call(root ? 'R' : 'L','U',8,8,a.data.data(),b.data.data(),c.data.data(),1.,0.);});
        reject([&]{call('L','U',8,8,a.data.data(),b.data.data(),c.data.data(),root ? 0. : 1.,0.);});
    }
    reject([&]{phemm<double>(nullptr,'L','U',0,0,0.,nullptr,a.d,nullptr,b.d,1.,nullptr,c.d);});
#if DDLA_HAS_GPU
    constexpr auto Other=Backend==DdlaBackend::CPU ? DdlaBackend::GPU : DdlaBackend::CPU;
    reject([&]{phemm<Other,double>(h,'L','U',0,0,0.,nullptr,a.d,nullptr,b.d,1.,nullptr,c.d);});
#endif
}

#if defined(__unix__)
template <typename T>
void protected_triangle(DdlaHandle_t h,char side,char u) {
    const int q=17,m=side=='L' ? q : 9,n=side=='L' ? 9 : q;
    const long page=sysconf(_SC_PAGESIZE);const std::size_t bytes=std::size_t(2*q+1)*page;
    void* mem=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(mem==MAP_FAILED) throw std::runtime_error("mmap failed");
    const int lda=2*page/sizeof(T)-1;
    T* a=reinterpret_cast<T*>(static_cast<char*>(mem)+(u=='U' ? page-sizeof(T) : 0));
    for(int j=0;j<q;++j) {
        if(mprotect(static_cast<char*>(mem)+std::size_t(2*j)*page,page,PROT_READ|PROT_WRITE)) throw std::runtime_error("mprotect failed");
        for(int i=0;i<q;++i) if(u=='U' ? i<=j : i>=j)
            a[i+std::size_t(j)*lda]=i==j ? scalar<T>(herm<T>(u,i,j).real(),std::numeric_limits<double>::quiet_NaN()) : value<T>(i,j,1);
        if(mprotect(static_cast<char*>(mem)+std::size_t(2*j)*page,page,PROT_READ)) throw std::runtime_error("mprotect read-only failed");
    }
    int da[DDLA_DLEN_],db[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,q,q,4,4,0,0,lda));DDLA_CHECK(ddlaDescInit(db,h,m,n,4,4,0,0));
    std::vector<T> b(std::size_t(m)*n),c(b.size(),scalar<T>(std::numeric_limits<double>::quiet_NaN()));
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) b[i+std::size_t(j)*m]=value<T>(i,j,2);
    phemm<T>(h,side,u,m,n,T(1),a,da,b.data(),db,T(0),c.data(),db);
    double worst=0;
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
        std::complex<double> ref=0;double bound=0;
        for(int k=0;k<q;++k) {auto av=side=='L' ? herm<T>(u,i,k) : herm<T>(u,k,j);
            auto bv=wide(side=='L' ? value<T>(k,j,2) : value<T>(i,k,2));ref+=av*bv;bound+=std::abs(av)*std::abs(bv);}
        const double err=std::abs(wide(c[i+std::size_t(j)*m])-ref)/(1+bound);
        if(!std::isfinite(err)) throw std::runtime_error("protected triangle NaN");worst=std::max(worst,err);
    }
    munmap(mem,bytes);
    constexpr bool single=std::is_same_v<typename ddla::real_type<T>::type,float>;
    if(worst>(single ? 3e-6 : 5e-14)) throw std::runtime_error("protected triangle result failed");
}
#endif

// Broadcast must accept undefined receive storage, including the host tunnel.
template <DdlaBackend Backend,typename T>
void uninitialized_broadcast(DdlaHandle_t h) {
    for(auto scope:{CommScope::Grid,CommScope::Row,CommScope::Col}) {
        const auto comm=Communicator::mpi_comm(h,scope);
        int rank,size;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
        for(int root:{0,size-1}) {
            ddla::vector<T> buf(h,17);
            std::vector<T> expected(17),out(17);
            for(int i=0;i<17;++i) expected[i]=value<T>(i,root,7);
            if(rank==root) DDLA_CHECK(ddlaMemcpy(h,buf.data(),expected.data(),17*sizeof(T)));
            commBcast<Backend>(h,scope,buf.data(),17,root);
            DDLA_CHECK(ddlaMemcpy(h,out.data(),buf.data(),17*sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(h));
            int bad=std::memcmp(out.data(),expected.data(),17*sizeof(T))!=0,any;
            MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
            if(any) throw std::runtime_error("broadcast receive-buffer check failed");
        }
    }
}

template <DdlaBackend Backend>
void run(MPI_Comm comm,int pr,int pc,char major,bool quick,bool uninitialized,bool stress) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,comm,pr,pc,major));
    if(uninitialized) {
        uninitialized_broadcast<Backend,float>(h);
        uninitialized_broadcast<Backend,double>(h);
        uninitialized_broadcast<Backend,std::complex<float>>(h);
        uninitialized_broadcast<Backend,std::complex<double>>(h);
        if(ddlaGetRank(h)==0) std::cout<<"PHEMM_BCAST_RECEIVE_PASS backend="<<int(Backend)<<std::endl;
    }
    int cases=0;double worst=0;
    scalar_cases<Backend,float>(h,quick,uninitialized,stress,cases,worst);
    scalar_cases<Backend,double>(h,quick,uninitialized,stress,cases,worst);
    scalar_cases<Backend,std::complex<float>>(h,quick,uninitialized,stress,cases,worst);
    scalar_cases<Backend,std::complex<double>>(h,quick,uninitialized,stress,cases,worst);
    invalid_cases<Backend>(h);
#if defined(__unix__)
    if constexpr(Backend==DdlaBackend::CPU) if(pr*pc==1 && !uninitialized) {
        for(char side:{'L','R'}) for(char u:{'U','L'}) {
            protected_triangle<float>(h,side,u);protected_triangle<double>(h,side,u);
            protected_triangle<std::complex<float>>(h,side,u);protected_triangle<std::complex<double>>(h,side,u);
        }
        std::cout<<"PHEMM_PROTECTED_TRIANGLES_PASS cases=16\n";
    }
#endif
    if(ddlaGetRank(h)==0) std::cout<<"PHEMM_PASS backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc<<" major="<<major<<" cases="<<cases<<" worst="<<worst<<std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace

int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);int size,rank;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    bool quick=false,uninitialized=false,stress=false;int ac=1;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--quick") quick=true;else if(arg=="--uninitialized") uninitialized=true;
        else if(arg=="--stress") stress=true;else argv[ac++]=argv[i];
    }
    api_grid_test::TestOptions opts;std::string error;
    if(!api_grid_test::parse_options(ac,argv,size,opts,error)) {std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Comm reversed;MPI_Comm_split(MPI_COMM_WORLD,0,size-rank,&reversed);
    try {
        for(auto grid:opts.grids) for(char major:{'R','C'}) {
            run<DdlaBackend::CPU>(reversed,grid.first,grid.second,major,quick,uninitialized,stress);
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(reversed,grid.first,grid.second,major,quick,uninitialized,stress);
#endif
        }
    } catch(const std::exception& e) {std::cerr<<"PHEMM_TEST_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Comm_free(&reversed);MPI_Finalize();return 0;
}
