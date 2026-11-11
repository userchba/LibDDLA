#include "pblas_structured_test_common.h"
#include "workspace_test_common.h"
#include <helpers.h>
#include "BLAS/hemmt_impl.h"
#include "PBLAS/pgemmt_impl.h"
#include "PBLAS/phemm_impl.h"
#include "PBLAS/pblas_workspace.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include "mixed_products_test_common.h"
using namespace ddla;
using structured_test::Matrix;
namespace {
template <typename T> T hermitian(int i,int j) {
    T x=0;for(int a=0;a<2;++a) {
        const T right=feature<T>(j,a,1);
        x+=feature<T>(i,a,1)*ddla::detail::host_conj(right);
    }
    return x;
}

template <DdlaBackend Backend,typename R,typename TA,typename TB>
void matrix_case(DdlaHandle_t h,bool local,bool structured,char side,char ua,char uc,char ta,char tb,
    int m,int n,int k,int nb,int mode,int& cases)
{
    using T=std::complex<R>;
    const bool left=side=='L';
    if(structured)k=left?m:n;
    const int ar=structured?k:ta=='N'?m:k,ac=structured?k:ta=='N'?k:m;
    const int br=structured?m:tb=='N'?k:n,bc=structured?n:tb=='N'?n:k;
    const T alpha=mode>=2?T(0):T(.75,.125),beta=mode==1||mode==3?T(0):mode==4?T(1):T(.25,-.125);
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    Matrix<TA> a(h,ar,ac,nb,pr-1,pc-1,true);
    Matrix<TB> b(h,br,bc,nb,0,pc-1,true);
    Matrix<T> c(h,m,n,nb,pr/2,pc/2,true);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    a.fill(ar,ac,[&](int i,int j){
        if(!structured)return general<TA>(i,j,1);
        if(ua=='U'?i>j:i<j)return scalar<TA>(nan,nan);
        TA x=hermitian<TA>(i,j);
        if constexpr(!std::is_arithmetic_v<TA>)if(i==j)x.imag(R(nan));
        return x;
    });
    b.fill(br,bc,[&](int i,int j){return general<TB>(i,j,3);});
    c.fill(m,n,[&](int i,int j){return beta==T(0)?T(nan,nan):general<T>(i,j,5);});
    a.upload();b.upload();c.upload();
    const TA* ap=alpha==T(0)?nullptr:a.data.data();
    const TB* bp=alpha==T(0)?nullptr:b.data.data();
    auto public_call=[&] {
        if(structured) {
            if(local) {
                if(uc=='F')hemm<Backend,R>(h,side,ua,m,n,alpha,ap,a.d[DDLA_LLD_],bp,b.d[DDLA_LLD_],beta,c.data.data(),c.d[DDLA_LLD_]);
                else hemmt<Backend,R>(h,side,ua,uc,n,alpha,ap,a.d[DDLA_LLD_],bp,b.d[DDLA_LLD_],beta,c.data.data(),c.d[DDLA_LLD_]);
            } else {
                if(uc=='F')phemm<Backend,R>(h,side,ua,m,n,alpha,ap,a.d,bp,b.d,beta,c.data.data(),c.d);
                else phemmt<Backend,R>(h,side,ua,uc,n,alpha,ap,a.d,bp,b.d,beta,c.data.data(),c.d);
            }
        } else if(local)gemmt<Backend,R>(h,uc,ta,tb,n,k,alpha,ap,a.d[DDLA_LLD_],bp,b.d[DDLA_LLD_],beta,c.data.data(),c.d[DDLA_LLD_]);
        else pgemmt<Backend,R>(h,uc,ta,tb,n,k,alpha,ap,a.d,bp,b.d,beta,c.data.data(),c.d);
    };
    Wide dot[2][2]={};
    if(alpha!=T(0)) {
        if(structured && !left)contraction<TB,TA>(false,true,'N','N',k,3,1,dot);
        else contraction<TA,TB>(structured,false,ta,tb,k,1,3,dot);
    }
    auto check=[&] {
        const auto actual=c.download();bool ok=true;double worst=0;
        for(std::size_t pos=0;pos<actual.size();++pos) {
            const int il=pos%c.d[DDLA_LLD_],jl=pos/c.d[DDLA_LLD_];
            const int i=il<c.nr?c.gi(il):m,j=jl<c.nc?c.gj(jl):n;
            const bool selected=i<m&&j<n&&(uc=='F'||(uc=='U'?i<=j:i>=j));
            if(!selected) {if(std::memcmp(&actual[pos],&c.host[pos],sizeof(T)))ok=false;continue;}
            Wide ref=0;
            if(alpha!=T(0)) ref=wide(alpha)*(structured&&!left
                ? product<TB,TA>(false,true,'N','N',i,j,3,1,dot)
                : product<TA,TB>(structured,false,ta,tb,i,j,1,3,dot));
            if(beta!=T(0))ref+=wide(beta)*wide(general<T>(i,j,5));
            const double error=double(std::abs(wide(actual[pos])-ref)/(1+std::abs(ref)));
            if(!std::isfinite(error))ok=false;else worst=std::max(worst,error);
        }
        const double tol=std::is_same_v<R,float>?3e-5:3e-12;
        require(h,ok&&worst<tol,"mixed matrix result/triangle/padding failure local="+std::to_string(local)+" structured="+std::to_string(structured)+" side="+side+" output="+uc+" n="+std::to_string(n)+" error="+std::to_string(worst));
    };
    public_call();check(); // Warm vendor and communication caches before tracing reuse.
    std::size_t bytes=0,host_bytes=0;
    {
        workspace_test::Scope scope(true);
        if(structured) {
            if(local)blas_impl::hemmt_workspace_size<T,TA,TB>(h,k,alpha!=T(0),&bytes,left?n:m,side,uc);
            else phemm_impl::phemm_workspace_size<T,TA,TB>(h,side,m,n,alpha==T(0),a.d,b.d,c.d,true,&bytes,&host_bytes,uc);
        } else {
            if(local)blas_impl::gemmt_workspace_size<T,TA,TB>(h,ta,tb,n,alpha==T(0)?0:k,&bytes);
            else pgemmt_impl::workspace_size<T,TA,TB>(h,uc,ta,tb,alpha!=T(0)&&k>0,n,k,a.d,b.d,c.d,&bytes,&host_bytes);
        }
        query_pure(h,scope.finish());
    }
    vector<char> work(h,bytes+512);std::vector<char> host(host_bytes+512,char(0x5a));
    DDLA_CHECK(ddlaMemset(h,work.data(),0x5a,bytes+512));
    for(int repeat=0;repeat<2;++repeat) {
        c.upload();
        workspace_test::Scope scope;
        if(structured) {
            if(local)blas_impl::hemmt_run<Backend,T>(h,side,ua,uc,k,alpha,ap,a.d[DDLA_LLD_],bp,b.d[DDLA_LLD_],beta,c.data.data(),c.d[DDLA_LLD_],work.data()+256,bytes,true,left?n:m);
            else phemm_impl::phemm_run<Backend,T>(h,side,ua,m,n,alpha,ap,a.d,bp,b.d,beta,c.data.data(),c.d,true,work.data()+256,bytes,host.data()+256,host_bytes,uc);
        } else {
            if(local)blas_impl::gemmt_run<Backend,T>(h,uc,ta,tb,n,k,alpha,ap,a.d[DDLA_LLD_],bp,b.d[DDLA_LLD_],beta,c.data.data(),c.d[DDLA_LLD_],work.data()+256,bytes);
            else pgemmt_impl::run<Backend,T>(h,uc,ta,tb,alpha!=T(0)&&k>0,n,k,alpha,ap,a.d,bp,b.d,beta,c.data.data(),c.d,work.data()+256,bytes,host.data()+256,host_bytes);
        }
        run_reused(h,scope.finish());check();
    }
    std::array<char,256> first,last,guard;guard.fill(char(0x5a));
    DDLA_CHECK(ddlaMemcpy(h,first.data(),work.data(),256));DDLA_CHECK(ddlaMemcpy(h,last.data(),work.data()+256+bytes,256));DDLA_CHECK(ddlaSynchronize(h));
    require(h,first==guard&&last==guard&&std::all_of(host.begin(),host.begin()+256,[](char x){return x==char(0x5a);})&&std::all_of(host.end()-256,host.end(),[](char x){return x==char(0x5a);}),"mixed workspace guard changed");
    const auto aa=a.download(),bb=b.download();
    require(h,!std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(TA))&&!std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(TB)),"mixed input was modified");
    ++cases;
}

template <DdlaBackend Backend,typename R,typename TA,typename TB>
void types(DdlaHandle_t h,bool local,bool large,int& cases) {
    for(char u:{'U','L'})for(char ta:{'N','T','C'})for(char tb:{'N','T','C'})
        matrix_case<Backend,R,TA,TB>(h,local,false,'L','U',u,ta,tb,17,17,13,4,0,cases);
    for(char side:{'L','R'})for(char ua:{'U','L'})for(char uc:{'U','L','F'}) {
        matrix_case<Backend,R,TA,TB>(h,local,true,side,ua,uc,'N','N',17,uc=='F'?11:17,0,4,0,cases);
        matrix_case<Backend,R,TA,TB>(h,local,true,side,ua,uc,'N','N',17,uc=='F'?11:17,0,4,1,cases);
    }
    for(int mode=1;mode<5;++mode) {
        matrix_case<Backend,R,TA,TB>(h,local,false,'L','U','L','C','T',3,3,mode==2?0:7,4,mode,cases);
        matrix_case<Backend,R,TA,TB>(h,local,true,'R','L','U','N','N',3,3,0,4,mode,cases);
    }
    matrix_case<Backend,R,TA,TB>(h,local,false,'L','U','U','T','C',0,0,0,4,0,cases);
    if(large)for(int n:{129,513,1031}) {
        matrix_case<Backend,R,TA,TB>(h,local,false,'L','U','L','C','T',n,n,n+6,129,1,cases);
        for(char side:{'L','R'})for(char uc:{'U','F'})matrix_case<Backend,R,TA,TB>(h,local,true,side,'L',uc,'N','N',n,n,0,129,0,cases);
    }
    if(local) {
        for(char ta:{'N','T','C'})for(char tb:{'N','T','C'})batch_case<Backend,R,TA,TB>(h,19,23,17,35,ta,tb,0,cases);
        for(int mode=1;mode<5;++mode)batch_case<Backend,R,TA,TB>(h,7,5,mode==2?0:13,3,'C','C',mode,cases);
        // Cover a long fused reduction and the packed 512-column tail.
        batch_case<Backend,R,TA,TB>(h,33,29,257,3,'C','N',1,cases);
        batch_case<Backend,R,TA,TB>(h,7,513,17,2,'C','C',0,cases);
        // These rectangular batches take the wide register tile on GPU.
        batch_case<Backend,R,TA,TB>(h,129,97,521,40,'C','N',0,cases);
        batch_case<Backend,R,TA,TB>(h,97,129,1031,40,'N','C',1,cases);
        batch_case<Backend,R,TA,TB>(h,3,2,4,0,'N','N',0,cases);
        if(large){batch_case<Backend,R,TA,TB>(h,257,265,269,3,'C','N',1,cases);batch_case<Backend,R,TA,TB>(h,1031,65,513,3,'N','C',0,cases);batch_case<Backend,R,TA,TB>(h,1031,1031,65,1,'C','T',1,cases);}
    }
}
template <DdlaBackend Backend> void invalid_cases(DdlaHandle_t h,bool local) {
    using T=std::complex<double>;
    Matrix<double> a(h,7,7,4);Matrix<T> b(h,7,7,4),c(h,7,7,4);
    auto reject=[&](auto call) {bool caught=false;try{call();}catch(const std::exception&){caught=true;}require(h,caught,"mixed invalid argument was accepted");};
    const double* no_real=nullptr;const T* no_complex=nullptr;
    if(local) {
        gemmt<Backend,double>(h,'U','N','N',7,0,T(0),no_real,7,no_complex,7,T(1),nullptr,7);
        hemm<Backend,double>(h,'L','U',7,7,T(0),no_real,7,no_complex,7,T(1),nullptr,7);
        hemmt<Backend,double>(h,'L','U','L',7,T(0),no_real,7,no_complex,7,T(1),nullptr,7);
        gemmBatched<Backend,double>(h,'N','N',7,7,7,T(0),static_cast<const double* const*>(nullptr),7,static_cast<const T* const*>(nullptr),7,T(1),nullptr,7,3);
        reject([&]{gemmt<Backend,double>(h,'U','N','N',7,7,T(1),reinterpret_cast<const double*>(c.data.data()),7,b.data.data(),7,T(0),c.data.data(),7);});
        reject([&]{hemm<Backend,double>(h,'L','U',7,-1,T(1),a.data.data(),7,b.data.data(),7,T(0),c.data.data(),7);});
        reject([&]{hemmt<Backend,double>(h,'L','U','F',7,T(1),a.data.data(),7,b.data.data(),7,T(0),c.data.data(),7);});
        reject([&]{gemmBatched<Backend,double>(h,'N','N',7,7,7,T(0),static_cast<const double* const*>(nullptr),7,static_cast<const T* const*>(nullptr),7,T(1),nullptr,7,-1);});
    } else {
        pgemmt<Backend,double>(h,'U','N','N',7,0,T(0),no_real,a.d,no_complex,b.d,T(1),nullptr,c.d);
        phemm<Backend,double>(h,'R','L',7,7,T(0),no_real,a.d,no_complex,b.d,T(1),nullptr,c.d);
        phemmt<Backend,double>(h,'L','U','L',7,T(0),no_real,a.d,no_complex,b.d,T(1),nullptr,c.d);
        int rank;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
        reject([&]{pgemmt<Backend,double>(h,rank==0?'?':'U','N','N',7,7,T(1),a.data.data(),a.d,b.data.data(),b.d,T(0),c.data.data(),c.d);});
        reject([&]{phemm<Backend,double>(h,'L','U',7,7,T(1),reinterpret_cast<const double*>(c.data.data()),c.d,b.data.data(),b.d,T(0),c.data.data(),c.d);});
        reject([&]{phemmt<Backend,double>(h,'L','U','F',7,T(1),a.data.data(),a.d,b.data.data(),b.d,T(0),c.data.data(),c.d);});
    }
}
// For a square product with enough descriptor blocks, neither prepared
// operand panel may exceed this rank's full-matrix ownership. This catches
// whole-operand replication on 1 x P grids at the GPU grouping threshold.
void panel_storage(DdlaHandle_t h) {
    int pr,pc,r,c;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
    if(ddlaGetBackend(h)!=DdlaBackend::GPU)return;
    using T=std::complex<double>;
    {
        std::size_t bytes=0;
        workspace_test::Scope query(true);
        blas_impl::hemmt_workspace_size<T,double,T>(h,8192,true,&bytes,8192,'R','F');
        const auto stats=query.finish();query_pure(h,stats);
        require(h,bytes<=256u*1024*1024,"right mixed HEMM scratch contains a full square complex plane");
    }
    if(pr*pc==1)return;
    // Full-output reflected transfers stay one block wide at any order.
    // A square enclosing transfer query would grow almost fourfold here.
    for(char side : {'L','R'}) for(bool cr : {false,true}) {
        std::size_t sizes[2]={0,0};
        for(int i=0;i<2;++i) {
            const int order=4096*(i+1);
            int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,order,order,128,128,0,0));
            std::size_t device=0,host=0;
            workspace_test::Scope query(true);
            if(cr)phemm_impl::phemm_workspace_size<T,T,double>(h,side,order,order,false,desc,desc,desc,true,&device,&host);
            else phemm_impl::phemm_workspace_size<T,double,T>(h,side,order,order,false,desc,desc,desc,true,&device,&host);
            const auto stats=query.finish();query_pure(h,stats);
            sizes[i]=device+host;
        }
        require(h,sizes[1]<=3*sizes[0],"mixed PHEMM scratch grows quadratically with matrix order");
    }
    constexpr int n=1024,nb=128;
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,n,n,nb,nb,0,0));
    const std::size_t owned=std::size_t(num_loc(n,nb,r,0,pr))*num_loc(n,nb,c,0,pc);
    std::size_t count[4],add[2][2];
    workspace_test::Scope scope(true);
    pblas::product_workspace_size<std::complex<double>,double,std::complex<double>>(
        h,'N','N',true,false,n,n,d,d,d,count,add);
    const auto stats=scope.finish();query_pure(h,stats);
    require(h,count[0]<=owned && count[1]<=owned,"mixed operand panels exceed local matrix ownership");
    std::size_t maximum=0,local=std::max(count[0],count[1]);
    MPI_Allreduce(&local,&maximum,1,MPI_UNSIGNED_LONG,MPI_MAX,ddlaGetCommunicator(h));
    int rank;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
    if(!rank)std::cout<<"MIXED_PANEL_STORAGE_PASS grid="<<pr<<'x'<<pc
        <<" n="<<n<<" max_panel_elements="<<maximum<<std::endl;
}
template <DdlaBackend Backend> void run(int pr,int pc,char order,bool large) {
    DdlaHandle_t h=nullptr;int cases=0;
    DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,order));
    panel_storage(h);
    invalid_cases<Backend>(h,false);
    types<Backend,float,float,std::complex<float>>(h,false,large,cases);types<Backend,float,std::complex<float>,float>(h,false,large,cases);
    types<Backend,double,double,std::complex<double>>(h,false,large,cases);types<Backend,double,std::complex<double>,double>(h,false,large,cases);
    DDLA_CHECK(ddlaDestroy(h));DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF,1,1));
    invalid_cases<Backend>(h,true);
    types<Backend,float,float,std::complex<float>>(h,true,large,cases);types<Backend,float,std::complex<float>,float>(h,true,large,cases);
    types<Backend,double,double,std::complex<double>>(h,true,large,cases);types<Backend,double,std::complex<double>,double>(h,true,large,cases);
    DDLA_CHECK(ddlaDestroy(h));int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(!rank)std::cout<<"MIXED_PRODUCTS_PASS backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc<<" cases="<<cases<<" large="<<large<<std::endl;
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
        const std::string mode=argc>1?argv[1]:"all";const int pr=argc>2?std::stoi(argv[2]):1,pc=argc>3?std::stoi(argv[3]):size;
        const char order=argc>4?argv[4][0]:'R';const bool large=std::getenv("DDLA_MIXED_LARGE_TEST")!=nullptr;
        if(pr*pc!=size||pr<1||pc<1)throw std::invalid_argument("invalid test grid");
        if(mode!="gpu")run<DdlaBackend::CPU>(pr,pc,order,large);
#if DDLA_HAS_GPU
        if(mode!="cpu")run<DdlaBackend::GPU>(pr,pc,order,large);
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"MIXED_PRODUCTS_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
