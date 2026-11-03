#include "pgemm_test_common.h"
#include "ddla_stream.h"

// A scalar oracle is independent of both real and complex vendor GEMM.
template <typename T> std::complex<long double> extended(T z) {
    if constexpr (std::is_floating_point_v<T>) return {z,0};
    else return {z.real(),z.imag()};
}
template <typename T> auto operand(char op,int i,int j,int salt) {
    auto z=extended(op=='N'?value<T>(i,j,salt):value<T>(j,i,salt));
    return op=='C'?std::conj(z):z;
}

bool owns(DdlaHandle_t h,int rows,int cols,const int* d) {
    int pr,pc,r,c;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
    return num_loc(rows,d[DDLA_MB_],r,d[DDLA_RSRC_],pr)>0 &&
           num_loc(cols,d[DDLA_NB_],c,d[DDLA_CSRC_],pc)>0;
}

template <DdlaBackend Backend,typename R,bool CR>
void one(DdlaHandle_t h,int m,int n,int k,char ta,char tb,bool shifted,int mode,int& cases) {
    using Z=std::complex<R>;
    using AType=std::conditional_t<CR,Z,R>;
    using BType=std::conditional_t<CR,R,Z>;
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    const int mb=4,nb=5,kb=3;
    const int ms=shifted?(ta=='N'?pr-1:std::min(pr,pc)-1):0;
    const int ns=shifted?(tb=='N'?pc-1:std::min(pr,pc)-1):0;
    const int kr=shifted?pr-1:0,kc=shifted?pc-1:0;
    const bool no_product=mode>=2 || k==0, zero_beta=mode==1 || mode==3;
    const Z alpha=mode>=2?Z(0):Z(R(.7),R(-.2));
    const Z beta=zero_beta?Z(0):Z(R(-.3),R(.15));
    Matrix<AType> a(h,ta=='N'?m:k,ta=='N'?k:m,ta=='N'?mb:kb,ta=='N'?kb:mb,
                    ta=='N'?ms:kr,ta=='N'?kc:ms,1,no_product);
    Matrix<BType> b(h,tb=='N'?k:n,tb=='N'?n:k,tb=='N'?kb:nb,tb=='N'?nb:kb,
                    tb=='N'?kr:ns,tb=='N'?ns:kc,2,no_product);
    Matrix<Z> c(h,m,n,mb,nb,ms,ns,3,zero_beta);
    const AType* ap=!no_product && owns(h,ta=='N'?m:k,ta=='N'?k:m,a.d)?a.data:nullptr;
    const BType* bp=!no_product && owns(h,tb=='N'?k:n,tb=='N'?n:k,b.d)?b.data:nullptr;
    for(auto alg:{Algorithm::AB,Algorithm::AC,Algorithm::BC,Algorithm::Auto}) {
        c.reset();
        Z* cp=owns(h,m,n,c.d)?c.data:nullptr;
        const int threads=h->host_threads;
        if(alg==Algorithm::Auto) {
            if constexpr (Backend==default_backend_v)
                pgemm<>(h,ta,tb,m,n,k,alpha,ap,a.d,bp,b.d,beta,cp,c.d);
            else pgemm<Backend,R>(h,ta,tb,m,n,k,alpha,ap,a.d,bp,b.d,beta,cp,c.d);
        } else pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,ap,a.d,bp,b.d,beta,cp,c.d,alg);
        if(h->host_threads!=threads)throw std::runtime_error("mixed pgemm changed host thread budget");
        auto out=c.download();long double worst=0;int bad=0;
        for(std::size_t p=0;p<out.size();++p) {
            const int i=p%c.d[DDLA_LLD_],j=p/c.d[DDLA_LLD_];
            const int gi=indxl2g(i,mb,c.r,ms,c.pr),gj=indxl2g(j,nb,c.c,ns,c.pc);
            if(i<c.nr && j<c.nc && gi<m && gj<n) {
                std::complex<long double> sum=0;long double scale=0;
                if(!no_product)for(int l=0;l<k;++l) {
                    auto av=operand<AType>(ta,gi,l,1),bv=operand<BType>(tb,l,gj,2);
                    sum+=av*bv;scale+=std::abs(av)*std::abs(bv);
                }
                auto ref=no_product?std::complex<long double>(0):extended(alpha)*sum;
                if(!no_product)scale*=std::abs(extended(alpha));
                if(!zero_beta){auto z=extended(beta)*extended(value<Z>(gi,gj,3));ref+=z;scale+=std::abs(z);}
                const auto error=std::abs(extended(out[p])-ref);
                const auto normalized=error/std::max(1.L,scale);
                if(!std::isfinite(error))bad=1;
                else worst=std::max(worst,normalized);
            } else if(std::memcmp(&out[p],&c.host[p],sizeof(Z)))bad=1;
        }
        long double error=0;int any=0;
        MPI_Allreduce(&worst,&error,1,MPI_LONG_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
        MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
        if(any || error>64*std::numeric_limits<R>::epsilon()) {
            if(ddlaGetRank(h)==0)std::cerr<<"mixed pgemm FAIL cr="<<CR<<" bytes="<<sizeof(R)<<" flags="<<ta<<tb<<" shape="<<m<<','<<n<<','<<k<<" shift="<<shifted<<" mode="<<mode<<" alg="<<int(alg)<<" error="<<double(error)<<" guard="<<any<<'\n';
            throw std::runtime_error("mixed pgemm scalar oracle/guard failure");
        }
        ++cases;
    }
    auto aa=a.download();auto bb=b.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(AType)) ||
            std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(BType)),any=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if(any)throw std::runtime_error("mixed pgemm modified inputs");
}

template <DdlaBackend Backend,typename R,bool CR>
void checks(DdlaHandle_t h,int& cases) {
    for(char ta:{'N','T','C'})for(char tb:{'N','T','C'})for(bool shifted:{false,true}) {
        one<Backend,R,CR>(h,17,13,23,ta,tb,shifted,0,cases);
        one<Backend,R,CR>(h,1,2,1,ta,tb,shifted,1,cases);
    }
    for(int mode:{1,2,3})one<Backend,R,CR>(h,11,7,13,'C','C',true,mode,cases);
    one<Backend,R,CR>(h,13,9,0,'N','N',true,0,cases);
    one<Backend,R,CR>(h,0,7,9,'N','C',false,0,cases);
    one<Backend,R,CR>(h,7,0,9,'C','N',false,0,cases);
}

template <DdlaBackend Backend> void invalid_cases(DdlaHandle_t h) {
    using Z=std::complex<double>;const int rank=ddlaGetRank(h);
    Matrix<double>a(h,17,23,4,3,0,0,1,false);
    Matrix<Z>b(h,23,13,3,5,0,0,2,false),c(h,17,13,4,5,0,0,3,false);
    auto reject=[&](auto&& fn) {
        int caught=0;try{fn();}catch(const std::exception&){caught=1;}
        int all=0;MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all)throw std::runtime_error("mixed pgemm validation was not collective");
    };
    reject([&]{pgemm<Backend,double>(h,rank==0?'X':'N','N',17,13,23,Z(1),a.data,a.d,b.data,b.d,Z(0),c.data,c.d);});
    reject([&]{pgemm<Backend,double>(h,'N','N',17,13,23,Z(1),rank==0?nullptr:a.data,a.d,b.data,b.d,Z(0),c.data,c.d);});
    reject([&]{pgemm<Backend,double>(h,'N','N',17,13,23,Z(1),a.data,a.d,b.data,b.d,Z(0),rank==0?nullptr:c.data,c.d);});
    int bad_desc[DDLA_DLEN_];std::copy(b.d,b.d+DDLA_DLEN_,bad_desc);if(rank==0)bad_desc[DDLA_LLD_]=0;
    reject([&]{pgemm<Backend,double>(h,'N','N',17,13,23,Z(1),a.data,a.d,b.data,bad_desc,Z(0),c.data,c.d);});
    int np=1;MPI_Comm_size(ddlaGetCommunicator(h),&np);
    if(np>1){
        setenv("DDLA_PGEMM_KPANEL",rank==0?"6":"9",1);
        reject([&]{pgemm<Backend,double>(h,'N','N',17,13,23,Z(1),a.data,a.d,b.data,b.d,Z(0),c.data,c.d);});
        setenv("DDLA_PGEMM_KPANEL","6",1);
    }
    // A zero product and beta=1 must not even form an address from null C.
    pgemm<Backend,double>(h,'N','N',17,13,23,Z(0),static_cast<const double*>(nullptr),a.d,
                         static_cast<const Z*>(nullptr),b.d,Z(1),static_cast<Z*>(nullptr),c.d);
}

template <DdlaBackend Backend> void run(int pr,int pc) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    int cases=0;
    checks<Backend,float,false>(h,cases);checks<Backend,double,false>(h,cases);
    checks<Backend,float,true>(h,cases);checks<Backend,double,true>(h,cases);
    invalid_cases<Backend>(h);
    if(ddlaGetRank(h)==0)std::cout<<"PASS mixed pgemm backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc<<" cases="<<cases<<'\n';
    DDLA_CHECK(ddlaDestroy(h));
}
int main(int argc,char**argv) {
    MPI_Init(&argc,&argv);
    try {
        const std::string mode=argc>1?argv[1]:"all";
        if(mode!="all" && mode!="cpu" && mode!="gpu")throw std::runtime_error("invalid backend mode");
        if((mode=="gpu" && !DDLA_HAS_GPU))throw std::runtime_error("backend unavailable");
        int np=1;MPI_Comm_size(MPI_COMM_WORLD,&np);
        const int pr=argc>2?std::stoi(argv[2]):1,pc=argc>3?std::stoi(argv[3]):np;
        if(pr*pc!=np)throw std::runtime_error("grid does not match ranks");
        setenv("DDLA_PGEMM_KPANEL","6",1);
        if(mode!="gpu")run<DdlaBackend::CPU>(pr,pc);
#if DDLA_HAS_GPU
        if(mode!="cpu")run<DdlaBackend::GPU>(pr,pc);
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
