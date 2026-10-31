#include "pgemm_test_common.h"
#include "ddla_stream.h"

void check_selection() {
    int d[DDLA_DLEN_] = {};
    d[DDLA_MB_] = d[DDLA_NB_] = 128;
    for(char ta : {'N','T','C'}) for(char tb : {'N','T','C'}) {
        const int shapes[][3]={{4096,4096,16},{4096,16,4096},{16,4096,4096}};
        const Algorithm expected[]={Algorithm::AB,Algorithm::BC,Algorithm::AC};
        for(int i=0;i<3;++i) {
            const auto* s=shapes[i];
            if(pgemm_impl::select(ta,tb,s[0],s[1],s[2],d,d,d,2,2)!=expected[i] ||
               pgemm_impl::select(ta,tb,s[0],s[1],s[2],d,d,d,1,1)!=Algorithm::AB)
                throw std::runtime_error("pgemm automatic selection regression");
        }
    }
}

template <DdlaBackend Backend, typename T>
void check_case(DdlaHandle_t h, int m, int n, int k, char ta, char tb, bool shifted, int mode, int& cases, int unaligned = 0)
{
    int pr,pc; ddlaGetGridDims(h,pr,pc);
    const int mb=4, nb=5, kb=3;
    const int ms=shifted ? (ta=='N' ? pr-1 : std::min(pr,pc)-1) : 0;
    const int ns=shifted ? (tb=='N' ? pc-1 : std::min(pr,pc)-1) : 0;
    const int kr=shifted ? pr-1 : 0, kc=shifted ? pc-1 : 0;
    const bool zero_a=mode>=2, zero_b=mode==1 || mode==3;
    const T alpha=zero_a ? T(0) : scalar<T>(0.7,-0.2);
    const T beta=zero_b ? T(0) : scalar<T>(-0.3,0.15);
    Matrix<T> a(h,ta=='N'?m:k,ta=='N'?k:m,ta=='N'?mb:kb,ta=='N'?kb:mb,
                ((ta=='N'?ms:kr)+(unaligned>1?1:0))%pr,((ta=='N'?kc:ms)+(unaligned?1:0))%pc,1,zero_a);
    Matrix<T> b(h,tb=='N'?k:n,tb=='N'?n:k,tb=='N'?kb:nb,tb=='N'?nb:kb,
                ((tb=='N'?kr:ns)+(unaligned?1:0))%pr,((tb=='N'?ns:kc)+(unaligned>1?1:0))%pc,2,zero_a);
    Matrix<T> c(h,m,n,mb,nb,ms,ns,3,zero_b);
    constexpr bool single=std::is_same_v<T,float> || std::is_same_v<T,std::complex<float>>;
    for(auto alg : {Algorithm::AB,Algorithm::AC,Algorithm::BC,Algorithm::Auto}) {
        if (unaligned && (alg==Algorithm::AB || alg==Algorithm::Auto)) continue;
        c.reset();
        const int host_threads = h->host_threads;
        const bool keep_a=alg==Algorithm::BC;
        const bool k_rows=keep_a?ta!='N':tb=='N';
        const bool thin_k_rows=keep_a?tb=='N':ta!='N';
        const int* fixed=keep_a?a.d:b.d;
        const int* thin=keep_a?b.d:a.d;
        const bool reject=alpha!=T(0) && m>0 && n>0 && k>0 && (alg==Algorithm::AC || alg==Algorithm::BC) &&
            ((thin_k_rows==k_rows && thin[k_rows?DDLA_RSRC_:DDLA_CSRC_]!=fixed[k_rows?DDLA_RSRC_:DDLA_CSRC_]) ||
             (keep_a!=k_rows && fixed[k_rows?DDLA_CSRC_:DDLA_RSRC_]!=c.d[keep_a?DDLA_RSRC_:DDLA_CSRC_]));
        bool caught=false;
        try { pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a.data,a.d,b.data,b.d,beta,c.data,c.d,alg); }
        catch(const std::invalid_argument&) { if(!reject)throw; caught=true; }
        if(caught!=reject)throw std::runtime_error("pgemm stationary owner-alignment validation");
        if(caught) {
            auto out=c.download();
            if(std::memcmp(out.data(),c.host.data(),out.size()*sizeof(T)))
                throw std::runtime_error("pgemm rejected layout after modifying C");
            ++cases;continue;
        }
        if(h->host_threads != host_threads) throw std::runtime_error("pgemm changed the host thread budget");
        auto out=c.download();
        double err=0; int bad=0;
        for(std::size_t idx=0;idx<out.size();++idx) {
            const int i=idx%c.d[DDLA_LLD_],j=idx/c.d[DDLA_LLD_];
            const int gi=indxl2g(i,mb,c.r,ms,c.pr),gj=indxl2g(j,nb,c.c,ns,c.pc);
            if(i<c.nr && j<c.nc && gi<m && gj<n) {
                std::complex<double> ref=zero_b ? 0.0 : wide(beta)*wide(value<T>(gi,gj,3));
                if(!zero_a) for(int l=0;l<k;++l) ref+=wide(alpha)*op_value<T>(ta,gi,l,1)*op_value<T>(tb,l,gj,2);
                const double delta=std::abs(wide(out[idx])-ref)/(1+std::abs(ref));
                if(!std::isfinite(delta)) bad=1;
                else err=std::max(err,delta);
            } else if(std::memcmp(&out[idx],&c.host[idx],sizeof(T))!=0) bad=1;
        }
        double worst=0; int any=0;
        MPI_Allreduce(&err,&worst,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
        MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
        if(any || worst>(single?3e-5:2e-11)) {
            if(ddlaGetRank(h)==0) std::cerr << "FAIL backend=" << int(Backend) << " type_bytes=" << sizeof(T)
                << " shape=" << m << ',' << n << ',' << k << " trans=" << ta << tb << " shifted=" << shifted
                << " mode=" << mode << " unaligned=" << unaligned << " branch=" << int(alg) << " error=" << worst << " sentinel/nonfinite=" << any << '\n';
            throw std::runtime_error("pgemm branch regression");
        }
        ++cases;
    }
    auto aa=a.download(), bb=b.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T))!=0 ||
            std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T))!=0, any=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if(any) throw std::runtime_error("pgemm modified an input");
}

template <DdlaBackend Backend, typename T>
void check_pgeadd_transpose_sources(DdlaHandle_t h) {
    int pr,pc; ddlaGetGridDims(h,pr,pc);
    for (bool conjugate : {false,true}) for (bool reverse : {false,true}) {
        const int ar=reverse?0:pr-1, ac=reverse?pc-1:0;
        const int cr=reverse?pr-1:0, cc=reverse?0:pc-1;
        Matrix<T> a(h,9,7,3,4,ar,ac,1,false), c(h,7,9,4,3,cr,cc,3,false);
        int da[DDLA_DLEN_],dc[DDLA_DLEN_];
        std::copy(a.d,a.d+DDLA_DLEN_,da); std::copy(c.d,c.d+DDLA_DLEN_,dc);
        da[DDLA_M_]=9; da[DDLA_N_]=7; dc[DDLA_M_]=7; dc[DDLA_N_]=9;
        pgeadd<Backend>(h,conjugate?'C':'T','N',7,9,T(1),a.data, 1, 1,da,
                        T(0),c.data, 1, 1,dc,c.data, 1, 1,dc);
        auto out=c.download(),input=a.download(); int bad=0,any=0;
        for (std::size_t idx=0;idx<out.size();++idx) {
            const int i=idx/c.d[DDLA_LLD_],j=idx%c.d[DDLA_LLD_];
            const int gi=indxl2g(j,4,c.r,cr,pr),gj=indxl2g(i,3,c.c,cc,pc);
            if(j<c.nr && i<c.nc && gi<7 && gj<9) {
                const auto expected=op_value<T>(conjugate?'C':'T',gi,gj,1);
                const double error=std::abs(wide(out[idx])-expected);
                if(!std::isfinite(error) || error!=0) bad=1;
            } else if(std::memcmp(&out[idx],&c.host[idx],sizeof(T))!=0) bad=1;
        }
        if(std::memcmp(input.data(),a.host.data(),input.size()*sizeof(T))!=0) bad=1;
        MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
        if(any) throw std::runtime_error("pgeadd_transpose source-coordinate regression");
    }
}

template <DdlaBackend Backend, typename T>
void scalar_cases(DdlaHandle_t h, bool quick, int& cases) {
    check_pgeadd_transpose_sources<Backend,T>(h);
    for(char ta : {'N','T','C'}) for(char tb : {'N','T','C'})
        for(int unaligned : {1,2})
            check_case<Backend,T>(h,17,13,21,ta,tb,true,0,cases,unaligned);
    const int shapes[][3]={{31,3,29},{3,31,29},{23,19,5},{1,1,1},{0,9,7},{9,0,7},{11,7,0},{17,13,21}};
    for(int s=0;s<(quick?2:8);++s)
        for(char ta : {'N','T','C'}) for(char tb : {'N','T','C'})
            for(bool shifted : {false,true})
                for(int mode=0;mode<(quick?2:4);++mode)
                    check_case<Backend,T>(h,shapes[s][0],shapes[s][1],shapes[s][2],ta,tb,shifted,mode,cases);
}

template <DdlaBackend Backend>
void backend_cases(int pr,int pc,bool quick) {
    DdlaHandle_t h=nullptr; DDLA_CHECK(ddlaInit(h,Backend)); DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    int cases=0;
    scalar_cases<Backend,float>(h,quick,cases); scalar_cases<Backend,double>(h,quick,cases);
    scalar_cases<Backend,std::complex<float>>(h,quick,cases); scalar_cases<Backend,std::complex<double>>(h,quick,cases);
    if(ddlaGetRank(h)==0) std::cout << "PASS backend=" << int(Backend) << " grid=" << pr << 'x' << pc << " cases=" << cases << '\n';
    DDLA_CHECK(ddlaDestroy(h));
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        check_selection();
        int np; MPI_Comm_size(MPI_COMM_WORLD,&np);
        int pr=argc>1?std::stoi(argv[1]):1,pc=argc>2?std::stoi(argv[2]):np;
        bool quick=argc>3 && std::string(argv[3])=="quick";
        if(pr*pc!=np) throw std::runtime_error("grid does not match ranks");
#if DDLA_HAS_CPU
        backend_cases<DdlaBackend::CPU>(pr,pc,quick);
#endif
#if DDLA_HAS_GPU
        backend_cases<DdlaBackend::GPU>(pr,pc,quick);
#endif
        MPI_Finalize(); return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n'; MPI_Abort(MPI_COMM_WORLD,1); return 1;
    }
}
