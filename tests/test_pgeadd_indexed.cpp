#include "matrix_test_common.h"

// This oracle addresses global elements directly, independently of the block
// planner. Guards check every element outside C's requested rectangle.
template <DdlaBackend Backend,typename T>
void check_case(DdlaHandle_t h,char ta,char tb,int m,int n,int mode,int alias,bool shifted,int& cases,int pointer_shift=0,bool colocate=false)
{
    int pr,pc; ddlaGetGridDims(h,pr,pc);
    const int mb=alias?3:4,nb=3;
    const int ic=shifted?(alias?5:6):1,jc=shifted?5:1;
    const int ia=shifted?(ta=='N'?ic+mb:jc+nb):1;
    const int ja=shifted?(ta=='N'?jc+2*nb:ic+2*mb):1;
    const int ib=shifted?(tb=='N'?ic+2*mb:jc+2*nb):1;
    const int jb=shifted?(tb=='N'?jc+nb:ic+mb):1;
    const int rs=pr-1,cs=pc-1;
    const T alpha=mode==1||mode==3||mode==5?T(0):mode==4?T(1):scalar<T>(0.7,-0.2);
    const T beta=mode==2||mode==3||mode==4?T(0):mode==5?T(1):scalar<T>(-0.3,0.15);
    const int extent=pointer_shift?128:48;
    // colocate (both transposed): B takes A's process origin shifted by whole
    // process cycles, so the operands share owners. alias 3 reads A and B from C.
    const bool co=colocate && ta!='N' && tb!='N';
    const bool a_in_c=alias==1||alias==3,b_in_c=alias==2||alias==3;
    Matrix<T> a(h,extent,extent,ta=='N'?mb:nb,ta=='N'?nb:mb,
        ta=='N'?(rs+pr-(shifted?1:0))%pr:0,
        ta=='N'?(cs+2*pc-(shifted?2:0))%pc:pc-1,1,alpha==T(0));
    Matrix<T> b(h,extent,extent,tb=='N'?mb:nb,tb=='N'?nb:mb,
        co?(a_in_c?rs:0):tb=='N'?(rs+2*pr-(shifted?2:0))%pr:pr-1,
        co?(a_in_c?cs:pc-1):tb=='N'?(cs+pc-(shifted?1:0))%pc:0,2,beta==T(0));
    Matrix<T> c(h,extent,extent,mb,nb,rs,cs,3,true);
    const T* av=a.data; const int* da=a.d; int ai=ia,aj=ja;
    const T* bv=b.data; const int* db=b.d; int bi=ib,bj=jb;
    if(a_in_c) { av=c.data+pointer_shift;da=c.d;ai=ta=='N'?ic+mb*pr:jc;aj=ta=='N'?jc:ic; }
    if(alias==2) { bv=c.data+pointer_shift;db=c.d;bi=tb=='N'?ic:jc;bj=tb=='N'?jc+nb*pc:ic; }
    if(co) { bi=ai+nb*pr;bj=aj+mb*pc; }
    if(alias==3) { bv=c.data;db=c.d; }
    // Aliased inputs start with finite values. Other C cases start poisoned.
    if(alias) { for(int j=0;j<c.nc;++j) for(int i=0;i<c.nr;++i) {
        int gi=indxl2g(i,mb,c.r,c.d[DDLA_RSRC_],pr),gj=indxl2g(j,nb,c.c,c.d[DDLA_CSRC_],pc);
        c.host[i+std::size_t(j)*c.d[DDLA_LLD_]]=value<T>(gi,gj,3);
    } c.reset(); }
    pgeadd<Backend>(h,ta,tb,m,n,alpha,av,ai,aj,da,beta,bv,bi,bj,db,c.data,ic,jc,c.d);
    auto out=c.download(),aa=a.download(),bb=b.download();
    double error=0;int bad=0,any=0;
    for(std::size_t ix=0;ix<out.size();++ix) {
        int i=ix%c.d[DDLA_LLD_],j=ix/c.d[DDLA_LLD_];
        int gi=indxl2g(i,mb,c.r,c.d[DDLA_RSRC_],pr),gj=indxl2g(j,nb,c.c,c.d[DDLA_CSRC_],pc);
        if(i<c.nr && j<c.nc && gi>=ic-1 && gi<ic-1+m && gj>=jc-1 && gj<jc-1+n) {
            const int r=gi-(ic-1),col=gj-(jc-1);
            auto operand=[&](char op,int i0,int j0,int salt) {
                int gi=i0-1+(op=='N'?r:col),gj=j0-1+(op=='N'?col:r);
                if(pointer_shift && salt==3) {
                    const int owner=indxg2p(gi,mb,rs,pr);
                    gi=indxl2g(indxg2l(gi,mb,pr)+pointer_shift,mb,owner,rs,pr);
                }
                auto x=wide(value<T>(gi,gj,salt));
                return op=='C'?std::conj(x):x;
            };
            std::complex<double> ref=0;
            if(alpha!=T(0)) ref+=wide(alpha)*operand(ta,ai,aj,a_in_c?3:1);
            if(beta!=T(0)) ref+=wide(beta)*operand(tb,bi,bj,b_in_c?3:2);
            const double e=std::abs(wide(out[ix])-ref)/(1+std::abs(ref));
            if(!std::isfinite(e))bad=1;else error=std::max(error,e);
        }else if(std::memcmp(&out[ix],&c.host[ix],sizeof(T)))bad=1;
    }
    if(std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T)) ||
       std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T)))bad=1;
    double worst=0;MPI_Allreduce(&error,&worst,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<T,float>||std::is_same_v<T,std::complex<float>>;
    if(any || worst>(single?3e-5:2e-11)) {
        if(ddlaGetRank(h)==0)std::cerr<<"pgeadd FAIL ops="<<ta<<tb<<" shape="<<m<<','<<n<<" mode="<<mode<<" alias="<<alias<<" shifted="<<shifted<<" pointer_shift="<<pointer_shift<<" colocate="<<co<<" bytes="<<sizeof(T)<<" error="<<worst<<" guard="<<any<<'\n';
        throw std::runtime_error("pgeadd shifted regression");
    }
    ++cases;
}

template <DdlaBackend Backend,typename T>
void scalar_cases(DdlaHandle_t h,int& cases) {
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    for(char ta:{'N','T','C'})for(char tb:{'N','T','C'}) {
        for(int mode=0;mode<4;++mode)for(bool shifted:{false,true}) {
            check_case<Backend,T>(h,ta,tb,9,7,mode,0,shifted,cases);
            check_case<Backend,T>(h,ta,tb,1,2,mode,0,shifted,cases);
        }
        for(int alias:{1,2}) for(int mode:{0,1,2})
            check_case<Backend,T>(h,ta,tb,9,7,mode,alias,true,cases);
        for(int alias:{1,2}) if((alias==1?ta:tb)!='N')
            for(int mode:{0,alias==1?2:1,alias==1?4:5})
                check_case<Backend,T>(h,ta,tb,73,67,mode,alias,true,cases,1);
        for(int mode:{4,5}) {
            for(bool shifted:{false,true}) {
                check_case<Backend,T>(h,ta,tb,9,7,mode,0,shifted,cases);
                check_case<Backend,T>(h,ta,tb,1,2,mode,0,shifted,cases);
            }
            for(int alias:{1,2}) check_case<Backend,T>(h,ta,tb,9,7,mode,alias,true,cases);
        }
        check_case<Backend,T>(h,ta,tb,0,7,0,0,true,cases);
        check_case<Backend,T>(h,ta,tb,9,0,0,0,true,cases);
        if(ta!='N' && tb!='N') {
            for(bool shifted:{false,true}) {
                check_case<Backend,T>(h,ta,tb,9,7,0,0,shifted,cases,0,true);
                check_case<Backend,T>(h,ta,tb,1,2,0,0,shifted,cases,0,true);
            }
            for(int alias:{1,3}) check_case<Backend,T>(h,ta,tb,9,7,0,alias,true,cases,0,true);
        }
    }
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,48,48,4,3,0,0));
    auto rejected=[&](char ta,char tb,const int* da,int ai,int aj,const int* db,int bi,int bj,int ci,int cj,int m=2) {
        bool caught=false;
        try { pgeadd<Backend,T>(h,ta,tb,m,2,T(0),nullptr,ai,aj,da,T(0),nullptr,bi,bj,db,nullptr,ci,cj,d); }
        catch(const std::invalid_argument&){caught=true;}
        if(!caught)throw std::runtime_error("pgeadd failed to reject incompatible layout or invalid region");
    };
    rejected('N','N',d,0,1,d,1,1,1,1);rejected('N','N',d,1,0,d,1,1,1,1);
    rejected('N','N',d,48,1,d,1,1,1,1);rejected('N','N',d,1,1,d,1,1,48,1);
    rejected('?','N',d,1,1,d,1,1,1,1);
    int bad[DDLA_DLEN_],trans[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(bad,h,48,48,5,3,0,0));
    DDLA_CHECK(ddlaDescInit(trans,h,48,48,3,4,0,0));
    // Validate both input layouts even when their coefficients are zero.
    rejected('N','N',bad,1,1,d,1,1,1,1);
    rejected('N','N',d,1,1,bad,1,1,1,1);
    rejected('T','N',d,1,1,d,1,1,1,1);
    rejected('N','C',d,1,1,d,1,1,1,1);
    rejected('N','N',d,2,1,d,1,1,1,1);
    rejected('N','N',d,1,1,d,1,2,1,1);
    rejected('T','N',trans,1,2,d,1,1,1,1);
    rejected('N','C',d,1,1,trans,2,1,1,1);
    rejected('N','N',bad,1,1,d,1,1,1,1,0);
    if(pr>1) {
        DDLA_CHECK(ddlaDescInit(bad,h,48,48,4,3,1,0));
        rejected('N','N',bad,1,1,d,1,1,1,1);
        rejected('N','N',d,1,1,bad,1,1,1,1);
        rejected('N','N',d,5,1,d,1,1,1,1);
    }
    if(pc>1) {
        DDLA_CHECK(ddlaDescInit(bad,h,48,48,4,3,0,1));
        rejected('N','N',bad,1,1,d,1,1,1,1);
        rejected('N','N',d,1,1,bad,1,1,1,1);
        rejected('N','N',d,1,4,d,1,1,1,1);
    }
}

template <DdlaBackend Backend> void run(int pr,int pc) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    int cases=0;scalar_cases<Backend,float>(h,cases);scalar_cases<Backend,double>(h,cases);
    scalar_cases<Backend,std::complex<float>>(h,cases);scalar_cases<Backend,std::complex<double>>(h,cases);
    if(ddlaGetRank(h)==0)std::cout<<"PGEADD_PASS backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc<<" cases="<<cases<<" invalid="<<4*(14+3*(pr>1)+3*(pc>1))<<'\n';
    DDLA_CHECK(ddlaDestroy(h));
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);try {
        int np;MPI_Comm_size(MPI_COMM_WORLD,&np);int pr=argc>1?std::stoi(argv[1]):1,pc=argc>2?std::stoi(argv[2]):np;
        if(pr*pc!=np)throw std::runtime_error("wrong grid");
        run<DdlaBackend::CPU>(pr,pc);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(pr,pc);
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
