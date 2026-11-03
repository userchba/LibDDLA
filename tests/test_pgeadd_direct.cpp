#include "matrix_test_common.h"

static int alltoalls=0,sendrecvs=0,max_send=0;
extern "C" int MPI_Alltoallv(const void* s,const int* sc,const int* sd,MPI_Datatype st,
    void* r,const int* rc,const int* rd,MPI_Datatype rt,MPI_Comm c) {
    ++alltoalls;return PMPI_Alltoallv(s,sc,sd,st,r,rc,rd,rt,c);
}
extern "C" int MPI_Sendrecv(const void* s,int sc,MPI_Datatype st,int dest,int stag,
    void* r,int rc,MPI_Datatype rt,int source,int rtag,MPI_Comm c,MPI_Status* status) {
    ++sendrecvs;max_send=std::max(max_send,sc);
    return PMPI_Sendrecv(s,sc,st,dest,stag,r,rc,rt,source,rtag,c,status);
}

template<DdlaBackend Backend,typename T>
void check_case(DdlaHandle_t h,char ta,char tb,bool shifted,int mode,bool expect_block,int& cases,bool colocate=false)
{
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    const int mb=16,nb=24,m=513,n=385,rs=pr-1,cs=pc-1;
    const int ic=shifted?19:1,jc=shifted?27:1;
    const int ia=shifted?(ta=='N'?ic+mb:jc+nb):1,ja=shifted?(ta=='N'?jc+2*nb:ic+2*mb):1;
    // colocate (both transposed): B takes A's process origin shifted by whole
    // process cycles, so the operands share owners and pre-add into one message.
    const bool co=colocate && ta!='N' && tb!='N';
    const int ib=co?ia+nb*pr:shifted?(tb=='N'?ic+2*mb:jc+2*nb):1;
    const int jb=co?ja+mb*pc:shifted?(tb=='N'?jc+nb:ic+mb):1;
    const T alpha=mode>=2?T(0):mode==1?T(1):scalar<T>(0.7,-0.2);
    const T beta=mode==1||mode==3?T(0):mode==2?T(1):scalar<T>(-0.3,0.15);
    Matrix<T> a(h,ia-1+(ta=='N'?m:n),ja-1+(ta=='N'?n:m),ta=='N'?mb:nb,ta=='N'?nb:mb,
        ta=='N'?(rs+pr-(shifted?1:0))%pr:0,ta=='N'?(cs+2*pc-(shifted?2:0))%pc:0,1,alpha==T(0));
    Matrix<T> b(h,ib-1+(tb=='N'?m:n),jb-1+(tb=='N'?n:m),tb=='N'?mb:nb,tb=='N'?nb:mb,
        co?0:tb=='N'?(rs+2*pr-(shifted?2:0))%pr:pr-1,co?0:tb=='N'?(cs+pc-(shifted?1:0))%pc:pc-1,2,beta==T(0));
    Matrix<T> c(h,ic-1+m,jc-1+n,mb,nb,rs,cs,3,true);
    DDLA_CHECK(ddlaSynchronize(h));
    alltoalls=sendrecvs=max_send=0;
    pgeadd<Backend>(h,ta,tb,m,n,alpha,a.data,ia,ja,a.d,beta,b.data,ib,jb,b.d,c.data,ic,jc,c.d);
    const int a2a=alltoalls,p2p=sendrecvs,count=max_send;
    int counts[3]={a2a,p2p,count},sums[3]={};
    MPI_Allreduce(counts,sums,2,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
    MPI_Allreduce(counts+2,sums+2,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    const int trans=(ta!='N' && alpha!=T(0))+(tb!='N' && beta!=T(0));
    // Square grids exchange a pre-added co-located pair as one message.
    const int messages=co && trans==2 ? 1 : trans;
    constexpr bool mpi=Backend==DdlaBackend::CPU
#ifdef DDLA_USE_GPU_CPU_TUNNEL
        || Backend==DdlaBackend::GPU
#endif
        ;
    if constexpr(mpi) {
        const bool blocks=expect_block || pr!=pc;
        if(sums[0]!=(blocks?trans*pr*pc:0) || p2p>(blocks?0:messages) ||
           (!blocks && pr>1 && trans && (!sums[1] || sums[2]<16384)))
            throw std::runtime_error("pgeadd direct communication regression");
    }
    auto out=c.download(),aa=a.download(),bb=b.download();
    double error=0;int bad=0,any=0;
    for(std::size_t ix=0;ix<out.size();++ix) {
        const int i=ix%c.d[DDLA_LLD_],j=ix/c.d[DDLA_LLD_];
        const int gi=indxl2g(i,mb,c.r,rs,pr),gj=indxl2g(j,nb,c.c,cs,pc);
        if(i<c.nr && j<c.nc && gi>=ic-1 && gi<ic-1+m && gj>=jc-1 && gj<jc-1+n) {
            const int r=gi-(ic-1),col=gj-(jc-1);
            auto operand=[&](char op,int ai,int aj,int salt) {
                auto x=wide(value<T>(ai-1+(op=='N'?r:col),aj-1+(op=='N'?col:r),salt));
                return op=='C'?std::conj(x):x;
            };
            std::complex<double> ref=0;
            if(alpha!=T(0))ref+=wide(alpha)*operand(ta,ia,ja,1);
            if(beta!=T(0))ref+=wide(beta)*operand(tb,ib,jb,2);
            const double delta=std::abs(wide(out[ix])-ref)/(1+std::abs(ref));
            if(!std::isfinite(delta))bad=1;else error=std::max(error,delta);
        } else if(std::memcmp(&out[ix],&c.host[ix],sizeof(T)))bad=1;
    }
    if(std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T)) ||
       std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T)))bad=1;
    double worst=0;MPI_Allreduce(&error,&worst,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<T,float>||std::is_same_v<T,std::complex<float>>;
    if(any || worst>(single?3e-5:2e-11))throw std::runtime_error("pgeadd direct numerical/guard regression");
    ++cases;
}
template<DdlaBackend Backend,typename T>
void scalar_cases(DdlaHandle_t h,bool before,int& cases) {
    for(char ta:{'N','T','C'})for(char tb:{'N','T','C'})for(bool shifted:{false,true})
        for(int mode=0;mode<4;++mode)check_case<Backend,T>(h,ta,tb,shifted,mode,before,cases);
    for(char ta:{'T','C'})for(char tb:{'T','C'})for(bool shifted:{false,true})
        check_case<Backend,T>(h,ta,tb,shifted,0,before,cases,true);
}
template<DdlaBackend Backend>void run(int pr,int pc,bool before) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    int cases=0;scalar_cases<Backend,float>(h,before,cases);scalar_cases<Backend,double>(h,before,cases);
    scalar_cases<Backend,std::complex<float>>(h,before,cases);scalar_cases<Backend,std::complex<double>>(h,before,cases);
    if(ddlaGetRank(h)==0)std::cout<<"PGEADD_DIRECT_PASS backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc<<" cases="<<cases<<" route="<<(before?"block-before":"automatic")<<'\n';
    DDLA_CHECK(ddlaDestroy(h));
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);try {
        int np;MPI_Comm_size(MPI_COMM_WORLD,&np);const int pr=argc>1?std::stoi(argv[1]):1,pc=argc>2?std::stoi(argv[2]):np;
        const bool before=argc>3 && std::string(argv[3])=="expect-block";
        if(pr*pc!=np)throw std::runtime_error("wrong grid");
        run<DdlaBackend::CPU>(pr,pc,before);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(pr,pc,before);
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
