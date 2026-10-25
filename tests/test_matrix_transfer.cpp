#include "api_grid_test_common.h"
#include <cstdint>
#include <cstring>
#include <type_traits>

using namespace api_grid_test;
namespace {
// Embed exceptional representations as well as coordinates. Integer copies
// below let this test detect lost NaN payloads and signed zeros exactly.
template<class R> R component(int key) {
    using U=std::conditional_t<sizeof(R)==4,std::uint32_t,std::uint64_t>;
    U bits;
    if(key%17==0) bits=U(1)<<(sizeof(R)*8-1); // negative zero
    else if(key%19==0) {
        if constexpr(sizeof(R)==4) bits=U(0x7fc00000u)|U(key&0x3fffff);
        else bits=U(0x7ff8000000000000ULL)|U(key);
    } else if(key%23==0) {
        if constexpr(sizeof(R)==4) bits=U(0xff800000u);
        else bits=U(0xfff0000000000000ULL);
    } else return R(key)+R(0.25);
    R out;std::memcpy(&out,&bits,sizeof(R));return out;
}
template<class T> T value(int r,int c,int salt=0) {
    int key=1+r+97*c+salt;
    if constexpr(std::is_same_v<T,float> || std::is_same_v<T,double>) return component<T>(key);
    else {using R=typename T::value_type;R parts[2]={component<R>(key),component<R>(key+11)};
          T out;std::memcpy(&out,parts,sizeof(T));return out;}
}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
template<DdlaBackend B,class T>
void one(const DdlaHandle_t& h,int M,int N,int mb,int nb,int rs,int cs,int ia,int ja,int m,int n,int root) {
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,M,N,mb,nb,rs,cs));d[DDLA_LLD_]+=3;
    auto original=make_local<T>(h,d,[](int i,int j){return value<T>(i,j);});
    const T sentinel=value<T>(7,9,301);
    for(int c=0;c<ddla_test::n_loc(h,d);++c)
        for(int r=ddla_test::m_loc(h,d);r<d[DDLA_LLD_];++r)original[r+std::size_t(c)*d[DDLA_LLD_]]=sentinel;
    DeviceBuffer<T> a(h,original.size());upload(h,a.ptr,original);
    int rank;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
    int pr,pc,myr,myc;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,myr,myc);
    const int lr=num_loc(ia-1+m,mb,myr,rs,pr)-num_loc(ia-1,mb,myr,rs,pr);
    const int lc=num_loc(ja-1+n,nb,myc,cs,pc)-num_loc(ja-1,nb,myc,cs,pc);
    T* const distributed=lr && lc?a.ptr:nullptr;
    const int ldw=m+5;
    std::vector<T> window(rank==root ? std::size_t(ldw)*n : 0,sentinel);
    DeviceBuffer<T> w(h,window.size());upload(h,w.ptr,window);
    ddla::gather<B,T>(h,m,n,distributed,ia,ja,d,root,rank==root?w.ptr:nullptr,rank==root?ldw:-7);
    check_ddla_sync(h);
    auto got=download(h,w.ptr,window.size());auto expected=window;
    if(rank==root)for(int c=0;c<n;++c)for(int r=0;r<m;++r)expected[r+std::size_t(c)*ldw]=value<T>(ia-1+r,ja-1+c);
    require_close(h,"gather exact natural order and padding",same(got,expected)?0:1,0);
    require_close(h,"gather preserves source",same(download(h,a.ptr,original.size()),original)?0:1,0);
    if(rank==root)for(int c=0;c<n;++c)for(int r=0;r<m;++r)window[r+std::size_t(c)*ldw]=value<T>(r,c,7001);
    upload(h,w.ptr,window);
    ddla::scatter<B,T>(h,m,n,distributed,ia,ja,d,root,rank==root?w.ptr:nullptr,rank==root?ldw:-7);
    check_ddla_sync(h);
    auto want=original;
    for(int c=0;c<ddla_test::n_loc(h,d);++c)for(int r=0;r<ddla_test::m_loc(h,d);++r) {
        int gr=indx_l2g_r(d,h,r),gc=indx_l2g_c(d,h,c);
        if(gr>=ia-1 && gr<ia-1+m && gc>=ja-1 && gc<ja-1+n)
            want[r+std::size_t(c)*d[DDLA_LLD_]]=value<T>(gr-ia+1,gc-ja+1,7001);
    }
    require_close(h,"scatter exact selected entries and untouched exterior/padding",same(download(h,a.ptr,want.size()),want)?0:1,0);
    require_close(h,"scatter preserves root source",same(download(h,w.ptr,window.size()),window)?0:1,0);
    // Empty operations do not inspect storage, even on root.
    ddla::gather<B,T>(h,0,n,nullptr,M+1,ja,d,root,nullptr,-1);
    ddla::scatter<B,T>(h,m,0,nullptr,ia,N+1,d,root,nullptr,-1);
}
template<DdlaBackend B,class T> void cases(const DdlaHandle_t& h,int size,int pr,int pc) {
    for(int source=0;source<2;++source)for(int root=0;root<size;++root) {
        int rs=source?pr-1:0,cs=source?pc-1:0;
        one<B,T>(h,31,27,3,5,rs,cs,2,4,27,21,root); // both axes, unaligned
        one<B,T>(h,31,27,3,5,rs,cs,5,7,19,3,root); // column fast path or arbitrary root
        one<B,T>(h,2,1,3,5,rs,cs,1,1,2,1,root);   // root may own nothing
    }
    int empty[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(empty,h,0,0,3,5,0,0));
    ddla::gather<B,T>(h,0,0,nullptr,1,1,empty,size-1,nullptr,-1);
    ddla::scatter<B,T>(h,0,0,nullptr,1,1,empty,size-1,nullptr,-1);
    // Exact alias of a strided local view; selected rectangle all on root.
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,9,11,9,11,pr-1,pc-1));d[DDLA_LLD_]+=4;
    auto host=make_local<T>(h,d,[](int i,int j){return value<T>(i,j);});
    DeviceBuffer<T> a(h,host.size());upload(h,a.ptr,host);
    int rr=0;
    // Avoid relying on the grid's major order to identify the owner rank.
    int myr,myc,rank;ddlaGetGridCoords(h,myr,myc);MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
    if(myr==pr-1 && myc==pc-1)rr=rank;
    MPI_Allreduce(MPI_IN_PLACE,&rr,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    T* local=rank==rr?a.ptr+1+d[DDLA_LLD_]*2:nullptr;
    ddla::gather<B,T>(h,6,7,host.empty()?nullptr:a.ptr,2,3,d,rr,local,d[DDLA_LLD_]);
    ddla::scatter<B,T>(h,6,7,host.empty()?nullptr:a.ptr,2,3,d,rr,local,d[DDLA_LLD_]);
    require_close(h,"matrix transfer exact local alias",same(download(h,a.ptr,host.size()),host)?0:1,0);
}
template<DdlaBackend B> void invalid(const DdlaHandle_t& h,int rank,int size) {
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,7,8,2,3,0,0));
    DeviceBuffer<double>a(h,local_size(h,d)),w(h,7*8);
    std::vector<double> host(56);
    for(int kind=0;kind<(B==DdlaBackend::GPU?13:11);++kind) {
        int bad[DDLA_DLEN_];std::memcpy(bad,d,sizeof(bad));
        int m=7,n=8,ia=1,ja=1,root=0,ld=7;
        const int* desc=bad;double* ap=a.ptr;double* wp=w.ptr;
        if(kind==0 && rank==size-1)bad[DDLA_MB_]=0;
        if(kind==1 && rank==0)ld=6;
        if(kind==2 && rank==0)wp=nullptr;
        if(kind==3 && rank==0)ap=nullptr;
        if(kind==4){ia=0;}
        if(kind==5){ja=2;}
        if(kind==6){root=size;}
        if(kind==7 && rank==size-1)desc=nullptr;
        if(kind==8){m=-1;}
        if(kind==9 && rank==size-1)bad[DDLA_LLD_]=0;
        if(kind==10){m=n=bad[DDLA_M_]=bad[DDLA_N_]=bad[DDLA_LLD_]=ld=std::numeric_limits<int>::max();}
        if(kind==11 && rank==0)wp=host.data();
        if(kind==12 && rank==0)ap=host.data();
        bool caught=false;
        try{ddla::gather<B,double>(h,m,n,ap,ia,ja,desc,root,wp,ld);}catch(const std::invalid_argument&){caught=true;}
        require_close(h,"collective gather validation "+std::to_string(kind),caught?0:1,0);
        caught=false;
        try{ddla::scatter<B,double>(h,m,n,ap,ia,ja,desc,root,wp,ld);}catch(const std::invalid_argument&){caught=true;}
        require_close(h,"collective scatter validation "+std::to_string(kind),caught?0:1,0);
    }
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    bool caught=false;
    try {
        if constexpr(B==DdlaBackend::GPU)ddla::gather<DdlaBackend::CPU,double>(h,0,0,nullptr,1,1,d,0,nullptr,0);
        else ddla::gather<DdlaBackend::GPU,double>(h,0,0,nullptr,1,1,d,0,nullptr,0);
    }catch(const std::invalid_argument&){caught=true;}
    require_close(h,"backend mismatch rejected",caught?0:1,0);
#endif
}
template<DdlaBackend B> void run(MPI_Comm comm,int pr,int pc,char major) {
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    int rank,size;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
    cases<B,float>(h,size,pr,pc);cases<B,double>(h,size,pr,pc);
    cases<B,std::complex<float>>(h,size,pr,pc);cases<B,std::complex<double>>(h,size,pr,pc);
    invalid<B>(h,rank,size);
    if(rank==0)std::cout<<"matrix transfer PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char**argv) {
    MPI_Init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
    TestOptions opts;std::string error;
    if(!parse_options(argc,argv,size,opts,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:opts.grids)for(char major:{'R','C'}) {
#if DDLA_HAS_CPU
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    if(size>=4) {
        int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm sub;
        MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&sub);
        int subsize;MPI_Comm_size(sub,&subsize);
#if DDLA_HAS_CPU
        if(rank%2==0)run<DdlaBackend::CPU>(sub,subsize,1,'C');
#endif
#if DDLA_HAS_GPU
        if(rank%2==0)run<DdlaBackend::GPU>(sub,subsize,1,'C');
#endif
        MPI_Comm_free(&sub);MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
