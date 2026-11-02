#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "ddla_stream.h"
#include "punmtr_impl.h"
#include "plan_util.h"
#include <ddla/ddla_vector.h>
#include <array>
#include <climits>
#include <complex>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace api_grid_test;
namespace {
using cd=std::complex<double>;
template<class T>T scalar(double r,double i=0) {
    if constexpr(std::is_same<T,typename real_type<T>::type>::value)return T(r);
    else return T(r,i);
}
template<class T>cd wide(T v){return cd(std::real(v),std::imag(v));}
template<class T>T ventry(int i,int j){return scalar<T>(((i*7+j*11)%19-9)/37.,((i*3+j*5)%13-6)/41.);}
template<class T>T tentry(int i,int j){return scalar<T>((i==j?0.4:0.03*(i+j+1)),0.02*(i-j+1));}
template<class T>T centry(int i,int j){return scalar<T>(((i*13+j*7)%31-15)/17.,((i*5+j*3)%17-8)/19.);}
bool no_allocation(const workspace_test::Snapshot& s) {
    return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;
}
template<DdlaBackend B,class T>void exercise(DdlaHandle_t h,char side,char uplo,char trans,int m,int n,int k) {
    const int pr=h->nprows_,pc=h->npcols_,mr=h->myprow_,mc=h->mypcol_;
    int dc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(dc,h,m+2,n+3,4,3,pr-1,pc-1));
    dc[DDLA_LLD_]+=2+h->myid%2;
    const int rows=num_loc(m,dc[DDLA_MB_],mr,dc[DDLA_RSRC_],pr);
    const int cols=num_loc(n,dc[DDLA_NB_],mc,dc[DDLA_CSRC_],pc);
    const int vr=side=='L'?rows:cols,ldv=std::max(1,vr)+2+h->myid%2,ldt=std::max(1,k)+3+h->myid%2;
    const T guard=scalar<T>(-937,31);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<T> v(std::size_t(ldv)*std::max(k,1),guard),t(std::size_t(ldt)*std::max(k,1),guard);
    for(int c=0;c<k;++c) {
        for(int i=0;i<vr;++i) {
            const int gi=indxl2g(i,side=='L'?dc[DDLA_MB_]:dc[DDLA_NB_],
                side=='L'?mr:mc,side=='L'?dc[DDLA_RSRC_]:dc[DDLA_CSRC_],side=='L'?pr:pc);
            v[i+std::size_t(c)*ldv]=ventry<T>(gi,c);
        }
        for(int r=0;r<k;++r)t[r+std::size_t(c)*ldt]=(uplo=='U'?r<=c:r>=c)?tentry<T>(r,c):scalar<T>(nan,nan);
    }
    auto original=make_local<T>(h,dc,[&](int i,int j){return i<m&&j<n?centry<T>(i,j):guard;});
    const int full_rows=ddla_test::m_loc(h,dc);
    for(int j=0;j<ddla_test::n_loc(h,dc);++j)
        for(int i=full_rows;i<dc[DDLA_LLD_];++i)original[i+std::size_t(j)*dc[DDLA_LLD_]]=guard;
    DeviceBuffer<T> V(h,v.size()),Tmat(h,t.size()),C(h,std::max<std::size_t>(1,original.size()));
    upload(h,V.ptr,v);upload(h,Tmat.ptr,t);upload(h,C.ptr,original);

    // Form the dense H explicitly, independently of the projection/TRMM update.
    const int order=side=='L'?m:n;
    std::vector<cd> H(std::size_t(order)*order),expected(std::size_t(m)*n);
    for(int j=0;j<order;++j)for(int i=0;i<order;++i) {
        cd value=i==j?cd(1):cd(0);
        for(int b=0;b<k;++b)for(int a=0;a<k;++a) {
            const int r=trans=='N'?a:b,c=trans=='N'?b:a;
            if(uplo=='U'?r>c:r<c)continue;
            cd coefficient=wide(tentry<T>(r,c));if(trans!='N')coefficient=std::conj(coefficient);
            value-=wide(ventry<T>(i,a))*coefficient*std::conj(wide(ventry<T>(j,b)));
        }
        H[i+std::size_t(j)*order]=value;
    }
    for(int j=0;j<n;++j)for(int i=0;i<m;++i)for(int a=0;a<order;++a)
        expected[i+std::size_t(j)*m]+=side=='L'?H[i+std::size_t(a)*order]*wide(centry<T>(a,j)):
                                                               wide(centry<T>(i,a))*H[a+std::size_t(j)*order];
    auto verify=[&] {
        const auto result=download(h,C.ptr,original.size());
        double error=0,changed=0;
        for(int j=0;j<ddla_test::n_loc(h,dc);++j)for(int i=0;i<dc[DDLA_LLD_];++i) {
            const auto at=i+std::size_t(j)*dc[DDLA_LLD_];
            if(i<rows&&j<cols) {
                const int gi=indxl2g(i,dc[DDLA_MB_],mr,dc[DDLA_RSRC_],pr);
                const int gj=indxl2g(j,dc[DDLA_NB_],mc,dc[DDLA_CSRC_],pc);
                const auto ref=expected[gi+std::size_t(gj)*m];
                error=std::max(error,workspace_test::finite_error(std::abs(wide(result[at])-ref)/(1+std::abs(ref))));
            } else if(std::memcmp(&result[at],&original[at],sizeof(T)))changed=1;
        }
        require_close(h,"punmtr dense H oracle",error,std::is_same<typename real_type<T>::type,float>::value?8e-5:4e-12);
        require_close(h,"punmtr C parent and padding",changed,0);
        const auto after_v=download(h,V.ptr,v.size()),after_t=download(h,Tmat.ptr,t.size());
        require_close(h,"punmtr readonly V/T including poison",std::memcmp(after_v.data(),v.data(),v.size()*sizeof(T))||
            std::memcmp(after_t.data(),t.data(),t.size()*sizeof(T))?1:0,0);
    };
    punmtr<B,T>(h,side,uplo,trans,m,n,k,V.ptr,ldv,Tmat.ptr,ldt,C.ptr,dc);check_ddla_sync(h);verify();

    detail::punmtr_check<T>(h,B,side,uplo,trans,m,n,k,V.ptr,ldv,Tmat.ptr,ldt,C.ptr,dc);
    const int active_k=(!m||!n)?0:k;
    const bool empty=active_k==0||(side=='L'?cols==0:rows==0);
    const bool reduce=(side=='L'?h->nprows_:h->npcols_)>1;
    const int ldw=(side=='L'?std::max(k,1):std::max(rows,1))+(empty?0:3);
    const std::size_t elements=empty?0:std::size_t(ldw)*(side=='L'?cols:k);
    std::size_t bytes=0;
    MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid) {
        workspace_test::Scope scope(true);detail::punmtr_workspace_size<T>(h,side,rows,cols,active_k,ldw,reduce,&bytes);const auto s=scope.finish();
        pure=no_allocation(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;
    }
    MPI_Barrier(h->comm);require_close(h,"punmtr one-rank query purity",pure?0:1,0);
    detail::punmtr_workspace_size<T>(h,side,rows,cols,active_k,ldw,reduce,&bytes);
    constexpr std::size_t canary=256;
    ddla::vector<unsigned char> scratch(h,bytes+2*canary);
    DDLA_CHECK(ddlaMemset(h,scratch.data(),0x6b,scratch.size()));
    char* next=reinterpret_cast<char*>(scratch.data()+canary);
    T* work=nullptr;T* reduced=nullptr;
    if(!empty){work=detail::take<T>(next,elements);if(reduce)reduced=detail::take<T>(next,elements);}
    require_close(h,"punmtr workspace carving matches size",std::size_t(next-reinterpret_cast<char*>(scratch.data()+canary))==bytes?0:1,0);
    // Warm vendor and host-staged transport paths before measuring reusable run.
    detail::punmtr_run<B,T>(h,side,uplo,trans,rows,cols,active_k,ldv,ldt,dc[DDLA_LLD_],ldw,reduce,V.ptr,Tmat.ptr,C.ptr,work,reduced);check_ddla_sync(h);
    for(int pass=0;pass<2;++pass) {
        upload(h,C.ptr,original);check_ddla_sync(h);
        workspace_test::Scope scope;
        detail::punmtr_run<B,T>(h,side,uplo,trans,rows,cols,active_k,ldv,ldt,dc[DDLA_LLD_],ldw,reduce,V.ptr,Tmat.ptr,C.ptr,work,reduced);
        const auto s=scope.finish();
        require_close(h,"punmtr no-check run has no scratch/query/setup",no_allocation(s)?0:1,0);
        if(workspace_test::available())require_close(h,"punmtr run only needs its data reduction",
            std::abs(double(s.collectives)-(reduce&&!empty?1:0)),0);
        check_ddla_sync(h);verify();
    }
    std::array<unsigned char,2*canary> guards{};
    DDLA_CHECK(ddlaMemcpy(h,guards.data(),scratch.data(),canary));
    DDLA_CHECK(ddlaMemcpy(h,guards.data()+canary,scratch.data()+canary+bytes,canary));check_ddla_sync(h);
    require_close(h,"punmtr exact workspace canaries",std::all_of(guards.begin(),guards.end(),[](unsigned char x){return x==0x6b;})?0:1,0);
}

template<DdlaBackend B>void rejections(DdlaHandle_t h) {
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,9,9,3,3,0,0));
    DeviceBuffer<double> buffer(h,512);int pr=h->nprows_,pc=h->npcols_;
    const int nr=num_loc(7,3,h->myprow_,0,pr),nc=num_loc(5,3,h->mypcol_,0,pc);
    auto reject=[&](char side,char uplo,char trans,int m,int n,int k,int ldv,int ldt,const double* v,const double* t,double* c) {
        bool caught=false;try{punmtr<B,double>(h,side,uplo,trans,m,n,k,v,ldv,t,ldt,c,d);}
        catch(const std::exception&){caught=true;}
        require_close(h,"punmtr invalid arguments rejected collectively",caught?0:1,0);
    };
    auto* x=buffer.ptr;
    reject('X','U','N',7,5,2,9,2,x,x+128,x+256);
    reject('L','X','N',7,5,2,9,2,x,x+128,x+256);
    reject('L','U','X',7,5,2,9,2,x,x+128,x+256);
    reject('L','U','N',-1,5,2,9,2,x,x+128,x+256);
    reject('L','U','N',7,10,2,9,2,x,x+128,x+256);
    reject('L','U','N',7,5,-1,9,2,x,x+128,x+256);
    reject('L','U','N',7,5,INT_MAX/2+1,9,INT_MAX/2+1,x,x+128,x+256);
    reject('L','U','N',7,5,2,h->myid==0?0:9,2,x,x+128,x+256);
    reject('L','U','N',7,5,2,9,h->myid==0?1:2,x,x+128,x+256);
    reject('L','U','N',7,5,2,9,2,nr>0&&h->myid==0?nullptr:x,x+128,x+256);
    reject('L','U','N',7,5,2,9,2,x,h->myid==0?nullptr:x+128,x+256);
    reject('L','U','N',7,5,2,9,2,x,x+128,nr>0&&nc>0&&h->myid==0?nullptr:x+256);
    if(h->nprocs>1)reject('L','U','N',h->myid==0?0:7,5,2,9,2,x,x+128,x+256);
    // Identity calls permit all data pointers and unused V/T strides to be zero.
    punmtr<B,double>(h,'l','u','n',7,5,0,nullptr,0,nullptr,0,nullptr,d);
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    constexpr auto other=B==DdlaBackend::CPU?DdlaBackend::GPU:DdlaBackend::CPU;
    bool caught=false;
    try{punmtr<other,double>(h,'L','U','N',0,0,0,nullptr,0,nullptr,0,nullptr,d);}
    catch(const std::exception&){caught=true;}
    require_close(h,"punmtr backend mismatch",caught?0:1,0);
#endif
}
template<DdlaBackend B,class T>void cases(DdlaHandle_t h) {
    for(char side:{'L','R'})for(char uplo:{'U','L'})for(char trans:{'N','C'})
        exercise<B,T>(h,side,uplo,trans,17,13,5);
    for(char side:{'L','R'})for(auto shape:{std::array<int,3>{0,7,3},{6,0,3},{5,7,0},{1,1,1},{3,2,2}})
        exercise<B,T>(h,side,'U','C',shape[0],shape[1],shape[2]);
    if constexpr(std::is_same<T,typename real_type<T>::type>::value)
        exercise<B,T>(h,'R','L','T',7,9,3);
    else {
        int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,1,1,1,1,0,0));bool caught=false;
        try{punmtr<B,T>(h,'L','U','T',1,1,1,nullptr,1,nullptr,1,nullptr,d);}
        catch(const std::exception&){caught=true;}
        require_close(h,"punmtr complex transpose rejected",caught?0:1,0);
    }
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major) {
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    cases<B,float>(h);cases<B,double>(h);cases<B,std::complex<float>>(h);cases<B,std::complex<double>>(h);
    rejections<B>(h);
    if(!h->myid)std::cout<<"punmtr PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
} // namespace
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}) {
#if DDLA_HAS_CPU
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
