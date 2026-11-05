#include "api_grid_test_common.h"
#include "pblas_structured_test_common.h"
#include <cstring>
#include <limits>

using namespace ddla;
using namespace structured_test;
namespace {
template <DdlaBackend Backend,typename T>
void check_case(DdlaHandle_t h,Operation op,char side,char uplo,char trans,
    int m,int n,int k,int nb,int mode,bool shifted,int& cases,double& worst) {
    const bool sym=op==Operation::Symm,herm=hermitian(op);
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    const int ar=sym ? (side=='L' ? m : n) : trans=='N' ? n : k;
    const int ac=sym ? ar : trans=='N' ? k : n;
    const int br=sym ? m : ar,bc=sym ? n : ac;
    const int cr=sym ? m : n;
    Matrix<T> a(h,ar,ac,nb,shifted ? pr-1 : 0,shifted ? pc-1 : 0,true);
    Matrix<T> b(h,br,bc,nb,0,shifted ? pc-1 : 0,true);
    Matrix<T> c(h,cr,n,nb,shifted && pr>2 ? 1 : 0,shifted && pc>2 ? 1 : 0,true);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const T alpha=(mode>=2 && mode<=4) ? T(0) : scalar<T>(-.75,op==Operation::Herk ? 0 : .25);
    const T beta=(mode==1 || mode==3) ? T(0) : mode==4 ? T(1) : scalar<T>(.5,herm ? 0 : -.25);
    a.fill(ar,ac,[&](int x,int y) {return sym && (uplo=='U' ? x>y : x<y) ? scalar<T>(nan,nan) : value<T>(x,y,1);});
    b.fill(br,bc,[](int x,int y){return value<T>(x,y,2);});
    c.fill(cr,n,[&](int x,int y) {
        if(!sym && (uplo=='U' ? x>y : x<y))return scalar<T>(-789.5,456.25);
        if(beta==T(0))return scalar<T>(nan,nan);
        if(herm && x==y)return scalar<T>(wide(value<T>(x,y,3)).real(),nan);
        return value<T>(x,y,3);
    });
    a.upload();b.upload();c.upload();
    const bool empty=cr==0 || n==0,product=alpha!=T(0) && (sym || k>0);
    invoke<Backend,T>(h,op,mode==5 ? char(std::tolower(side)) : side,
        mode==5 ? char(std::tolower(uplo)) : uplo,mode==5 ? char(std::tolower(trans)) : trans,
        m,n,k,alpha,product ? a.data.data() : nullptr,a.d,
        product && two_inputs(op) ? b.data.data() : nullptr,b.d,beta,
        empty ? nullptr : c.data.data(),c.d);
    const auto out=c.download(),aa=a.download(),bb=b.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T)) || std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T));
    double error=0;
    for(std::size_t p=0;p<out.size();++p) {
        const int i=p%c.d[DDLA_LLD_],j=p/c.d[DDLA_LLD_];
        const bool valid=i<c.nr && j<c.nc && c.gi(i)<cr && c.gj(j)<n;
        const bool selected=valid && (sym || (uplo=='U' ? c.gi(i)<=c.gj(j) : c.gi(i)>=c.gj(j)));
        if(selected) {
            const auto ref=reference<T>(op,side,uplo,trans,c.gi(i),c.gj(j),m,n,k,alpha,beta);
            const double e=std::abs(wide(out[p])-ref.first)/(1+ref.second);
            if(!std::isfinite(e))bad=1;else error=std::max(error,e);
            if(herm && c.gi(i)==c.gj(j) && wide(out[p]).imag()!=0)bad=1;
        } else if(std::memcmp(&out[p],&c.host[p],sizeof(T)))bad=1;
    }
    int any;double global;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&error,&global,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<typename real_type<T>::type,float>;
    if(any || global>(single ? 3e-6 : 5e-14)) {
        if(ddlaGetRank(h)==0)std::cerr<<"STRUCTURED_FAIL op="<<name(op)<<" type="<<type_name<T>()<<" flags="<<side<<uplo<<trans<<" m="<<m<<" n="<<n<<" k="<<k<<" nb="<<nb<<" mode="<<mode<<" shifted="<<shifted<<" guard="<<any<<" error="<<global<<std::endl;
        throw std::runtime_error("structured PBLAS numerical/input/triangle/padding check failed");
    }
    ++cases;worst=std::max(worst,global);
}

template <DdlaBackend Backend,typename T>
void type_cases(DdlaHandle_t h,int& cases,double& worst) {
    std::vector<Operation> ops={Operation::Symm,Operation::Syrk,Operation::Syr2k};
    if constexpr(!std::is_arithmetic_v<T>) {ops.push_back(Operation::Herk);ops.push_back(Operation::Her2k);}
    for(auto op:ops) for(char uplo:{'U','L'}) {
        const std::vector<char> flags=op==Operation::Symm ? std::vector<char>{'L','R'} : std::vector<char>{'N',hermitian(op) ? 'C' : 'T'};
        for(char flag:flags) {
            const char side=op==Operation::Symm ? flag : 'L',trans=op==Operation::Symm ? 'N' : flag;
            check_case<Backend,T>(h,op,side,uplo,trans,13,9,7,4,0,true,cases,worst);
            check_case<Backend,T>(h,op,side,uplo,trans,13,9,7,4,0,false,cases,worst);
            if constexpr(std::is_arithmetic_v<T>) {
                if(op!=Operation::Symm && trans=='T')
                    check_case<Backend,T>(h,op,side,uplo,'C',13,9,7,4,0,true,cases,worst);
            }
            check_case<Backend,T>(h,op,side,uplo,trans,13,9,7,4,1,true,cases,worst);
            check_case<Backend,T>(h,op,side,uplo,trans,1,2,1,4,0,true,cases,worst);
            check_case<Backend,T>(h,op,side,uplo,trans,7,5,3,32,1,false,cases,worst);
            check_case<Backend,T>(h,op,side,uplo,trans,8,12,9,1,5,true,cases,worst);
            for(int mode:{2,3,4})check_case<Backend,T>(h,op,side,uplo,trans,13,9,7,4,mode,true,cases,worst);
            if(op!=Operation::Symm) {
                check_case<Backend,T>(h,op,side,uplo,trans,17,17,1031,32,0,true,cases,worst);
                check_case<Backend,T>(h,op,side,uplo,trans,0,9,0,4,0,true,cases,worst);
                check_case<Backend,T>(h,op,side,uplo,trans,0,9,0,4,1,true,cases,worst);
            }
            check_case<Backend,T>(h,op,side,uplo,trans,13,0,7,4,0,true,cases,worst);
            if(op==Operation::Symm)check_case<Backend,T>(h,op,side,uplo,trans,0,9,7,4,0,true,cases,worst);
        }
    }
    if constexpr (Backend==DdlaBackend::GPU && std::is_arithmetic_v<T>) {
        // Exercise the batched real-GPU branch above its small-matrix cutoff,
        // including partial output tiles and both normalized panel layouts.
        check_case<Backend,T>(h,Operation::Syrk,'L','U','N',0,1057,7,128,1,true,cases,worst);
        check_case<Backend,T>(h,Operation::Syr2k,'L','L','T',0,1057,7,128,0,true,cases,worst);
    }

}

template <DdlaBackend Backend>
void invalid_cases(DdlaHandle_t h) {
    int pr,pc;ddlaGetGridDims(h,pr,pc);const int n=4*std::max(pr,pc)+1;
    using T=std::complex<double>;
    Matrix<T> a(h,n,n,4),b(h,n,n,4),c(h,n,n,4);
    const bool root=ddlaGetRank(h)==0;
    auto reject=[&](auto call) {
        int caught=0;try{call();}catch(const std::exception&){caught=1;}
        int all;MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all)throw std::runtime_error("structured PBLAS invalid argument not rejected on every rank");
    };
    for(auto op:{Operation::Symm,Operation::Syrk,Operation::Syr2k,Operation::Herk,Operation::Her2k}) {
        auto call=[&](char side,char u,char tr,int nn,int kk,const T* ap,const T* bp,T* cp,T alpha=T(1)) {
            invoke<Backend,T>(h,op,side,u,tr,nn,nn,kk,alpha,ap,a.d,bp,b.d,T(0),cp,c.d);
        };
        reject([&]{call('L','?','N',n,n,a.data.data(),b.data.data(),c.data.data());});
        reject([&]{call('L','U','N',-1,n,a.data.data(),b.data.data(),c.data.data());});
        reject([&]{call('L','U','N',n+1,n,a.data.data(),b.data.data(),c.data.data());});
        if(op==Operation::Symm)reject([&]{call('?','U','N',n,n,a.data.data(),b.data.data(),c.data.data());});
        else {
            reject([&]{call('L','U','?',n,n,a.data.data(),b.data.data(),c.data.data());});
            reject([&]{call('L','U',hermitian(op) ? 'T' : 'C',n,n,a.data.data(),b.data.data(),c.data.data());});
            reject([&]{call('L','U','N',n,-1,a.data.data(),b.data.data(),c.data.data());});
        }
        reject([&]{call('L','U','N',n,n,root ? nullptr : a.data.data(),b.data.data(),c.data.data());});
        if(two_inputs(op))reject([&]{call('L','U','N',n,n,a.data.data(),root ? nullptr : b.data.data(),c.data.data());});
        reject([&]{call('L','U','N',n,n,a.data.data(),b.data.data(),root ? nullptr : c.data.data());});
        reject([&]{call('L','U','N',n,n,a.data.data(),b.data.data(),root ? a.data.data() : c.data.data());});
        if(pr*pc>1) {
            reject([&]{call('L',root ? 'U' : 'L','N',n,n,a.data.data(),b.data.data(),c.data.data());});
            reject([&]{call('L','U','N',n,n,a.data.data(),b.data.data(),c.data.data(),root ? T(0) : T(1));});
        }
    }
}

template <DdlaBackend Backend>
void run(MPI_Comm comm,int pr,int pc,char major) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,comm,pr,pc,major));
    int cases=0;double worst=0;
    type_cases<Backend,float>(h,cases,worst);type_cases<Backend,double>(h,cases,worst);
    type_cases<Backend,std::complex<float>>(h,cases,worst);type_cases<Backend,std::complex<double>>(h,cases,worst);
    invalid_cases<Backend>(h);
    if(ddlaGetRank(h)==0)std::cout<<"STRUCTURED_PBLAS_PASS backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc<<" major="<<major<<" cases="<<cases<<" worst="<<worst<<std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);int size,rank;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    api_grid_test::TestOptions opts;std::string error;
    if(!api_grid_test::parse_options(argc,argv,size,opts,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Comm reversed;MPI_Comm_split(MPI_COMM_WORLD,0,size-rank,&reversed);
    try {
        for(auto grid:opts.grids)for(char major:{'R','C'}) {
            run<DdlaBackend::CPU>(reversed,grid.first,grid.second,major);
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(reversed,grid.first,grid.second,major);
#endif
        }
    }catch(const std::exception& e){std::cerr<<"STRUCTURED_PBLAS_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Comm_free(&reversed);MPI_Finalize();return 0;
}
