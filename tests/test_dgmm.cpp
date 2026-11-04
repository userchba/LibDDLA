#include "matrix_test_common.h"
#include <ddla/ddla_vector.h>
#include "BLAS/blas_impl.h"
#include "PBLAS/pdgmm_impl.h"
#include <climits>
#include <functional>

namespace {
void require_all(DdlaHandle_t h, bool good, const char* message)
{
    int bad=good ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if (bad) throw std::runtime_error(message);
}

template <typename U>
std::vector<U> diagonal_values(int length, int incx)
{
    const int step=std::abs(incx);
    std::vector<U> values(std::max(1,1+(length-1)*step),scalar<U>(-23,7));
    for (int i=0;i<length;++i) {
        const int offset=incx<0 ? (length-1-i)*step : i*step;
        values[offset]=value<U>(incx==0 ? 0 : i,0,7);
    }
    return values;
}

template <typename T, typename U>
void verify(DdlaHandle_t h, const std::vector<T>& actual, const std::vector<T>& seed,
    int m, int n, int lda, const std::vector<U>& diagonal, int incx, char side,
    int mb=1, int nb=1, int row=0, int column=0, int rs=0, int cs=0, int pr=1, int pc=1)
{
    bool good=true;
    const int length=side=='L' ? m : n;
    for (std::size_t offset=0;offset<actual.size();++offset) {
        const int local_i=offset%lda, local_j=offset/lda;
        const int i=indxl2g(local_i,mb,row,rs,pr);
        const int j=indxl2g(local_j,nb,column,cs,pc);
        // The local row index may be padding beyond this rank's logical rows.
        const bool active=local_i<num_loc(m,mb,row,rs,pr) &&
            local_j<num_loc(n,nb,column,cs,pc);
        if (active) {
            const int logical=side=='L' ? i : j;
            const int x_index=incx<0 ? (length-1-logical)*-incx : logical*incx;
            const auto expected=wide(value<T>(i,j,1))*wide(diagonal[x_index]);
            const double error=std::abs(wide(actual[offset])-expected);
            using R=typename real_type<T>::type;
            good=good && std::isfinite(error) && error<=8*std::numeric_limits<R>::epsilon()*std::max(1.,std::abs(expected));
        } else {
            good=good && std::memcmp(&actual[offset],&seed[offset],sizeof(T))==0;
        }
    }
    require_all(h,good,"DGMM scalar reference or padding check failed");
}

template <typename T, typename U>
void verify_inputs(DdlaHandle_t h, const T* A, const std::vector<T>& saved,
    const U* x, const std::vector<U>& diagonal)
{
    std::vector<T> actual(saved.size());
    std::vector<U> actual_x(diagonal.size());
    DDLA_CHECK(ddlaMemcpy(h,actual.data(),A,actual.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,actual_x.data(),x,actual_x.size()*sizeof(U)));
    DDLA_CHECK(ddlaSynchronize(h));
    require_all(h,std::memcmp(actual.data(),saved.data(),actual.size()*sizeof(T))==0 &&
        std::memcmp(actual_x.data(),diagonal.data(),actual_x.size()*sizeof(U))==0,
        "DGMM modified a read-only input");
}

template <DdlaBackend B, typename T, typename U>
void local_case(DdlaHandle_t h, char side, int m, int n, int incx, bool inplace)
{
    const int lda=std::max(1,m)+3,ldc=inplace ? lda : lda+2;
    std::vector<T> input(std::size_t(lda)*(n+2),scalar<T>(-17,5));
    std::vector<T> seed(std::size_t(ldc)*(n+2),scalar<T>(-29,3));
    for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
        input[i+std::size_t(j)*lda]=value<T>(i,j,1);
        seed[i+std::size_t(j)*ldc]=scalar<T>(std::numeric_limits<double>::quiet_NaN());
    }
    auto diagonal=diagonal_values<U>(side=='L' ? m : n,incx);
    ddla::vector<T> A(h,input.size()),C(h,seed.size());
    ddla::vector<U> x(h,diagonal.size());
    DDLA_CHECK(ddlaMemcpy(h,x.data(),diagonal.data(),diagonal.size()*sizeof(U)));
    for (int mode=0;mode<2;++mode) {
        DDLA_CHECK(ddlaMemcpy(h,A.data(),input.data(),input.size()*sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h,C.data(),seed.data(),seed.size()*sizeof(T)));
        T* output=inplace ? A.data() : C.data();
        if (mode==0) dgmm<B,T,U>(h,side,m,n,A.data(),lda,x.data(),incx,output,ldc);
        else blas_impl::dgmm_run<B,T,U>(h,side,m,n,A.data(),lda,x.data(),incx,output,ldc);
        std::vector<T> actual(inplace ? input.size() : seed.size());
        DDLA_CHECK(ddlaMemcpy(h,actual.data(),output,actual.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        verify(h,actual,inplace ? input : seed,m,n,ldc,diagonal,incx,side);
        if (!inplace) verify_inputs(h,A.data(),input,x.data(),diagonal);
        else verify_inputs(h,x.data(),diagonal,x.data(),diagonal);
    }
}

template <DdlaBackend B, typename T, typename U>
void distributed_case(DdlaHandle_t h, char side, int m, int n, int incx, bool inplace, int rs, int cs)
{
    Matrix<T> A(h,m+2,n+1,2,3,rs,cs,1,false);
    Matrix<T> C(h,m,n,2,3,rs,cs,2,true);
    auto diagonal=diagonal_values<U>(side=='L' ? m : n,incx);
    ddla::vector<U> x(h,diagonal.size());
    DDLA_CHECK(ddlaMemcpy(h,x.data(),diagonal.data(),diagonal.size()*sizeof(U)));
    const int rows=num_loc(m,2,A.r,rs,A.pr),columns=num_loc(n,3,A.c,cs,A.pc);
    const bool active=rows>0 && columns>0;
    for (int mode=0;mode<2;++mode) {
        A.reset(); C.reset();
        T* output=inplace ? A.data : C.data;
        const int* desc=inplace ? A.d : C.d;
        if (mode==0) pdgmm<B,T,U>(h,side,m,n,active ? A.data : nullptr,A.d,
            active ? x.data() : nullptr,incx,active ? output : nullptr,desc);
        else pdgmm_impl::pdgmm_run<B,T,U>(h,side,m,n,active ? A.data : nullptr,A.d,
            active ? x.data() : nullptr,incx,active ? output : nullptr,desc);
        auto actual=inplace ? A.download() : C.download();
        verify(h,actual,inplace ? A.host : C.host,m,n,desc[DDLA_LLD_],diagonal,incx,side,
            2,3,A.r,A.c,rs,cs,A.pr,A.pc);
        if (!inplace) verify_inputs(h,A.data,A.host,x.data(),diagonal);
        else verify_inputs(h,x.data(),diagonal,x.data(),diagonal);
    }
}

template <typename Call>
void rejected(DdlaHandle_t h, Call call)
{
    bool caught=false;
    try {call();} catch (const std::exception&) {caught=true;}
    require_all(h,caught,"DGMM invalid arguments were accepted on some rank");
}

template <DdlaBackend B>
void invalid_cases(DdlaHandle_t h)
{
    ddla::vector<double> A(h,128),x(h,128),C(h,128);
    rejected(h,[&]{dgmm<B>(h,'X',3,4,A.data(),3,x.data(),1,C.data(),3);});
    rejected(h,[&]{dgmm<B>(h,'L',-1,4,A.data(),3,x.data(),1,C.data(),3);});
    rejected(h,[&]{dgmm<B>(h,'L',3,4,A.data(),2,x.data(),1,C.data(),3);});
    rejected(h,[&]{dgmm<B,double,double>(h,'L',3,4,nullptr,3,x.data(),1,C.data(),3);});
    rejected(h,[&]{dgmm<B,double,double>(h,'L',3,4,A.data(),3,x.data(),1,nullptr,3);});
    rejected(h,[&]{dgmm<B>(h,'L',3,4,A.data(),3,x.data(),1,A.data(),4);});
    rejected(h,[&]{dgmm<B>(h,'R',3,4,A.data(),3,x.data(),1,A.data()+1,3);});
    rejected(h,[&]{dgmm<B>(h,'R',3,4,A.data(),3,C.data(),1,C.data(),3);});
    rejected(h,[&]{dgmm<B>(h,'L',INT_MAX,1,A.data(),INT_MAX,x.data(),INT_MIN,C.data(),INT_MAX);});
    dgmm<B,double,double>(h,'l',0,4,nullptr,1,nullptr,0,nullptr,1);
    dgmm<B,double,double>(h,'r',3,0,nullptr,3,nullptr,-2,nullptr,3);
    Matrix<double> a(h,7,11,2,3,0,0,1,false),c(h,7,11,2,3,0,0,2,true);
    int bad[DDLA_DLEN_]; std::copy(c.d,c.d+DDLA_DLEN_,bad);
    if (ddlaGetRank(h)==0) bad[DDLA_MB_]=0;
    rejected(h,[&]{pdgmm<B>(h,'L',7,11,a.data,a.d,x.data(),1,c.data,bad);});
    rejected(h,[&]{pdgmm<B,double,double>(h,'L',7,11,a.data,a.d,
        ddlaGetRank(h)==0 ? nullptr : x.data(),1,c.data,c.d);});
    if (ddlaGetSize(h)>1) {
        rejected(h,[&]{pdgmm<B>(h,'L',7,ddlaGetRank(h)==0 ? 10 : 11,a.data,a.d,x.data(),1,c.data,c.d);});
        rejected(h,[&]{pdgmm<B>(h,ddlaGetRank(h)==0 ? 'L' : 'R',7,11,a.data,a.d,x.data(),1,c.data,c.d);});
    }
}

template <DdlaBackend B, typename T, typename U>
void scalar_cases(DdlaHandle_t h)
{
    int pr=0,pc=0;ddlaGetGridDims(h,pr,pc);
    for (char side : {'L','R'}) for (int incx : {0,1,2,-1,-2}) for (bool inplace : {false,true}) {
        local_case<B,T,U>(h,side,5,7,incx,inplace);
        local_case<B,T,U>(h,side,1,1,incx,inplace);
        distributed_case<B,T,U>(h,side,7,11,incx,inplace,pr-1,pc-1);
        distributed_case<B,T,U>(h,side,1,1,incx,inplace,0,0);
    }
    for (char side : {'L','R'}) {
        local_case<B,T,U>(h,side,0,7,1,false);
        local_case<B,T,U>(h,side,5,0,-1,false);
        distributed_case<B,T,U>(h,side,0,11,1,false,0,0);
        distributed_case<B,T,U>(h,side,7,0,-1,false,0,0);
    }
}

template <DdlaBackend B>
void run(int pr, int pc)
{
    DdlaHandle_t h=nullptr;
    DDLA_CHECK(ddlaInit(h,B)); DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    scalar_cases<B,float,float>(h);
    scalar_cases<B,double,double>(h);
    scalar_cases<B,std::complex<float>,std::complex<float>>(h);
    scalar_cases<B,std::complex<double>,std::complex<double>>(h);
    scalar_cases<B,std::complex<float>,float>(h);
    scalar_cases<B,std::complex<double>,double>(h);
    invalid_cases<B>(h);
    if (ddlaGetRank(h)==0) std::cout<<"DGMM_PASS backend="<<int(B)<<" grid="<<pr<<'x'<<pc<<" type_pairs=6"<<std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    try {
        int size=1; MPI_Comm_size(MPI_COMM_WORLD,&size);
        const int pr=argc>2 ? std::stoi(argv[2]) : 1;
        const int pc=argc>3 ? std::stoi(argv[3]) : size;
        if (pr*pc!=size) throw std::invalid_argument("test_dgmm grid does not match rank count");
        const bool gpu=argc>1 && std::string(argv[1])=="gpu";
        if (gpu) {
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(pr,pc);
#else
            throw std::runtime_error("GPU unavailable");
#endif
        } else run<DdlaBackend::CPU>(pr,pc);
        MPI_Finalize(); return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<std::endl; MPI_Abort(MPI_COMM_WORLD,1); return 1;
    }
}
