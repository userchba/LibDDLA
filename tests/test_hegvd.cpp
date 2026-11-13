#include "generalized_eigen_test_common.h"
#include <cstring>
using namespace api_grid_test;
namespace {
template<class T>void cases(DdlaHandle_t h) {
    using R=typename real_type<T>::type;
    const int rank=ddlaGetRank(h);
    for(int size:{0,1,9,17,33,65})for(char uplo:{'U','L'})for(char job:{'N','V'}) {
        const int n=size?size+rank:0,lda=std::max(1,n+3),ldb=std::max(1,n+5);
        const T sentinel=T(12345),nan=T(std::numeric_limits<R>::quiet_NaN());
        std::vector<T> initial_a(std::size_t(lda)*(n+2),sentinel),initial_b(std::size_t(ldb)*(n+2),sentinel);
        for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
            const bool stored=uplo=='U'?i<=j:i>=j;
            initial_a[i+std::size_t(j)*lda]=stored?generalized_test::a<T>(i,j,n):nan;
            initial_b[i+std::size_t(j)*ldb]=stored?generalized_test::b<T>(i,j,n):nan;
        }
        DeviceBuffer<T> a(h,initial_a.size()),b(h,initial_b.size());
        DeviceBuffer<R> values(h,n+2);DeviceBuffer<int> status(h,3);
        upload(h,a.ptr,initial_a);upload(h,b.ptr,initial_b);
        upload(h,values.ptr,std::vector<R>(n+2,R(6789)));upload(h,status.ptr,std::vector<int>(3,9182));
        if constexpr(std::is_same_v<T,R>)sygvd<T>(h,job,uplo,n,n?a.ptr:nullptr,lda,n?b.ptr:nullptr,ldb,n?values.ptr+1:nullptr,status.ptr+1);
        else hegvd<T>(h,job,uplo,n,n?a.ptr:nullptr,lda,n?b.ptr:nullptr,ldb,n?values.ptr+1:nullptr,status.ptr+1);
        const auto out=download(h,a.ptr,initial_a.size()),bout=download(h,b.ptr,initial_b.size());
        const auto w=download(h,values.ptr,n+2);const auto info=download(h,status.ptr,3);
        bool okay=info[0]==9182&&info[1]==0&&info[2]==9182&&w.front()==R(6789)&&w.back()==R(6789);
        for(int j=0;j<n+2;++j) {
            for(int i=0;i<lda;++i)if(i>=n||j>=n)
                okay=okay&&!std::memcmp(&out[i+std::size_t(j)*lda],&initial_a[i+std::size_t(j)*lda],sizeof(T));
            for(int i=0;i<ldb;++i)if(i>=n||j>=n)
                okay=okay&&!std::memcmp(&bout[i+std::size_t(j)*ldb],&initial_b[i+std::size_t(j)*ldb],sizeof(T));
        }
        require_close(h,"local generalized info/eigenvalue/padding guards",okay?0:1,0);
        std::vector<T> z(std::size_t(n)*(job=='V'?n:0));
        if(job=='V')for(int j=0;j<n;++j)for(int i=0;i<n;++i)z[i+std::size_t(j)*n]=out[i+std::size_t(j)*lda];
        generalized_test::verify<T>(h,n,job=='V'?n:0,w.data()+1,z);
    }
    const int n=3;
    DeviceBuffer<T> a(h,n*n),b(h,n*n);DeviceBuffer<R> w(h,n);DeviceBuffer<int> status(h,1);
    std::vector<T> input(n*n,T(0)),bad(n*n,T(0));
    for(int i=0;i<n;++i){input[i+i*n]=T(i+1);bad[i+i*n]=T(i?1:-1);}
    upload(h,a.ptr,input);upload(h,b.ptr,bad);
    auto call=[&](T* bb,char job) {
        if constexpr(std::is_same_v<T,R>)sygvd<T>(h,job,'U',n,a.ptr,n,bb,n,w.ptr,status.ptr);
        else hegvd<T>(h,job,'U',n,a.ptr,n,bb,n,w.ptr,status.ptr);
    };
    call(b.ptr,'V');require_close(h,"local non-positive B info=n+i",download(h,status.ptr,1)[0]==n+1?0:1,0);
    for(bool alias:{false,true}) {
        bool rejected=false;
        try{call(alias?a.ptr:b.ptr,alias?'V':'?');}catch(const std::invalid_argument&){rejected=true;}
        require_close(h,"local generalized invalid arguments rejected",rejected?0:1,0);
    }
}
void body(const DdlaHandle_t& h,const Shape&) {
    cases<float>(h);cases<double>(h);cases<std::complex<float>>(h);cases<std::complex<double>>(h);
}
}
int main(int argc,char** argv){return run_grid_test(argc,argv,"test_hegvd",body);}
