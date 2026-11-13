#include "api_grid_test_common.h"

#include <complex>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace api_grid_test;
namespace {
template<class T> constexpr bool complex_type = !std::is_same<T, typename ddla::real_type<T>::type>::value;

template<class T> T value(int r, int c)
{
    using R = typename ddla::real_type<T>::type;
    const R real = R(2 * r + 3 * c + 1) / R(8);
    if constexpr (complex_type<T>) {
        // A symmetric diagonal must retain even the NaN payload in its imaginary part.
        const R imag = r == c ? std::numeric_limits<R>::quiet_NaN() : R(r - 5 * c - 2) / R(16);
        return T(real, imag);
    } else return real;
}
template<class T> T poison()
{
    using R = typename ddla::real_type<T>::type;
    const R q = std::numeric_limits<R>::quiet_NaN();
    if constexpr (complex_type<T>) return T(q, q);
    else return q;
}
template<class T> T expected(int r, int c, bool upper, bool hermitian)
{
    if (r == c) {
        const T x = value<T>(r, c);
        if (hermitian) return T(std::real(x));
        return x;
    }
    const bool stored = upper ? r < c : r > c;
    const T x = stored ? value<T>(r, c) : value<T>(c, r);
    if constexpr (complex_type<T>) return !stored && hermitian ? std::conj(x) : x;
    else return x;
}
template<class T> bool same(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template<class T> void local(ddla::DdlaHandle_t h, int n, bool upper, bool hermitian)
{
    constexpr int guard = 7;
    const int lda = n + 5, columns = n + 2;
    std::vector<T> input(guard + std::size_t(lda) * columns + guard, T(-713));
    for (int c = 0; c < n; ++c) for (int r = 0; r < n; ++r)
        input[guard + r + std::size_t(c) * lda] = (upper ? r <= c : r >= c) ? value<T>(r,c) : poison<T>();
    DeviceBuffer<T> a(h, input.size());
    upload(h, a.ptr, input);
    // No explicit synchronization: completion must follow the upload on the handle stream.
    if (hermitian) ddla::herm_complete(h, upper ? 'U' : 'L', n, a.ptr + guard, lda);
    else ddla::syrm_complete(h, upper ? 'U' : 'L', n, a.ptr + guard, lda);
    const auto got = download(h, a.ptr, input.size());
    double bad = 0;
    for (std::size_t k = 0; k < input.size(); ++k) {
        T want = input[k];
        if (k >= guard && k < guard + std::size_t(lda) * columns) {
            const int r = int((k - guard) % lda), c = int((k - guard) / lda);
            if (r < n && c < n) want = expected<T>(r,c,upper,hermitian);
        }
        if (!same(got[k], want)) ++bad;
    }
    require_close(h, "local completion exact values/diagonal/guards", bad, 0);
}

template<class T> void distributed(ddla::DdlaHandle_t h, int n, int nb, bool upper, bool hermitian)
{
    int d[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddla::ddlaDescInit(d,h,n+3,n+5,nb,nb,0,0));
    d[ddla::DDLA_LLD_] += 3;
    int pr,pc,mr,mc; ddla::ddlaGetGridDims(h,pr,pc); ddla::ddlaGetGridCoords(h,mr,mc);
    const int cols = ddla_test::n_loc(h,d), rows = ddla_test::m_loc(h,d), lda=d[ddla::DDLA_LLD_];
    std::vector<T> input(local_size(h,d) + 8, T(-917));
    for (int c=0;c<cols;++c) for (int r=0;r<rows;++r) {
        const int gr=ddla::indxl2g(r,nb,mr,0,pr),gc=ddla::indxl2g(c,nb,mc,0,pc);
        if (gr<n && gc<n)
            input[r+std::size_t(c)*lda]=(upper?gr<=gc:gr>=gc)?value<T>(gr,gc):poison<T>();
    }
    DeviceBuffer<T> a(h,input.size()); upload(h,a.ptr,input);
    if (hermitian) ddla::pherm_complete(h,upper?'U':'L',n,a.ptr,d);
    else ddla::psyrm_complete(h,upper?'U':'L',n,a.ptr,d);
    const auto got=download(h,a.ptr,input.size());
    double bad=0;
    for(std::size_t k=0;k<input.size();++k) {
        T want=input[k];
        const int r=int(k%lda),c=int(k/lda);
        if(r<rows && c<cols) {
            const int gr=ddla::indxl2g(r,nb,mr,0,pr),gc=ddla::indxl2g(c,nb,mc,0,pc);
            if(gr<n && gc<n)want=expected<T>(gr,gc,upper,hermitian);
        }
        if(!same(got[k],want))++bad;
    }
    require_close(h,"distributed completion exact values/diagonal/parent/LLD",bad,0);
}

template<class F> void rejected(ddla::DdlaHandle_t h, const F& f)
{
    bool caught=false;try{f();}catch(const std::exception&){caught=true;}
    require_close(h,"invalid completion rejected",caught?0:1,0);
}
void invalid(ddla::DdlaHandle_t h)
{
    for(bool hermitian:{false,true}) {
        auto call=[&](ddla::DdlaHandle_t handle,char u,int n,double* a,int ld) {
            if(hermitian)ddla::herm_complete(handle,u,n,a,ld);
            else ddla::syrm_complete(handle,u,n,a,ld);
        };
        call(h,'U',0,nullptr,0);
        rejected(h,[&]{call(nullptr,'U',0,nullptr,0);});
        rejected(h,[&]{call(h,'X',0,nullptr,0);});
        rejected(h,[&]{call(h,'U',-1,nullptr,0);});
        rejected(h,[&]{call(h,'L',1,nullptr,1);});
        DeviceBuffer<double>a(h,4);
        rejected(h,[&]{call(h,'L',2,a.ptr,1);});
    }
    int d[ddla::DDLA_DLEN_];DDLA_CHECK(ddla::ddlaDescInit(d,h,3,3,2,2,0,0));
    DeviceBuffer<double>a(h,local_size(h,d));
    const int rank=ddla::ddlaGetRank(h);
    rejected(h,[&]{ddla::psyrm_complete(h,rank==0?'X':'U',0,a.ptr,d);});
    rejected(h,[&]{ddla::psyrm_complete(h,'U',4,a.ptr,d);});
    if(ddla::ddlaGetSize(h)>1) {
        rejected(h,[&]{ddla::psyrm_complete(h,'U',rank==0?0:1,a.ptr,d);});
        rejected(h,[&]{
            if(rank==0)ddla::pherm_complete(h,'U',0,a.ptr,d);
            else ddla::psyrm_complete(h,'U',0,a.ptr,d);
        });
    }
    DDLA_CHECK(ddla::ddlaDescInit(d,h,1,1,8,8,0,0));
    const bool active=ddla_test::m_loc(h,d)>0&&ddla_test::n_loc(h,d)>0;
    ddla::psyrm_complete(h,'U',1,active?a.ptr:nullptr,d);
    rejected(h,[&]{ddla::psyrm_complete<double>(h,'U',1,nullptr,d);});
    d[ddla::DDLA_NB_]=4;
    rejected(h,[&]{ddla::psyrm_complete<double>(h,'U',0,nullptr,d);});
}
template<class T> void type(ddla::DdlaHandle_t h)
{
    for(bool upper:{false,true})for(bool hermitian:{false,true}) {
        for(int n:{0,1,2,7,31,32,33,65,257,513})local<T>(h,n,upper,hermitian);
        for(int nb:{1,5,33})for(int n:{0,1,2,7,67,101})distributed<T>(h,n,nb,upper,hermitian);
    }
}
void body(ddla::DdlaHandle_t h,const Shape&)
{
#if DDLA_HAS_GPU
    void* original=nullptr;
    gpuStream_t alternate=nullptr;
    if(ddla::ddlaGetBackend(h)==ddla::DdlaBackend::GPU) {
        DDLA_CHECK(ddla::ddlaSynchronize(h));original=ddla::ddlaGetStream(h);
        GPU_CHECK(gpuStreamCreateWithFlags(&alternate,gpuStreamNonBlocking));
        DDLA_CHECK(ddla::ddlaSetStream(h,static_cast<void*>(alternate)));
    }
#endif
    type<float>(h);type<double>(h);type<std::complex<float>>(h);type<std::complex<double>>(h);
    invalid(h);
#if DDLA_HAS_GPU
    if(alternate) {
        DDLA_CHECK(ddla::ddlaSynchronize(h));
        DDLA_CHECK(ddla::ddlaSetStream(h,original));
        GPU_CHECK(gpuStreamDestroy(alternate));
    }
#endif
}
}
int main(int argc,char** argv) { return run_grid_test(argc,argv,"test_matrix_complete",body); }
