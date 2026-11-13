#include "api_grid_test_common.h"
#include "eig_test_common.h"
#include <cstring>
#include <limits>
#include <type_traits>

using namespace api_grid_test;
namespace {
struct Environment {
    const char* name;
    bool present;
    std::string saved;
    explicit Environment(const char* key) : name(key), present(std::getenv(key) != nullptr),
        saved(present ? std::getenv(key) : "") {}
    ~Environment() { set(present ? saved.c_str() : nullptr); }
    void set(const char* value) const {
        if (value) setenv(name, value, 1);
        else unsetenv(name);
    }
};
template<class T> T entry(int i, int j) {
    using R = typename ddla::real_type<T>::type;
    if (i == j) return T(R(3 + 0.03 * i));
    const int lo = std::min(i, j), hi = std::max(i, j);
    const R re = R(0.07 * ((lo * 13 + hi * 7) % 17 - 8));
    if constexpr (std::is_same_v<T, R>) return re;
    else {
        const R im = R(0.09 * ((lo * 5 + hi * 3) % 11 - 5));
        return T(re, i < j ? im : -im);
    }
}
template<class T> void solve(const ddla::DdlaHandle_t& h, const Environment& method,
                            int n, char uplo, bool one, bool alias, int automatic = -1, int block = 16) {
    using R = typename ddla::real_type<T>::type;
    int da[ddla::DDLA_DLEN_], dz[ddla::DDLA_DLEN_];
    ddlaDescInit(da, h, n + 2, n + 3, block, block, 0, 0);
    da[ddla::DDLA_LLD_] += 3;
    std::copy_n(da, ddla::DDLA_DLEN_, dz);
    const T sentinel = T(12345);
    const R nan = std::numeric_limits<R>::quiet_NaN();
    auto full = make_local<T>(h, da, [&](int i, int j) {
        return i < n && j < n ? entry<T>(i, j) : sentinel;
    });
    auto input = make_local<T>(h, da, [&](int i, int j) {
        if (i >= n || j >= n) return sentinel;
        if (uplo == 'U' ? i > j : i < j) return T(nan);
        if constexpr (!std::is_same_v<T, R>)
            if (i == j) return T(std::real(entry<T>(i, j)), nan);
        return entry<T>(i, j);
    });
    DeviceBuffer<T> a(h, input.size()), z(h, input.size()), reference(h, input.size());
    const std::vector<T> output(input.size(), sentinel);
    upload(h, reference.ptr, full);
    std::vector<R> w(std::max(n, 1)), direct(std::max(n, 1)), values(std::max(n, 1));
    int info = -1;
    // Explicit entries must ignore the selector, including an invalid value.
    method.set("invalid-for-explicit-entry");
    upload(h, a.ptr, input);
    if (one) ddla::pheevd_one_stage(h, 'N', uplo, n,n, a.ptr, da, direct.data(), static_cast<T*>(nullptr), nullptr, info);
    else ddla::pheevd_two_stage(h, 'N', uplo, n,n, a.ptr, da, direct.data(), static_cast<T*>(nullptr), nullptr, info);
    require_close(h, "explicit method info", info, 0);
    if (automatic < 0) method.set(one ? "one_stage" : "two_stage");
    else method.set(automatic == 0 ? nullptr : automatic == 1 ? "" : "auto");
    upload(h, a.ptr, input); upload(h, z.ptr, output);
    T* vectors = alias ? a.ptr : z.ptr;
    ddla::pheevd(h, 'V', uplo, n,n, a.ptr, da, w.data(), vectors, dz, info);
    require_close(h, "selected method V info", info, 0);
    if (n) {
        require_close(h, "selected method residual", eig_test::residual_metric<T>(h, reference.ptr, da, vectors, dz, w.data(), n), 50);
        require_close(h, "selected method orthogonality", eig_test::orthogonality_metric<T>(h, vectors, dz, n), 50);
    }
    const auto after_a = download(h, a.ptr, input.size()), after_z = download(h, vectors, input.size());
    double bad = 0;
    const int rows = ddla_test::m_loc(h, da), cols = ddla_test::n_loc(h, da), ld = da[ddla::DDLA_LLD_];
    for (int j = 0; j < cols; ++j) for (int i = 0; i < ld; ++i) {
        const std::size_t at = i + std::size_t(j) * ld;
        const bool active = i < rows && ddla::indx_l2g_r(da, h, i) < n && ddla::indx_l2g_c(da, h, j) < n;
        if (active) bad += std::isfinite(std::abs(after_z[at])) ? 0 : 1;
        else {
            bad += std::memcmp(&after_a[at], &input[at], sizeof(T)) ? 1 : 0;
            const T expected = alias ? input[at] : output[at];
            bad += std::memcmp(&after_z[at], &expected, sizeof(T)) ? 1 : 0;
        }
    }
    if(bad)std::cout<<"PHEEVD_GUARD_FAILURE n="<<n<<" scalar_bytes="<<sizeof(T)<<" uplo="<<uplo<<" alias="<<alias<<" automatic="<<automatic<<std::endl;
    require_close(h, "selected method finite vectors and parent/LLD guards", bad, 0);
    upload(h, a.ptr, input);
    ddla::pheevd(h, 'N', uplo, n,n, a.ptr, da, values.data(), static_cast<T*>(nullptr), nullptr, info);
    require_close(h, "selected method N info", info, 0);
    double error = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(w[i]) || !std::isfinite(values[i]) || !std::isfinite(direct[i]) || (i && w[i] < w[i-1])) error = INFINITY;
        const double scale = std::max(1., double(std::abs(w[i]))) * std::max(1, n) * std::numeric_limits<R>::epsilon();
        error = std::max(error, double(std::abs(w[i] - values[i])) / scale);
        error = std::max(error, double(std::abs(w[i] - direct[i])) / scale);
    }
    require_close(h, "selected N/V and explicit eigenvalues", error, 50);
}
void selection(const ddla::DdlaHandle_t& h, const Environment& method, const Environment& band) {
    int d[ddla::DDLA_DLEN_]; ddlaDescInit(d, h, 17, 17, 16, 16, 0, 0);
    DeviceBuffer<double> a(h, local_size(h, d));
    const auto input = make_local<double>(h, d, [](int i, int j) { return entry<double>(i, j); });
    std::vector<double> w(17);
    int info = -1;
    auto invoke = [&](int n, bool reject) {
        upload(h, a.ptr, input);
        bool caught = false;
        try { ddla::pheevd(h, 'N', 'U', n,n, a.ptr, d, w.data(), static_cast<double*>(nullptr), nullptr, info); }
        catch (const std::exception&) { caught = true; }
        require_close(h, "selector expected rejection", caught == reject ? 0 : 1, 0);
        if (reject) {
            const auto after = download(h, a.ptr, input.size());
            require_close(h, "selector rejection precedes input mutation", std::memcmp(after.data(), input.data(), input.size()*sizeof(double)) ? 1 : 0, 0);
        }
    };
    for (int n : {0, 17}) { method.set("invalid"); invoke(n, true); }
    int ranks = 0; MPI_Comm_size(ddlaGetCommunicator(h), &ranks);
    if (ranks > 1) {
        for (int n : {0, 17}) {
            method.set(ddlaGetRank(h) == 0 ? "one_stage" : "two_stage"); invoke(n, true);
            method.set(ddlaGetRank(h) == 0 ? "invalid" : "one_stage"); invoke(n, true);
        }
    }
    // Only the two-stage solver validates band width. This proves routing,
    // including the unset/empty defaults and per-call changes in one process.
    band.set("3");
    for (const char* value : {static_cast<const char*>(nullptr), "", "auto", "two_stage", "one_stage", "two_stage", "one_stage"}) {
        method.set(value);
        const bool automatic = !value || !*value || std::strcmp(value,"auto")==0;
        invoke(17, (!automatic || ranks>1) && (!value || std::strcmp(value,"one_stage")!=0));
    }
    band.set("16");
}
template<class T> void cases(const ddla::DdlaHandle_t& h, const Environment& method) {
    for (bool one : {false, true}) for (char uplo : {'U', 'L'}) {
        for (int n : {0, 1, 17, 33}) solve<T>(h, method, n, uplo, one, false);
        solve<T>(h, method, 33, uplo, one, true);
    }
}
template<class T> void overlap_local(const ddla::DdlaHandle_t& h, const Environment& method) {
    using R=typename ddla::real_type<T>::type;
    int ranks=0;MPI_Comm_size(ddlaGetCommunicator(h),&ranks);if(ranks!=1)return;
    const int n=17;
    for(char uplo:{'U','L'})for(int shift:{0,1}) {
        int da[ddla::DDLA_DLEN_],dz[ddla::DDLA_DLEN_];
        ddlaDescInit(da,h,n+2,n+3,16,16,0,0);da[ddla::DDLA_LLD_]+=4;
        std::copy_n(da,ddla::DDLA_DLEN_,dz);dz[ddla::DDLA_LLD_]+=2;
        const auto size=std::max(local_size(h,da),std::size_t(shift)+local_size(h,dz));
        std::vector<T> initial(size,T(12345));
        for(int j=0;j<n;++j)for(int i=0;i<n;++i)
            initial[i+std::size_t(j)*da[ddla::DDLA_LLD_]]= (uplo=='U'?i<=j:i>=j)?entry<T>(i,j):T(std::numeric_limits<R>::quiet_NaN());
        DeviceBuffer<T> a(h,size),reference(h,local_size(h,da));
        upload(h,a.ptr,initial);
        const auto full=make_local<T>(h,da,[&](int i,int j){return i<n&&j<n?entry<T>(i,j):T(12345);});
        upload(h,reference.ptr,full);
        std::vector<R> w(n+2,R(9876));int info=-1;
        method.set("auto");T* z=a.ptr+shift;
        ddla::pheevd(h,'V',uplo,n,n,a.ptr,da,w.data()+1,z,dz,info);
        require_close(h,"local overlapping output info",info,0);
        require_close(h,"local overlapping output residual",eig_test::residual_metric<T>(h,reference.ptr,da,z,dz,w.data()+1,n),50);
        require_close(h,"local overlapping output orthogonality",eig_test::orthogonality_metric<T>(h,z,dz,n),50);
        auto after=download(h,a.ptr,size);std::vector<bool> active(size,false);
        for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
            active[i+std::size_t(j)*da[ddla::DDLA_LLD_]]=true;
            active[shift+i+std::size_t(j)*dz[ddla::DDLA_LLD_]]=true;
        }
        bool guards=w.front()==R(9876)&&w.back()==R(9876);
        for(std::size_t i=0;i<size;++i)if(!active[i])guards=guards&&!std::memcmp(&after[i],&initial[i],sizeof(T));
        for(int i=1;i<=n;++i)guards=guards&&std::isfinite(w[i])&&(i==1||w[i]>=w[i-1]);
        require_close(h,"local overlapping matrix/scalar guards",guards?0:1,0);
    }
}
void body(const ddla::DdlaHandle_t& h, const Shape&) {
    Environment method("DDLA_EIGEN_METHOD"), band("DDLA_EIG_NBW");
    selection(h, method, band);
    cases<float>(h, method); cases<double>(h, method);
    cases<std::complex<float>>(h, method); cases<std::complex<double>>(h, method);
    band.set(nullptr);
    const auto automatic = [&](auto scalar) {
        using T = decltype(scalar);
        for (char uplo : {'U', 'L'}) for (int selector : {0, 1, 2})
            solve<T>(h, method, 65, uplo, false, selector == 1, selector, 64);
    };
    automatic(float{}); automatic(double{});
    automatic(std::complex<float>{}); automatic(std::complex<double>{});
    overlap_local<float>(h,method);overlap_local<double>(h,method);
    overlap_local<std::complex<float>>(h,method);overlap_local<std::complex<double>>(h,method);
}
}
int main(int argc, char** argv) { return run_grid_test(argc, argv, "test_pheevd_method", body); }
