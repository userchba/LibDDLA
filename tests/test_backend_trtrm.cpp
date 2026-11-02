#include "trtrm_test_common.h"
#include <array>
#include <functional>
using namespace trtrm_test;

template <DdlaBackend Backend, typename T>
int sampled_large_case(DdlaHandle_t h, int n, char uplo)
{
    const bool upper = uplo == 'U';
    using W = typename Scalar<T>::wide;
    using R = typename Scalar<T>::real;
    const int lda=n+3, ldb=n+5, ldc=n+7;
    std::vector<T> a(std::size_t(lda)*(n+1),poison<T>()), b(std::size_t(ldb)*(n+1),poison<T>());
    std::vector<T> c(std::size_t(ldc)*(n+1),value<T>(-77,19));
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) if(upper ? i<=j : i>=j) {
        a[i+std::size_t(j)*lda]=data<T>(i,j,1);
        b[i+std::size_t(j)*ldb]=data<T>(i,j,2);
        c[i+std::size_t(j)*ldc]=data<T>(i,j,3);
    }
    Buffer<T> da(h,a.size(),true), db(h,b.size(),true), dc(h,c.size(),true);
    da.put(a); db.put(b); dc.put(c);
    const T alpha=value<T>(0.75,-0.25), beta=value<T>(-0.5,0.3);
    trtrm<Backend,T>(h,uplo,'N',uplo,'N',n,alpha,da.p,lda,db.p,ldb,beta,dc.p,ldc);
    const auto got=dc.get();
    bool ok=true; long double worst=0;
    auto check=[&](int i,int j) {
        W sum=0; long double scale=0;
        for(int k=upper?i:j;k<=(upper?j:i);++k) {
            const W term=W(a[i+std::size_t(k)*lda])*W(b[k+std::size_t(j)*ldb]);
            sum+=term; scale+=std::abs(term);
        }
        const auto p=i+std::size_t(j)*ldc;
        const W want=W(alpha)*sum+W(beta)*W(c[p]);
        scale=std::abs(W(alpha))*scale+std::abs(W(beta)*W(c[p]));
        const long double err=std::abs(W(got[p])-want)/std::max(1.0L,scale);
        ok &= std::isfinite(err) && err<=64*std::numeric_limits<R>::epsilon();
        if(std::isfinite(err)) worst=std::max(worst,err);
    };
    const std::vector<int> edges={0,15,16,31,32,63,64,127,128,255,256,n-2,n-1};
    for(int j:edges) for(int i:edges) if(upper ? i<=j : i>=j) check(i,j);
    for(int q=0;q<256;++q) { const int j=(q*991+17)%n; check(upper ? (q*193+3)%(j+1) : j+(q*193+3)%(n-j),j); }
    for(int j=0;j<=n;++j) for(int i=0;i<ldc;++i) if(j==n || i>=n || !(upper ? i<=j : i>=j)) {
        const auto p=i+std::size_t(j)*ldc;
        ok &= std::memcmp(&got[p],&c[p],sizeof(T))==0;
    }
    std::cout<<"TRTRM_LARGE uplo="<<uplo<<" type="<<Scalar<T>::name<<" n="<<n<<" max_scaled_error="<<double(worst)
             <<" failures="<<!ok<<std::endl;
    return ok?0:1;
}

template <DdlaBackend Backend, typename T>
int run_type(DdlaHandle_t h, bool quick, char uplo)
{
    const bool upper = uplo == 'U';
    using W = typename Scalar<T>::wide;
    using R = typename Scalar<T>::real;
    int failures = 0, cases = 0;
    const std::vector<int> sizes = quick ? std::vector<int>{1, 17, 65, 129} :
        std::vector<int>{0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257};
    long double worst = 0;
    for (int n : sizes) {
        const int lda = std::max(1, n) + 3, ldb = lda + 2, ldc = ldb + 2;
        // Offset bases exercise unaligned views; trailing guards catch overwrites.
        const int off = 3;
        std::vector<T> a(std::size_t(lda) * (n + 1) + off, poison<T>());
        std::vector<T> b(std::size_t(ldb) * (n + 1) + off, poison<T>());
        std::vector<T> c(std::size_t(ldc) * (n + 1) + off, value<T>(-77, 19));
        for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) if (upper ? i <= j : i >= j) {
            a[off + i + std::size_t(j) * lda] = data<T>(i, j, 1);
            b[off + i + std::size_t(j) * ldb] = data<T>(i, j, 2);
        }
        Buffer<T> da(h, a.size(), true), db(h, b.size(), true), dc(h, c.size(), true);
        da.put(a); db.put(b);
        for (int mode = 0; mode < 6; ++mode) {
            const T alpha = (mode >= 2 && mode <= 4) ? T(0) : value<T>(0.75, -0.25);
            const T beta = mode == 0 || mode == 2 ? T(0) : mode == 4 ? T(1) : value<T>(-0.5, 0.3);
            for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) if (upper ? i <= j : i >= j)
                c[off + i + std::size_t(j) * ldc] = beta == T(0) ? poison<T>() : data<T>(i, j, 3);
            dc.put(c);
            const char triangle = mode%2 ? (upper?'u':'l') : uplo;
            trtrm<Backend,T>(h, triangle,'N',triangle,'N', n, alpha, alpha == T(0) ? nullptr : da.p + off, lda,
                            alpha == T(0) ? nullptr : (mode == 5 ? da.p : db.p) + off, mode == 5 ? lda : ldb, beta, dc.p + off, ldc);
            auto got = dc.get();
            bool ok = true;
            for (std::size_t p = 0; p < c.size(); ++p) {
                const std::size_t q = p >= off ? p - off : c.size();
                const int i = int(q % ldc), j = int(q / ldc);
                if (p < off || j >= n || i >= n || !(upper ? i <= j : i >= j)) {
                    ok &= std::memcmp(&got[p], &c[p], sizeof(T)) == 0;
                    continue;
                }
                W sum = W(0); long double scale = 0;
                if (alpha != T(0)) for (int k = upper ? i : j; k <= (upper ? j : i); ++k) {
                    const W term = W(a[off + i + std::size_t(k) * lda]) * W(mode == 5 ? a[off + k + std::size_t(j) * lda] : b[off + k + std::size_t(j) * ldb]);
                    sum += term; scale += std::abs(term);
                }
                W want = W(alpha) * sum;
                scale *= std::abs(W(alpha));
                if (beta != T(0)) { want += W(beta) * W(c[p]); scale += std::abs(W(beta) * W(c[p])); }
                const long double err = std::abs(W(got[p]) - want) / std::max(1.0L, scale);
                if (!std::isfinite(err) || err > 64 * std::numeric_limits<R>::epsilon()) ok = false;
                if (std::isfinite(err)) worst = std::max(worst, err);
            }
            if (!ok) { ++failures; std::cerr << "FAIL n=" << n << " uplo=" << uplo << " type=" << Scalar<T>::name << " mode=" << mode << '\n'; }
            ++cases;
        }
        const auto aa = da.get(), bb = db.get();
        if (std::memcmp(aa.data(), a.data(), a.size()*sizeof(T)) || std::memcmp(bb.data(), b.data(), b.size()*sizeof(T))) ++failures;
    }
    // alpha == 0 must scale signed zeros directly, without adding a +0
    // product that could change their sign. Padding/opposite triangle stay.
    {
        const T scale = value<T>(-2.0);
        std::vector<T> before(6, value<T>(-77, 19));
        for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
            if (upper ? i <= j : i >= j)
                before[i + 3*j] = value<T>(i == j ? 0.0 : -0.0, i == j ? -0.0 : 0.0);
        Buffer<T> zeros(h, before.size(), true);
        zeros.put(before);
        trtrm<Backend,T>(h, uplo,'N',uplo,'N', 2, T(0), nullptr, 2, nullptr, 2, scale, zeros.p, 3);
        const auto after = zeros.get();
        for (int j = 0; j < 2; ++j) for (int i = 0; i < 3; ++i) {
            const int q = i + 3*j;
            const T want = i < 2 && (upper ? i <= j : i >= j) ? scale*before[q] : before[q];
            if (std::memcmp(&after[q], &want, sizeof(T)) != 0) ++failures;
        }
        ++cases;
    }
    std::cout << "TRTRM backend=" << (Backend == DdlaBackend::GPU ? "gpu" : "cpu")
              << " uplo=" << uplo << " type=" << Scalar<T>::name << " cases=" << cases << " max_scaled_error=" << double(worst)
              << " failures=" << failures << std::endl;
    if constexpr (Backend == DdlaBackend::GPU) {
        failures += sampled_large_case<Backend,T>(h,quick?2053:513,uplo);
        if (!quick) failures += sampled_large_case<Backend,T>(h,2053,uplo);
    }
    return failures;
}


// ---------------------------------------------------------------------------
// General form: per-operand uplo and N/T/C flags.  36 flag combinations;
// matching effective triangles keep a triangular result, mixed ones fill C.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
int general_one(DdlaHandle_t h, int n, char ua, char xa, char ub, char xb,
                const T& alpha, const T& beta, bool component_alignment = true)
{
    using W = typename Scalar<T>::wide;
    using R = typename Scalar<T>::real;
    const char call_ua=ua, call_xa=xa, call_ub=ub, call_xb=xb;
    ua = char(std::toupper(static_cast<unsigned char>(ua)));
    xa = char(std::toupper(static_cast<unsigned char>(xa)));
    ub = char(std::toupper(static_cast<unsigned char>(ub)));
    xb = char(std::toupper(static_cast<unsigned char>(xb)));
    const bool upa = ua == 'U', upb = ub == 'U';
    const bool ta = xa != 'N', tb = xb != 'N';
    const bool eua = upa != ta, eub = upb != tb;
    const bool full = eua != eub;
    const int lda = std::max(1, n) + 3, ldb = std::max(1, n) + 5, ldc = std::max(1, n) + 7;
    const int off = 3;
    std::vector<T> a(std::size_t(lda) * (n + 1) + off, poison<T>());
    std::vector<T> b(std::size_t(ldb) * (n + 1) + off, poison<T>());
    std::vector<T> c(std::size_t(ldc) * (n + 1) + off, value<T>(-77, 19));
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
        if (upa ? i <= j : i >= j) a[off + i + std::size_t(j) * lda] = data<T>(i, j, 1);
        if (upb ? i <= j : i >= j) b[off + i + std::size_t(j) * ldb] = data<T>(i, j, 2);
    }
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        c[off + i + std::size_t(j) * ldc] =
            beta==T(0) && (full || (eua ? i<=j : i>=j)) ? poison<T>() : data<T>(i, j, 3);
    Buffer<T> da(h, a.size(), component_alignment), db(h, b.size(), component_alignment),
              dc(h, c.size(), component_alignment);
    da.put(a); db.put(b); dc.put(c);
    trtrm<Backend,T>(h, call_ua, call_xa, call_ub, call_xb, n, alpha, alpha == T(0) ? nullptr : da.p + off, lda,
                     alpha == T(0) ? nullptr : db.p + off, ldb, beta, dc.p + off, ldc);
    const auto got = dc.get();
    // Reference op(): zero outside the stored triangle, swapped/conjugated in.
    // op(X)(r,c) reads the stored element (r,c) or, under a transpose, (c,r);
    // the storage-triangle mask applies to that stored pair.
    auto op = [&](const std::vector<T>& m, int ld, bool upper, char tr, int r, int c) {
        const int sr = tr == 'N' ? r : c, sc = tr == 'N' ? c : r;
        if (upper ? sr > sc : sc > sr) return W(0);
        W v = W(m[off + sr + std::size_t(sc) * ld]);
        if constexpr (!std::is_arithmetic_v<T>) if (tr == 'C') v = std::conj(v);
        return v;
    };
    const auto aa=da.get(), bb=db.get();
    bool ok = std::memcmp(aa.data(),a.data(),a.size()*sizeof(T))==0 &&
              std::memcmp(bb.data(),b.data(),b.size()*sizeof(T))==0;
    long double worst = 0;
    for (std::size_t p = 0; p < c.size(); ++p) {
        const std::size_t q = p >= off ? p - off : c.size();
        const int i = int(q % ldc), j = int(q / ldc);
        if (p < off || j >= n || i >= n || (!full && (eua ? i > j : i < j))) {
            ok &= std::memcmp(&got[p], &c[p], sizeof(T)) == 0;
            continue;
        }
        W sum = W(0); long double scale = 0;
        if (alpha != T(0)) for (int k = 0; k < n; ++k) {
            const W term = op(a, lda, upa, xa, i, k) * op(b, ldb, upb, xb, k, j);
            sum += term; scale += std::abs(term);
        }
        W want = W(alpha) * sum;
        scale *= std::abs(W(alpha));
        if (beta != T(0)) { want += W(beta) * W(c[p]); scale += std::abs(W(beta) * W(c[p])); }
        const long double err = std::abs(W(got[p]) - want) / std::max(1.0L, scale);
        if (!std::isfinite(err) || err > 64 * std::numeric_limits<R>::epsilon()) ok = false;
        if (alpha==T(0) && beta==T(0) && got[p]!=T(0)) ok=false;
        if (std::isfinite(err)) worst = std::max(worst, err);
    }
    if (!ok) std::cerr << "FAIL general n=" << n << " " << ua << xa << " " << ub << xb
                       << " type=" << Scalar<T>::name << " alpha=" << alpha << " beta=" << beta << '\n';
    return ok ? 0 : 1;
}

template <DdlaBackend Backend, typename T>
int general_run_type(DdlaHandle_t h, bool quick)
{
    int failures = 0, cases = 0;
    const std::vector<int> sizes = quick ? std::vector<int>{1, 17} : std::vector<int>{1, 2, 7, 8, 9, 15, 16, 17, 33};
    const T alpha = value<T>(0.75, -0.25), beta = value<T>(-0.5, 0.3);
    for (char ua : {'U', 'L'}) for (char xa : {'N', 'T', 'C'})
        for (char ub : {'U', 'L'}) for (char xb : {'N', 'T', 'C'})
            for (int n : sizes) {
                failures += general_one<Backend,T>(h, n, ua, xa, ub, xb, alpha, beta);
                ++cases;
            }
    // Edge semantics on a mixed and a transposed-triangular combination:
    // alpha=0 (no A/B reference), beta=0 (C not read), beta=1 no-op shortcut.
    for (auto flags : std::vector<std::array<char,4>>{
             {'U','N','L','N'}, {'L','N','U','N'}, {'L','C','L','T'}, {'U','T','U','C'}})
        for (int n : {0, 1, 17}) {
            failures += general_one<Backend,T>(h, n, flags[0], flags[1], flags[2], flags[3], T(0), beta); ++cases;
            failures += general_one<Backend,T>(h, n, flags[0], flags[1], flags[2], flags[3], alpha, T(0)); ++cases;
            failures += general_one<Backend,T>(h, n, flags[0], flags[1], flags[2], flags[3], T(0), T(0)); ++cases;
            failures += general_one<Backend,T>(h, n, flags[0], flags[1], flags[2], flags[3], T(0), T(1)); ++cases;
        }
    // Whole-complex alignment selects the full-output GEAM path on GPUs.
    // Both-zero coefficients must overwrite NaNs with exact zeros there too.
    for (auto flags : std::vector<std::array<char,4>>{
             {'U','N','L','N'}, {'L','N','U','N'}}) {
        failures += general_one<Backend,T>(h,17,flags[0],flags[1],flags[2],flags[3],T(0),T(0),false);
        ++cases;
    }
    // Case-insensitive flags.
    failures += general_one<Backend,T>(h, 17, 'u', 'c', 'l', 't', alpha, beta); ++cases;
    std::cout << "TRTRM_GENERAL backend=" << (Backend == DdlaBackend::GPU ? "gpu" : "cpu")
              << " type=" << Scalar<T>::name << " cases=" << cases << " failures=" << failures << std::endl;
    if constexpr (Backend == DdlaBackend::GPU) {
        failures += general_one<Backend,T>(h, quick ? 777 : 513, 'U', 'N', 'L', 'T', alpha, beta);
        failures += general_one<Backend,T>(h, quick ? 513 : 513, 'L', 'C', 'L', 'N', alpha, beta);
    }
    return failures;
}

template <DdlaBackend Backend>
int run(bool quick, const std::string& uplos)
{
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, Backend)); DDLA_CHECK(ddlaSet(h, MPI_COMM_SELF));
    int bad = 0;
    for (char uplo : uplos) {
        bad += run_type<Backend,float>(h,quick,uplo) + run_type<Backend,double>(h,quick,uplo) +
            run_type<Backend,std::complex<float>>(h,quick,uplo) + run_type<Backend,std::complex<double>>(h,quick,uplo);
    }
    bad += general_run_type<Backend,float>(h,quick) + general_run_type<Backend,double>(h,quick) +
        general_run_type<Backend,std::complex<float>>(h,quick) + general_run_type<Backend,std::complex<double>>(h,quick);
    auto rejects = [&](std::function<void()> f) { try { f(); ++bad; } catch (const std::exception&) {} };
    double a[12] = {}, b[12] = {}, c[12] = {};
    for (char uplo : uplos) {
        rejects([&] { trtrm<Backend,double>(nullptr, uplo,'N',uplo,'N', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', -1, 1., a, 1, b, 1, 0., c, 1); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 1, b, 2, 0., c, 2); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 2, b, 1, 0., c, 2); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 2, b, 2, 0., c, 1); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., nullptr, 2, b, 2, 0., c, 2); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 2, nullptr, 2, 0., c, 2); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 2, b, 2, 0., nullptr, 2); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 2, b, 2, 0., a, 2); });
        rejects([&] { trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 1., a, 2, b, 2, 0., b + 1, 2); });
        trtrm<Backend,double>(h, uplo,'N',uplo,'N', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1);
        trtrm<Backend,double>(h, uplo,'N',uplo,'N', 2, 0., nullptr, 2, nullptr, 2, 1., nullptr, 2);
    }
    for (char invalid : {'X','N','\0'}) {
        try {
            trtrm<Backend,double>(h,invalid,'N','U','N',0,1.,nullptr,1,nullptr,1,0.,nullptr,1);
            ++bad;
        } catch (const std::invalid_argument&) {}
    }
    // Per-operand flag rejects: bad flags, dimensions, nulls, overlap.
    rejects([&] { trtrm<Backend,double>(h, 'X','N','U','N', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1); });
    rejects([&] { trtrm<Backend,double>(h, 'U','X','U','N', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1); });
    rejects([&] { trtrm<Backend,double>(h, 'U','N','X','N', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1); });
    rejects([&] { trtrm<Backend,double>(h, 'U','N','U','X', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1); });
    rejects([&] { trtrm<Backend,double>(h, 'U','T','L','N', -1, 1., a, 1, b, 1, 0., c, 1); });
    rejects([&] { trtrm<Backend,double>(h, 'U','T','L','N', 2, 1., a, 1, b, 2, 0., c, 2); });
    rejects([&] { trtrm<Backend,double>(h, 'U','T','L','N', 2, 1., a, 2, b, 2, 0., nullptr, 2); });
    rejects([&] { trtrm<Backend,double>(h, 'U','T','L','N', 2, 1., nullptr, 2, b, 2, 0., c, 2); });
    rejects([&] { trtrm<Backend,double>(h, 'U','T','L','N', 2, 1., a, 2, b, 2, 0., a, 2); });
    rejects([&] { trtrm<Backend,double>(h, 'U','N','L','N', 2, 1., a, 2, b, 2, 0., a, 2); });
    rejects([&] { trtrm<Backend,double>(h, 'L','N','U','N', 2, 1., a, 2, b, 2, 0., b+1, 2); });
    trtrm<Backend,double>(h, 'U','T','L','C', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1);
    trtrm<Backend,double>(h, 'L','C','U','T', 2, 0., nullptr, 2, nullptr, 2, 1., nullptr, 2);
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    constexpr DdlaBackend Other = Backend == DdlaBackend::CPU ? DdlaBackend::GPU : DdlaBackend::CPU;
    rejects([&] { trtrm<Other,double>(h, 'L','N','L','N', 0, 1., nullptr, 1, nullptr, 1, 0., nullptr, 1); });
#endif
    DDLA_CHECK(ddlaSynchronize(h)); DDLA_CHECK(ddlaDestroy(h));
    return bad;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int bad = 0;
    try {
        bool quick = false;
        std::string uplos = "UL";
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--quick") quick=true;
            else if (arg=="--uplo" && i+1<argc) {
                const std::string value=argv[++i];
                if (value=="U" || value=="u") uplos="U";
                else if (value=="L" || value=="l") uplos="L";
                else if (value=="both") uplos="UL";
                else throw std::invalid_argument("--uplo requires U, L or both");
            } else throw std::invalid_argument("unknown test option");
        }
#if DDLA_HAS_CPU
        bad += run<DdlaBackend::CPU>(quick,uplos);
#endif
#if DDLA_HAS_GPU
        bad += run<DdlaBackend::GPU>(quick,uplos);
#endif
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; ++bad; }
    int total = 0; MPI_Allreduce(&bad, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "TRTRM_CHECKS_FAILED=" << total << std::endl;
    MPI_Finalize(); return total ? 1 : 0;
}
