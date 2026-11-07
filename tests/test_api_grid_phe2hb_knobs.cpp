#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"

// ---------------------------------------------------------------------------
// phe2hb (stage 1 of pheevd) runs its QL panel through the shared QR/QL driver
// (qr_make_plan('L') / qr_factor_block).  The public pgeqrf reads one tuning knob,
// DDLA_QR_LOOKAHEAD (the 1x1 split update / aux-stream lookahead), and it must not
// leak into stage 1: with the knob set to 0 or 1, phe2hb's output (the band/reflector
// storage A and the T slots) must equal the unset run bit for bit.  (DDLA_QR_NB1 and
// DDLA_QR_IB were removed 2026-10-19.)
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

std::string g_case_tag_suffix;

template <typename T> const char* type_tag() { return "?"; }
template <> const char* type_tag<float>() { return "s"; }
template <> const char* type_tag<double>() { return "d"; }
template <> const char* type_tag<std::complex<float>>() { return "c"; }
template <> const char* type_tag<std::complex<double>>() { return "z"; }

template <typename T> T tau_sentinel()
{
    if constexpr (is_cplx<T>::value) return T(-12345.0, 777.0);
    else return T(-12345.0);
}

// splitmix64 uniform in [-0.5, 0.5).
inline double unit_hash(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

template <typename T>
T rnd_value(int i, int j, std::uint64_t salt)
{
    using rt = typename ddla::real_type<T>::type;
    const std::uint64_t key = (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(j) ^ (salt << 56);
    if constexpr (is_cplx<T>::value)
        return T(static_cast<rt>(unit_hash(key)), static_cast<rt>(unit_hash(key ^ 0xA5A5A5A5ULL)));
    else
        return T(static_cast<rt>(unit_hash(key)));
}

// Backend dispatch (DDLA_HAS_* are VALUE checks).
template <typename Fn>
void on_backend(const ddla::DdlaHandle_t& handle, Fn fn)
{
#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU) fn(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
    else fn(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#elif DDLA_HAS_GPU
    fn(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#else
    fn(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
#endif
}

template <typename T>
struct RunOut {
    std::vector<T> a, tmat;
    bool threw = false;
};

template <typename T>
double bits_differ(const RunOut<T>& x, const RunOut<T>& y)
{
    double d = (x.a.size() != y.a.size() || x.tmat.size() != y.tmat.size()) ? 1.0 : 0.0;
    for (std::size_t i = 0; d == 0.0 && i < x.a.size(); ++i)
        if (std::memcmp(&x.a[i], &y.a[i], sizeof(T)) != 0) d += 1.0;
    for (std::size_t i = 0; i < x.tmat.size() && i < y.tmat.size(); ++i)
        if (std::memcmp(&x.tmat[i], &y.tmat[i], sizeof(T)) != 0) d += 1.0;
    return d;
}

// Sets DDLA_QR_LOOKAHEAD (nullptr = unset) and restores the previous value.
struct QrEnvScope {
    static constexpr const char* kName = "DDLA_QR_LOOKAHEAD";
    bool had;
    std::string old;
    explicit QrEnvScope(const char* la)
    {
        const char* v = std::getenv(kName);
        had = (v != nullptr);
        if (v != nullptr) old = v;
        set(la);
    }
    static void set(const char* la)
    {
        if (la == nullptr) unsetenv(kName);
        else setenv(kName, la, 1);
    }
    ~QrEnvScope()
    {
        if (had) setenv(kName, old.c_str(), 1);
        else unsetenv(kName);
    }
    QrEnvScope(const QrEnvScope&) = delete;
    QrEnvScope& operator=(const QrEnvScope&) = delete;
};

template <typename T>
T herm_conj(T v) { return v; }
template <typename U>
std::complex<U> herm_conj(std::complex<U> v) { return std::conj(v); }

template <typename T>
void check_phe2hb_ignores_knobs(const ddla::DdlaHandle_t& handle)
{
    struct Geo { int n, nb, nbw; };
    const Geo geos[] = {{96, 16, 8}, {100, 16, 16}, {77, 8, 8}, {120, 12, 12}, {200, 64, 16}};
    const QrEnvScope restore(nullptr);
    for (const Geo& gm : geos) {
        const int n = gm.n, nb = gm.nb, nbw = gm.nbw;
        int desc[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));
        auto h_A = make_local<T>(handle, desc, [&](int i, int j) -> T {
            const int lo = std::min(i, j), hi = std::max(i, j);
            const T v = rnd_value<T>(hi, lo, 9);
            if (i == j) return T(std::real(v));
            return i > j ? v : herm_conj(v);
        });
        const int nsteps = (n - 1) / nbw;
        const std::size_t ntmat = static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1);
        auto run = [&]() {
            RunOut<T> r;
            DeviceBuffer<T> d_A(handle, h_A.size());
            upload(handle, d_A.ptr, h_A);
            DeviceBuffer<T> d_tmat(handle, ntmat);
            const std::vector<T> init(ntmat, tau_sentinel<T>());
            upload(handle, d_tmat.ptr, init);
            check_ddla_sync(handle);
            int info = -1;
            try {
                on_backend(handle, [&](auto bt) {
                    constexpr DdlaBackend B = decltype(bt)::value;
                    ddla::phe2hb<B, T>(handle, n, d_A.ptr, desc, nbw, d_tmat.ptr, info);
                });
            } catch (const std::exception&) { r.threw = true; }
            check_ddla_sync(handle);
            r.a = download<T>(handle, d_A.ptr, h_A.size());
            r.tmat = download<T>(handle, d_tmat.ptr, ntmat);
            return r;
        };
        QrEnvScope::set(nullptr);
        const RunOut<T> ref = run();
        for (const char* ks : {"0", "1"}) {
            QrEnvScope::set(ks);
            const RunOut<T> x = run();
            require_close(handle, std::string(type_tag<T>()) + " phe2hb ignores-knobs n=" + std::to_string(n)
                                      + " nb=" + std::to_string(nb) + " nbw=" + std::to_string(nbw)
                                      + " [" + ks + "]" + g_case_tag_suffix,
                          (ref.threw || x.threw) ? 1.0 : bits_differ(ref, x), 0.0);
        }
    }
}

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    // The case list is grid-independent (every case builds its own descriptors): run it ONCE per grid.
    static int seen_nprows = -1, seen_npcols = -1;
    int nprows_g = 0, npcols_g = 0;
    ddlaGetGridDims(handle, nprows_g, npcols_g);
    if (nprows_g == seen_nprows && npcols_g == seen_npcols) return;
    seen_nprows = nprows_g;
    seen_npcols = npcols_g;
    (void)base;
    check_phe2hb_ignores_knobs<double>(handle);
    check_phe2hb_ignores_knobs<Complex>(handle);
    check_phe2hb_ignores_knobs<float>(handle);
    check_phe2hb_ignores_knobs<std::complex<float>>(handle);

#if DDLA_HAS_GPU
    // Dual lane: the same legs on a CPU handle.
    if (ddlaGetBackend(handle) == DdlaBackend::GPU && ddlaBackendAvailable(DdlaBackend::CPU)) {
        ddla::DdlaHandle_t h_cpu = nullptr;
        ddla::ddlaInit(h_cpu, DdlaBackend::CPU);
        ddla::ddlaSet(h_cpu, MPI_COMM_WORLD, nprows_g, npcols_g, 'R');
        const std::string save = g_case_tag_suffix;
        g_case_tag_suffix = " [cpu-arm]";
        check_phe2hb_ignores_knobs<double>(h_cpu);
        check_phe2hb_ignores_knobs<Complex>(h_cpu);
        check_phe2hb_ignores_knobs<float>(h_cpu);
        check_phe2hb_ignores_knobs<std::complex<float>>(h_cpu);
        g_case_tag_suffix = save;
        ddla::ddlaDestroy(h_cpu);
    }
#endif
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_phe2hb_knobs", check_body);
}
