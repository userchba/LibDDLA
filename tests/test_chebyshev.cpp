// Isolated private prototype fixture; no global n-by-n reference is allocated.
// CPU is the default, including GPU builds. Use --gpu explicitly for a device.
#include "Eigensolver/chebyshev_impl.h"
#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include "workspace_test_common.h"
#include "check_agree.h"
#include <array>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <type_traits>

using namespace ddla;
using namespace api_grid_test;
namespace ch = ddla::spectral_impl::chebyshev;
namespace
{
std::string transport = "default";
template<DdlaBackend B> void select_transport(DdlaHandle_t h) {
    if constexpr (B == DdlaBackend::GPU) {
        if (transport != "default") {
            const auto mode = transport == "ccl" ? DdlaCommunication::CCL :
                transport == "staged" ? DdlaCommunication::HostStagedMPI : DdlaCommunication::MPI;
            DDLA_CHECK(ddlaSetCommunication(h, mode));
            if (ddlaGetCommunication(h) != mode)
                throw std::runtime_error("requested GPU transport was not selected");
        }
    }
}
// Compile this same analytic fixture twice: one target tests borrowed execution,
// the other enters exclusively through the scalar-only public owning API.
#ifdef DDLA_CHEBYSHEV_PUBLIC_TEST
constexpr bool public_entry = true;
#else
constexpr bool public_entry = false;
#endif
using Wide = std::complex<double>;
template <class T> Wide wide(T x)
{
    return Wide(double(x), 0);
}
template <class R> Wide wide(std::complex<R> x)
{
    return Wide(x.real(), x.imag());
}
template <class T> T narrow(Wide x)
{
    using R = typename real_type<T>::type;
    if constexpr (std::is_arithmetic<T>::value)
        return R(x.real());
    else
        return T(R(x.real()), R(x.imag()));
}
template <class T> T value(double r, double i = 0)
{
    return narrow<T>(Wide(r, i));
}

// Q=I-2uu^H is an analytic unitary. Its columns and the spectrum define
// independent spectral/output oracles using only one length-n vector at a time.
template <class T> Wide u(int i, int n)
{
    const double weight = 1 / std::sqrt(double(n));
    if constexpr (std::is_arithmetic<T>::value)
        return Wide(i % 2 ? -weight : weight, 0);
    else
        return std::polar(weight, .23 * i);
}
struct Case
{
    int n = 17, nev = 3, nex = 2, repeat = 0, maximum = 80, degree = 12, scale = 0;
    char triangle = 'L';
    bool dense = true, binary = false, scalar = false, warm = false, alias = false;
    int warm_kind = 0; // 0=correlated wanted; 1=exact unwanted; 2=zero; 3=duplicate
    double tolerance = 0;
};
double lambda(const Case &c, int j)
{
    if (c.scalar)
        return std::scalbn(-.75, c.scale);
    if (c.binary)
        return j < c.nev ? 0 : std::scalbn(1., c.scale);
    const double base = j < c.nev ? (-.6 + .3 * j) : (3. + .15 * j);
    return std::scalbn(base + .025 * c.repeat, c.scale);
}
template <class T> Wide eigenvector(const Case &c, int i, int j)
{
    if (!c.dense)
        return Wide(i == j ? 1 : 0, 0);
    return Wide(i == j ? 1 : 0, 0) - 2. * u<T>(i, c.n) * std::conj(u<T>(j, c.n));
}
template <class T> T entry(const Case &c, int i, int j)
{
    if (!c.dense)
        return value<T>(i == j ? lambda(c, i) : 0);
    double mean = 0;
    for (int k = 0; k < c.n; ++k)
        mean += lambda(c, k) / c.n;
    const Wide result =
        Wide(i == j ? lambda(c, i) : 0, 0) +
        u<T>(i, c.n) * std::conj(u<T>(j, c.n)) * (4 * mean - 2 * lambda(c, i) - 2 * lambda(c, j));
    return narrow<T>(result);
}

template <class T>
std::vector<T> vector_column(DdlaHandle_t h, const int *d, const std::vector<T> &local, int n,
                             int column)
{
    std::vector<T> result(n, T(0));
    for (int j = 0; j < ddla_test::n_loc(h, d); ++j)
    {
        if (indx_l2g_c(d, h, j) != column)
            continue;
        for (int i = 0; i < ddla_test::m_loc(h, d); ++i)
        {
            const int gi = indx_l2g_r(d, h, i);
            if (gi < n)
                result[gi] = local[i + std::size_t(j) * d[DDLA_LLD_]];
        }
    }
    if (n)
        MPI_Allreduce(MPI_IN_PLACE, result.data(), n, detail::mpi_datatype<T>(), MPI_SUM, h->comm);
    return result;
}

template <class T>
void verify(DdlaHandle_t h, const Case &c, const int *dz, const std::vector<T> &z,
            const std::vector<typename real_type<T>::type> &w, double tolerance)
{
    double eigerror = 0, residual = 0, orthogonality = 0;
    for (int j = 0; j < c.nev; ++j)
    {
        eigerror =
            std::max(eigerror, std::abs(double(w[4 + j]) - lambda(c, j)) /
                                   std::max(std::numeric_limits<double>::min(),
                                            std::abs(lambda(c, j)) + std::scalbn(1., c.scale)));
        const auto x = vector_column(h, dz, z, c.n, j);
        double norm = 0, rsq = 0, anorm = 0;
        for (int i = 0; i < c.n; ++i)
        {
            Wide ax = 0;
            double row = 0;
            // The oracle divides by the common binary scale BEFORE dot products
            // so subnormal/extreme input does not under/overflow its own test.
            for (int k = 0; k < c.n; ++k)
            {
                const Wide a = wide(entry<T>(c, i, k));
                const Wide scaled(std::scalbn(a.real(), -c.scale), std::scalbn(a.imag(), -c.scale));
                ax += scaled * wide(x[k]);
                row += std::abs(scaled);
            }
            const double l = std::scalbn(double(w[4 + j]), -c.scale);
            rsq += std::norm(ax - l * wide(x[i]));
            norm += std::norm(wide(x[i]));
            anorm = std::max(anorm, row);
        }
        const double denominator =
            (anorm + std::abs(std::scalbn(double(w[4 + j]), -c.scale))) * std::sqrt(norm);
        const double err = denominator ? std::sqrt(rsq) / denominator : (rsq == 0 ? 0 : INFINITY);
        residual = std::max(residual, workspace_test::finite_error(err));
        for (int k = 0; k <= j; ++k)
        {
            const auto y = vector_column(h, dz, z, c.n, k);
            Wide dot = 0;
            for (int i = 0; i < c.n; ++i)
                dot += std::conj(wide(y[i])) * wide(x[i]);
            orthogonality = std::max(orthogonality, std::abs(dot - Wide(j == k ? 1 : 0, 0)));
        }
    }
    const bool single = std::is_same<typename real_type<T>::type, float>::value;
    require_close(h, "Chebyshev lowest-index analytic oracle", eigerror, single ? 8e-4 : 2e-9);
    require_close(h, "Chebyshev independent residual", residual, 3 * tolerance);
    require_close(h, "Chebyshev orthogonality", orthogonality, single ? 8e-4 : 2e-10);
}

class Arena
{
    static constexpr std::size_t guard = 256;
    DdlaHandle_t handle_;
    std::size_t backend_bytes_, host_bytes_;
    ddla::vector<char> backend_;
    std::vector<char> host_;

  public:
    Arena(DdlaHandle_t h, std::size_t b, std::size_t s)
        : handle_(h), backend_bytes_(b), host_bytes_(s), backend_(h, b + 2 * guard),
          host_(s + 2 * guard, char(0xa7))
    {
        DDLA_CHECK(ddlaMemset(h, backend_.data(), 0xa7, backend_.size()));
    }
    char *backend()
    {
        return backend_.data() + guard;
    }
    char *host()
    {
        return host_.data() + guard;
    }
    void verify()
    {
        std::array<unsigned char, 2 * guard> result{};
        DDLA_CHECK(ddlaMemcpy(handle_, result.data(), backend_.data(), guard));
        DDLA_CHECK(ddlaMemcpy(handle_, result.data() + guard, backend() + backend_bytes_, guard));
        DDLA_CHECK(ddlaSynchronize(handle_));
        bool ok = true;
        for (std::size_t i = 0; i < guard; ++i)
            ok = ok && result[i] == 0xa7 && result[guard + i] == 0xa7 &&
                 static_cast<unsigned char>(host_[i]) == 0xa7 &&
                 static_cast<unsigned char>(host()[host_bytes_ + i]) == 0xa7;
        require_close(handle_, "Chebyshev exact-arena guards", ok ? 0 : 1, 0);
    }
};
bool no_owners(const workspace_test::Snapshot &s)
{
    return !s.ddla_alloc && !s.ddla_free && !s.raw_gpu_alloc && !s.pinned_alloc && !s.host_new &&
           !s.vendor_query && !s.resource_setup;
}

template <DdlaBackend B, class T> void run_case(DdlaHandle_t h, Case c, bool replay = false)
{
    using R = typename real_type<T>::type;
    const double tolerance =
        c.tolerance > 0 ? c.tolerance : (std::is_same<R, float>::value ? 4e-5 : 2e-11);
    int da[DDLA_DLEN_], dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da, h, c.n + 2, c.n + 2, 4, 4, 0, 0));
    DDLA_CHECK(ddlaDescInit(dz, h, c.n + 2, c.nev + 2, 3, 2, 0, h->npcols_ > 1 ? 1 : 0));
    da[DDLA_LLD_] += 2 + h->myid;
    dz[DDLA_LLD_] += 3 + h->myid;
    if (c.alias)
        std::copy_n(da, DDLA_DLEN_, dz);
    const auto sentinel = value<T>(-717, .75);
    auto original = make_local<T>(
        h, da,
        [&](int i, int j)
        {
            if (i >= c.n || j >= c.n)
                return sentinel;
            if (c.triangle == 'L' ? i < j : i > j)
                return c.alias && c.warm ? T(0) : value<T>(std::numeric_limits<R>::quiet_NaN());
            const T a = entry<T>(c, i, j);
            if constexpr (!std::is_arithmetic<T>::value)
            {
                if (i == j && !c.alias)
                    return T(a.real(), std::numeric_limits<R>::quiet_NaN());
            }
            return a;
        });
    const auto initial_z =
        make_local<T>(h, dz,
                      [&](int i, int j)
                      {
                          if (i >= c.n || j >= c.nev)
                              return sentinel;
                          if (!c.warm)
                              return value<T>(std::numeric_limits<R>::quiet_NaN());
                          if (c.warm_kind == 2)
                              return T(0);
                          const int column =
                              c.warm_kind == 1 ? c.n - 1 - j : (c.warm_kind == 3 ? 0 : j);
                          return narrow<T>(eigenvector<T>(c, i, column));
                      });
    DeviceBuffer<T> a(h, original.size()), separate_z(h, initial_z.size());
    T *z = c.alias ? a.ptr : separate_z.ptr;
    std::vector<R> values(std::size_t(c.nev) + 8, R(-313));
    ch::Plan p;
    std::optional<ch::Resources<T>> resources;
    std::optional<Arena> arena;
    std::size_t bb = 0, hb = 0;
    if constexpr (!public_entry)
    {
        p = ch::check<T>(h, B, c.triangle, c.n, c.nev, c.nex, a.ptr, da, values.data() + 4, z, dz,
                         tolerance, c.maximum, c.degree, c.warm);
        resources.emplace(h, p);
        ch::prepare<B>(h, p, *resources);
        workspace_test::Scope query_scope(true);
        ch::workspace_size(p, &bb, &hb);
        const auto query = query_scope.finish();
        require_close(h, "Chebyshev integer query is pure",
                      no_owners(query) && !query.copies && !query.launches && !query.syncs &&
                              !query.collectives
                          ? 0
                          : 1,
                      0);
        arena.emplace(h, bb, hb);
        const auto staging=ch::staging_bytes<T>(h,p);
        detail::agree_local(h,"Chebyshev fixture transport reservation",[&] {
            if(staging[0]||staging[1])h->host_buffers<std::byte>(staging[0],staging[1]);
        });
    }
    const int runs = replay ? 3 : 1;
    for (int repetition = 0; repetition < runs; ++repetition)
    {
        if (repetition)
        {
            ++c.repeat;
            original = make_local<T>(h, da,
                                     [&](int i, int j)
                                     {
                                         if (i >= c.n || j >= c.n)
                                             return sentinel;
                                         return (c.triangle == 'L' ? i >= j : i <= j)
                                                    ? entry<T>(c, i, j)
                                                    : value<T>(NAN);
                                     });
        }
        upload(h, a.ptr, original);
        // For diagonal warm aliases, the first columns of A themselves are
        // scaled coordinate eigenvectors; copy-before-mutation is essential.
        if (!c.alias)
            upload(h, z, initial_z);
        std::fill(values.begin(), values.end(), R(-313));
        int iterations = -1, converged = -1, info = -1;
        ch::Counters count;
        workspace_test::Scope scope;
        if constexpr (public_entry)
            ddla::pheevd_chebyshev<T>(h, c.triangle, c.n, c.nev, c.nex, a.ptr, da,
                                      values.data() + 4, z, dz, tolerance, c.maximum, c.degree,
                                      c.warm, iterations, converged, info);
        else
            ch::run<B>(h, p, *resources, a.ptr, values.data() + 4, z, iterations, converged, info,
                       &count, arena->backend(), bb, arena->host(), hb);
        const auto trace = scope.finish();
        if (!public_entry && repetition >= 1)
            require_close(h, "Chebyshev prewarmed raw has no owner/query/setup",
                          no_owners(trace) ? 0 : 1, 0);
        require_close(h, "Chebyshev converged status",
                      info == ch::success && converged == c.nev ? 0 : 1, 0);
        if (c.warm && c.nev + c.nex < c.n && !c.scalar)
            require_close(h, "Chebyshev warm performs genuine filter", iterations > 0 ? 0 : 1, 0);
        bool guarded = true;
        for (int j = 0; j < 4; ++j)
            guarded = guarded && values[j] == R(-313) && values[c.nev + 4 + j] == R(-313);
        require_close(h, "Chebyshev nev-sized value guards", guarded ? 0 : 1, 0);
        const auto actual = download(h, z, initial_z.size());
        if (c.nev)
            verify(h, c, dz, actual, values, tolerance);
        // Check local output parent/padding using only its descriptor partition.
        guarded = true;
        const auto &before = c.alias ? original : initial_z;
        for (int j = 0; j < ddla_test::n_loc(h, dz); ++j)
            for (int i = 0; i < dz[DDLA_LLD_]; ++i)
            {
                const int gi = indx_l2g_r(dz, h, i), gj = indx_l2g_c(dz, h, j);
                if (i >= ddla_test::m_loc(h, dz) || gi >= c.n || gj >= c.nev)
                {
                    // A/Z aliases may legitimately contain completed/scaled A.
                    if (c.alias && gi < c.n && gj < c.n)
                        continue;
                    const auto offset = i + std::size_t(j) * dz[DDLA_LLD_];
                    guarded =
                        guarded && std::memcmp(&actual[offset], &before[offset], sizeof(T)) == 0;
                }
            }
        require_close(h, "Chebyshev output parent/padding", guarded ? 0 : 1, 0);
        const auto after_a = download(h, a.ptr, original.size());
        guarded = true;
        for (int j = 0; j < ddla_test::n_loc(h, da); ++j)
            for (int i = 0; i < da[DDLA_LLD_]; ++i)
            {
                const int gi = indx_l2g_r(da, h, i), gj = indx_l2g_c(da, h, j);
                if (i >= ddla_test::m_loc(h, da) || gi >= c.n || gj >= c.n)
                {
                    const auto offset = i + std::size_t(j) * da[DDLA_LLD_];
                    guarded =
                        guarded && std::memcmp(&after_a[offset], &original[offset], sizeof(T)) == 0;
                }
            }
        require_close(h, "Chebyshev compact completion parent/padding", guarded ? 0 : 1, 0);
        if constexpr (!public_entry)
        {
            arena->verify();
            std::uint64_t memory[5] = {std::uint64_t(bb), std::uint64_t(hb),
                                       std::uint64_t(resources->retained_backend_bytes()),
                                       std::uint64_t(resources->retained_host_bytes()),
                                       std::uint64_t(bb + hb + resources->retained_backend_bytes() +
                                                     resources->retained_host_bytes())};
            MPI_Allreduce(MPI_IN_PLACE, memory, 5, MPI_UINT64_T, MPI_MAX, h->comm);
            std::uint64_t cold[4] = {trace.host_new, trace.raw_gpu_alloc, trace.ddla_alloc,
                                     trace.resource_setup};
            MPI_Allreduce(MPI_IN_PLACE, cold, 4, MPI_UINT64_T, MPI_MAX, h->comm);
            if (!h->myid)
                std::cout << "CHEB n=" << c.n << " q=" << p.q << " bq=" << p.bq
                          << " bytes=" << sizeof(T) << " grid=" << p.pr << "x" << p.pc
                          << " warm=" << c.warm << " kind=" << c.warm_kind
                          << " iteration=" << iterations << " residual=" << std::setprecision(6)
                          << count.maximum_residual << " arena_backend_max=" << memory[0]
                          << " arena_host_max=" << memory[1]
                          << " retained_backend_max=" << memory[2]
                          << " retained_host_max=" << memory[3]
                          << " tracked_total_rank_max=" << memory[4]
                          << " trace_host_new_max=" << cold[0] << " trace_gpu_alloc_max=" << cold[1]
                          << " trace_ddla_alloc_max=" << cold[2] << " trace_setup_max=" << cold[3]
                          << std::endl;
        }
        else if (!h->myid)
        {
            std::cout << "CHEB_PUBLIC n=" << c.n << " q=" << (c.nev ? c.nev + c.nex : 0)
                      << " bytes=" << sizeof(T) << " grid=" << h->nprows_ << "x" << h->npcols_
                      << " warm=" << c.warm << " kind=" << c.warm_kind << " alias=" << c.alias
                      << " iterations=" << iterations << " converged=" << converged
                      << " info=" << info << std::endl;
        }
    }
}

template <DdlaBackend B, class T> void negative_cases(DdlaHandle_t h)
{
    using R = typename real_type<T>::type;
    const int n = 5, nev = 1;
    int da[DDLA_DLEN_], dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da, h, n, n, 2, 2, 0, 0));
    DDLA_CHECK(ddlaDescInit(dz, h, n, nev, 2, 1, 0, 0));
    DeviceBuffer<T> a(h, local_size(h, da)), z(h, local_size(h, dz));
    std::vector<R> w(3, R(-313));
    const double tolerance = std::is_same<R, float>::value ? 4e-5 : 2e-11;
    for (int kind = 0; kind < 4; ++kind)
    {
        const R large = std::scalbn(R(.75), std::numeric_limits<R>::max_exponent);
        const R tiny = std::numeric_limits<R>::denorm_min();
        const auto input =
            make_local<T>(h, da,
                          [&](int i, int j)
                          {
                              if (i < j)
                                  return value<T>(NAN);
                              if (i != j)
                                  return T(0);
                              if (kind == 0 && i == 0)
                                  return value<T>(NAN);
                              if (kind == 1)
                                  return i == 0 ? T(large) : (i == 1 ? T(tiny) : T(1));
                              return T(i + 1);
                          });
        auto initial = make_local<T>(h, dz, [&](int i, int)
                                     { return kind == 2 && i == 0 ? value<T>(NAN) : T(1); });
        upload(h, a.ptr, input);
        upload(h, z.ptr, initial);
        int iterations = 0, converged = 0, info = -1;
        ddla::pheevd_chebyshev<T>(h, 'L', n, nev, 0, a.ptr, da, w.data() + 1, z.ptr, dz, tolerance,
                                  0, 8, kind == 2, iterations, converged, info);
        const int expected = kind == 0
                                 ? ch::nonfinite
                                 : (kind == 1 ? ch::scaling_loss
                                              : (kind == 2 ? ch::nonfinite : ch::iteration_limit));
        require_close(h, "Chebyshev explicit negative status", info == expected ? 0 : 1, 0);
        if (kind != 3)
        {
            const auto after = download(h, a.ptr, input.size());
            require_close(h, "Chebyshev failure before A mutation",
                          std::memcmp(after.data(), input.data(), input.size() * sizeof(T)) ? 1 : 0,
                          0);
        }
        require_close(h, "Chebyshev negative value guards",
                      w.front() == R(-313) && w.back() == R(-313) ? 0 : 1, 0);
    }
    // Public controls are required and invalid values fail before mutation.
    const auto valid_input = make_local<T>(
        h, da, [&](int i, int j) { return i < j ? value<T>(NAN) : T(i == j ? i + 1 : 0); });
    const auto valid_z = make_local<T>(h, dz, [&](int, int) { return T(1); });
    for (int bad = 0; bad < 5; ++bad)
    {
        upload(h, a.ptr, valid_input);
        upload(h, z.ptr, valid_z);
        std::fill(w.begin(), w.end(), R(-313));
        int iterations = -11, converged = -22, info = -33;
        bool threw = false;
        try
        {
            ddla::pheevd_chebyshev<T>(h, 'L', n, nev, bad == 0 ? n : 0, a.ptr, da, w.data() + 1,
                                      z.ptr, dz, bad == 1 ? 0 : (bad == 2 ? NAN : tolerance),
                                      bad == 3 ? -1 : 4, bad == 4 ? 0 : 8, false, iterations,
                                      converged, info);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        require_close(h, "Chebyshev public invalid controls reject", threw ? 0 : 1, 0);
        const auto after_a = download(h, a.ptr, valid_input.size());
        const auto after_z = download(h, z.ptr, valid_z.size());
        const bool unchanged =
            std::memcmp(after_a.data(), valid_input.data(), valid_input.size() * sizeof(T)) == 0 &&
            std::memcmp(after_z.data(), valid_z.data(), valid_z.size() * sizeof(T)) == 0 &&
            w[0] == R(-313) && w[1] == R(-313) && w[2] == R(-313) && iterations == -11 &&
            converged == -22 && info == -33;
        require_close(h, "Chebyshev public rejects before input/output writes", unchanged ? 0 : 1,
                      0);
    }
    // The stored components are finite/reversibly scalable, but this rank-one
    // matrix has eigenvalue 2*large outside the scalar output range.
    int small_a[DDLA_DLEN_], small_z[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(small_a, h, 2, 2, 1, 1, 0, 0));
    DDLA_CHECK(ddlaDescInit(small_z, h, 2, 2, 1, 1, 0, 0));
    DeviceBuffer<T> large_a(h, local_size(h, small_a)), large_z(h, local_size(h, small_z));
    const R large = std::scalbn(R(.75), std::numeric_limits<R>::max_exponent);
    const auto input =
        make_local<T>(h, small_a, [&](int i, int j) { return i >= j ? T(large) : value<T>(NAN); });
    upload(h, large_a.ptr, input);
    R large_w[2] = {0, 0};
    int iterations = 0, converged = 0, info = -1;
    ddla::pheevd_chebyshev<T>(h, 'L', 2, 2, 0, large_a.ptr, small_a, large_w, large_z.ptr, small_z,
                              tolerance, 2, 8, false, iterations, converged, info);
    require_close(h, "Chebyshev unrepresentable eigenvalue status",
                  info == ch::unrepresentable ? 0 : 1, 0);
    const R tiny = std::numeric_limits<R>::denorm_min();
    const auto subnormal =
        make_local<T>(h, small_a, [&](int i, int j)
                      { return i < j ? value<T>(NAN) : (i == 1 && j == 1 ? T(0) : T(tiny)); });
    upload(h, large_a.ptr, subnormal);
    ddla::pheevd_chebyshev<T>(h, 'L', 2, 2, 0, large_a.ptr, small_a, large_w, large_z.ptr, small_z,
                              tolerance, 2, 8, false, iterations, converged, info);
    require_close(h, "Chebyshev finite quantization failure status",
                  info == ch::unrepresentable ? 0 : 1, 0);
}

template <DdlaBackend B, class T> void suite(DdlaHandle_t h, bool quick)
{
    Case c;
    run_case<B, T>(h, c, true);
    c.triangle = 'U';
    c.warm = true;
    run_case<B, T>(h, c);
    c.warm_kind = 1;
    run_case<B, T>(h, c);
    c.warm_kind = 2;
    run_case<B, T>(h, c);
    c.warm_kind = 3;
    run_case<B, T>(h, c);
    c = Case{};
    c.n = 3;
    c.nev = 1;
    c.nex = 0;
    c.binary = true;
    c.dense = false;
    c.warm = true;
    c.warm_kind = 1;
    run_case<B, T>(h, c);
    c.tolerance = std::is_same<typename real_type<T>::type, float>::value ? 1e-20 : 1e-24;
    run_case<B, T>(h, c); // Exact binary action supports this true tight criterion.
    c = Case{};
    c.n = 3;
    c.nev = 3;
    c.nex = 0;
    c.binary = true;
    c.dense = false;
    run_case<B, T>(h, c); // True collapsed bounds: the matrix is identically zero.
    c = Case{};
    c.n = 5;
    c.nev = 2;
    c.nex = 1;
    run_case<B, T>(h, c); // q=3 tests the bounded 2-by-2 projected leaf floor.
    c = Case{};
    c.n = 1;
    c.nev = 1;
    c.nex = 0;
    c.scalar = true;
    run_case<B, T>(h, c);
    c = Case{};
    c.n = 5;
    c.nev = 5;
    c.nex = 0;
    run_case<B, T>(h, c);
    c = Case{};
    c.n = 0;
    c.nev = 0;
    c.nex = 0;
    run_case<B, T>(h, c);
    c = Case{};
    c.nev = 0;
    c.nex = 0;
    run_case<B, T>(h, c);
    negative_cases<B, T>(h);
    if (quick)
        return;
    c = Case{};
    c.alias = true;
    run_case<B, T>(h, c);
    c = Case{};
    c.n = 5;
    c.nev = 2;
    c.nex = 1;
    c.alias = true;
    c.warm = true;
    c.dense = false;
    run_case<B, T>(h, c);
    c = Case{};
    c.dense = false;
    c.n = 5;
    c.nev = 2;
    c.nex = 1;
    c.scale = std::numeric_limits<typename real_type<T>::type>::max_exponent - 5;
    run_case<B, T>(h, c);
    // Exactly representable diagonal powers near the bottom of the exponent
    // range avoid confusing entry-construction rounding with solver loss.
    c.scale = std::numeric_limits<typename real_type<T>::type>::min_exponent - 2;
    run_case<B, T>(h, c);
    c.scalar = true;
    c.scale = std::numeric_limits<typename real_type<T>::type>::min_exponent -
              std::numeric_limits<typename real_type<T>::type>::digits + 2;
    run_case<B, T>(h, c);
}

template <DdlaBackend B> void execute(int pr, int pc, bool quick)
{
    DdlaHandle_t h = nullptr;
    ddlaInit(h, B);
    select_transport<B>(h);
    ddlaSet(h, MPI_COMM_WORLD, pr, pc);
    if (!h->myid) std::cout << "CHEBYSHEV_TRANSPORT backend=" << int(B)
        << " requested=" << transport << " actual=" << int(ddlaGetCommunication(h)) << std::endl;
    suite<B, float>(h, quick);
    suite<B, double>(h, quick);
    suite<B, std::complex<float>>(h, quick);
    suite<B, std::complex<double>>(h, quick);
    ddlaDestroy(h);
}
} // namespace
int main(int argc, char **argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int ranks = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool gpu = false, quick = false;
    int pr = 1, pc = ranks;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--gpu")
                gpu = true;
            else if (arg == "--quick")
                quick = true;
            else if (arg == "--transport" && i + 1 < argc)
            {
                transport = argv[++i];
                if (transport != "ccl" && transport != "staged" && transport != "mpi")
                    throw std::invalid_argument("--transport requires ccl|staged|mpi");
            }
            else if (arg == "--grid" && i + 1 < argc)
            {
                if (!parse_grid_spec(argv[++i], pr, pc) || static_cast<long long>(pr) * pc != ranks)
                    throw std::invalid_argument("--grid must match the allocated MPI ranks");
            }
            else
                throw std::invalid_argument("usage: test_chebyshev [--gpu] [--quick] [--grid RxC] [--transport ccl|staged|mpi]");
        }
        if (gpu)
        {
#if DDLA_HAS_GPU
            execute<DdlaBackend::GPU>(pr, pc, quick);
#else
            throw std::runtime_error("--gpu requested in a CPU-only build");
#endif
        }
        else
            execute<DdlaBackend::CPU>(pr, pc, quick);
        if (!rank)
            std::cout << (public_entry ? "PHEEVD_CHEBYSHEV_PUBLIC_PASS"
                                       : "CHEBYSHEV_PROTOTYPE_PASS")
                      << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "rank " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
