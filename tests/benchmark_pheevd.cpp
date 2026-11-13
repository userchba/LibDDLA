#include "test_backend.h"
// ---------------------------------------------------------------------------
// Scale benchmark of the full Hermitian eigensolver pheevd('V'): time, per-rank memory (DDLA_EIG_PROFILE=1 prints
// the per-stage PHEEVDPROF / PHEEVDMEM lines) and, with --check, the residual ||A V - V Lambda||_F /
// (n ||A||_F eps) and orthogonality ||V^H V - I|| / (n eps) at the SAME size (a wrong pipeline still yields
// plausible timings -- every timing here has its correctness check).
//
//   mpirun -n 4 ./benchmark_pheevd [--grid RxC] [--nb NB] [--check] [--complex] [--reps R] [--uplo U|L] [--nopoison] n [n ...]
//
// A is a dense random symmetric (Hermitian with --complex) matrix, built per entry from a hash (no host copy of the
// matrix); only the --uplo triangle is stored -- the other triangle is poisoned (the ScaLAPACK contract), unless
// --nopoison fills it with the true values (for libraries predating the uplo fix, which read it).  The input is
// rebuilt before every repetition (pheevd destroys A).  Time = max over ranks.  The checks need A again (rebuilt)
// plus AZ and G, i.e. about 4 n^2/P elements per rank on top of Z: size n accordingly.  Heavy: never run by ctest.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"
#include "eig_test_common.h"

using namespace api_grid_test;

namespace {

double hash01(std::uint64_t a, std::uint64_t b, int tag)
{
    std::uint64_t x = (a * 0x9E3779B97F4A7C15ull) ^ ((b + 1) * 0xC2B2AE3D27D4EB4Full) ^ (static_cast<std::uint64_t>(tag) * 0x165667B19E3779F9ull);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0 - 0.5;
}

template <typename T> struct is_cx : std::false_type {};
template <typename R> struct is_cx<std::complex<R>> : std::true_type {};

// full Hermitian matrix entry
template <typename T>
T entry(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    if (i == j) return T(static_cast<rt>(hash01(i, i, 1)));
    const std::uint64_t lo = static_cast<std::uint64_t>(std::min(i, j)), hi = static_cast<std::uint64_t>(std::max(i, j));
    const rt re = static_cast<rt>(hash01(hi, lo, 2));
    if constexpr (!is_cx<T>::value) {
        return T(re);
    } else {
        const rt im = static_cast<rt>(hash01(hi, lo, 3));
        return (i > j) ? T(re, im) : T(re, -im);
    }
}

template <typename T>
void run(const ddla::DdlaHandle_t& handle, int n, int nb, bool check, int reps, char uplo, bool poison, bool show)
{
    using rt = typename ddla::real_type<T>::type;
    int descA[ddla::DDLA_DLEN_], descZ[ddla::DDLA_DLEN_];
    ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0);
    ddlaDescInit(descZ, handle, n, n, nb, nb, 0, 0);
    const bool upper = (uplo == 'U');
    DeviceBuffer<T> d_A(handle, local_size(handle, descA));
    DeviceBuffer<T> d_Z(handle, local_size(handle, descZ));
    std::vector<rt> w(static_cast<std::size_t>(n));

    // the stored triangle holds the matrix; the opposite strict triangle is poison (never to be read)
    auto fill = [&](bool full) {
        const auto local = make_local<T>(handle, descA, [&](int i, int j) {
            if (full || !poison || (upper ? i <= j : i >= j)) return entry<T>(i, j);
            return T(static_cast<rt>(1.0e6));
        });
        upload<T>(handle, d_A.ptr, local);
        check_ddla_sync(handle);
    };

    std::vector<double> times;
    int info = 0;
    for (int r = 0; r < reps; ++r) {
        fill(false);
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        ddla::pheevd(handle, 'V', uplo, n,n, d_A.ptr, descA, w.data(), d_Z.ptr, descZ, info);
        check_ddla_sync(handle);
        double dt = MPI_Wtime() - t0, dmax = 0.0;
        MPI_Allreduce(&dt, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        times.push_back(dmax);
    }
    double resid = -1.0, ortho = -1.0;
    int bad = (info != 0) ? 1 : 0;
    if (check) {
        fill(true);   // the residual needs the full matrix (pheevd destroyed A)
        resid = eig_test::residual_metric<T>(handle, d_A.ptr, descA, d_Z.ptr, descZ, w.data(), n);
        ortho = eig_test::orthogonality_metric<T>(handle, d_Z.ptr, descZ, n);
        if (!(resid <= 50.0) || !(ortho <= 50.0)) bad = 1;
    }
    for (int i = 1; i < n; ++i)
        if (!(w[static_cast<std::size_t>(i)] >= w[static_cast<std::size_t>(i) - 1])) bad = 1;   // ascending
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (show) {
        const double tmin = *std::min_element(times.begin(), times.end());
        std::vector<double> s(times);
        std::sort(s.begin(), s.end());
        std::cout << "PHEEVD_BENCH n=" << n << " type=" << (is_cx<T>::value ? "z" : "d") << " uplo=" << uplo
                  << " nb=" << nb << (poison ? "" : " nopoison") << " time_min=" << std::fixed << std::setprecision(3) << tmin
                  << " time_med=" << s[s.size() / 2];
        if (check) std::cout << std::scientific << std::setprecision(3) << " resid=" << resid << " ortho=" << ortho;
        std::cout << (bad ? "  FAIL" : "  ok") << std::endl;
    }
    if (bad) MPI_Abort(MPI_COMM_WORLD, 1);
}

} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nprows = 0, npcols = 0, nb = 64, reps = 1;
    bool check = false, cx = false, poison = true;
    char uplo = 'U';
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--grid" && i + 1 < argc) {
            const std::string g(argv[++i]);
            const std::size_t x = g.find('x');
            if (x != std::string::npos) { nprows = std::atoi(g.substr(0, x).c_str()); npcols = std::atoi(g.substr(x + 1).c_str()); }
        } else if (a == "--nb" && i + 1 < argc) nb = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--uplo" && i + 1 < argc) uplo = (argv[++i][0] == 'L') ? 'L' : 'U';
        else if (a == "--check") check = true;
        else if (a == "--nopoison") poison = false;
        else if (a == "--complex") cx = true;
        else sizes.push_back(std::atoi(a.c_str()));
    }
    if (sizes.empty()) sizes = {2000, 5000};
    if (nprows == 0) {   // most square divisor pair of nprocs
        nprows = 1; npcols = nprocs;
        for (int r = 1; r <= nprocs; ++r)
            if (nprocs % r == 0 && r * r <= nprocs) { nprows = r; npcols = nprocs / r; }
    }
    if (nprows * npcols != nprocs) {
        if (rank == 0) std::cerr << "--grid " << nprows << "x" << npcols << " needs " << nprows * npcols << " ranks" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddlaInit(handle, ddla_test::execution_backend);
    ddla::ddlaSet(handle, MPI_COMM_WORLD, nprows, npcols);
    if (rank == 0)
        std::cout << "=== pheevd scale benchmark: ranks=" << nprocs << " grid=" << nprows << "x" << npcols << " nb=" << nb
                  << (check ? " (with checks)" : "") << " ===" << std::endl;
    for (const int n : sizes) {
        if (cx) run<std::complex<double>>(handle, n, nb, check, reps, uplo, poison, rank == 0);
        else run<double>(handle, n, nb, check, reps, uplo, poison, rank == 0);
    }
    check_ddla_sync(handle);
    ddla::ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
