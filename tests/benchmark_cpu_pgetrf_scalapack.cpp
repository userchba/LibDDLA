/**
 * @file benchmark_cpu_pgetrf_scalapack.cpp
 * @brief DDLA CPU pgetrf vs ScaLAPACK getrf correctness and performance benchmark.
 *
 * Uses CPU handles in every build when a ScaLAPACK library is available.
 * Compares double (pdgetrf) and std::complex<double> (pzgetrf), full-matrix
 * factorization A(1:n,1:n), on a 2x2 process grid (4 ranks).
 * The DDLA panel method follows --method: gather (default), inplace, or both
 * (alternated across repetitions), applied through the DDLA_LU_PANEL contract.
 *
 * CSV columns:
 * n,nb,type,method,ranks,grid,warmup,repeats,ddla_s,scalapack_s,
 * ddla_gflops,scalapack_gflops,ddla_over_scalapack_time,
 * max_abs_error,max_rel_error,pivot_mismatches
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "mpi_test_init.h"

extern "C" {
/* BLACS C interface */
void Cblacs_get(int context, int request, int* val);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);

/* Fortran interface wrappers */
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc,
            const int* nprocs);
void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
               const int* irsrc, const int* icsrc, const int* ictxt, const int* lld,
               int* info);
void pdgetrf_(const int* m, const int* n, double* A, const int* ia, const int* ja,
              const int* descA, int* ipiv, int* info);
void pzgetrf_(const int* m, const int* n, std::complex<double>* A, const int* ia,
              const int* ja, const int* descA, int* ipiv, int* info);
}

#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"

using namespace ddla;

template <class Z> struct TypeTraits;

template <> struct TypeTraits<double> {
    static const char* name() { return "d"; }
    static bool is_complex() { return false; }
    /* LU with partial pivoting: (2/3) n^3 flops for real types. */
    static double flop_scale() { return 2.0 / 3.0; }
    static void getrf(const int* m, const int* n, double* A, const int* ia,
                      const int* ja, const int* desc, int* ipiv, int* info)
    {
        pdgetrf_(m, n, A, ia, ja, desc, ipiv, info);
    }
};

template <> struct TypeTraits<std::complex<double>> {
    static const char* name() { return "z"; }
    static bool is_complex() { return true; }
    /* Complex multiply-add costs four real FLOPs: (8/3) n^3. */
    static double flop_scale() { return 8.0 / 3.0; }
    static void getrf(const int* m, const int* n, std::complex<double>* A, const int* ia,
                      const int* ja, const int* desc, int* ipiv, int* info)
    {
        pzgetrf_(m, n, A, ia, ja, desc, ipiv, info);
    }
};

/* Deterministic element value for global index (ig, jg) in [-0.5, 0.5]. */
template <class Z>
static Z entry_val(int ig, int jg)
{
    std::uint64_t x = std::uint64_t(ig + 1) * 0x9e3779b97f4a7c15ULL
                    + std::uint64_t(jg + 1) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    double re = double(x & 0xffffff) / 16777216.0 - 0.5;
    if constexpr (std::is_same_v<Z, double>) return re;
    else {
        double im = double((x >> 24) & 0xffffff) / 16777216.0 - 0.5;
        return Z(re, im);
    }
}

template <class Z>
static void fill_local(Z* data, const DdlaHandle_t& handle, const int* desc)
{
    for (int jl = 0; jl < ddla_test::n_loc(handle, desc); ++jl) {
        int jg = indxl2g(jl, desc[DDLA_NB_], ddla_test::mypcol(handle), desc[DDLA_CSRC_],
                         ddla_test::npcols(handle));
        for (int il = 0; il < ddla_test::m_loc(handle, desc); ++il) {
            int ig = indxl2g(il, desc[DDLA_MB_], ddla_test::myprow(handle), desc[DDLA_RSRC_],
                             ddla_test::nprows(handle));
            data[il + std::size_t(jl) * desc[DDLA_LLD_]] = entry_val<Z>(ig, jg);
        }
    }
}

static std::vector<int> parse_sizes(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const std::size_t mid = v.size() / 2;
    return v.size() % 2 == 0 ? 0.5 * (v[mid - 1] + v[mid]) : v[mid];
}

template <class Z>
static int run_type(const DdlaHandle_t& handle, int ictxt, int myid,
                    const std::vector<int>& sizes, int nb, int warmup, int repeats,
                    const std::string& method_opt)
{
    using TT = TypeTraits<Z>;
    (void)ictxt;
    int exit_status = 0;
    const int ia = 1, ja = 1, irsrc = 0, icsrc = 0;

    for (int n : sizes) {
        /* DDLA descriptor */
        int descA[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, irsrc, icsrc));
        const int loc_m = ddla_test::m_loc(handle, descA);
        const int loc_n = ddla_test::n_loc(handle, descA);
        const int lld = descA[DDLA_LLD_];

        /* BLACS descriptor */
        int desc_sc[9];
        int info_desc = 0;
        descinit_(desc_sc, &n, &n, &nb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info_desc);
        int desc_failure = info_desc != 0;
        int any_desc_failure = 0;
        MPI_Allreduce(&desc_failure, &any_desc_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (any_desc_failure) {
            if (myid == 0) std::cerr << "descinit failed for n=" << n << " info=" << info_desc << "\n";
            exit_status = 1;
            break;
        }

        /* Local storage: pristine input plus one working copy per backend. */
        const std::size_t local_elems = std::size_t(lld) * loc_n;
        std::vector<Z> a0(local_elems), a_ddla(local_elems), a_sca(local_elems);
        /* ScaLAPACK wants IPIV of at least MB + numroc(...) local rows. */
        std::vector<int> ipiv_ddla(std::size_t(loc_m) + nb, 0);
        std::vector<int> ipiv_sca(std::size_t(loc_m) + nb, 0);

        fill_local(a0.data(), handle, descA);

        struct Samples {
            std::vector<double> ddla_gather, ddla_inplace, scalapack;
        } samples;

        /* Which panel methods this run exercises. */
        std::vector<std::string> methods;
        if (method_opt == "both") methods = {"gather", "inplace"};
        else methods = {method_opt};

        /* ---- Warmup: one untimed run per DDLA method and for ScaLAPACK. */
        for (const auto& method : methods) {
            std::copy(a0.begin(), a0.end(), a_ddla.begin());
            std::fill(ipiv_ddla.begin(), ipiv_ddla.end(), 0);
            setenv("DDLA_LU_PANEL", method.c_str(), 1);
            int info = -1;
            ddla::pgetrf(handle, n, n, a_ddla.data(), ia, ja, descA,
                                           ipiv_ddla.data(), info);
            if (info != 0) {
                if (myid == 0) std::cerr << "DDLA pgetrf warmup info=" << info << " n=" << n << "\n";
                exit_status = 1;
            }
        }
        std::copy(a0.begin(), a0.end(), a_sca.begin());
        std::fill(ipiv_sca.begin(), ipiv_sca.end(), 0);
        {
            int info = -1, nn = n;
            TT::getrf(&nn, &nn, a_sca.data(), &ia, &ja, desc_sc, ipiv_sca.data(), &info);
            if (info != 0) {
                if (myid == 0) std::cerr << "ScaLAPACK getrf warmup info=" << info << " n=" << n << "\n";
                exit_status = 1;
            }
        }
        if (exit_status != 0) break;
        (void)warmup;

        /* ---- Timed repetitions: alternate the DDLA method across reps; time
         *      ScaLAPACK every rep; flip the execution order with rep parity. */
        for (int r = 0; r < repeats; ++r) {
            const std::string& method = methods[r % methods.size()];
            setenv("DDLA_LU_PANEL", method.c_str(), 1);

            auto timed_ddla = [&]() {
                std::copy(a0.begin(), a0.end(), a_ddla.begin());
                std::fill(ipiv_ddla.begin(), ipiv_ddla.end(), 0);
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
                int info = -1;
                ddla::pgetrf(handle, n, n, a_ddla.data(), ia, ja, descA,
                                               ipiv_ddla.data(), info);
                double t1 = MPI_Wtime();
                int failed = info != 0;
                MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if (failed) {
                    if (myid == 0) std::cerr << "DDLA pgetrf info=" << info << " n=" << n << "\n";
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                double dt = t1 - t0;
                MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                (method == "gather" ? samples.ddla_gather : samples.ddla_inplace).push_back(dt);
            };

            auto timed_scalapack = [&]() {
                std::copy(a0.begin(), a0.end(), a_sca.begin());
                std::fill(ipiv_sca.begin(), ipiv_sca.end(), 0);
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
                int info = -1, nn = n;
                TT::getrf(&nn, &nn, a_sca.data(), &ia, &ja, desc_sc, ipiv_sca.data(), &info);
                double t1 = MPI_Wtime();
                int failed = info != 0;
                MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if (failed) {
                    if (myid == 0) std::cerr << "ScaLAPACK getrf info=" << info << " n=" << n << "\n";
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                double dt = t1 - t0;
                MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                samples.scalapack.push_back(dt);
            };

            if (r % 2 == 0) { timed_ddla(); timed_scalapack(); }
            else { timed_scalapack(); timed_ddla(); }
        }

        /* ---- Correctness: fresh runs, compare factors and pivots. */
        double glob_max_abs = 0.0, glob_max_ref = 0.0;
        int glob_pivot_mismatch = 0, glob_nonfinite = 0;
        {
            std::copy(a0.begin(), a0.end(), a_sca.begin());
            std::fill(ipiv_sca.begin(), ipiv_sca.end(), 0);
            int info = -1, nn = n;
            TT::getrf(&nn, &nn, a_sca.data(), &ia, &ja, desc_sc, ipiv_sca.data(), &info);
            int failed = info != 0;
            MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (failed) {
                if (myid == 0) std::cerr << "ScaLAPACK correctness run info=" << info << "\n";
                exit_status = 1;
                break;
            }

            double local_max_ref = 0.0;
            for (std::size_t i = 0; i < local_elems; ++i)
                local_max_ref = std::max(local_max_ref, std::abs(a0[i]));

            for (const auto& method : methods) {
                std::copy(a0.begin(), a0.end(), a_ddla.begin());
                std::fill(ipiv_ddla.begin(), ipiv_ddla.end(), 0);
                setenv("DDLA_LU_PANEL", method.c_str(), 1);
                int info = -1;
                ddla::pgetrf(handle, n, n, a_ddla.data(), ia, ja, descA,
                                               ipiv_ddla.data(), info);
                int failed = info != 0;
                MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if (failed) {
                    if (myid == 0) std::cerr << "DDLA correctness run info=" << info << "\n";
                    exit_status = 1;
                    break;
                }

                double local_max_abs = 0.0;
                int local_nonfinite = 0;
                for (std::size_t i = 0; i < local_elems; ++i) {
                    double d = std::abs(a_ddla[i] - a_sca[i]);
                    local_nonfinite = local_nonfinite || !std::isfinite(d);
                    if (std::isfinite(d)) local_max_abs = std::max(local_max_abs, d);
                }
                int local_pivot_mismatch = 0;
                for (int i = 0; i < loc_m; ++i)
                    local_pivot_mismatch += ipiv_ddla[i] != ipiv_sca[i];

                double g_abs = 0.0, g_ref = 0.0;
                int g_piv = 0, g_nf = 0;
                MPI_Allreduce(&local_max_abs, &g_abs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&local_max_ref, &g_ref, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&local_pivot_mismatch, &g_piv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_nonfinite, &g_nf, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

                /* CSV row per method. */
                if (myid == 0) {
                    double ddla_med = method == "gather" ? median(samples.ddla_gather)
                                                          : median(samples.ddla_inplace);
                    double sca_med = median(samples.scalapack);
                    double flops = TT::flop_scale() * std::pow(double(n), 3.0);
                    std::cout << std::scientific << std::setprecision(6)
                              << n << "," << nb << "," << TT::name() << "," << method << ","
                              << 4 << ",2x2," << warmup << "," << repeats << ","
                              << ddla_med << "," << sca_med << ","
                              << flops / ddla_med / 1e9 << "," << flops / sca_med / 1e9 << ","
                              << ddla_med / sca_med << ","
                              << g_abs << "," << (g_ref > 0 ? g_abs / g_ref : 0.0) << ","
                              << double(g_piv) << "\n" << std::fixed;
                }
                if (method == "gather") {
                    glob_max_abs = g_abs; glob_max_ref = g_ref;
                    glob_pivot_mismatch = g_piv; glob_nonfinite = g_nf;
                } else {
                    glob_max_abs = std::max(glob_max_abs, g_abs);
                    glob_pivot_mismatch = std::max(glob_pivot_mismatch, g_piv);
                    glob_nonfinite = std::max(glob_nonfinite, g_nf);
                }
            }
            if (exit_status != 0) break;
        }

        double max_rel = glob_max_ref > 0 ? glob_max_abs / glob_max_ref : 0.0;
        /* Factors from two independent LU implementations agree only to LU
         * roundoff (O(n·eps) growth): 1e-7 relative with EXACT pivot
         * agreement is the pass rule; a real defect shows O(1) deviations
         * or a pivot mismatch. */
        bool pass = !glob_nonfinite && max_rel <= 1e-7 && glob_pivot_mismatch == 0;
        if (!pass) {
            if (myid == 0)
                std::cerr << "Correctness FAIL n=" << n << " max_abs=" << glob_max_abs
                          << " max_rel=" << max_rel
                          << " pivot_mismatches=" << glob_pivot_mismatch
                          << " nonfinite=" << glob_nonfinite << "\n";
            exit_status = 1;
            break;
        }
    }
    return exit_status;
}

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);

    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nb = 128;
    int warmup = 1;
    int repeats = 4;
    std::string sizes_str = "2000,5000,10000,15000";
    std::string type = "z";
    std::string method = "gather";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc) sizes_str = argv[++i];
        else if (arg == "--nb" && i + 1 < argc) nb = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (arg == "--type" && i + 1 < argc) type = argv[++i];
        else if (arg == "--method" && i + 1 < argc) method = argv[++i];
        else {
            if (myid == 0)
                std::cerr << "Unknown option: " << arg << "\n"
                          << "Usage: " << argv[0]
                          << " [--sizes S1,S2,...] [--nb N] [--warmup N] [--repeats N]"
                          << " [--type d|z] [--method gather|inplace|both]\n";
            MPI_Finalize();
            return 1;
        }
    }
    if (nb < 1 || warmup < 0 || repeats < 1) {
        if (myid == 0) std::cerr << "Invalid nb/warmup/repeats\n";
        MPI_Finalize();
        return 1;
    }
    if (type != "d" && type != "z") {
        if (myid == 0) std::cerr << "--type must be d or z\n";
        MPI_Finalize();
        return 1;
    }
    if (method != "gather" && method != "inplace" && method != "both") {
        if (myid == 0) std::cerr << "--method must be gather, inplace or both\n";
        MPI_Finalize();
        return 1;
    }
    std::vector<int> sizes = parse_sizes(sizes_str);
    if (sizes.empty() || std::any_of(sizes.begin(), sizes.end(), [](int n) { return n < 1; })) {
        if (myid == 0) std::cerr << "Sizes must be positive integers\n";
        MPI_Finalize();
        return 1;
    }
    if (nprocs != 4) {
        if (myid == 0) std::cerr << "This benchmark requires exactly 4 MPI ranks (got " << nprocs << ")\n";
        MPI_Finalize();
        return 1;
    }

    /* ---- DDLA handle 2x2 row-major grid ----------------------------- */
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, DdlaBackend::CPU);
    ddlaSet(handle, MPI_COMM_WORLD, 2, 2, 'R');

    /* ---- BLACS grid 2x2 row-major ------------------------------------ */
    int ictxt;
    Cblacs_get(-1, 0, &ictxt);
    int blacs_nprow = 2, blacs_npcol = 2;
    Cblacs_gridinit(&ictxt, "R", blacs_nprow, blacs_npcol);
    int blacs_myrow, blacs_mycol;
    Cblacs_gridinfo(ictxt, &blacs_nprow, &blacs_npcol, &blacs_myrow, &blacs_mycol);

    int ddla_nprows = 0, ddla_npcols = 0, ddla_myrow = 0, ddla_mycol = 0;
    ddlaGetGridDims(handle, ddla_nprows, ddla_npcols);
    ddlaGetGridCoords(handle, ddla_myrow, ddla_mycol);
    int grid_mismatch = ddla_nprows != blacs_nprow || ddla_npcols != blacs_npcol
                     || ddla_myrow != blacs_myrow || ddla_mycol != blacs_mycol;
    int any_grid_mismatch = 0;
    MPI_Allreduce(&grid_mismatch, &any_grid_mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_grid_mismatch) {
        if (myid == 0) std::cerr << "DDLA/BLACS grid mismatch\n";
        Cblacs_gridexit(ictxt);
        ddlaDestroy(handle);
        MPI_Finalize();
        return 1;
    }

    if (myid == 0) {
        std::cout << "n,nb,type,method,ranks,grid,warmup,repeats,ddla_s,scalapack_s,"
                  << "ddla_gflops,scalapack_gflops,ddla_over_scalapack_time,"
                  << "max_abs_error,max_rel_error,pivot_mismatches\n";
        std::cout << "# DDLA CPU pgetrf vs ScaLAPACK " << (type == "d" ? "pdgetrf" : "pzgetrf")
                  << " (" << (type == "d" ? "double" : "complex<double>")
                  << "), 4 ranks, 2x2 grid, DDLA_LU_PANEL per --method\n";
    }

    int status;
    if (type == "d")
        status = run_type<double>(handle, ictxt, myid, sizes, nb, warmup, repeats, method);
    else
        status = run_type<std::complex<double>>(handle, ictxt, myid, sizes, nb, warmup, repeats, method);

    Cblacs_gridexit(ictxt);
    ddlaDestroy(handle);
    MPI_Finalize();
    return status;
}
