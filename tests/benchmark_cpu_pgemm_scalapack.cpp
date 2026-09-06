/**
 * @file benchmark_cpu_pgemm_scalapack.cpp
 * @brief DDLA CPU pgemm vs ScaLAPACK pzgemm correctness and performance benchmark.
 *
 * Only built for DDLA_USE_CPU when a ScaLAPACK library is available.
 * Compares std::complex<double>, N,N only, on a 2x2 process grid (4 ranks).
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <mpi.h>

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
void pzgemm_(const char* transa, const char* transb, const int* m, const int* n,
             const int* k, const std::complex<double>* alpha,
             const std::complex<double>* A, const int* ia, const int* ja, const int* descA,
             const std::complex<double>* B, const int* ib, const int* jb, const int* descB,
             const std::complex<double>* beta,
             std::complex<double>* C, const int* ic, const int* jc, const int* descC);
}

#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "ddla_stream.h"

using namespace ddla;
using Z = std::complex<double>;

/* ------------------------------------------------------------------ */
/* Deterministic element value for global index (ig, jg)               */
/* ------------------------------------------------------------------ */
static Z elem_val_A(int ig, int jg, int /*ncols*/) {
    double re = ((ig * 1000 + jg    + 1) % 13) - 6.0;
    double im = ((ig * 1000 + jg * 2 + 5) % 13) - 6.0;
    return Z(re, im);
}
static Z elem_val_B(int ig, int jg, int /*ncols*/) {
    double re = ((ig * 2000 + jg * 3 + 7) % 17) - 8.0;
    double im = ((ig * 1500 + jg * 5 + 3) % 17) - 8.0;
    return Z(re, im);
}
static Z elem_val_C0(int ig, int jg, int /*ncols*/) {
    double re = ((ig * 700  + jg * 7 + 11) % 11) - 5.0;
    double im = ((ig * 900  + jg * 9 + 13) % 11) - 5.0;
    return Z(re, im);
}

/* ------------------------------------------------------------------ */
/* Parse comma-separated sizes                                          */
/* ------------------------------------------------------------------ */
static std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Helper: allocate and fill local array from global indices           */
/* ------------------------------------------------------------------ */
static void fill_local(Z* data, const DdlaHandle_t& handle, const int* desc,
                       Z (*elem)(int,int,int))
{
    int ncols = desc[DDLA_N_];   // used for elem_val variants if needed
    for (int jl = 0; jl < ddla_test::n_loc(handle, desc); ++jl) {
        int jg = indxl2g(jl, desc[DDLA_NB_], ddla_test::mypcol(handle), desc[DDLA_CSRC_], ddla_test::npcols(handle));
        for (int il = 0; il < ddla_test::m_loc(handle, desc); ++il) {
            int ig = indxl2g(il, desc[DDLA_MB_], ddla_test::myprow(handle), desc[DDLA_RSRC_], ddla_test::nprows(handle));
            data[il + jl * desc[DDLA_LLD_]] = elem(ig, jg, ncols);
        }
    }
}

/* copy local array (with lld stride)                                   */
static void copy_local(Z* dst, const Z* src, int m_loc, int n_loc, int lld) {
    for (int j = 0; j < n_loc; ++j)
        for (int i = 0; i < m_loc; ++i)
            dst[i + j * lld] = src[i + j * lld];
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- parse CLI -------------------------------------------------- */
    int nb = 128;
    int warmup = 1;
    int repeats = 3;
    std::string sizes_str = "1024,2048,4096";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc) {
            sizes_str = argv[++i];
        } else if (arg == "--nb" && i + 1 < argc) {
            nb = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else {
            if (myid == 0) {
                std::cerr << "Unknown option: " << arg << "\n";
                std::cerr << "Usage: " << argv[0]
                          << " [--sizes S1,S2,...] [--nb N] [--warmup N] [--repeats N]\n";
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (nb < 1) { if (myid == 0) std::cerr << "Invalid nb\n"; MPI_Finalize(); return 1; }
    if (warmup < 0) { if (myid == 0) std::cerr << "Invalid warmup\n"; MPI_Finalize(); return 1; }
    if (repeats < 1) { if (myid == 0) std::cerr << "Invalid repeats\n"; MPI_Finalize(); return 1; }

    std::vector<int> sizes = parse_sizes(sizes_str);
    if (sizes.empty() || std::any_of(sizes.begin(), sizes.end(), [](int n) { return n < 1; })) {
        if (myid == 0) std::cerr << "Sizes must be positive integers\n";
        MPI_Finalize();
        return 1;
    }

    /* exactly 4 ranks required */
    if (nprocs != 4) {
        if (myid == 0)
            std::cerr << "This benchmark requires exactly 4 MPI ranks (got " << nprocs << ")\n";
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

    /* verify coordinates match */
    int ddla_nprows = 0, ddla_npcols = 0, ddla_myrow = 0, ddla_mycol = 0;
    ddlaGetGridDims(handle, ddla_nprows, ddla_npcols);
    ddlaGetGridCoords(handle, ddla_myrow, ddla_mycol);
    int grid_mismatch =
        ddla_nprows != blacs_nprow || ddla_npcols != blacs_npcol ||
        ddla_myrow != blacs_myrow || ddla_mycol != blacs_mycol;
    int any_grid_mismatch = 0;
    MPI_Allreduce(&grid_mismatch, &any_grid_mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_grid_mismatch) {
        std::cerr << "Rank " << myid << ": grid mismatch DDLA("
                  << ddla_myrow << "," << ddla_mycol << ") vs BLACS("
                  << blacs_myrow << "," << blacs_mycol << ")\n";
        Cblacs_gridexit(ictxt);
        ddlaDestroy(handle);
        MPI_Finalize();
        return 1;
    }

    /* ---- CSV header (rank 0 only) ------------------------------------ */
    if (myid == 0) {
        std::cout << "n,nb,ranks,grid,warmup,repeats,ddla_s,scalapack_s,"
                  << "ddla_gflops,scalapack_gflops,ddla_over_scalapack_time,"
                  << "max_abs_error,max_rel_error\n";
    }

    Z alpha(1.1, -0.3);
    Z beta(-0.4, 0.2);
    int exit_status = 0;

    for (int n : sizes) {
        const int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1; /* 1-based for ScaLAPACK */
        const int irsrc = 0, icsrc = 0;  /* zero-based */

        /* DDLA descriptors */
        int descA[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, irsrc, icsrc));
        int descB[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(descB, handle, n, n, nb, nb, irsrc, icsrc));
        int descC[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(descC, handle, n, n, nb, nb, irsrc, icsrc));

        /* verify local dimensions */
        int loc_m_a = ddla_test::m_loc(handle, descA), loc_n_a = ddla_test::n_loc(handle, descA), lld_a = descA[DDLA_LLD_];
        int loc_m_b = ddla_test::m_loc(handle, descB), loc_n_b = ddla_test::n_loc(handle, descB), lld_b = descB[DDLA_LLD_];
        int loc_m_c = ddla_test::m_loc(handle, descC), loc_n_c = ddla_test::n_loc(handle, descC), lld_c = descC[DDLA_LLD_];

        int expected_m = numroc_(&n, &nb, &blacs_myrow, &irsrc, &blacs_nprow);
        int expected_n = numroc_(&n, &nb, &blacs_mycol, &icsrc, &blacs_npcol);
        int local_size_mismatch =
            loc_m_a != expected_m || loc_n_a != expected_n ||
            loc_m_b != expected_m || loc_n_b != expected_n ||
            loc_m_c != expected_m || loc_n_c != expected_n ||
            lld_a != std::max(1, expected_m) ||
            lld_b != std::max(1, expected_m) ||
            lld_c != std::max(1, expected_m);
        int any_size_mismatch = 0;
        MPI_Allreduce(&local_size_mismatch, &any_size_mismatch, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (any_size_mismatch) {
            if (myid == 0) std::cerr << "BLACS/DDLA local-size mismatch for n=" << n << "\n";
            exit_status = 1;
            break;
        }

        /* BLACS descriptors */
        int descA_sc[9], descB_sc[9], descC_sc[9];
        int info_a = 0, info_b = 0, info_c = 0;
        descinit_(descA_sc, &n, &n, &nb, &nb, &irsrc, &icsrc, &ictxt, &lld_a, &info_a);
        descinit_(descB_sc, &n, &n, &nb, &nb, &irsrc, &icsrc, &ictxt, &lld_b, &info_b);
        descinit_(descC_sc, &n, &n, &nb, &nb, &irsrc, &icsrc, &ictxt, &lld_c, &info_c);
        int local_desc_failure = info_a != 0 || info_b != 0 || info_c != 0;
        int any_desc_failure = 0;
        MPI_Allreduce(&local_desc_failure, &any_desc_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (any_desc_failure) {
            if (myid == 0) {
                std::cerr << "descinit failed for n=" << n << " (A=" << info_a
                          << ", B=" << info_b << ", C=" << info_c << ")\n";
            }
            exit_status = 1;
            break;
        }

        /* Verify descriptor internal layout matches DDLA */
        /* descinit_ internal layout: DTYPE_A_, CTXT_, M_, N_, MB_, NB_, IRSRC_, ICSRC_, LLD_ */
        {
            bool ok = true;
            ok = ok && (descA_sc[0] == 1);              /* dense */
            ok = ok && (descA_sc[1] == ictxt);
            ok = ok && (descA_sc[2] == n && descA_sc[3] == n);
            ok = ok && (descA_sc[4] == nb && descA_sc[5] == nb);
            ok = ok && (descA_sc[6] == irsrc && descA_sc[7] == icsrc);
            ok = ok && (descA_sc[8] == lld_a);
            int fail_flag = ok ? 0 : 1;
            int global_fail = 0;
            MPI_Allreduce(&fail_flag, &global_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (global_fail) {
                if (myid == 0) std::cerr << "Descriptor verification failed for n=" << n << "\n";
                exit_status = 1;
            }
        }
        if (exit_status != 0) break;

        /* Allocate local matrices */
        int sz_a = lld_a * loc_n_a;
        int sz_b = lld_b * loc_n_b;
        int sz_c = lld_c * loc_n_c;

        std::vector<Z> h_A(sz_a), h_B(sz_b), h_C0(sz_c);
        std::vector<Z> h_C_work(sz_c);
        std::vector<Z> h_C_ddla(sz_c);
        std::vector<Z> h_C_sca(sz_c);

        /* Fill deterministically */
        fill_local(h_A.data(), handle, descA, elem_val_A);
        fill_local(h_B.data(), handle, descB, elem_val_B);
        fill_local(h_C0.data(), handle, descC, elem_val_C0);

        /* ---- Timing data structures ---- */
        struct Timings {
            std::vector<double> ddla_times;
            std::vector<double> scalapack_times;
        } timings;
        timings.ddla_times.reserve(repeats);
        timings.scalapack_times.reserve(repeats);

        char Nchar = 'N';

        /* Warmup runs */
        for (int w = 0; w < warmup; ++w) {
            copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
            pgemm<DdlaBackend::CPU>(handle, 
                Nchar, Nchar, n, n, n, alpha,
                h_A.data(), descA,
                h_B.data(), descB,
                beta,
                h_C_work.data(), descC);
            copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
            pzgemm_(&Nchar, &Nchar, &n, &n, &n, &alpha,
                    h_A.data(), &ia, &ja, descA_sc,
                    h_B.data(), &ib, &jb, descB_sc,
                    &beta,
                    h_C_work.data(), &ic, &jc, descC_sc);
        }

        /* Timed repetitions */
        for (int r = 0; r < repeats; ++r) {
            bool ddla_first = (r % 2 == 0);
            double t0, t1;

            if (ddla_first) {
                /* DDLA */
                copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                pgemm<DdlaBackend::CPU>(handle, 
                    Nchar, Nchar, n, n, n, alpha,
                    h_A.data(), descA,
                    h_B.data(), descB,
                    beta,
                    h_C_work.data(), descC);
                t1 = MPI_Wtime();
                double ddla_t = t1 - t0;
                MPI_Allreduce(MPI_IN_PLACE, &ddla_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                timings.ddla_times.push_back(ddla_t);

                /* ScaLAPACK */
                copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                pzgemm_(&Nchar, &Nchar, &n, &n, &n, &alpha,
                        h_A.data(), &ia, &ja, descA_sc,
                        h_B.data(), &ib, &jb, descB_sc,
                        &beta,
                        h_C_work.data(), &ic, &jc, descC_sc);
                t1 = MPI_Wtime();
                double sca_t = t1 - t0;
                MPI_Allreduce(MPI_IN_PLACE, &sca_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                timings.scalapack_times.push_back(sca_t);
            } else {
                /* ScaLAPACK first */
                copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                pzgemm_(&Nchar, &Nchar, &n, &n, &n, &alpha,
                        h_A.data(), &ia, &ja, descA_sc,
                        h_B.data(), &ib, &jb, descB_sc,
                        &beta,
                        h_C_work.data(), &ic, &jc, descC_sc);
                t1 = MPI_Wtime();
                double sca_t = t1 - t0;
                MPI_Allreduce(MPI_IN_PLACE, &sca_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                timings.scalapack_times.push_back(sca_t);

                /* DDLA */
                copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                pgemm<DdlaBackend::CPU>(handle, 
                    Nchar, Nchar, n, n, n, alpha,
                    h_A.data(), descA,
                    h_B.data(), descB,
                    beta,
                    h_C_work.data(), descC);
                t1 = MPI_Wtime();
                double ddla_t = t1 - t0;
                MPI_Allreduce(MPI_IN_PLACE, &ddla_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                timings.ddla_times.push_back(ddla_t);
            }
        }

        /* Compute median times */
        std::sort(timings.ddla_times.begin(), timings.ddla_times.end());
        std::sort(timings.scalapack_times.begin(), timings.scalapack_times.end());
        auto median = [](const std::vector<double>& values) {
            const size_t mid = values.size() / 2;
            return values.size() % 2 == 0
                ? 0.5 * (values[mid - 1] + values[mid])
                : values[mid];
        };
        double ddla_med = median(timings.ddla_times);
        double sca_med  = median(timings.scalapack_times);

        double n3 = static_cast<double>(n);
        n3 = n3 * n3 * n3;
        double gflops = (8.0 * n3) / 1e9;
        double ddla_gflops = gflops / ddla_med;
        double sca_gflops  = gflops / sca_med;
        double ratio = ddla_med / sca_med;

        /* ---- Correctness: fresh runs --------------------------------- */
        copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
        /* DDLA */
        pgemm<DdlaBackend::CPU>(handle, 
            Nchar, Nchar, n, n, n, alpha,
            h_A.data(), descA,
            h_B.data(), descB,
            beta,
            h_C_work.data(), descC);
        copy_local(h_C_ddla.data(), h_C_work.data(), loc_m_c, loc_n_c, lld_c);

        copy_local(h_C_work.data(), h_C0.data(), loc_m_c, loc_n_c, lld_c);
        /* ScaLAPACK */
        pzgemm_(&Nchar, &Nchar, &n, &n, &n, &alpha,
                h_A.data(), &ia, &ja, descA_sc,
                h_B.data(), &ib, &jb, descB_sc,
                &beta,
                h_C_work.data(), &ic, &jc, descC_sc);
        copy_local(h_C_sca.data(), h_C_work.data(), loc_m_c, loc_n_c, lld_c);

        /* Compare local elements */
        double local_max_abs = 0.0;
        double local_max_ref = 0.0;
        int local_nonfinite = 0;
        for (int j = 0; j < loc_n_c; ++j) {
            for (int i = 0; i < loc_m_c; ++i) {
                Z d = h_C_ddla[i + j * lld_c] - h_C_sca[i + j * lld_c];
                double abs_d = std::abs(d);
                local_nonfinite = local_nonfinite || !std::isfinite(abs_d);
                if (std::isfinite(abs_d) && abs_d > local_max_abs) local_max_abs = abs_d;
                double abs_c = std::abs(h_C_ddla[i + j * lld_c]);
                local_nonfinite = local_nonfinite || !std::isfinite(abs_c);
                if (std::isfinite(abs_c) && abs_c > local_max_ref) local_max_ref = abs_c;
                double abs_s = std::abs(h_C_sca[i + j * lld_c]);
                local_nonfinite = local_nonfinite || !std::isfinite(abs_s);
                if (std::isfinite(abs_s) && abs_s > local_max_ref) local_max_ref = abs_s;
            }
        }

        double glob_max_abs = 0.0;
        double glob_max_ref = 0.0;
        int glob_nonfinite = 0;
        MPI_Allreduce(&local_max_abs, &glob_max_abs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max_ref, &glob_max_ref, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_nonfinite, &glob_nonfinite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        double max_rel_error = glob_max_abs / std::max(1.0, glob_max_ref);
        bool pass = !glob_nonfinite &&
                    std::isfinite(ddla_med) && ddla_med > 0.0 &&
                    std::isfinite(sca_med) && sca_med > 0.0 &&
                    glob_max_abs <= 1e-10 * std::max(1.0, glob_max_ref);

        /* ---- CSV output (rank 0) ---- */
        if (myid == 0) {
            std::cout << std::scientific << std::setprecision(12)
                      << n << "," << nb << ","
                      << nprocs << "," << "2x2" << ","
                      << warmup << "," << repeats << ","
                      << ddla_med << "," << sca_med << ","
                      << ddla_gflops << "," << sca_gflops << ","
                      << ratio << ","
                      << glob_max_abs << "," << max_rel_error << "\n";
        }

        if (!pass) {
            if (myid == 0) {
                std::cerr << "Correctness FAIL for n=" << n
                          << " max_abs=" << glob_max_abs
                          << " max_rel=" << max_rel_error
                          << " nonfinite=" << glob_nonfinite << "\n";
            }
            exit_status = 1;
            break;
        }
    }

    Cblacs_gridexit(ictxt);
    ddlaDestroy(handle);
    MPI_Finalize();
    return exit_status;
}
