/**
 * @file test_cpu_pgemm.cpp
 * @brief Correctness test for CPU-backend pgemm (DDLA_USE_CPU).
 *
 * Only compiles with CPU builds. Tests N,N N,T T,N T,C C,N C,T matrix
 * multiplication correctness across all 4 scalar types at various sizes.
 */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <mpi.h>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

// Helper: conj for complex, identity for real
template <typename T> T maybe_conj(T v) { return v; }
template <> std::complex<float>  maybe_conj(std::complex<float> v)  { return std::conj(v); }
template <> std::complex<double> maybe_conj(std::complex<double> v) { return std::conj(v); }

// C++11 has no `if constexpr`: complex-valued alpha/beta are assigned only
// for complex T; the tag overload keeps the two-argument T(...) constructor
// from ever being instantiated for real T.
template <typename T>
void set_complex_alpha_beta_impl(T& alpha, T& beta,
                                 double ar, double ai, double br, double bi,
                                 std::true_type)
{
    alpha = T(ar, ai);
    beta = T(br, bi);
}

template <typename T>
void set_complex_alpha_beta_impl(T&, T&, double, double, double, double,
                                 std::false_type)
{
}

template <typename T>
void set_complex_alpha_beta(T& alpha, T& beta,
                            double ar, double ai, double br, double bi)
{
    set_complex_alpha_beta_impl(
        alpha, beta, ar, ai, br, bi,
        std::integral_constant<bool,
            std::is_same<T, std::complex<float>>::value
            || std::is_same<T, std::complex<double>>::value>());
}

// ---------------------------------------------------------------------------
// Sequential reference GEMM: C = alpha * op(A) * op(B) + beta * C
// A is MxK, B is KxN, C is MxN (all column-major).
// ---------------------------------------------------------------------------
template <typename T>
static void ref_gemm(char transa, char transb,
                     int M, int N, int K,
                     T alpha,
                     const T* A, int lda,
                     const T* B, int ldb,
                     T beta,
                     T* C, int ldc)
{
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T sum = T(0);
            for (int p = 0; p < K; ++p) {
                T a_val;
                if (transa == 'N')
                    a_val = A[i + p * lda];
                else if (transa == 'T')
                    a_val = A[p + i * lda];
                else // 'C'
                    a_val = maybe_conj(A[p + i * lda]);

                T b_val;
                if (transb == 'N')
                    b_val = B[p + j * ldb];
                else if (transb == 'T')
                    b_val = B[j + p * ldb];
                else // 'C'
                    b_val = maybe_conj(B[j + p * ldb]);

                sum += a_val * b_val;
            }
            C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic element value: A_ij = (i * Ncols + j + 1) mod 13
// ---------------------------------------------------------------------------
template <typename T>
static T elem_val(int i, int j, int salt) {
    double v = ((i * 1009 + j * 917 + salt * 37 + 17) % 29) - 14.0;
    return T(v);
}
template <>
std::complex<float> elem_val<std::complex<float>>(int i, int j, int salt) {
    double re = ((i * 1009 + j * 917 + salt * 37 + 17) % 29) - 14.0;
    double im = ((i * 811 + j * 613 + salt * 43 + 11) % 31) - 15.0;
    return std::complex<float>(float(re), float(im));
}
template <>
std::complex<double> elem_val<std::complex<double>>(int i, int j, int salt) {
    double re = ((i * 1009 + j * 917 + salt * 37 + 17) % 29) - 14.0;
    double im = ((i * 811 + j * 613 + salt * 43 + 11) % 31) - 15.0;
    return std::complex<double>(re, im);
}

// ---------------------------------------------------------------------------
// Check correctness for one (type, transa, transb, M, N, K, nb) tuple
// ---------------------------------------------------------------------------
template <typename T>
static int check_one(
    DdlaHandle_t handle,
    char transa, char transb,
    int M, int N, int K, int nb,
    int irsrc, int icsrc,
    bool use_default_backend = false)
{
    int nprows = 0, npcols = 0;
    ddla_get_grid_dims(handle, nprows, npcols);
    int myid = ddla_get_rank(handle);
    int nprocs = ddla_get_size(handle);

    int rowsA = (transa == 'N') ? M : K;
    int colsA = (transa == 'N') ? K : M;
    int rowsB = (transb == 'N') ? K : N;
    int colsB = (transb == 'N') ? N : K;

    DdlaDesc descA(handle); descA.init(rowsA, colsA, nb, nb, irsrc, icsrc);
    DdlaDesc descB(handle); descB.init(rowsB, colsB, nb, nb, irsrc, icsrc);
    DdlaDesc descC(handle); descC.init(M, N, nb, nb, irsrc, icsrc);

    std::vector<T> h_A(std::max(1, descA.lld() * descA.n_loc()));
    std::vector<T> h_B(std::max(1, descB.lld() * descB.n_loc()));
    std::vector<T> h_C(std::max(1, descC.lld() * descC.n_loc()));

    // fill local blocks deterministically from global indices
    for (int jl = 0; jl < descA.n_loc(); ++jl) {
        int jg = indxl2g(jl, descA.nb(), descA.mypcol(), descA.icsrc(), descA.npcols());
        for (int il = 0; il < descA.m_loc(); ++il) {
            int ig = indxl2g(il, descA.mb(), descA.myprow(), descA.irsrc(), descA.nprows());
            h_A[il + jl * descA.lld()] = elem_val<T>(ig, jg, 0);
        }
    }
    for (int jl = 0; jl < descB.n_loc(); ++jl) {
        int jg = indxl2g(jl, descB.nb(), descB.mypcol(), descB.icsrc(), descB.npcols());
        for (int il = 0; il < descB.m_loc(); ++il) {
            int ig = indxl2g(il, descB.mb(), descB.myprow(), descB.irsrc(), descB.nprows());
            h_B[il + jl * descB.lld()] = elem_val<T>(ig, jg, 1000);
        }
    }
    for (int jl = 0; jl < descC.n_loc(); ++jl) {
        int jg = indxl2g(jl, descC.nb(), descC.mypcol(), descC.icsrc(), descC.npcols());
        for (int il = 0; il < descC.m_loc(); ++il) {
            int ig = indxl2g(il, descC.mb(), descC.myprow(), descC.irsrc(), descC.nprows());
            h_C[il + jl * descC.lld()] = elem_val<T>(ig, jg, 2000);
        }
    }

    T alpha = T(1.25);
    T beta  = T(-0.5);
    set_complex_alpha_beta(alpha, beta, 1.25, -0.375, -0.5, 0.25);

    // ---- compute with ddla::pgemm ----
    if (use_default_backend) {
        pgemm<>(transa, transb, M, N, K, alpha,
                h_A.data(), descA,
                h_B.data(), descB,
                beta,
                h_C.data(), descC);
    } else {
        pgemm<DdlaBackend::CPU>(
            transa, transb, M, N, K, alpha,
            h_A.data(), descA,
            h_B.data(), descB,
            beta,
            h_C.data(), descC);
    }

    // ---- gather global C from all ranks to rank 0 for checking ----
    auto gather_global = [&](const DdlaDesc& desc, const std::vector<T>& local) -> std::vector<T> {
        int mg = desc.m(), ng = desc.n();
        std::vector<T> global;
        if (myid == 0) global.resize(mg * ng, T(0));

        // determine maximum local size to send
        int loc_m = desc.m_loc();
        int loc_n = desc.n_loc();
        int loc_ld = desc.lld();

        // For simplicity, serialize local into contiguous (loc_m * loc_n) buffer
        std::vector<T> sendbuf(loc_m * loc_n);
        for (int j = 0; j < loc_n; ++j)
            for (int i = 0; i < loc_m; ++i)
                sendbuf[i + j * loc_m] = local[i + j * loc_ld];

        int elem_count = static_cast<int>(sendbuf.size());
        int byte_count = elem_count * static_cast<int>(sizeof(T));

        int* recvcounts = nullptr;
        int* displs = nullptr;
        if (myid == 0) {
            recvcounts = new int[nprocs];
            displs = new int[nprocs];
        }

        // Gather element counts, then convert to byte offsets for MPI_BYTE Gatherv
        MPI_Gather(&byte_count, 1, MPI_INT,
                   recvcounts, 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        if (myid == 0) {
            displs[0] = 0;
            for (int r = 1; r < nprocs; ++r)
                displs[r] = displs[r-1] + recvcounts[r-1];
        }

        // gather flattened data as bytes
        std::vector<T> recvbuf;
        if (myid == 0) {
            int total_bytes = displs[nprocs-1] + recvcounts[nprocs-1];
            recvbuf.resize(total_bytes / sizeof(T) + 1);
        }
        MPI_Gatherv(sendbuf.data(), byte_count, MPI_BYTE,
                    recvbuf.data(), recvcounts, displs, MPI_BYTE,
                    0, MPI_COMM_WORLD);

        if (myid == 0) {
            // scatter received segments back to global (contiguous) layout
            for (int r = 0; r < nprocs; ++r) {
                int r_row, r_col;
                ddla_rank_to_rc(handle, r, r_row, r_col);
                int r_loc_m = num_loc(mg, desc.mb(), r_row, desc.irsrc(), desc.nprows());
                int r_loc_n = num_loc(ng, desc.nb(), r_col, desc.icsrc(), desc.npcols());
                // byte offset -> element offset
                const T* src = recvbuf.data() + displs[r] / static_cast<int>(sizeof(T));
                for (int jr = 0; jr < r_loc_n; ++jr) {
                    int jg = indxl2g(jr, desc.nb(), r_col, desc.icsrc(), desc.npcols());
                    for (int ir = 0; ir < r_loc_m; ++ir) {
                        int ig = indxl2g(ir, desc.mb(), r_row, desc.irsrc(), desc.nprows());
                        global[ig + jg * mg] = src[ir + jr * r_loc_m];
                    }
                }
            }
            delete[] recvcounts;
            delete[] displs;
        }
        return global;
    };

    std::vector<T> global_C = gather_global(descC, h_C);

    if (myid != 0) return 0; // only root checks

    // ---- reference computation (sequential on rank 0) ----
    // build full A, B on rank 0 using the same elem_val function
    std::vector<T> full_A(rowsA * colsA);
    std::vector<T> full_B(rowsB * colsB);
    std::vector<T> ref_C(M * N);

    for (int j = 0; j < colsA; ++j)
        for (int i = 0; i < rowsA; ++i)
            full_A[i + j * rowsA] = elem_val<T>(i, j, 0);
    for (int j = 0; j < colsB; ++j)
        for (int i = 0; i < rowsB; ++i)
            full_B[i + j * rowsB] = elem_val<T>(i, j, 1000);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
            ref_C[i + j * M] = elem_val<T>(i, j, 2000);

    ref_gemm(transa, transb, M, N, K,
             alpha, full_A.data(), rowsA,
             full_B.data(), rowsB,
             beta, ref_C.data(), M);

    // ---- compare ----
    // Determine reference magnitude for scale-aware tolerance
    double max_ref = 0.0;
    bool has_nonfinite = false;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i) {
            double ref_abs = static_cast<double>(std::abs(ref_C[i + j * M]));
            has_nonfinite = has_nonfinite || !std::isfinite(ref_abs);
            if (std::isfinite(ref_abs)) max_ref = std::max(max_ref, ref_abs);
        }

    double max_abs_err = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T diff = global_C[i + j * M] - ref_C[i + j * M];
            double abs_err = static_cast<double>(std::abs(diff));
            has_nonfinite = has_nonfinite || !std::isfinite(abs_err);
            if (std::isfinite(abs_err) && abs_err > max_abs_err) max_abs_err = abs_err;
        }
    }

    // Classify by component precision: float/complex<float> = single,
    // double/complex<double> = double
    constexpr bool is_single = std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value;
    double rel_tol = is_single ? 1e-4 : 1e-10;
    double threshold = rel_tol * std::max(1.0, max_ref);
    if (has_nonfinite || max_abs_err > threshold) {
        std::cout << "  FAIL [" << transa << "," << transb
                  << "] M=" << M << " N=" << N << " K=" << K
                  << " nb=" << nb
                  << " max_abs_err=" << max_abs_err
                  << " max_ref=" << max_ref
                  << " rel_tol=" << rel_tol
                  << " threshold=" << threshold
                  << " nonfinite=" << has_nonfinite << std::endl;
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Run all test sizes for one type
// ---------------------------------------------------------------------------
template <typename T>
static int run_tests_for_type(const char* type_name,
                              DdlaHandle_t handle,
                              int irsrc, int icsrc)
{
    int nfailed = 0;

    // Test configurations: (transa, transb)
    const char trans_pairs[][2] = {
        {'N','N'}, {'N','T'}, {'N','C'},
        {'T','N'}, {'T','T'}, {'T','C'},
        {'C','N'}, {'C','T'}, {'C','C'}
    };
    const int npairs = sizeof(trans_pairs) / sizeof(trans_pairs[0]);

    // Problem sizes
    const int sizes[] = {17, 64, 128, 257};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    const int nb = 32; // block size

    for (int si = 0; si < nsizes; ++si) {
        int M = sizes[si];
        int N = sizes[si] + 1;  // non-square for extra coverage
        int K = sizes[si] + 17;

        for (int pi = 0; pi < npairs; ++pi) {
            char ta = trans_pairs[pi][0];
            char tb = trans_pairs[pi][1];

            // For Trans/C cases, ensure K is appropriate
            // pgemm<>() (bracket-free, defaulted Backend) only resolves to
            // CPU when this build has no GPU capability -- in a dual build
            // default_backend_v prefers GPU (see ddla_handle_t.h), so
            // exercising the bare-default call site against this test's CPU
            // handle would throw a backend-mismatch std::runtime_error there.
            const bool use_default_backend =
                (si == 0 && pi == 0) && (default_backend_v == DdlaBackend::CPU);
            int nf = check_one<T>(handle, ta, tb, M, N, K, nb,
                                  irsrc, icsrc, use_default_backend);
            nfailed += nf;
        }
    }

    // F2 regression: K == 0 must reduce to C := beta*C, leaving A/B
    // untouched (the k-loop that used to perform the scale never runs
    // when K == 0).
    {
        const int M = 17, N = 20, nb = 8;
        DdlaDesc descA(handle); descA.init(M, 0, nb, nb, irsrc, icsrc);   // K == 0
        DdlaDesc descB(handle); descB.init(0, N, nb, nb, irsrc, icsrc);   // K == 0
        DdlaDesc descC(handle); descC.init(M, N, nb, nb, irsrc, icsrc);

        std::vector<T> h_A(1), h_B(1);
        std::vector<T> h_C(std::max(1, descC.lld() * descC.n_loc()));
        for (int jl = 0; jl < descC.n_loc(); ++jl) {
            int jg = indxl2g(jl, descC.nb(), descC.mypcol(), descC.icsrc(), descC.npcols());
            for (int il = 0; il < descC.m_loc(); ++il) {
                int ig = indxl2g(il, descC.mb(), descC.myprow(), descC.irsrc(), descC.nprows());
                h_C[il + jl * descC.lld()] = elem_val<T>(ig, jg, 3000);
            }
        }
        std::vector<T> h_C_orig = h_C;

        T alpha = T(2.0);
        T beta = T(-0.75);
        set_complex_alpha_beta(alpha, beta, 2.0, -1.0, -0.75, 0.5);

        pgemm<DdlaBackend::CPU>('N', 'N', M, N, 0, alpha,
                                h_A.data(), descA,
                                h_B.data(), descB,
                                beta,
                                h_C.data(), descC);

        bool k_zero_ok = true;
        constexpr bool is_single = std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value;
        double tol = is_single ? 1e-5 : 1e-12;
        for (size_t idx = 0; idx < h_C.size(); ++idx) {
            T expected = beta * h_C_orig[idx];
            double err = static_cast<double>(std::abs(h_C[idx] - expected));
            if (err > tol) { k_zero_ok = false; break; }
        }
        if (!k_zero_ok) {
            nfailed += 1;
            if (ddla_get_rank(handle) == 0)
                std::cout << "  FAIL [" << type_name << "] K=0 scaling check" << std::endl;
        }
    }

    int myid = ddla_get_rank(handle);
    if (nfailed == 0) {
        if (myid == 0)
            std::cout << "  [" << type_name << "] ALL PASSED" << std::endl;
    } else {
        if (myid == 0)
            std::cout << "  [" << type_name << "] " << nfailed << " FAILED" << std::endl;
    }
    return nfailed;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    // Parse optional grid dimensions from command line
    int nprows = 0, npcols = 0;
    if (argc >= 3) {
        nprows = std::atoi(argv[1]);
        npcols = std::atoi(argv[2]);
    }

    // Parse optional --src IR IC (nonzero irsrc/icsrc probe; default 0 0).
    int irsrc = 0, icsrc = 0;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--src" && i + 2 < argc) {
            irsrc = std::atoi(argv[i + 1]);
            icsrc = std::atoi(argv[i + 2]);
            i += 2;
        }
    }

    DdlaHandle_t handle = nullptr;
    ddla_init(handle, DdlaBackend::CPU);
    if (nprows > 0 && npcols > 0)
        ddla_set(handle, MPI_COMM_WORLD, nprows, npcols, 'R');
    else
        ddla_set(handle, MPI_COMM_WORLD, 'R');

    int myid = ddla_get_rank(handle);
    int g_nprows = 0, g_npcols = 0;
    ddla_get_grid_dims(handle, g_nprows, g_npcols);

    if (myid == 0) {
        std::cout << "=== CPU pgemm correctness test ===" << std::endl;
        std::cout << "Grid: " << g_nprows << "x" << g_npcols << std::endl;
        std::cout << "Procs: " << ddla_get_size(handle) << std::endl;
    }

    int total_failed = 0;
    total_failed += run_tests_for_type<float>("float", handle, irsrc, icsrc);
    total_failed += run_tests_for_type<double>("double", handle, irsrc, icsrc);
    total_failed += run_tests_for_type<std::complex<float>>("complex<float>", handle, irsrc, icsrc);
    total_failed += run_tests_for_type<std::complex<double>>("complex<double>", handle, irsrc, icsrc);

    if (myid == 0) {
        if (total_failed == 0) {
            std::cout << "ALL TESTS PASSED" << std::endl;
        } else {
            std::cout << total_failed << " TEST(S) FAILED" << std::endl;
        }
    }

    ddla_destroy(handle);
    MPI_Finalize();
    return (total_failed > 0) ? 1 : 0;
}
