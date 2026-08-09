/**
 * @file test_dual_pgemm.cpp
 * @brief Dual-backend (CPU + GPU) pgemm comparison test.
 *
 * Built only for DDLA_DUAL_BUILD configurations (both CPU and GPU compiled).
 * Compares CPU pgemm against GPU pgemm on deterministic inputs.
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <mpi.h>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/transport_block.h>

using namespace ddla;

static int myid = -1;
static int nprocs = 0;

#define TEST(name, cond)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "FAIL [" << myid << "]: " << (name) << std::endl;    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
        if (myid == 0) std::cout << "  PASS: " << (name) << std::endl;        \
    } while (0)

// Deterministic element value
template <typename T>
static T elem_val(int i, int j, int salt) {
    double v = ((i * 1009 + j * 917 + salt * 37 + 17) % 29) - 14.0;
    return T(v);
}
template <>
std::complex<float> elem_val<std::complex<float>>(int i, int j, int salt) {
    return std::complex<float>(
        float(((i * 1009 + j * 917 + salt * 37 + 17) % 29) - 14.0),
        float(((i * 811 + j * 613 + salt * 43 + 11) % 31) - 15.0));
}
template <>
std::complex<double> elem_val<std::complex<double>>(int i, int j, int salt) {
    return std::complex<double>(
        ((i * 1009 + j * 917 + salt * 37 + 17) % 29) - 14.0,
        ((i * 811 + j * 613 + salt * 43 + 11) % 31) - 15.0);
}

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

template <typename T>
static int check_one(
    DdlaHandle_t h_cpu, DdlaHandle_t h_gpu,
    char transa, char transb,
    int M, int N, int K, int nb,
    bool cpu_first = false,
    bool use_default_gpu = false)
{
    int nprows = 0, npcols = 0;
    ddla_get_grid_dims(h_cpu, nprows, npcols);

    int rowsA = (transa == 'N') ? M : K;
    int colsA = (transa == 'N') ? K : M;
    int rowsB = (transb == 'N') ? K : N;
    int colsB = (transb == 'N') ? N : K;

    DdlaDesc descA_cpu(h_cpu); descA_cpu.init(rowsA, colsA, nb, nb, 0, 0);
    DdlaDesc descB_cpu(h_cpu); descB_cpu.init(rowsB, colsB, nb, nb, 0, 0);
    DdlaDesc descC_cpu(h_cpu); descC_cpu.init(M, N, nb, nb, 0, 0);

    DdlaDesc descA_gpu(h_gpu); descA_gpu.init(rowsA, colsA, nb, nb, 0, 0);
    DdlaDesc descB_gpu(h_gpu); descB_gpu.init(rowsB, colsB, nb, nb, 0, 0);
    DdlaDesc descC_gpu(h_gpu); descC_gpu.init(M, N, nb, nb, 0, 0);

    // Allocate CPU host buffers (local sizes)
    const std::size_t szA = static_cast<std::size_t>(descA_cpu.lld() * descA_cpu.n_loc());
    const std::size_t szB = static_cast<std::size_t>(descB_cpu.lld() * descB_cpu.n_loc());
    const std::size_t szC = static_cast<std::size_t>(descC_cpu.lld() * descC_cpu.n_loc());
    const std::size_t szDA = static_cast<std::size_t>(descA_gpu.lld() * descA_gpu.n_loc());
    const std::size_t szDB = static_cast<std::size_t>(descB_gpu.lld() * descB_gpu.n_loc());
    const std::size_t szDC = static_cast<std::size_t>(descC_gpu.lld() * descC_gpu.n_loc());
    TEST("CPU/GPU local storage matches", szA == szDA && szB == szDB && szC == szDC);
    std::vector<T> h_A(std::max<std::size_t>(1, szA));
    std::vector<T> h_B(std::max<std::size_t>(1, szB));
    std::vector<T> h_C_cpu(std::max<std::size_t>(1, szC));
    std::vector<T> h_C_gpu(std::max<std::size_t>(1, szC));

    // Fill local blocks deterministically
    auto fill = [](const DdlaDesc& desc, std::vector<T>& buf, int salt) {
        for (int jl = 0; jl < desc.n_loc(); ++jl) {
            int jg = indxl2g(jl, desc.nb(), desc.mypcol(), desc.icsrc(), desc.npcols());
            for (int il = 0; il < desc.m_loc(); ++il) {
                int ig = indxl2g(il, desc.mb(), desc.myprow(), desc.irsrc(), desc.nprows());
                buf[il + jl * desc.lld()] = elem_val<T>(ig, jg, salt);
            }
        }
    };
    fill(descA_cpu, h_A, 0);
    fill(descB_cpu, h_B, 1000);
    fill(descC_cpu, h_C_cpu, 2000);
    fill(descC_cpu, h_C_gpu, 2000); // same initial C

    T alpha = T(1.25);
    T beta  = T(-0.5);
    set_complex_alpha_beta(alpha, beta, 1.25, -0.375, -0.5, 0.25);

    // --- GPU device allocation and upload ---
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    TEST("GPU malloc A", ddla_malloc(reinterpret_cast<void**>(&d_A),
                                     std::max<std::size_t>(1, szDA) * sizeof(T), h_gpu) == 0);
    TEST("GPU malloc B", ddla_malloc(reinterpret_cast<void**>(&d_B),
                                     std::max<std::size_t>(1, szDB) * sizeof(T), h_gpu) == 0);
    TEST("GPU malloc C", ddla_malloc(reinterpret_cast<void**>(&d_C),
                                     std::max<std::size_t>(1, szDC) * sizeof(T), h_gpu) == 0);

    // Transfer sizes must match the CPU-side storage
    std::size_t xferA = szA * sizeof(T);
    std::size_t xferB = szB * sizeof(T);
    std::size_t xferC = szC * sizeof(T);
    int ret = 0;
    ret |= ddla_memcpy(d_A, h_A.data(), xferA, DdlaMemoryCopyKind::HostToDevice, h_gpu);
    ret |= ddla_memcpy(d_B, h_B.data(), xferB, DdlaMemoryCopyKind::HostToDevice, h_gpu);
    ret |= ddla_memcpy(d_C, h_C_gpu.data(), xferC, DdlaMemoryCopyKind::HostToDevice, h_gpu);
    TEST("GPU memcpy upload", ret == 0);
    TEST("GPU sync after upload", ddla_synchronize(h_gpu) == 0);

    // --- Compute in requested order ---
    auto run_cpu = [&]() {
        pgemm<DdlaBackend::CPU>(
            transa, transb, M, N, K, alpha,
            h_A.data(), descA_cpu, h_B.data(), descB_cpu,
            beta, h_C_cpu.data(), descC_cpu);
    };
    auto run_gpu_and_download = [&]() {
        if (use_default_gpu) {
            pgemm<>(transa, transb, M, N, K, alpha,
                    d_A, descA_gpu, d_B, descB_gpu, beta, d_C, descC_gpu);
        } else {
            pgemm<DdlaBackend::GPU>(
                transa, transb, M, N, K, alpha,
                d_A, descA_gpu, d_B, descB_gpu, beta, d_C, descC_gpu);
        }
        int ret_dl = ddla_memcpy(h_C_gpu.data(), d_C, xferC, DdlaMemoryCopyKind::DeviceToHost, h_gpu);
        TEST("GPU download after compute", ret_dl == 0);
    };

    if (cpu_first) {
        run_cpu();
        run_gpu_and_download();
    } else {
        run_gpu_and_download();
        run_cpu();
    }

    TEST("GPU sync after download", ddla_synchronize(h_gpu) == 0);

    int mf = 0;
    mf |= ddla_free(d_A, h_gpu);
    mf |= ddla_free(d_B, h_gpu);
    mf |= ddla_free(d_C, h_gpu);
    TEST("GPU free", mf == 0);

    // --- Compare CPU vs GPU results ---
    // Detect nonfinite values before they disappear through std::max/std::abs
    int rank_nonfinite_flag = 0;
    int nloc = descC_cpu.n_loc();
    int mloc = descC_cpu.m_loc();
    int lld = descC_cpu.lld();
    for (int jl = 0; jl < nloc; ++jl) {
        for (int il = 0; il < mloc; ++il) {
            T cpu_val = h_C_cpu[il + jl * lld];
            T gpu_val = h_C_gpu[il + jl * lld];
            bool cpu_finite = std::isfinite(static_cast<double>(std::abs(cpu_val)));
            bool gpu_finite = std::isfinite(static_cast<double>(std::abs(gpu_val)));
            T diff = cpu_val - gpu_val;
            bool diff_finite = std::isfinite(static_cast<double>(std::abs(diff)));
            if (!cpu_finite || !gpu_finite || !diff_finite) {
                rank_nonfinite_flag = 1;
            }
        }
    }
    {
        int global_nonfinite = 0;
        MPI_Comm comm = ddla_get_communicator(h_cpu);
        MPI_Allreduce(&rank_nonfinite_flag, &global_nonfinite, 1, MPI_INT, MPI_MAX, comm);
        if (global_nonfinite != 0) {
            if (myid == 0)
                std::cerr << "  FAIL: nonfinite data detected in CPU vs GPU comparison"
                          << std::endl;
            return 1;
        }
    }

    double max_ref = 0.0, max_err = 0.0;
    for (int jl = 0; jl < nloc; ++jl) {
        for (int il = 0; il < mloc; ++il) {
            T cpu_val = h_C_cpu[il + jl * lld];
            T gpu_val = h_C_gpu[il + jl * lld];
            double cpu_abs = static_cast<double>(std::abs(cpu_val));
            max_ref = std::max(max_ref, cpu_abs);
            max_err = std::max(max_err, static_cast<double>(std::abs(cpu_val - gpu_val)));
        }
    }

    double global_ref = 0.0, global_err = 0.0;
    {
        MPI_Comm comm = ddla_get_communicator(h_cpu);
        MPI_Allreduce(&max_ref, &global_ref, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&max_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, comm);
    }

    constexpr bool is_single = std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value;
    double tol = is_single ? 2e-4 : 2e-10;
    if (myid == 0 && global_err > tol * std::max(1.0, global_ref)) {
        std::cout << "  FAIL [" << transa << "," << transb
                  << "] M=" << M << " N=" << N << " K=" << K
                  << " CPUvsGPU_max_err=" << global_err << std::endl;
        return 1;
    }
    return 0;
}

// F2 regression: k==0 must reduce to C := beta*C on both backends (the
// k-loop that used to perform the scale never runs when k==0), and the
// CPU and GPU results must still agree with each other.
template <typename T>
static int check_k_zero(DdlaHandle_t h_cpu, DdlaHandle_t h_gpu)
{
    const int M = 17, N = 20, nb = 8;

    DdlaDesc descA_cpu(h_cpu); descA_cpu.init(M, 0, nb, nb, 0, 0);  // K == 0
    DdlaDesc descB_cpu(h_cpu); descB_cpu.init(0, N, nb, nb, 0, 0);  // K == 0
    DdlaDesc descC_cpu(h_cpu); descC_cpu.init(M, N, nb, nb, 0, 0);

    DdlaDesc descA_gpu(h_gpu); descA_gpu.init(M, 0, nb, nb, 0, 0);  // K == 0
    DdlaDesc descB_gpu(h_gpu); descB_gpu.init(0, N, nb, nb, 0, 0);  // K == 0
    DdlaDesc descC_gpu(h_gpu); descC_gpu.init(M, N, nb, nb, 0, 0);

    const std::size_t szC =
        static_cast<std::size_t>(descC_cpu.lld() * descC_cpu.n_loc());
    std::vector<T> h_A(1), h_B(1);
    std::vector<T> h_C_cpu(std::max<std::size_t>(1, szC));
    std::vector<T> h_C_gpu(std::max<std::size_t>(1, szC));

    for (int jl = 0; jl < descC_cpu.n_loc(); ++jl) {
        int jg = indxl2g(jl, descC_cpu.nb(), descC_cpu.mypcol(), descC_cpu.icsrc(), descC_cpu.npcols());
        for (int il = 0; il < descC_cpu.m_loc(); ++il) {
            int ig = indxl2g(il, descC_cpu.mb(), descC_cpu.myprow(), descC_cpu.irsrc(), descC_cpu.nprows());
            T v = elem_val<T>(ig, jg, 3000);
            h_C_cpu[il + jl * descC_cpu.lld()] = v;
            h_C_gpu[il + jl * descC_cpu.lld()] = v;
        }
    }

    T alpha = T(2.0);
    T beta = T(-0.75);
    set_complex_alpha_beta(alpha, beta, 2.0, -1.0, -0.75, 0.5);

    pgemm<DdlaBackend::CPU>('N', 'N', M, N, 0, alpha,
                            h_A.data(), descA_cpu, h_B.data(), descB_cpu,
                            beta, h_C_cpu.data(), descC_cpu);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    TEST("k=0 GPU malloc A", ddla_malloc(reinterpret_cast<void**>(&d_A), sizeof(T), h_gpu) == 0);
    TEST("k=0 GPU malloc B", ddla_malloc(reinterpret_cast<void**>(&d_B), sizeof(T), h_gpu) == 0);
    TEST("k=0 GPU malloc C", ddla_malloc(reinterpret_cast<void**>(&d_C),
                                        std::max<std::size_t>(1, szC) * sizeof(T), h_gpu) == 0);
    int up = ddla_memcpy(d_C, h_C_gpu.data(), szC * sizeof(T),
                         DdlaMemoryCopyKind::HostToDevice, h_gpu);
    TEST("k=0 GPU upload C", up == 0);
    TEST("k=0 GPU sync after upload", ddla_synchronize(h_gpu) == 0);

    pgemm<DdlaBackend::GPU>('N', 'N', M, N, 0, alpha,
                            d_A, descA_gpu, d_B, descB_gpu,
                            beta, d_C, descC_gpu);
    int dl = ddla_memcpy(h_C_gpu.data(), d_C, szC * sizeof(T),
                        DdlaMemoryCopyKind::DeviceToHost, h_gpu);
    TEST("k=0 GPU download", dl == 0);
    TEST("k=0 GPU sync after download", ddla_synchronize(h_gpu) == 0);

    int mf = 0;
    mf |= ddla_free(d_A, h_gpu);
    mf |= ddla_free(d_B, h_gpu);
    mf |= ddla_free(d_C, h_gpu);
    TEST("k=0 GPU free", mf == 0);

    constexpr bool is_single = std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value;
    const double tol = is_single ? 2e-4 : 2e-10;
    int local_failed = 0;
    double local_max_err = 0.0;
    for (int jl = 0; jl < descC_cpu.n_loc(); ++jl) {
        int jg = indxl2g(jl, descC_cpu.nb(), descC_cpu.mypcol(), descC_cpu.icsrc(), descC_cpu.npcols());
        for (int il = 0; il < descC_cpu.m_loc(); ++il) {
            int ig = indxl2g(il, descC_cpu.mb(), descC_cpu.myprow(), descC_cpu.irsrc(), descC_cpu.nprows());
            T expected = beta * elem_val<T>(ig, jg, 3000);
            T cpu_val = h_C_cpu[il + jl * descC_cpu.lld()];
            T gpu_val = h_C_gpu[il + jl * descC_cpu.lld()];
            local_max_err = std::max(local_max_err,
                static_cast<double>(std::abs(cpu_val - gpu_val)));
            if (static_cast<double>(std::abs(cpu_val - expected)) > tol ||
                static_cast<double>(std::abs(gpu_val - expected)) > tol) {
                local_failed = 1;
            }
        }
    }

    MPI_Comm comm = ddla_get_communicator(h_cpu);
    double global_max_err = 0.0;
    int global_failed = 0;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, comm);
    if (global_failed || global_max_err > tol) {
        if (myid == 0)
            std::cerr << "  FAIL: pgemm(k=0) CPU/GPU parity, max_err="
                      << global_max_err << std::endl;
        return 1;
    }
    return 0;
}

template <typename T>
static int run_type(const char* name, DdlaHandle_t h_cpu, DdlaHandle_t h_gpu,
                    bool cpu_first = false)
{
    const char pairs[][2] = {
        {'N','N'}, {'N','T'}, {'N','C'},
        {'T','N'}, {'T','T'}, {'T','C'},
        {'C','N'}, {'C','T'}, {'C','C'}
    };
    const int npairs = 9;
    int nf = 0;
    for (int pi = 0; pi < npairs; ++pi) {
        char ta = pairs[pi][0], tb = pairs[pi][1];
        // Test a few sizes
        int sizes[][3] = {{23, 19, 31}, {47, 53, 37}};
        for (int si = 0; si < 2; ++si) {
            auto& sz = sizes[si];
            const bool use_default_gpu = (pi == 0 && si == 0);
            nf += check_one<T>(h_cpu, h_gpu, ta, tb,
                               sz[0], sz[1], sz[2], 8,
                               cpu_first, use_default_gpu);
        }
    }
    if (myid == 0) {
        if (nf == 0)
            std::cout << "  [" << name << "] ALL PASSED" << std::endl;
        else
            std::cout << "  [" << name << "] " << nf << " FAILED" << std::endl;
    }
    return nf;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nprows = 2, npcols = 2;
    if (argc >= 3) { nprows = std::atoi(argv[1]); npcols = std::atoi(argv[2]); }

    if (myid == 0) std::cout << "=== Dual CPU+GPU pgemm Test ===" << std::endl;

    TEST("Both backends available", ddla_backend_available(DdlaBackend::CPU) &&
                                    ddla_backend_available(DdlaBackend::GPU));

    // --- Test descriptor-handle mismatch rejection ---
    {
        DdlaHandle_t h_c = nullptr, h_g = nullptr;
        ddla_init(h_c, DdlaBackend::CPU);
        ddla_init(h_g, DdlaBackend::GPU);
        ddla_set(h_c, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddla_set(h_g, MPI_COMM_WORLD, nprows, npcols, 'R');

        DdlaDesc descA_g(h_g); descA_g.init(10, 10, 4, 4, 0, 0);
        DdlaDesc descB_g(h_g); descB_g.init(10, 10, 4, 4, 0, 0);
        DdlaDesc descC_c(h_c); descC_c.init(10, 10, 4, 4, 0, 0);

        // Use local descriptor sizes, not global 10*10
        const std::size_t rawA = static_cast<std::size_t>(descA_g.lld() * descA_g.n_loc());
        const std::size_t rawB = static_cast<std::size_t>(descB_g.lld() * descB_g.n_loc());
        const std::size_t rawC = static_cast<std::size_t>(descC_c.lld() * descC_c.n_loc());
        const std::size_t locA = std::max<std::size_t>(1, rawA);
        const std::size_t locB = std::max<std::size_t>(1, rawB);
        const std::size_t locC = std::max<std::size_t>(1, rawC);
        std::vector<float> bufA(locA, 0.0f), bufB(locB, 0.0f), bufC(locC, 0.0f);
        float *dA = nullptr, *dB = nullptr;
        int mv = 0;
        mv |= ddla_malloc(reinterpret_cast<void**>(&dA), locA * sizeof(float), h_g);
        mv |= ddla_malloc(reinterpret_cast<void**>(&dB), locB * sizeof(float), h_g);
        TEST("Mismatch-test malloc", mv == 0);

        // Upload A, B as device pointers (matching GPU handle)
        int mr = 0;
        mr |= ddla_memcpy(dA, bufA.data(), rawA * sizeof(float), DdlaMemoryCopyKind::HostToDevice, h_g);
        mr |= ddla_memcpy(dB, bufB.data(), rawB * sizeof(float), DdlaMemoryCopyKind::HostToDevice, h_g);
        mr |= ddla_synchronize(h_g);
        TEST("Mismatch-test memcpy/sync", mr == 0);

        bool mismatch_caught = false;
        bool mismatch_msg_ok = false;
        try {
            pgemm('N', 'N', 10, 10, 10, 1.0f,
                  dA, descA_g, dB, descB_g, 0.0f, bufC.data(), descC_c);
        } catch (const std::runtime_error& e) {
            mismatch_caught = true;
            std::string msg(e.what());
            mismatch_msg_ok = (msg.find("pgemm") != std::string::npos &&
                               msg.find("descriptor handle mismatch") != std::string::npos);
        } catch (...) {
            // Other exceptions must not satisfy this test
        }
        TEST("Descriptor-handle mismatch rejects with std::runtime_error",
             mismatch_caught && mismatch_msg_ok);

        bool backend_mismatch_caught = false;
        bool backend_mismatch_msg_ok = false;
        try {
            pgemm<DdlaBackend::CPU>(
                'N', 'N', 10, 10, 10, 1.0f,
                dA, descA_g, dB, descB_g, 0.0f, dA, descA_g);
        } catch (const std::runtime_error& e) {
            backend_mismatch_caught = true;
            const std::string msg(e.what());
            backend_mismatch_msg_ok =
                msg.find("template backend CPU") != std::string::npos &&
                msg.find("handle backend GPU") != std::string::npos;
        } catch (...) {
        }
        TEST("Explicit pgemm backend mismatch rejects with std::runtime_error",
             backend_mismatch_caught && backend_mismatch_msg_ok);

        int mf = 0;
        mf |= ddla_free(dA, h_g);
        mf |= ddla_free(dB, h_g);
        TEST("Mismatch-test free", mf == 0);
        ddla_destroy(h_c);
        ddla_destroy(h_g);
    }

    // --- Test transport_block<DdlaBackend::CPU> succeeds on a CPU handle ---
    // (Contract intentionally changed: transport_block used to reject CPU
    // handles outright; it is now backend-unified via CommTraits<Backend>
    // and genuinely supports CPU. Full CPU-path correctness across varied
    // grid shapes/transposes/k==0 is already covered indirectly by
    // test_cpu_pgemm's existing extensive coverage, exercised through
    // CpuOps::transport_panel -> transport_block<DdlaBackend::CPU,T>; this
    // is a focused API-level check that the explicit-Backend CPU
    // instantiation completes without throwing and produces the correct
    // (uniform-input) result for the portion of the buffer it actually
    // writes.)
    {
        DdlaHandle_t h_cpu_tb = nullptr;
        ddla_init(h_cpu_tb, DdlaBackend::CPU);
        ddla_set(h_cpu_tb, MPI_COMM_WORLD, nprows, npcols, 'R');

        DdlaDesc desc_tb(h_cpu_tb);
        desc_tb.init(8, 8, 4, 4, 0, 0);
        std::size_t sz = static_cast<std::size_t>(desc_tb.lld() * desc_tb.n_loc());
        const std::complex<double> fill_value(3.0, -2.0);
        std::vector<std::complex<double>> buf(sz > 0 ? sz : 1, fill_value);
        // Panel row-extent (m_panel) is one block height (mb), not the full
        // 8 global rows: for sData='R', transport_block reads m_panel LOCAL
        // rows directly from buf on the owning row-rank (matching how it's
        // actually used in pgemm's SUMMA loop and in
        // test_api_grid_transport_block.cpp's row_panel_rows=mb) -- using the
        // full 8 here would over-read buf's local storage on any grid where a
        // row-rank owns fewer than 8 local rows. block_buf holds the gathered
        // m_panel x cols panel (pitch m_panel, not desc_tb's local lld/n_loc);
        // 8*8 is always big enough regardless of grid shape or m_panel choice.
        const int m_panel = desc_tb.mb();
        std::vector<std::complex<double>> block_buf(
            64, std::complex<double>(-999.0, -999.0));

        bool tb_ok = true;
        try {
            transport_block<DdlaBackend::CPU>('R', 'N', m_panel, 8,
                            buf.data(), 0, 0, desc_tb, block_buf.data());
        } catch (...) {
            tb_ok = false;
        }

        // Only the portion transport_block actually writes for this
        // (sData='R', trans='N') case: an m_panel x cols panel, cols = the
        // local column count this rank owns out of the full n=8 columns.
        const int j_loc = num_loc(0, desc_tb.nb(), desc_tb.mypcol(), desc_tb.icsrc(), desc_tb.npcols());
        const int n_loc_val = num_loc(8, desc_tb.nb(), desc_tb.mypcol(), desc_tb.icsrc(), desc_tb.npcols());
        const int cols = n_loc_val - j_loc;
        double tb_err = 0.0;
        if (cols > 0) {
            for (int k = 0; k < m_panel * cols; ++k)
                tb_err = std::max(tb_err, std::abs(block_buf[k] - fill_value));
        }
        TEST("transport_block<DdlaBackend::CPU> succeeds and returns uniform input unchanged",
             tb_ok && tb_err < 1e-12);

        ddla_destroy(h_cpu_tb);
    }

    // --- Create CPU and GPU handles from same MPI_COMM_WORLD ---
    DdlaHandle_t h_cpu = nullptr, h_gpu = nullptr;
    ddla_init(h_cpu, DdlaBackend::CPU);
    ddla_init(h_gpu, DdlaBackend::GPU);
    ddla_set(h_cpu, MPI_COMM_WORLD, nprows, npcols, 'R');
    ddla_set(h_gpu, MPI_COMM_WORLD, nprows, npcols, 'R');

    TEST("CPU backend resolves", ddla_get_backend(h_cpu) == DdlaBackend::CPU);
    TEST("GPU backend resolves", ddla_get_backend(h_gpu) == DdlaBackend::GPU);
    TEST("CPU stream is null", ddla_get_stream(h_cpu) == nullptr);
    TEST("GPU stream is non-null after set",
         ddla_get_stream(h_gpu) != nullptr);

    // Memory round-trip per backend
    {
        void* cpu_ptr = nullptr;
        void* gpu_ptr = nullptr;
        TEST("CPU malloc", ddla_malloc(&cpu_ptr, 256, h_cpu) == 0);
        TEST("GPU malloc", ddla_malloc(&gpu_ptr, 256, h_gpu) == 0);

        char pattern[256], readback[256];
        for (int i = 0; i < 256; ++i) pattern[i] = static_cast<char>(i);
        std::memset(readback, 0, 256);

        // CPU round-trip
        int rc = 0;
        rc |= ddla_memcpy(cpu_ptr, pattern, 256, DdlaMemoryCopyKind::HostToDevice, h_cpu);
        rc |= ddla_memcpy(readback, cpu_ptr, 256, DdlaMemoryCopyKind::DeviceToHost, h_cpu);
        TEST("CPU memcpy round-trip", rc == 0);
        bool cpu_ok = (std::memcmp(pattern, readback, 256) == 0);
        TEST("CPU memory round-trip", cpu_ok);

        // GPU round-trip
        std::memset(readback, 0, 256);
        rc = 0;
        rc |= ddla_memcpy(gpu_ptr, pattern, 256, DdlaMemoryCopyKind::HostToDevice, h_gpu);
        rc |= ddla_memcpy(readback, gpu_ptr, 256, DdlaMemoryCopyKind::DeviceToHost, h_gpu);
        rc |= ddla_synchronize(h_gpu);
        TEST("GPU memcpy round-trip", rc == 0);
        bool gpu_ok = (std::memcmp(pattern, readback, 256) == 0);
        TEST("GPU memory round-trip", gpu_ok);

        int mf = 0;
        mf |= ddla_free(cpu_ptr, h_cpu);
        mf |= ddla_free(gpu_ptr, h_gpu);
        TEST("Round-trip free", mf == 0);
    }

    // Destroy the memory-roundtrip handle pair before creating execution handles
    ddla_destroy(h_cpu);
    ddla_destroy(h_gpu);

    // --- Section 1: GPU→CPU execution order, destroy GPU first ---
    int nf = 0;
    {
        DdlaHandle_t h_cpu_s1 = nullptr, h_gpu_s1 = nullptr;
        ddla_init(h_cpu_s1, DdlaBackend::CPU);
        ddla_init(h_gpu_s1, DdlaBackend::GPU);
        ddla_set(h_cpu_s1, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddla_set(h_gpu_s1, MPI_COMM_WORLD, nprows, npcols, 'R');
        nf += run_type<float>("float (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        nf += run_type<double>("double (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        nf += run_type<std::complex<float>>("cfloat (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        nf += run_type<std::complex<double>>("cdouble (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        // Destroy GPU first, then CPU
        ddla_destroy(h_gpu_s1);
        ddla_destroy(h_cpu_s1);
    }

    // --- Section 2: CPU→GPU execution order, destroy CPU first ---
    {
        DdlaHandle_t h_cpu_s2 = nullptr, h_gpu_s2 = nullptr;
        ddla_init(h_cpu_s2, DdlaBackend::CPU);
        ddla_init(h_gpu_s2, DdlaBackend::GPU);
        ddla_set(h_cpu_s2, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddla_set(h_gpu_s2, MPI_COMM_WORLD, nprows, npcols, 'R');
        nf += run_type<float>("float (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        nf += run_type<double>("double (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        nf += run_type<std::complex<float>>("cfloat (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        nf += run_type<std::complex<double>>("cdouble (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        // Destroy CPU first, then GPU
        ddla_destroy(h_cpu_s2);
        ddla_destroy(h_gpu_s2);
    }

    // --- Section 3: k==0 CPU/GPU parity (F2 regression) ---
    {
        DdlaHandle_t h_cpu_s3 = nullptr, h_gpu_s3 = nullptr;
        ddla_init(h_cpu_s3, DdlaBackend::CPU);
        ddla_init(h_gpu_s3, DdlaBackend::GPU);
        ddla_set(h_cpu_s3, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddla_set(h_gpu_s3, MPI_COMM_WORLD, nprows, npcols, 'R');
        int nf3 = 0;
        nf3 += check_k_zero<float>(h_cpu_s3, h_gpu_s3);
        nf3 += check_k_zero<double>(h_cpu_s3, h_gpu_s3);
        nf3 += check_k_zero<std::complex<float>>(h_cpu_s3, h_gpu_s3);
        nf3 += check_k_zero<std::complex<double>>(h_cpu_s3, h_gpu_s3);
        if (myid == 0) {
            if (nf3 == 0)
                std::cout << "  [k=0 parity] ALL PASSED" << std::endl;
            else
                std::cout << "  [k=0 parity] " << nf3 << " FAILED" << std::endl;
        }
        nf += nf3;
        ddla_destroy(h_cpu_s3);
        ddla_destroy(h_gpu_s3);
    }

    if (myid == 0) {
        if (nf == 0)
            std::cout << "ALL DUAL Pgemm TESTS PASSED" << std::endl;
        else
            std::cout << nf << " DUAL Pgemm TEST(S) FAILED" << std::endl;
    }

    MPI_Finalize();
    return (nf > 0) ? 1 : 0;
}
