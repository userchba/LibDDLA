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
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "transport_block.h"

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

template <typename T>
static int check_one(
    DdlaHandle_t h_cpu, DdlaHandle_t h_gpu,
    char transa, char transb,
    int M, int N, int K, int nb,
    bool cpu_first = false,
    bool use_default_gpu = false)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(h_cpu, nprows, npcols);

    int rowsA = (transa == 'N') ? M : K;
    int colsA = (transa == 'N') ? K : M;
    int rowsB = (transb == 'N') ? K : N;
    int colsB = (transb == 'N') ? N : K;

    int descA_cpu[DDLA_DLEN_], descB_cpu[DDLA_DLEN_], descC_cpu[DDLA_DLEN_];
    int descA_gpu[DDLA_DLEN_], descB_gpu[DDLA_DLEN_], descC_gpu[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA_cpu, h_cpu, rowsA, colsA, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descB_cpu, h_cpu, rowsB, colsB, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descC_cpu, h_cpu, M, N, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descA_gpu, h_gpu, rowsA, colsA, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descB_gpu, h_gpu, rowsB, colsB, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descC_gpu, h_gpu, M, N, nb, nb, 0, 0));

    // Allocate CPU host buffers (local sizes)
    const std::size_t szA = static_cast<std::size_t>(descA_cpu[DDLA_LLD_] * ddla_test::n_loc(h_cpu, descA_cpu));
    const std::size_t szB = static_cast<std::size_t>(descB_cpu[DDLA_LLD_] * ddla_test::n_loc(h_cpu, descB_cpu));
    const std::size_t szC = static_cast<std::size_t>(descC_cpu[DDLA_LLD_] * ddla_test::n_loc(h_cpu, descC_cpu));
    const std::size_t szDA = static_cast<std::size_t>(descA_gpu[DDLA_LLD_] * ddla_test::n_loc(h_gpu, descA_gpu));
    const std::size_t szDB = static_cast<std::size_t>(descB_gpu[DDLA_LLD_] * ddla_test::n_loc(h_gpu, descB_gpu));
    const std::size_t szDC = static_cast<std::size_t>(descC_gpu[DDLA_LLD_] * ddla_test::n_loc(h_gpu, descC_gpu));
    TEST("CPU/GPU local storage matches", szA == szDA && szB == szDB && szC == szDC);
    std::vector<T> h_A(std::max<std::size_t>(1, szA));
    std::vector<T> h_B(std::max<std::size_t>(1, szB));
    std::vector<T> h_C_cpu(std::max<std::size_t>(1, szC));
    std::vector<T> h_C_gpu(std::max<std::size_t>(1, szC));

    // Fill local blocks deterministically
    auto fill = [](const DdlaHandle_t& handle, const int* desc, std::vector<T>& buf, int salt) {
        for (int jl = 0; jl < ddla_test::n_loc(handle, desc); ++jl) {
            int jg = indxl2g(jl, desc[DDLA_NB_], ddla_test::mypcol(handle), desc[DDLA_CSRC_], ddla_test::npcols(handle));
            for (int il = 0; il < ddla_test::m_loc(handle, desc); ++il) {
                int ig = indxl2g(il, desc[DDLA_MB_], ddla_test::myprow(handle), desc[DDLA_RSRC_], ddla_test::nprows(handle));
                buf[il + jl * desc[DDLA_LLD_]] = elem_val<T>(ig, jg, salt);
            }
        }
    };
    fill(h_cpu, descA_cpu, h_A, 0);
    fill(h_cpu, descB_cpu, h_B, 1000);
    fill(h_cpu, descC_cpu, h_C_cpu, 2000);
    fill(h_cpu, descC_cpu, h_C_gpu, 2000); // same initial C

    T alpha = T(1.25);
    T beta  = T(-0.5);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        alpha = T(1.25, -0.375);
        beta = T(-0.5, 0.25);
    }

    // --- GPU device allocation and upload ---
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    TEST("GPU malloc A", ddlaMalloc(reinterpret_cast<void**>(&d_A),
                                     std::max<std::size_t>(1, szDA) * sizeof(T), h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
    TEST("GPU malloc B", ddlaMalloc(reinterpret_cast<void**>(&d_B),
                                     std::max<std::size_t>(1, szDB) * sizeof(T), h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
    TEST("GPU malloc C", ddlaMalloc(reinterpret_cast<void**>(&d_C),
                                     std::max<std::size_t>(1, szDC) * sizeof(T), h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);

    // Transfer sizes must match the CPU-side storage
    std::size_t xferA = szA * sizeof(T);
    std::size_t xferB = szB * sizeof(T);
    std::size_t xferC = szC * sizeof(T);
    int ret = 0;
    ret |= static_cast<int>(ddlaMemcpy(d_A, h_A.data(), xferA, DdlaMemoryCopyKind::HostToDevice, h_gpu));
    ret |= static_cast<int>(ddlaMemcpy(d_B, h_B.data(), xferB, DdlaMemoryCopyKind::HostToDevice, h_gpu));
    ret |= static_cast<int>(ddlaMemcpy(d_C, h_C_gpu.data(), xferC, DdlaMemoryCopyKind::HostToDevice, h_gpu));
    TEST("GPU memcpy upload", ret == 0);
    TEST("GPU sync after upload", ddlaSynchronize(h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);

    // --- Compute in requested order ---
    auto run_cpu = [&]() {
        pgemm<DdlaBackend::CPU>(
            h_cpu, transa, transb, M, N, K, alpha,
            h_A.data(), descA_cpu, h_B.data(), descB_cpu,
            beta, h_C_cpu.data(), descC_cpu);
    };
    auto run_gpu_and_download = [&]() {
        if (use_default_gpu) {
            pgemm<>(h_gpu, transa, transb, M, N, K, alpha,
                    d_A, descA_gpu, d_B, descB_gpu, beta, d_C, descC_gpu);
        } else {
            pgemm<DdlaBackend::GPU>(
                h_gpu, transa, transb, M, N, K, alpha,
                d_A, descA_gpu, d_B, descB_gpu, beta, d_C, descC_gpu);
        }
        int ret_dl = static_cast<int>(ddlaMemcpy(h_C_gpu.data(), d_C, xferC, DdlaMemoryCopyKind::DeviceToHost, h_gpu));
        TEST("GPU download after compute", ret_dl == 0);
    };

    if (cpu_first) {
        run_cpu();
        run_gpu_and_download();
    } else {
        run_gpu_and_download();
        run_cpu();
    }

    TEST("GPU sync after download", ddlaSynchronize(h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);

    int mf = 0;
    mf |= static_cast<int>(ddlaFree(d_A, h_gpu));
    mf |= static_cast<int>(ddlaFree(d_B, h_gpu));
    mf |= static_cast<int>(ddlaFree(d_C, h_gpu));
    TEST("GPU free", mf == 0);

    // --- Compare CPU vs GPU results ---
    // Detect nonfinite values before they disappear through std::max/std::abs
    int rank_nonfinite_flag = 0;
    int nloc = ddla_test::n_loc(h_cpu, descC_cpu);
    int mloc = ddla_test::m_loc(h_cpu, descC_cpu);
    int lld = descC_cpu[DDLA_LLD_];
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
        MPI_Comm comm = ddlaGetCommunicator(h_cpu);
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
        MPI_Comm comm = ddlaGetCommunicator(h_cpu);
        MPI_Allreduce(&max_ref, &global_ref, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&max_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, comm);
    }

    constexpr bool is_single = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
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

    int descA_cpu[DDLA_DLEN_], descB_cpu[DDLA_DLEN_], descC_cpu[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA_cpu, h_cpu, M, 0, nb, nb, 0, 0));  // K == 0
    DDLA_CHECK(ddlaDescInit(descB_cpu, h_cpu, 0, N, nb, nb, 0, 0));  // K == 0
    DDLA_CHECK(ddlaDescInit(descC_cpu, h_cpu, M, N, nb, nb, 0, 0));

    int descA_gpu[DDLA_DLEN_], descB_gpu[DDLA_DLEN_], descC_gpu[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA_gpu, h_gpu, M, 0, nb, nb, 0, 0));  // K == 0
    DDLA_CHECK(ddlaDescInit(descB_gpu, h_gpu, 0, N, nb, nb, 0, 0));  // K == 0
    DDLA_CHECK(ddlaDescInit(descC_gpu, h_gpu, M, N, nb, nb, 0, 0));

    const std::size_t szC =
        static_cast<std::size_t>(descC_cpu[DDLA_LLD_] * ddla_test::n_loc(h_cpu, descC_cpu));
    std::vector<T> h_A(1), h_B(1);
    std::vector<T> h_C_cpu(std::max<std::size_t>(1, szC));
    std::vector<T> h_C_gpu(std::max<std::size_t>(1, szC));

    for (int jl = 0; jl < ddla_test::n_loc(h_cpu, descC_cpu); ++jl) {
        int jg = indxl2g(jl, descC_cpu[DDLA_NB_], ddla_test::mypcol(h_cpu), descC_cpu[DDLA_CSRC_], ddla_test::npcols(h_cpu));
        for (int il = 0; il < ddla_test::m_loc(h_cpu, descC_cpu); ++il) {
            int ig = indxl2g(il, descC_cpu[DDLA_MB_], ddla_test::myprow(h_cpu), descC_cpu[DDLA_RSRC_], ddla_test::nprows(h_cpu));
            T v = elem_val<T>(ig, jg, 3000);
            h_C_cpu[il + jl * descC_cpu[DDLA_LLD_]] = v;
            h_C_gpu[il + jl * descC_cpu[DDLA_LLD_]] = v;
        }
    }

    T alpha = T(2.0);
    T beta = T(-0.75);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        alpha = T(2.0, -1.0);
        beta = T(-0.75, 0.5);
    }

    pgemm<DdlaBackend::CPU>(h_cpu, 'N', 'N', M, N, 0, alpha,
                            h_A.data(), descA_cpu, h_B.data(), descB_cpu,
                            beta, h_C_cpu.data(), descC_cpu);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    TEST("k=0 GPU malloc A", ddlaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T), h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
    TEST("k=0 GPU malloc B", ddlaMalloc(reinterpret_cast<void**>(&d_B), sizeof(T), h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
    TEST("k=0 GPU malloc C", ddlaMalloc(reinterpret_cast<void**>(&d_C),
                                        std::max<std::size_t>(1, szC) * sizeof(T), h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
    int up = static_cast<int>(ddlaMemcpy(d_C, h_C_gpu.data(), szC * sizeof(T),
                         DdlaMemoryCopyKind::HostToDevice, h_gpu));
    TEST("k=0 GPU upload C", up == 0);
    TEST("k=0 GPU sync after upload", ddlaSynchronize(h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);

    pgemm<DdlaBackend::GPU>(h_gpu, 'N', 'N', M, N, 0, alpha,
                            d_A, descA_gpu, d_B, descB_gpu,
                            beta, d_C, descC_gpu);
    int dl = static_cast<int>(ddlaMemcpy(h_C_gpu.data(), d_C, szC * sizeof(T),
                        DdlaMemoryCopyKind::DeviceToHost, h_gpu));
    TEST("k=0 GPU download", dl == 0);
    TEST("k=0 GPU sync after download", ddlaSynchronize(h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);

    int mf = 0;
    mf |= static_cast<int>(ddlaFree(d_A, h_gpu));
    mf |= static_cast<int>(ddlaFree(d_B, h_gpu));
    mf |= static_cast<int>(ddlaFree(d_C, h_gpu));
    TEST("k=0 GPU free", mf == 0);

    constexpr bool is_single = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
    const double tol = is_single ? 2e-4 : 2e-10;
    int local_failed = 0;
    double local_max_err = 0.0;
    for (int jl = 0; jl < ddla_test::n_loc(h_cpu, descC_cpu); ++jl) {
        int jg = indxl2g(jl, descC_cpu[DDLA_NB_], ddla_test::mypcol(h_cpu), descC_cpu[DDLA_CSRC_], ddla_test::npcols(h_cpu));
        for (int il = 0; il < ddla_test::m_loc(h_cpu, descC_cpu); ++il) {
            int ig = indxl2g(il, descC_cpu[DDLA_MB_], ddla_test::myprow(h_cpu), descC_cpu[DDLA_RSRC_], ddla_test::nprows(h_cpu));
            T expected = beta * elem_val<T>(ig, jg, 3000);
            T cpu_val = h_C_cpu[il + jl * descC_cpu[DDLA_LLD_]];
            T gpu_val = h_C_gpu[il + jl * descC_cpu[DDLA_LLD_]];
            local_max_err = std::max(local_max_err,
                static_cast<double>(std::abs(cpu_val - gpu_val)));
            if (static_cast<double>(std::abs(cpu_val - expected)) > tol ||
                static_cast<double>(std::abs(gpu_val - expected)) > tol) {
                local_failed = 1;
            }
        }
    }

    MPI_Comm comm = ddlaGetCommunicator(h_cpu);
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

    TEST("Both backends available", ddlaBackendAvailable(DdlaBackend::CPU) &&
                                    ddlaBackendAvailable(DdlaBackend::GPU));

    // --- Test descriptor-handle mismatch rejection ---
    {
        DdlaHandle_t h_c = nullptr, h_g = nullptr;
        ddlaInit(h_c, DdlaBackend::CPU);
        ddlaInit(h_g, DdlaBackend::GPU);
        ddlaSet(h_c, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddlaSet(h_g, MPI_COMM_WORLD, nprows, npcols, 'R');

        int descA_g[DDLA_DLEN_], descB_g[DDLA_DLEN_], descC_c[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA_g, h_g, 10, 10, 4, 4, 0, 0));
        DDLA_CHECK(ddlaDescInit(descB_g, h_g, 10, 10, 4, 4, 0, 0));
        DDLA_CHECK(ddlaDescInit(descC_c, h_c, 10, 10, 4, 4, 0, 0));

        // Use local descriptor sizes, not global 10*10
        const std::size_t rawA = static_cast<std::size_t>(descA_g[DDLA_LLD_] * ddla_test::n_loc(h_g, descA_g));
        const std::size_t rawB = static_cast<std::size_t>(descB_g[DDLA_LLD_] * ddla_test::n_loc(h_g, descB_g));
        const std::size_t rawC = static_cast<std::size_t>(descC_c[DDLA_LLD_] * ddla_test::n_loc(h_c, descC_c));
        const std::size_t locA = std::max<std::size_t>(1, rawA);
        const std::size_t locB = std::max<std::size_t>(1, rawB);
        const std::size_t locC = std::max<std::size_t>(1, rawC);
        std::vector<float> bufA(locA, 0.0f), bufB(locB, 0.0f), bufC(locC, 0.0f);
        float *dA = nullptr, *dB = nullptr;
        int mv = 0;
        mv |= static_cast<int>(ddlaMalloc(reinterpret_cast<void**>(&dA), locA * sizeof(float), h_g));
        mv |= static_cast<int>(ddlaMalloc(reinterpret_cast<void**>(&dB), locB * sizeof(float), h_g));
        TEST("Mismatch-test malloc", mv == 0);

        // Upload A, B as device pointers (matching GPU handle)
        int mr = 0;
        mr |= static_cast<int>(ddlaMemcpy(dA, bufA.data(), rawA * sizeof(float), DdlaMemoryCopyKind::HostToDevice, h_g));
        mr |= static_cast<int>(ddlaMemcpy(dB, bufB.data(), rawB * sizeof(float), DdlaMemoryCopyKind::HostToDevice, h_g));
        mr |= static_cast<int>(ddlaSynchronize(h_g));
        TEST("Mismatch-test memcpy/sync", mr == 0);

        bool mismatch_caught = false;
        bool mismatch_msg_ok = false;
        int desc_foreign[DDLA_DLEN_];
        std::copy(descA_g, descA_g + DDLA_DLEN_, desc_foreign);
        desc_foreign[DDLA_LLD_] -= 1;  // built against a different process grid
        try {
            pgemm(h_g, 'N', 'N', 10, 10, 10, 1.0f,
                  dA, desc_foreign, dB, descB_g, 0.0f, bufC.data(), descC_c);
        } catch (const std::invalid_argument& e) {
            mismatch_caught = true;
            std::string msg(e.what());
            mismatch_msg_ok = (msg.find("LLD_A") != std::string::npos &&
                               msg.find("process grid") != std::string::npos);
        } catch (...) {
            // Other exceptions must not satisfy this test
        }
        TEST("Descriptor-grid mismatch rejects with std::invalid_argument",
             mismatch_caught && mismatch_msg_ok);

        bool backend_mismatch_caught = false;
        bool backend_mismatch_msg_ok = false;
        try {
            pgemm<DdlaBackend::CPU>(
                h_g, 'N', 'N', 10, 10, 10, 1.0f,
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
        mf |= static_cast<int>(ddlaFree(dA, h_g));
        mf |= static_cast<int>(ddlaFree(dB, h_g));
        TEST("Mismatch-test free", mf == 0);
        ddlaDestroy(h_c);
        ddlaDestroy(h_g);
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
        ddlaInit(h_cpu_tb, DdlaBackend::CPU);
        ddlaSet(h_cpu_tb, MPI_COMM_WORLD, nprows, npcols, 'R');

        int desc_tb[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc_tb, h_cpu_tb, 8, 8, 4, 4, 0, 0));
        std::size_t sz = static_cast<std::size_t>(desc_tb[DDLA_LLD_] * ddla_test::n_loc(h_cpu_tb, desc_tb));
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
        const int m_panel = desc_tb[DDLA_MB_];
        std::vector<std::complex<double>> block_buf(
            64, std::complex<double>(-999.0, -999.0));

        bool tb_ok = true;
        try {
            transport_block<DdlaBackend::CPU>(h_cpu_tb, 'R', 'N', m_panel, 8,
                            buf.data(), 0, 0, desc_tb, block_buf.data());
        } catch (...) {
            tb_ok = false;
        }

        // Only the portion transport_block actually writes for this
        // (sData='R', trans='N') case: an m_panel x cols panel, cols = the
        // local column count this rank owns out of the full n=8 columns.
        const int j_loc = num_loc(0, desc_tb[DDLA_NB_], ddla_test::mypcol(h_cpu_tb), desc_tb[DDLA_CSRC_], ddla_test::npcols(h_cpu_tb));
        const int n_loc_val = num_loc(8, desc_tb[DDLA_NB_], ddla_test::mypcol(h_cpu_tb), desc_tb[DDLA_CSRC_], ddla_test::npcols(h_cpu_tb));
        const int cols = n_loc_val - j_loc;
        double tb_err = 0.0;
        if (cols > 0) {
            for (int k = 0; k < m_panel * cols; ++k)
                tb_err = std::max(tb_err, std::abs(block_buf[k] - fill_value));
        }
        TEST("transport_block<DdlaBackend::CPU> succeeds and returns uniform input unchanged",
             tb_ok && tb_err < 1e-12);

        ddlaDestroy(h_cpu_tb);
    }

    // --- Create CPU and GPU handles from same MPI_COMM_WORLD ---
    DdlaHandle_t h_cpu = nullptr, h_gpu = nullptr;
    ddlaInit(h_cpu, DdlaBackend::CPU);
    ddlaInit(h_gpu, DdlaBackend::GPU);
    ddlaSet(h_cpu, MPI_COMM_WORLD, nprows, npcols, 'R');
    ddlaSet(h_gpu, MPI_COMM_WORLD, nprows, npcols, 'R');

    TEST("CPU backend resolves", ddlaGetBackend(h_cpu) == DdlaBackend::CPU);
    TEST("GPU backend resolves", ddlaGetBackend(h_gpu) == DdlaBackend::GPU);
    TEST("CPU stream is null", ddlaGetStream(h_cpu) == nullptr);
    TEST("GPU stream is non-null after set",
         ddlaGetStream(h_gpu) != nullptr);

    // Memory round-trip per backend
    {
        void* cpu_ptr = nullptr;
        void* gpu_ptr = nullptr;
        TEST("CPU malloc", ddlaMalloc(&cpu_ptr, 256, h_cpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
        TEST("GPU malloc", ddlaMalloc(&gpu_ptr, 256, h_gpu) == ddlaStatus_t::DDLA_STATUS_SUCCESS);

        char pattern[256], readback[256];
        for (int i = 0; i < 256; ++i) pattern[i] = static_cast<char>(i);
        std::memset(readback, 0, 256);

        // CPU round-trip
        int rc = 0;
        rc |= static_cast<int>(ddlaMemcpy(cpu_ptr, pattern, 256, DdlaMemoryCopyKind::HostToDevice, h_cpu));
        rc |= static_cast<int>(ddlaMemcpy(readback, cpu_ptr, 256, DdlaMemoryCopyKind::DeviceToHost, h_cpu));
        TEST("CPU memcpy round-trip", rc == 0);
        bool cpu_ok = (std::memcmp(pattern, readback, 256) == 0);
        TEST("CPU memory round-trip", cpu_ok);

        // GPU round-trip
        std::memset(readback, 0, 256);
        rc = 0;
        rc |= static_cast<int>(ddlaMemcpy(gpu_ptr, pattern, 256, DdlaMemoryCopyKind::HostToDevice, h_gpu));
        rc |= static_cast<int>(ddlaMemcpy(readback, gpu_ptr, 256, DdlaMemoryCopyKind::DeviceToHost, h_gpu));
        rc |= static_cast<int>(ddlaSynchronize(h_gpu));
        TEST("GPU memcpy round-trip", rc == 0);
        bool gpu_ok = (std::memcmp(pattern, readback, 256) == 0);
        TEST("GPU memory round-trip", gpu_ok);

        int mf = 0;
        mf |= static_cast<int>(ddlaFree(cpu_ptr, h_cpu));
        mf |= static_cast<int>(ddlaFree(gpu_ptr, h_gpu));
        TEST("Round-trip free", mf == 0);
    }

    // Destroy the memory-roundtrip handle pair before creating execution handles
    ddlaDestroy(h_cpu);
    ddlaDestroy(h_gpu);

    // --- Section 1: GPU→CPU execution order, destroy GPU first ---
    int nf = 0;
    {
        DdlaHandle_t h_cpu_s1 = nullptr, h_gpu_s1 = nullptr;
        ddlaInit(h_cpu_s1, DdlaBackend::CPU);
        ddlaInit(h_gpu_s1, DdlaBackend::GPU);
        ddlaSet(h_cpu_s1, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddlaSet(h_gpu_s1, MPI_COMM_WORLD, nprows, npcols, 'R');
        nf += run_type<float>("float (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        nf += run_type<double>("double (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        nf += run_type<std::complex<float>>("cfloat (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        nf += run_type<std::complex<double>>("cdouble (GPU->CPU)", h_cpu_s1, h_gpu_s1, false);
        // Destroy GPU first, then CPU
        ddlaDestroy(h_gpu_s1);
        ddlaDestroy(h_cpu_s1);
    }

    // --- Section 2: CPU→GPU execution order, destroy CPU first ---
    {
        DdlaHandle_t h_cpu_s2 = nullptr, h_gpu_s2 = nullptr;
        ddlaInit(h_cpu_s2, DdlaBackend::CPU);
        ddlaInit(h_gpu_s2, DdlaBackend::GPU);
        ddlaSet(h_cpu_s2, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddlaSet(h_gpu_s2, MPI_COMM_WORLD, nprows, npcols, 'R');
        nf += run_type<float>("float (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        nf += run_type<double>("double (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        nf += run_type<std::complex<float>>("cfloat (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        nf += run_type<std::complex<double>>("cdouble (CPU->GPU)", h_cpu_s2, h_gpu_s2, true);
        // Destroy CPU first, then GPU
        ddlaDestroy(h_cpu_s2);
        ddlaDestroy(h_gpu_s2);
    }

    // --- Section 3: k==0 CPU/GPU parity (F2 regression) ---
    {
        DdlaHandle_t h_cpu_s3 = nullptr, h_gpu_s3 = nullptr;
        ddlaInit(h_cpu_s3, DdlaBackend::CPU);
        ddlaInit(h_gpu_s3, DdlaBackend::GPU);
        ddlaSet(h_cpu_s3, MPI_COMM_WORLD, nprows, npcols, 'R');
        ddlaSet(h_gpu_s3, MPI_COMM_WORLD, nprows, npcols, 'R');
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
        ddlaDestroy(h_cpu_s3);
        ddlaDestroy(h_gpu_s3);
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
