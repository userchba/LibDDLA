// Run only with libpblas_alloc_failure.so preloaded; see the matching test recipe.
#include "mpi_test_init.h"
#include "test_backend.h"
#include "LU/lu_internal.h"
#include "test_desc_helpers.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ddla;
namespace {
template <typename F> F symbol(const char* name)
{
    void* address = dlsym(RTLD_DEFAULT, name);
    F function = nullptr;
    static_assert(sizeof(function) == sizeof(address), "function pointer ABI");
    std::memcpy(&function, &address, sizeof(function));
    if (!function) throw std::runtime_error("allocator failure preload is missing");
    return function;
}
struct Matrix {
    int d[DDLA_DLEN_];
    std::vector<double> original, actual;
    ddla::vector<double> storage;
    explicit Matrix(const DdlaHandle_t& h, int extent, double value) : storage(h)
    {
        DDLA_CHECK(ddlaDescInit(d, h, extent, extent, 4, 4, 0, 0));
        d[DDLA_LLD_] += 3;
        const std::size_t count = std::size_t(d[DDLA_LLD_]) * ddla_test::n_loc(h, d);
        original.assign(count + 16, -12345.0);
        const int rows = ddla_test::m_loc(h, d), cols = ddla_test::n_loc(h, d);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                original[8 + i + std::size_t(j) * d[DDLA_LLD_]] = value;
        actual.resize(original.size());
        storage.resize(original.size());
        DDLA_CHECK(ddlaMemcpy(h, storage.data(), original.data(), original.size() * sizeof(double)));
    }
    double* data() { return storage.data() + 8; }
    void read(const DdlaHandle_t& h)
    {
        DDLA_CHECK(ddlaMemcpy(h, actual.data(), storage.data(), actual.size() * sizeof(double)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    bool unchanged(const DdlaHandle_t& h)
    {
        read(h);
        return std::memcmp(actual.data(), original.data(), actual.size() * sizeof(double)) == 0;
    }
};
}
int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { MPI_Finalize(); return 77; }
    try {
        // CLI: OP KIND [N [GRID_ROWS [FAULT_RANK]]], KIND=backend|host.
        const std::string op = argc > 1 ? argv[1] : "pgemm";
        const std::string kind = argc > 2 ? argv[2] : "backend";
        const int n = argc > 3 ? std::atoi(argv[3]) : 17;
        const int rows = argc > 4 ? std::atoi(argv[4]) : (size % 2 == 0 ? 2 : 1);
        const int target = argc > 5 ? std::atoi(argv[5]) : 0;
        if (n < 1 || rows < 1 || size % rows || target < 0 || target >= size ||
            (kind != "backend" && kind != "host"))
            throw std::invalid_argument("invalid fault-test arguments");
        const auto prewarm = symbol<int (*)()>("pblas_alloc_failure_prewarm");
        const auto arm = symbol<void (*)(int)>("pblas_alloc_failure_arm");
        const auto disarm = symbol<void (*)()>("pblas_alloc_failure_disarm");
        const auto fired = symbol<unsigned (*)()>("pblas_alloc_failure_fired");
        const auto bytes = symbol<std::size_t (*)()>("pblas_alloc_failure_bytes");
        DdlaHandle_t h = nullptr;
        DDLA_CHECK(ddlaInit(h, ddla_test::execution_backend));
        DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, rows, size / rows));
        {
            Matrix a(h, n + 4, 1), b(h, n + 4, 2), c(h, n + 4, 5);
            std::vector<int> piv(std::size_t(n) + 8, -777);
            const auto original_piv = piv;
            std::vector<double> tau(std::size_t(n) + 8, -888.);
            const auto original_tau = tau;
            int info = -1;
            ddla::vector<int> block_piv(h, piv.size());
            DDLA_CHECK(ddlaMemcpy(h, block_piv.data(), piv.data(), piv.size() * sizeof(int)));
            DDLA_CHECK(ddlaSynchronize(h));
            auto call = [&] {
                if (op == "pgemm")
                    pgemm<double>(h, 'N', 'N', n, n, n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "pgeadd")
                    pgeadd<double>(h, 'T', 'N', n, n, 1., a.data(), 2, 3, a.d,
                        1., b.data(), 1, 2, b.d, c.data(), 3, 2, c.d);
                else if (op == "phemm")
                    phemm<double>(h, 'L', 'L', n, n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "psymm")
                    psymm<double>(h, 'L', 'L', n, n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "psyrk")
                    psyrk<double>(h, 'L', 'N', n, n, 1., a.data(), a.d, 1., c.data(), c.d);
                else if (op == "psyr2k")
                    psyr2k<double>(h, 'L', 'N', n, n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "pgemmt")
                    pgemmt<double>(h, 'L', 'N', 'N', n, n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "phemmt")
                    phemmt<double>(h, 'L', 'L', 'L', n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "psymmt")
                    psymmt<double>(h, 'L', 'L', 'L', n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "ptrtrm")
                    ptrtrm<double>(h, 'L', 'N', 'L', 'N', n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
                else if (op == "ptrmm")
                    ptrmm<double>(h, 'L', 'L', 'N', 'U', n, n, 1., a.data(), a.d, c.data(), c.d);
                else if (op == "ptrtrs")
                    ptrtrs<double>(h, 'L', 'L', 'N', 'U', n, n, a.data(), a.d, c.data(), c.d);
                else if (op == "ppotrf")
                    ppotrf<double>(h, 'L', n, a.data(), 1, 1, a.d, info);
                else if (op == "ppotrs")
                    ppotrs<double>(h, 'L', 'L', 'N', n, n, a.data(), a.d, c.data(), c.d);
                else if (op == "pposv")
                    pposv<double>(h, 'L', 'L', 'N', n, n, a.data(), 1, 1, a.d, c.data(), 1, 1, c.d, info);
                else if (op == "pswap")
                    pswap<double>(h, n, a.data(), 1, 1, a.d, 1, c.data(), 1, 1, c.d, 1);
                else if (op == "pgeqrf")
                    pgeqrf<double>(h, n, n, a.data(), a.d, tau.data(), info);
                else if (op == "pgeqrs")
                    pgeqrs<double>(h, n, n, n, a.data(), a.d, tau.data(), c.data(), c.d, info);
                else if (op == "pgetrf")
                    pgetrf<double>(h, n, n, a.data(), 1, 1, a.d, piv.data(), info);
                else if (op == "pgetf2")
                    pgetf2<ddla_test::execution_backend, double>(h, n, n, 4, a.data(), 1, 1, a.d, piv.data(), info);
                else if (op == "pgetrf_bpiv")
                    pgetrf_bpiv<double>(h, n, n, a.data(), a.d, block_piv.data(), info);
                else if (op == "pgetrf_nopiv")
                    pgetrf_nopiv<double>(h, n, n, a.data(), a.d, info);
                else if (op == "pgetrs")
                    pgetrs<double>(h, 'L', 'N', n, n, a.data(), a.d, piv.data(), c.data(), c.d);
                else if (op == "pgetrs_bpiv")
                    pgetrs_bpiv<double>(h, 'L', 'N', n, n, a.data(), a.d, block_piv.data(), c.data(), c.d);
                else if (op == "pgetrs_nopiv")
                    pgetrs_nopiv<double>(h, 'L', 'N', n, n, a.data(), a.d, c.data(), c.d);
                else if (op == "pgesv")
                    pgesv<double>(h, 'L', 'N', n, n, a.data(), a.d, c.data(), c.d);
                else if (op == "pgesv_bpiv")
                    pgesv_bpiv<double>(h, 'L', 'N', n, n, a.data(), a.d, c.data(), c.d);
                else if (op == "pgesv_nopiv")
                    pgesv_nopiv<double>(h, 'L', 'N', n, n, a.data(), a.d, c.data(), c.d);
                else if (op == "plapiv")
                    plapiv<double>(h, 'F', 'R', 'C', n, n, a.data(), a.d, piv.data(), a.d, nullptr);
                else if (op == "ppotrf_bottom_right")
                    ppotrf_bottom_right<double>(h, 'L', n, a.data(), a.d, info);
                else throw std::invalid_argument("unknown operation");
            };
            if (!prewarm()) throw std::runtime_error("allocator preload failed to resolve ddlaMalloc");
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == target) arm(kind == "backend" ? 1 : 2);
            int caught = 0;
            try { call(); }
            catch (const std::exception& error) {
                caught = 1;
                // This line and the preload's fault event survive an expected
                // baseline hang before this test's following Allreduce.
                std::fprintf(stderr, "PBLAS_FAILURE_RETURN rank=%d fired=%u bytes=%zu error=%s\n",
                    rank, fired(), bytes(), error.what());
                std::fflush(stderr);
            }
            disarm();
            const unsigned observed = fired();
            const std::size_t observed_bytes = bytes();
            int local = caught && observed == unsigned(rank == target) &&
                (rank != target || observed_bytes > 0);
            // Verify both inputs, output, padded LLD, and prefix/suffix guards.
            local &= a.unchanged(h);
            local &= b.unchanged(h);
            local &= c.unchanged(h);
            local &= piv == original_piv && tau == original_tau;
            std::vector<int> actual_block_piv(piv.size());
            DDLA_CHECK(ddlaMemcpy(h, actual_block_piv.data(), block_piv.data(), piv.size() * sizeof(int)));
            DDLA_CHECK(ddlaSynchronize(h));
            local &= actual_block_piv == original_piv;
            int all = 0;
            MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (!all) throw std::runtime_error("fault did not fire once and preserve every rank's matrices");
            // An unarmed call must still work after all scratch has unwound.
            pgemm<double>(h, 'N', 'N', n, n, n, 1., a.data(), a.d, b.data(), b.d, 1., c.data(), c.d);
            c.read(h);
            int pr = 0, pc = 0, r = 0, col = 0;
            ddlaGetGridDims(h, pr, pc);
            ddlaGetGridCoords(h, r, col);
            for (int j = 0; j < ddla_test::n_loc(h, c.d); ++j)
                for (int i = 0; i < ddla_test::m_loc(h, c.d); ++i) {
                    const int gi = indxl2g(i, 4, r, 0, pr), gj = indxl2g(j, 4, col, 0, pc);
                    const double expected = gi < n && gj < n ? 2. * n + 5. : 5.;
                    if (c.actual[8 + i + std::size_t(j) * c.d[DDLA_LLD_]] != expected)
                        throw std::runtime_error("unarmed recovery GEMM returned the wrong result");
                }
        }
        DDLA_CHECK(ddlaDestroy(h));
        if (!rank) std::printf("PBLAS_ALLOC_FAILURE_PASS op=%s kind=%s n=%d grid=%dx%d target=%d\n",
            op.c_str(), kind.c_str(), n, rows, size / rows, target);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
