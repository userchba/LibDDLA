/**
 * @file test_backend_handle.cpp
 * @brief Test backend selection, handle lifecycle, and public accessors.
 *
 * Covers:
 *  - Default resolution to the compile-time default backend
 *  - Explicit available backend
 *  - Explicit unavailable backend (throws)
 *  - Backend query (ddlaGetBackend)
 *  - CPU stream returns nullptr (when CPU)
 *  - Two handles created from the same original MPI communicator
 *  - Cleanup in both orders
 *  - ddlaMalloc/ddlaFree/ddlaMemcpy/ddlaMemcpy2D (handle-first, no direction
 *    argument: a GPU handle derives host/device from the pointers), including
 *    host-to-host, pitched 2-D copies, interior, pinned and managed pointers,
 *    on every compiled backend, and the pointer classifier itself
 *  - All operations must be collective
 */

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <mpi.h>
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "ddla_stream.h"

using namespace ddla;

static_assert(std::is_same_v<decltype(DdlaStream::stream), void*> &&
              std::is_same_v<decltype(DdlaStream::stream_data), void*> &&
              std::is_same_v<decltype(DdlaStream::stream_level), void*> &&
              std::is_same_v<decltype(DdlaStream::blasH), void*> &&
              std::is_same_v<decltype(DdlaStream::solverH), void*> &&
              std::is_same_v<decltype(DdlaStream::nccl_comm), void*> &&
              std::is_same_v<decltype(DdlaStream::nccl_row_comm), void*> &&
              std::is_same_v<decltype(DdlaStream::nccl_col_comm), void*>);

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

// Pitched byte rectangles for the ddlaMemcpy2D checks.  Payload byte (r, c) of
// every rectangle is rect_byte(r, c), so a copy between rectangles of any
// pitch must reproduce the payload; every byte of the pitch padding holds `pad`
// and must still hold it afterwards.
static unsigned char rect_byte(std::size_t r, std::size_t c)
{
    return static_cast<unsigned char>((r * 31 + c * 7 + 1) & 0xFF);
}

static std::vector<unsigned char> make_rect(std::size_t pitch, std::size_t width,
                                            std::size_t height, unsigned char pad)
{
    std::vector<unsigned char> buf(pitch * height, pad);
    for (std::size_t r = 0; r < height; ++r)
        for (std::size_t c = 0; c < width; ++c)
            buf[r * pitch + c] = rect_byte(r, c);
    return buf;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myid == 0) std::cout << "=== Backend & Handle Test ===" << std::endl;

    // ---- 1. Default backend resolution --------------------------------
    {
        DdlaHandle_t h1 = nullptr;
        ddlaInit(h1);
        TEST("Default resolves to compile-time default backend",
             ddlaGetBackend(h1) == default_backend_v);
        TEST("Resolved backend is available",
             ddlaBackendAvailable(ddlaGetBackend(h1)));
        ddlaDestroy(h1);
    }

    // ---- 2. Explicit available backend ---------------------------------
    {
        // Determine which backends are available
        bool cpu_avail = ddlaBackendAvailable(DdlaBackend::CPU);
        bool gpu_avail = ddlaBackendAvailable(DdlaBackend::GPU);
        TEST("At least one backend available", cpu_avail || gpu_avail);

        DdlaBackend explicit_be = cpu_avail ? DdlaBackend::CPU : DdlaBackend::GPU;
        DdlaHandle_t h2 = nullptr;
        ddlaInit(h2, explicit_be);
        TEST("Explicit available backend matches request",
             ddlaGetBackend(h2) == explicit_be);
        ddlaDestroy(h2);
    }

    // ---- 3. Explicit unavailable backend (must throw) ------------------
    {
        DdlaBackend unavailable = ddlaBackendAvailable(DdlaBackend::CPU)
                                      ? DdlaBackend::GPU
                                      : DdlaBackend::CPU;
        if (!ddlaBackendAvailable(unavailable)) {
            bool threw = false;
            DdlaHandle_t h3 = nullptr;
            try {
                ddlaInit(h3, unavailable);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            TEST("Unavailable backend throws std::runtime_error", threw);
            // handle should remain nullptr after failed init
            TEST("Handle is nullptr after failed init", h3 == nullptr);
        } else {
            if (myid == 0)
                std::cout << "  SKIP: both backends available (unavailable test "
                             "not applicable)"
                          << std::endl;
        }
    }

    // ---- 4. Backend query on null handle -------------------------------
    {
        DdlaHandle_t null_h = nullptr;
        bool threw = false;
        try {
            ddlaGetBackend(null_h);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        TEST("ddlaGetBackend(nullptr) throws std::runtime_error", threw);
        TEST("ddlaGetStream(nullptr) returns nullptr",
             ddlaGetStream(null_h) == nullptr);
    }

    // ---- 5. CPU stream = nullptr (when CPU backend is available) -------
    if (ddlaBackendAvailable(DdlaBackend::CPU)) {
        DdlaHandle_t h_cpu = nullptr;
        ddlaInit(h_cpu, DdlaBackend::CPU);
        ddlaSet(h_cpu, MPI_COMM_WORLD, 'R');
        void* s = ddlaGetStream(h_cpu);
#if DDLA_HAS_CPU
        TEST("CPU handle stream is nullptr", s == nullptr);
        TEST("CPU handle has no native resources",
             h_cpu->stream == nullptr && h_cpu->stream_data == nullptr &&
             h_cpu->stream_level == nullptr && h_cpu->blasH == nullptr && h_cpu->solverH == nullptr &&
             h_cpu->nccl_comm == nullptr && h_cpu->nccl_row_comm == nullptr && h_cpu->nccl_col_comm == nullptr);
        TEST("CPU level queue stays null", h_cpu->level_queue() == nullptr);
#else
        // In a GPU-only build, CPU backend shouldn't be available
        (void)s;
#endif
        ddlaDestroy(h_cpu);
    } else {
        if (myid == 0)
            std::cout << "  SKIP: CPU backend not available" << std::endl;
    }

    // ---- 6. Public accessors after ddlaSet ----------------------------
    {
        DdlaHandle_t h = nullptr;
        ddlaInit(h);
        ddlaSet(h, MPI_COMM_WORLD, 'R');

        int rank = ddlaGetRank(h);
        int size = ddlaGetSize(h);
        TEST("ddlaGetRank in range", rank >= 0 && rank < size);
        TEST("ddlaGetSize matches nprocs", size == nprocs);

        MPI_Comm comm = ddlaGetCommunicator(h);
        TEST("ddlaGetCommunicator != MPI_COMM_NULL", comm != MPI_COMM_NULL);
        int comm_rank, comm_size;
        MPI_Comm_rank(comm, &comm_rank);
        MPI_Comm_size(comm, &comm_size);
        TEST("communicator rank matches", comm_rank == rank);
        TEST("communicator size matches", comm_size == size);

        int myprow = -1, mypcol = -1;
        ddlaGetGridCoords(h, myprow, mypcol);
        TEST("grid coords in range",
             myprow >= 0 && mypcol >= 0);

        int nprows = 0, npcols = 0;
        ddlaGetGridDims(h, nprows, npcols);
        TEST("grid dims product matches nprocs", nprows * npcols == nprocs);

        // rank_to_rc / rc_to_rank roundtrip
        for (int r = 0; r < nprocs; ++r) {
            int rr, rc;
            ddlaRankToRc(h, r, rr, rc);
            int back = ddlaRcToRank(h, rr, rc);
            TEST("rank_to_rc/rc_to_rank roundtrip", back == r);
        }

        ddlaDestroy(h);
    }

    // ---- 7. Two handles from the same original communicator ------------
    {
        DdlaHandle_t ha = nullptr, hb = nullptr;
        ddlaInit(ha);
        ddlaInit(hb);
        ddlaSet(ha, MPI_COMM_WORLD, 'R');
        ddlaSet(hb, MPI_COMM_WORLD, 'R');

        TEST("Both handles initialized", ha != nullptr && hb != nullptr);
        TEST("Both handles have same backend",
             ddlaGetBackend(ha) == ddlaGetBackend(hb));

        MPI_Comm ca = ddlaGetCommunicator(ha);
        MPI_Comm cb = ddlaGetCommunicator(hb);
        int cmp_result = MPI_UNEQUAL;
        MPI_Comm_compare(ca, cb, &cmp_result);
        TEST("Duplicated communicators are congruent (not identical)",
             cmp_result == MPI_CONGRUENT);

        int ra, rb;
        MPI_Comm_rank(ca, &ra);
        MPI_Comm_rank(cb, &rb);
        TEST("Same rank in both communicators", ra == rb);

        // Clean up in reverse order
        ddlaDestroy(hb);
        ddlaDestroy(ha);
    }

    // ---- 8. Cleanup in both orders --------------------------------------
    {
        DdlaHandle_t hx = nullptr, hy = nullptr;
        ddlaInit(hx);
        ddlaInit(hy);
        ddlaSet(hx, MPI_COMM_WORLD, 'R');
        ddlaSet(hy, MPI_COMM_WORLD, 'R');

        // Destroy in same order as creation
        ddlaDestroy(hx);
        ddlaDestroy(hy);
        TEST("Same-order destroy: handles null", hx == nullptr && hy == nullptr);
    }

    // ---- 9. Idempotent double-destroy -----------------------------------
    {
        DdlaHandle_t h = nullptr;
        ddlaInit(h);
        ddlaSet(h, MPI_COMM_WORLD, 'R');
        ddlaDestroy(h);
        ddlaDestroy(h);  // should be safe
        TEST("Double destroy is safe", h == nullptr);
    }

    // ---- 9b. Repeated ddlaSet (reinit lifecycle) -----------------------
    {
        DdlaHandle_t h = nullptr;
        ddlaInit(h);
        ddlaSet(h, MPI_COMM_WORLD, 'R');
        int rank1 = ddlaGetRank(h);
        int size1 = ddlaGetSize(h);
        MPI_Comm comm1 = ddlaGetCommunicator(h);
        MPI_Comm comm1_copy = MPI_COMM_NULL;
        MPI_Comm_dup(comm1, &comm1_copy);

        // Second ddlaSet on the same handle — must clean and reinitialize
        ddlaSet(h, MPI_COMM_WORLD, 'R');
        int rank2 = ddlaGetRank(h);
        int size2 = ddlaGetSize(h);
        MPI_Comm comm2 = ddlaGetCommunicator(h);

        TEST("Repeated ddlaSet preserves rank", rank1 == rank2);
        TEST("Repeated ddlaSet preserves size", size1 == size2);
        TEST("Repeated ddlaSet creates new communicator", comm2 != MPI_COMM_NULL);

        int cmp_result = MPI_UNEQUAL;
        MPI_Comm_compare(comm1_copy, comm2, &cmp_result);
        TEST("Repeated ddlaSet communicators are congruent",
             cmp_result == MPI_CONGRUENT);

        MPI_Comm_free(&comm1_copy);
        ddlaDestroy(h);
    }

    // ---- 10. Memory helpers smoke test ----------------------------------
    {
        DdlaHandle_t h = nullptr;
        ddlaInit(h);
        ddlaSet(h, MPI_COMM_WORLD, 'R');

        void* ptr = nullptr;
        ddlaStatus_t rc = ddlaMalloc(h, &ptr, 1024);
        TEST("ddlaMalloc succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS && ptr != nullptr);

        // Create host pattern buffer and fill
        std::vector<char> host_src(1024, 0);
        for (size_t i = 0; i < 1024; ++i)
            host_src[i] = static_cast<char>(i & 0xFF);

        // Copy host -> device
        rc = ddlaMemcpy(h, ptr, host_src.data(), 1024);
        TEST("ddlaMemcpy H2D succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);
        rc = ddlaSynchronize(h);
        TEST("ddlaSynchronize after H2D succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);

        // Allocate host buffer for readback
        std::vector<char> host(1024, 0);

        // Copy device -> host
        rc = ddlaMemcpy(h, host.data(), ptr, 1024);
        TEST("ddlaMemcpy D2H succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);

        rc = ddlaSynchronize(h);
        TEST("ddlaSynchronize after D2H succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);

        // Verify pattern
        bool pattern_ok = true;
        for (size_t i = 0; i < 1024; ++i) {
            if (static_cast<unsigned char>(host[i]) !=
                static_cast<unsigned char>(i & 0xFF)) {
                pattern_ok = false;
                break;
            }
        }
        TEST("Memory pattern preserved", pattern_ok);

        rc = ddlaFree(h, ptr);
        TEST("ddlaFree succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);

        ddlaDestroy(h);
    }

    // ---- 11. Boundary tests for every compiled/available backend ---------
    // Test zero-byte allocation, zero-byte copy and invalid args.
    {
        bool cpu_avail = ddlaBackendAvailable(DdlaBackend::CPU);
        bool gpu_avail = ddlaBackendAvailable(DdlaBackend::GPU);
        DdlaBackend backends_to_test[2];
        int nb = 0;
        if (cpu_avail) backends_to_test[nb++] = DdlaBackend::CPU;
        if (gpu_avail) backends_to_test[nb++] = DdlaBackend::GPU;

        for (int bi = 0; bi < nb; ++bi) {
            DdlaBackend be = backends_to_test[bi];
            const char* be_name = (be == DdlaBackend::CPU) ? "CPU" : "GPU";

            DdlaHandle_t h = nullptr;
            ddlaInit(h, be);
            ddlaSet(h, MPI_COMM_WORLD, 'R');

            TEST(std::string(be_name) + " native resource ownership",
                 be == DdlaBackend::CPU ?
                    h->stream == nullptr && h->stream_data == nullptr && h->blasH == nullptr && h->solverH == nullptr :
                    h->stream != nullptr && h->stream_data != nullptr && h->blasH != nullptr && h->solverH != nullptr);
            TEST(std::string(be_name) + " level queue starts lazy", h->stream_level == nullptr);
#ifdef DDLA_USE_CCL
            TEST(std::string(be_name) + " CCL resource ownership",
                 be == DdlaBackend::CPU ?
                    h->nccl_comm == nullptr && h->nccl_row_comm == nullptr && h->nccl_col_comm == nullptr :
                    h->nccl_comm != nullptr && h->nccl_row_comm != nullptr && h->nccl_col_comm != nullptr);
#else
            TEST(std::string(be_name) + " CCL resources disabled",
                 h->nccl_comm == nullptr && h->nccl_row_comm == nullptr && h->nccl_col_comm == nullptr);
#endif

            // 11a. Zero-byte allocation succeeds, returns nullptr
            {
                void* zp = reinterpret_cast<void*>(0x1); // non-null sentinel
                ddlaStatus_t rc = ddlaMalloc(h, &zp, 0);
                TEST(std::string(be_name) + " zero-byte malloc succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);
                TEST(std::string(be_name) + " zero-byte malloc returns nullptr",
                     zp == nullptr);
            }

            // 11b. Zero-byte copy with null pointers succeeds
            {
                ddlaStatus_t rc = ddlaMemcpy(h, nullptr, nullptr, 0);
                TEST(std::string(be_name) + " zero-byte memcpy(null,null) succeeds",
                     rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);
            }

            // 11c. Nonzero copy with null source fails
            {
                char buf[1] = {0};
                ddlaStatus_t rc = ddlaMemcpy(h, buf, nullptr, 1);
                TEST(std::string(be_name) + " nonzero memcpy null src fails",
                     rc != ddlaStatus_t::DDLA_STATUS_SUCCESS);
            }

            // 11d. Nonzero copy with null destination fails
            {
                char buf[1] = {0};
                ddlaStatus_t rc = ddlaMemcpy(h, nullptr, buf, 1);
                TEST(std::string(be_name) + " nonzero memcpy null dst fails",
                     rc != ddlaStatus_t::DDLA_STATUS_SUCCESS);
            }

            // Cleanup: free nullptr (valid no-op)
            {
                ddlaStatus_t rc = ddlaFree(h, nullptr);
                TEST(std::string(be_name) + " free nullptr succeeds", rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);
            }

            ddlaDestroy(h);
        }
    }

    // ---- 12. ddlaMemcpy / ddlaMemcpy2D on every backend -------------------
    // There is no direction argument: a CPU handle copies host to host, and a
    // GPU handle asks the runtime where each pointer lives.  ddlaMalloc gives
    // host memory on a CPU handle and device memory on a GPU handle, so the
    // same code path runs for both backends.
    {
        using Status = ddlaStatus_t;
        const Status ok = Status::DDLA_STATUS_SUCCESS;
        const Status bad_value = Status::DDLA_STATUS_INVALID_VALUE;

        DdlaBackend backends_to_test[2];
        int nb = 0;
        if (ddlaBackendAvailable(DdlaBackend::CPU)) backends_to_test[nb++] = DdlaBackend::CPU;
        if (ddlaBackendAvailable(DdlaBackend::GPU)) backends_to_test[nb++] = DdlaBackend::GPU;

        for (int bi = 0; bi < nb; ++bi) {
            const DdlaBackend be = backends_to_test[bi];
            const bool gpu = (be == DdlaBackend::GPU);
            const std::string nm = std::string(gpu ? "GPU" : "CPU") + " ";

            DdlaHandle_t h = nullptr;
            ddlaInit(h, be);
            ddlaSet(h, MPI_COMM_WORLD, 'R');

            // 12a. host -> host between two std::vector buffers (a plain memcpy on both handle kinds)
            {
                std::vector<unsigned char> src(257), dst(257, 0xEE);
                for (std::size_t i = 0; i < src.size(); ++i)
                    src[i] = static_cast<unsigned char>(i * 13 + 5);
                Status rc = ddlaMemcpy(h, dst.data(), src.data(), src.size());
                TEST(nm + "memcpy host->host succeeds", rc == ok);
                TEST(nm + "memcpy host->host copies every byte", dst == src);
            }

            // 12b. 1-D host -> device -> device -> host round trip through ddlaMalloc buffers
            {
                const std::size_t n = 300;
                std::vector<unsigned char> src(n), back(n, 0xEE);
                for (std::size_t i = 0; i < n; ++i)
                    src[i] = static_cast<unsigned char>(i * 7 + 3);
                void* d1 = nullptr;
                void* d2 = nullptr;
                bool good = ddlaMalloc(h, &d1, n) == ok && ddlaMalloc(h, &d2, n) == ok;
                good = good && ddlaMemcpy(h, d1, src.data(), n) == ok;
                good = good && ddlaMemcpy(h, d2, d1, n) == ok;
                good = good && ddlaMemcpy(h, back.data(), d2, n) == ok;
                good = good && ddlaSynchronize(h) == ok;
                TEST(nm + "memcpy host->device->device->host calls succeed", good);
                TEST(nm + "memcpy host->device->device->host preserves bytes", back == src);
                TEST(nm + "free after memcpy round trip",
                     ddlaFree(h, d1) == ok && ddlaFree(h, d2) == ok);
            }

            // 12c. interior pointers: a sub-range copied out of / into the middle of device blocks
            {
                const std::size_t n = 256, off = 96, len = 100;
                std::vector<unsigned char> src(n), fill(n, 0x11), back(n, 0);
                for (std::size_t i = 0; i < n; ++i)
                    src[i] = static_cast<unsigned char>(i * 5 + 2);
                unsigned char* d1 = nullptr;
                unsigned char* d2 = nullptr;
                bool good = ddlaMalloc(h, reinterpret_cast<void**>(&d1), n) == ok
                         && ddlaMalloc(h, reinterpret_cast<void**>(&d2), n) == ok;
                good = good && ddlaMemcpy(h, d1, src.data(), n) == ok;
                good = good && ddlaMemcpy(h, d2, fill.data(), n) == ok;
                good = good && ddlaMemcpy(h, d2 + off, d1 + off, len) == ok;          // device interior -> device interior
                good = good && ddlaMemcpy(h, back.data(), d2, n) == ok;
                good = good && ddlaSynchronize(h) == ok;
                std::vector<unsigned char> expect = fill;
                for (std::size_t i = off; i < off + len; ++i) expect[i] = src[i];
                TEST(nm + "memcpy interior pointers: calls succeed", good);
                TEST(nm + "memcpy interior pointers: only the sub-range changes", back == expect);
                TEST(nm + "free after interior-pointer memcpy",
                     ddlaFree(h, d1) == ok && ddlaFree(h, d2) == ok);
            }

#if DDLA_GPU_VENDOR
            // 12d. pinned host memory is host memory: host(pinned) -> device -> host(pinned)
            {
                const std::size_t n = 200;
                void* p1 = nullptr;
                void* p2 = nullptr;
                GPU_CHECK(gpuHostAlloc(&p1, n, gpuHostAllocDefault));
                GPU_CHECK(gpuHostAlloc(&p2, n, gpuHostAllocDefault));
                unsigned char* a = static_cast<unsigned char*>(p1);
                unsigned char* b = static_cast<unsigned char*>(p2);
                for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<unsigned char>(i * 3 + 1); b[i] = 0xEE; }
                void* d = nullptr;
                bool good = ddlaMalloc(h, &d, n) == ok;
                good = good && ddlaMemcpy(h, d, a, n) == ok;
                good = good && ddlaMemcpy(h, b, d, n) == ok;
                good = good && ddlaSynchronize(h) == ok;
                bool same = true;
                for (std::size_t i = 0; i < n; ++i) same = same && a[i] == b[i];
                TEST(nm + "memcpy pinned host buffers: calls succeed", good);
                TEST(nm + "memcpy pinned host buffers: bytes preserved", same);
                good = ddlaFree(h, d) == ok;
                GPU_CHECK(gpuFreeHost(p1));
                GPU_CHECK(gpuFreeHost(p2));
                TEST(nm + "free after pinned memcpy", good);
            }

            // 12e. managed memory counts as device memory (reachable from both sides)
            if (gpu) {
                const std::size_t n = 220;
                std::vector<unsigned char> src(n), back(n, 0xEE);
                for (std::size_t i = 0; i < n; ++i)
                    src[i] = static_cast<unsigned char>(i * 9 + 4);
                void* m1 = nullptr;
                void* m2 = nullptr;
                GPU_CHECK(gpuMallocManaged(&m1, n));
                GPU_CHECK(gpuMallocManaged(&m2, n));
                bool good = ddlaMemcpy(h, m1, src.data(), n) == ok;                  // host -> managed
                good = good && ddlaMemcpy(h, m2, m1, n) == ok;                       // managed -> managed
                good = good && ddlaMemcpy(h, back.data(), m2, n) == ok;              // managed -> host
                good = good && ddlaSynchronize(h) == ok;
                TEST(nm + "memcpy managed memory: calls succeed", good);
                TEST(nm + "memcpy managed memory: bytes preserved", back == src);
                GPU_CHECK(gpuFree(m1));
                GPU_CHECK(gpuFree(m2));
            }

            // 12f. the pointer classifier itself
            if (gpu) {
                std::vector<unsigned char> v(64);
                unsigned char on_stack[8] = {0};
                void* d = nullptr;
                void* pin = nullptr;
                void* man = nullptr;
                bool good = ddlaMalloc(h, &d, 4096) == ok;
                GPU_CHECK(gpuHostAlloc(&pin, 64, gpuHostAllocDefault));
                GPU_CHECK(gpuMallocManaged(&man, 64));
                TEST(nm + "classifier: std::vector is Host", gpuPointerMemType(v.data()) == GpuMemType::Host);
                TEST(nm + "classifier: stack array is Host", gpuPointerMemType(on_stack) == GpuMemType::Host);
                TEST(nm + "classifier: pinned host memory is Host", gpuPointerMemType(pin) == GpuMemType::Host);
                TEST(nm + "classifier: ddlaMalloc block is Device", good && gpuPointerMemType(d) == GpuMemType::Device);
                TEST(nm + "classifier: interior pointer of a ddlaMalloc block is Device",
                     good && gpuPointerMemType(static_cast<unsigned char*>(d) + 1000) == GpuMemType::Device);
                TEST(nm + "classifier: managed memory is Device", gpuPointerMemType(man) == GpuMemType::Device);
                TEST(nm + "classifier leaves no sticky runtime error", ddlaGetLastError(h) == ddlaStatus_t::DDLA_STATUS_SUCCESS);
                good = ddlaFree(h, d) == ok;
                GPU_CHECK(gpuFreeHost(pin));
                GPU_CHECK(gpuFree(man));
                TEST(nm + "free after classifier checks", good);
            }
#endif

            // 12g. 2-D chain with a different pitch at every hop.  The padding bytes
            // of each destination start as a sentinel and must survive.
            {
                const std::size_t width = 7, height = 5;
                const std::size_t p_src = 11, p_d1 = 13, p_d2 = 9, p_dst = 8;
                const unsigned char pad_src = 0xA5, pad_d1 = 0x5A, pad_d2 = 0x3C, pad_dst = 0xC3;

                const std::vector<unsigned char> h_src = make_rect(p_src, width, height, pad_src);
                const std::vector<unsigned char> fill_d1(p_d1 * height, pad_d1);
                const std::vector<unsigned char> fill_d2(p_d2 * height, pad_d2);
                std::vector<unsigned char> h_d1(p_d1 * height, 0), h_d2(p_d2 * height, 0);
                std::vector<unsigned char> h_dst(p_dst * height, pad_dst);

                void* d1 = nullptr;
                void* d2 = nullptr;
                bool good = ddlaMalloc(h, &d1, fill_d1.size()) == ok
                         && ddlaMalloc(h, &d2, fill_d2.size()) == ok;
                good = good && ddlaMemcpy(h, d1, fill_d1.data(), fill_d1.size()) == ok;
                good = good && ddlaMemcpy(h, d2, fill_d2.data(), fill_d2.size()) == ok;
                good = good && ddlaMemcpy2D(h, d1, p_d1, h_src.data(), p_src, width, height) == ok;   // host -> device
                good = good && ddlaMemcpy2D(h, d2, p_d2, d1, p_d1, width, height) == ok;             // device -> device
                good = good && ddlaMemcpy2D(h, h_dst.data(), p_dst, d2, p_d2, width, height) == ok;  // device -> host
                good = good && ddlaMemcpy(h, h_d1.data(), d1, h_d1.size()) == ok;
                good = good && ddlaMemcpy(h, h_d2.data(), d2, h_d2.size()) == ok;
                good = good && ddlaSynchronize(h) == ok;
                TEST(nm + "memcpy2D host->device->device->host calls succeed", good);
                TEST(nm + "memcpy2D host->device fills payload, keeps padding",
                     h_d1 == make_rect(p_d1, width, height, pad_d1));
                TEST(nm + "memcpy2D device->device fills payload, keeps padding",
                     h_d2 == make_rect(p_d2, width, height, pad_d2));
                TEST(nm + "memcpy2D device->host fills payload, keeps padding",
                     h_dst == make_rect(p_dst, width, height, pad_dst));
                TEST(nm + "free after memcpy2D chain",
                     ddlaFree(h, d1) == ok && ddlaFree(h, d2) == ok);
            }

            // 12h. 2-D host -> host: pitched -> pitched, then the contiguous (pitch == width) case.
            {
                const std::size_t width = 7, height = 5;
                const std::vector<unsigned char> src = make_rect(11, width, height, 0xA5);
                std::vector<unsigned char> dst(10 * height, 0xC3);
                Status rc = ddlaMemcpy2D(h, dst.data(), 10, src.data(), 11, width, height);
                TEST(nm + "memcpy2D host->host succeeds", rc == ok);
                TEST(nm + "memcpy2D host->host fills payload, keeps padding",
                     dst == make_rect(10, width, height, 0xC3));

                const std::vector<unsigned char> csrc = make_rect(width, width, 6, 0);
                std::vector<unsigned char> cdst(width * 6, 0xEE);
                rc = ddlaMemcpy2D(h, cdst.data(), width, csrc.data(), width, width, 6);
                TEST(nm + "memcpy2D host->host contiguous succeeds", rc == ok);
                TEST(nm + "memcpy2D host->host contiguous copies all rows", cdst == csrc);
            }

            // 12i. Argument validation: exact status codes
            {
                unsigned char a[16] = {0}, b[16] = {0};

                TEST(nm + "memcpy zero bytes (null ptrs) succeeds",
                     ddlaMemcpy(h, nullptr, nullptr, 0) == ok);
                TEST(nm + "memcpy null src is INVALID_VALUE",
                     ddlaMemcpy(h, a, nullptr, 4) == bad_value);
                TEST(nm + "memcpy null dst is INVALID_VALUE",
                     ddlaMemcpy(h, nullptr, b, 4) == bad_value);
                TEST(nm + "memcpy2D zero width (null ptrs) succeeds",
                     ddlaMemcpy2D(h, nullptr, 8, nullptr, 8, 0, 4) == ok);
                TEST(nm + "memcpy2D zero height (null ptrs) succeeds",
                     ddlaMemcpy2D(h, nullptr, 8, nullptr, 8, 4, 0) == ok);
                TEST(nm + "memcpy2D null src is INVALID_VALUE",
                     ddlaMemcpy2D(h, a, 8, nullptr, 8, 4, 2) == bad_value);
                TEST(nm + "memcpy2D null dst is INVALID_VALUE",
                     ddlaMemcpy2D(h, nullptr, 8, b, 8, 4, 2) == bad_value);
                TEST(nm + "memcpy2D dpitch < width is INVALID_VALUE",
                     ddlaMemcpy2D(h, a, 3, b, 8, 4, 2) == bad_value);
                TEST(nm + "memcpy2D spitch < width is INVALID_VALUE",
                     ddlaMemcpy2D(h, a, 8, b, 3, 4, 2) == bad_value);
            }

            // 13a. ddlaMemset: device bytes on a GPU handle, host bytes on a CPU handle
            {
                const std::size_t n = 100;
                void* d = nullptr;
                std::vector<unsigned char> back(n, 0x11);
                bool good = ddlaMalloc(h, &d, n) == ok;
                good = good && ddlaMemset(h, d, 0xAB, n) == ok;
                good = good && ddlaMemcpy(h, back.data(), d, n) == ok;
                good = good && ddlaSynchronize(h) == ok;
                TEST(nm + "memset: calls succeed", good);
                TEST(nm + "memset: every byte is set", back == std::vector<unsigned char>(n, 0xAB));

                // a sub-range leaves the rest alone
                good = ddlaMemset(h, static_cast<unsigned char*>(d) + 10, 0x5C, 20) == ok;
                good = good && ddlaMemcpy(h, back.data(), d, n) == ok;
                good = good && ddlaSynchronize(h) == ok;
                std::vector<unsigned char> expect(n, 0xAB);
                for (std::size_t i = 10; i < 30; ++i) expect[i] = 0x5C;
                TEST(nm + "memset: sub-range only", good && back == expect);
                TEST(nm + "memset: zero bytes (null ptr) succeeds", ddlaMemset(h, nullptr, 0, 0) == ok);
                TEST(nm + "memset: null ptr is INVALID_VALUE", ddlaMemset(h, nullptr, 0, 4) == bad_value);
                TEST(nm + "free after memset", ddlaFree(h, d) == ok);
            }

            // 13b. pinned host memory through the public API, used as source and target of device copies
            {
                const std::size_t n = 200;
                void* p = nullptr;
                double* typed = nullptr;
                bool good = ddlaMallocHost(h, &p, n) == ok && p != nullptr;
                good = good && ddlaMallocHost(h, &typed, 8 * sizeof(double)) == ok && typed != nullptr;
                TEST(nm + "pinned: void* and typed allocation succeed", good);

                unsigned char* a = static_cast<unsigned char*>(p);
                for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<unsigned char>(i * 3 + 1);
                std::vector<unsigned char> expect(a, a + n);
                void* d = nullptr;
                good = ddlaMalloc(h, &d, n) == ok;
                good = good && ddlaMemcpy(h, d, p, n) == ok;                                   // pinned -> device
                good = good && ddlaMemset(h, p, 0, n) == ok;                                   // clobber the source after the copy was queued...
                good = good && ddlaSynchronize(h) == ok;                                       // ...and only then (stream-ordered copy done)
                good = good && ddlaMemcpy(h, p, d, n) == ok;                                   // device -> pinned
                good = good && ddlaSynchronize(h) == ok;
                TEST(nm + "pinned: round trip through a device buffer", good && std::vector<unsigned char>(a, a + n) == expect);
                TEST(nm + "pinned: free", ddlaFree(h, d) == ok && ddlaFreeHost(h, p) == ok && ddlaFreeHost(h, typed) == ok);

                int sentinel = 0;
                void* z = &sentinel;
                TEST(nm + "pinned: zero bytes sets nullptr and succeeds", ddlaMallocHost(h, &z, 0) == ok && z == nullptr);
                TEST(nm + "pinned: free of nullptr succeeds", ddlaFreeHost(h, nullptr) == ok);
                TEST(nm + "pinned: null output is INVALID_VALUE", ddlaMallocHost(h, static_cast<void**>(nullptr), 8) == bad_value);
            }

            // 13c. the handle's private level queue and the events that order it against the handle's own queue.  A GPU
            // handle has a real second vendor queue: work on it must not occupy the handle's own queue (and vice versa),
            // and an event orders the two.  A CPU handle has no queue: every call is a no-op.
            {
                const std::size_t n = gpu ? (std::size_t(128) << 20) : (std::size_t(1) << 20);
                void* const q = h->level_queue();
                ddlaEvent_t ev = nullptr, ev2 = nullptr;
                bool good = ddlaEventCreate(h, &ev) == ok && ddlaEventCreate(h, &ev2) == ok;
                TEST(nm + "event create succeeds", good);
                TEST(nm + "level queue: a GPU handle gets a private queue, a CPU handle none",
                     gpu == (q != nullptr) && gpu == (ev != nullptr) && (!gpu || q != static_cast<gpuStream_t>(h->stream)));

                void* d1 = nullptr;
                void* d2 = nullptr;
                good = ddlaMalloc(h, &d1, n) == ok && ddlaMalloc(h, &d2, n) == ok;
                good = good && ddlaMemset(h, d1, 0x00, n) == ok && ddlaMemset(h, d2, 0x00, n) == ok;
                good = good && ddlaSynchronize(h) == ok;
#if DDLA_GPU_VENDOR
                if (gpu) {
                    // a long copy on the private queue leaves the handle's own queue idle
                    good = good && detail::memcpy_on(h, d2, d1, n, q) == ok;
                    const bool own_queue_idle = gpuStreamQuery(static_cast<gpuStream_t>(h->stream)) == gpuSuccess;
                    good = good && detail::synchronize_on(h, q) == ok;
                    const bool q_drained = gpuStreamQuery(static_cast<gpuStream_t>(q)) == gpuSuccess;
                    TEST(nm + "private queue: a copy leaves the handle's own queue idle", good && own_queue_idle);
                    TEST(nm + "synchronize_on drains that queue", good && q_drained);
                    // a long memset on the handle's own queue leaves the private queue idle
                    good = ddlaMemset(h, d1, 0x22, n) == ok;
                    const bool q_idle = gpuStreamQuery(static_cast<gpuStream_t>(q)) == gpuSuccess;
                    good = good && ddlaDeviceSynchronize(h) == ok;
                    const bool device_drained = gpuStreamQuery(static_cast<gpuStream_t>(h->stream)) == gpuSuccess;
                    TEST(nm + "own queue: a memset leaves the private queue idle", good && q_idle);
                    TEST(nm + "deviceSynchronize drains the handle's own queue", good && device_drained);
                    // an event recorded on the private queue: the host wait covers the work before it
                    good = detail::memcpy_on(h, d2, d1, n, q) == ok && detail::event_record_on(h, ev, q) == ok;
                    good = good && ddlaEventSynchronize(h, ev) == ok;
                    const bool event_drained = gpuStreamQuery(static_cast<gpuStream_t>(q)) == gpuSuccess;
                    TEST(nm + "eventSynchronize waits for the work before the event", good && event_drained);
                    (void)ddlaGetLastError(h);
                }
#endif
                good = ddlaMemset(h, d1, 0x00, n) == ok && ddlaSynchronize(h) == ok;
                // the handle's own queue produces (16 passes over d1, the last one writes 0x11) and marks its end
                for (int pass = 1; pass <= 15; ++pass)
                    good = good && ddlaMemset(h, d1, pass, n) == ok;
                good = good && ddlaMemset(h, d1, 0x11, n) == ok;
                good = good && ddlaEventRecord(h, ev) == ok;
                good = good && detail::event_wait_on(h, ev, q) == ok;          // the private queue must not start before that
                good = good && detail::memcpy_on(h, d2, d1, n, q) == ok;
                good = good && detail::event_record_on(h, ev2, q) == ok;
                good = good && ddlaEventWait(h, ev2) == ok;                    // the handle's own queue waits for it (no host wait)
                TEST(nm + "event ordering: calls succeed", good);

                // sample the start, the middle and the end of d2, read on the handle's own queue
                const std::size_t probe = 4096;
                const std::size_t offs[3] = {0, n / 2, n - probe};
                bool all_set = true;
                for (int k = 0; k < 3; ++k) {
                    std::vector<unsigned char> back(probe, 0xEE);
                    good = ddlaMemcpy(h, back.data(), static_cast<unsigned char*>(d2) + offs[k], probe) == ok;
                    good = good && ddlaSynchronize(h) == ok;
                    all_set = all_set && good && back == std::vector<unsigned char>(probe, 0x11);
                }
                TEST(nm + "event ordering: the private queue's copy saw the finished memset", all_set);
                TEST(nm + "free after the event ordering", ddlaFree(h, d1) == ok && ddlaFree(h, d2) == ok
                     && ddlaSynchronize(h) == ok);
                TEST(nm + "event destroy", ddlaEventDestroy(h, ev) == ok && ddlaEventDestroy(h, ev2) == ok);
                TEST(nm + "event create: null output is INVALID_VALUE", ddlaEventCreate(h, nullptr) == bad_value);
            }

            // 13d. error query, device queries, status names
            {
                TEST(nm + "getLastError starts clean", ddlaGetLastError(h) == ok);
#if DDLA_GPU_VENDOR
                if (gpu) {
                    // an invalid vendor call leaves a sticky error; ddlaGetLastError reports it once and clears it
                    (void)gpuMemcpyAsync(nullptr, nullptr, 8, gpuMemcpyDeviceToDevice, static_cast<gpuStream_t>(ddlaGetStream(h)));
                    TEST(nm + "getLastError reports a failed vendor call", ddlaGetLastError(h) == Status::DDLA_STATUS_INTERNAL_ERROR);
                    TEST(nm + "getLastError clears the error", ddlaGetLastError(h) == ok);
                    const std::string msg = ddlaGetErrorString(Status::DDLA_STATUS_INTERNAL_ERROR);
                    TEST(nm + "error string carries the vendor message", msg.size() > std::string("DDLA_STATUS_INTERNAL_ERROR").size() + 2);
                }
#endif
                int count = 0;
                TEST(nm + "getDeviceCount", ddlaGetDeviceCount(h, &count) == ok && count >= 1);
                std::size_t fr = 7, tot = 7;
                TEST(nm + "memGetInfo", ddlaMemGetInfo(h, &fr, &tot) == ok && (gpu ? (tot > 0 && fr <= tot) : (fr == 0 && tot == 0)));
                TEST(nm + "setDevice", ddlaSetDevice(h) == ok);
                TEST(nm + "deviceSynchronize", ddlaDeviceSynchronize(h) == ok);
                TEST(nm + "queries: null outputs are INVALID_VALUE",
                     ddlaGetDeviceCount(h, nullptr) == bad_value && ddlaMemGetInfo(h, nullptr, &tot) == bad_value
                     && ddlaMemGetInfo(h, &fr, nullptr) == bad_value);
                TEST(nm + "status names", std::string(ddlaGetErrorString(ok)) == "DDLA_STATUS_SUCCESS"
                     && std::string(ddlaGetErrorString(bad_value)) == "DDLA_STATUS_INVALID_VALUE");
            }

            ddlaDestroy(h);
        }

        // 12j. Null handle: every memory entry point reports INVALID_HANDLE
        {
            DdlaHandle_t none = nullptr;
            unsigned char a[16] = {0}, b[16] = {0};
            void* p = nullptr;
            const Status bad_handle = Status::DDLA_STATUS_INVALID_HANDLE;
            TEST("null handle: ddlaMalloc is INVALID_HANDLE",
                 ddlaMalloc(none, &p, 8) == bad_handle);
            TEST("null handle: ddlaFree is INVALID_HANDLE",
                 ddlaFree(none, a) == bad_handle);
            TEST("null handle: ddlaMemcpy is INVALID_HANDLE",
                 ddlaMemcpy(none, a, b, 4) == bad_handle);
            TEST("null handle: ddlaMemcpy2D is INVALID_HANDLE",
                 ddlaMemcpy2D(none, a, 8, b, 8, 4, 2) == bad_handle);
            ddlaEvent_t ev = nullptr;
            int count = 0;
            std::size_t fr = 0, tot = 0;
            TEST("null handle: ddlaMemset / ddlaMallocHost / ddlaFreeHost are INVALID_HANDLE",
                 ddlaMemset(none, a, 0, 4) == bad_handle && ddlaMallocHost(none, &p, 8) == bad_handle
                 && ddlaFreeHost(none, a) == bad_handle);
            TEST("null handle: event calls are INVALID_HANDLE",
                 ddlaEventCreate(none, &ev) == bad_handle && ddlaEventDestroy(none, ev) == bad_handle
                 && ddlaEventRecord(none, ev) == bad_handle && ddlaEventWait(none, ev) == bad_handle
                 && ddlaEventSynchronize(none, ev) == bad_handle);
            TEST("null handle: the private-queue calls are INVALID_HANDLE",
                 detail::memcpy_on(none, a, b, 4, nullptr) == bad_handle && detail::synchronize_on(none, nullptr) == bad_handle
                 && detail::event_record_on(none, ev, nullptr) == bad_handle
                 && detail::event_wait_on(none, ev, nullptr) == bad_handle);
            TEST("null handle: synchronize, error and device queries are INVALID_HANDLE",
                 ddlaSynchronize(none) == bad_handle && ddlaDeviceSynchronize(none) == bad_handle
                 && ddlaGetLastError(none) == bad_handle && ddlaSetDevice(none) == bad_handle
                 && ddlaGetDeviceCount(none, &count) == bad_handle && ddlaMemGetInfo(none, &fr, &tot) == bad_handle);
        }
    }

    if (myid == 0) std::cout << "ALL BACKEND HANDLE TESTS PASSED" << std::endl;
    MPI_Finalize();
    return 0;
}
