#include "api_grid_test_common.h"
#include "ddla_connector.h"
#include <array>
#include <climits>
#include <complex>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ddla;
using namespace api_grid_test;

namespace {
struct Odd3 { unsigned char bytes[3]; };
struct Words3 { unsigned words[3]; };
int cases = 0;

void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool selected(char u, bool diagonal, int i, int j)
{
    return u == 'U' ? (diagonal ? i <= j : i < j) : (diagonal ? i >= j : i > j);
}

template <class T>
void reference(std::vector<unsigned char>& dst, int ld, std::size_t offset,
               const std::vector<unsigned char>& src, int lds, int m, int n,
               char u, bool diagonal, int fill = -1)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (selected(u, diagonal, i, j)) {
                auto* out = dst.data() + offset + (i + std::size_t(j) * ld) * sizeof(T);
                if (fill < 0)
                    std::memcpy(out, src.data() + offset + (i + std::size_t(j) * lds) * sizeof(T), sizeof(T));
                else
                    std::memset(out, fill, sizeof(T));
            }
}

template <class T>
void matrix_case(DdlaHandle_t h, int m, int n, char u, bool diagonal)
{
    const int lds = m + 3, ldd = m + 5;
    // Deliberately avoid stronger-than-T alignment (notably complex<float>).
    const std::size_t offset = 64 + alignof(T);
    std::vector<unsigned char> src(offset + std::size_t(lds) * (n + 2) * sizeof(T) + 64);
    std::vector<unsigned char> initial(offset + std::size_t(ldd) * (n + 2) * sizeof(T) + 64, 0xcd);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = (i * 37 + 11) % 251;
    // Explicit NaN payloads, negative zero and infinities must survive copies
    // unchanged, including both components of complex scalars.
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        auto* element = src.data() + offset + (i + std::size_t(j) * lds) * sizeof(T);
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
            const std::uint32_t patterns[] = {0x7fc12345u, 0x80000000u, 0x7f800000u, 0xff800000u};
            for (std::size_t c = 0; c < sizeof(T) / 4; ++c) {
                const auto word = patterns[(i + 2*j + c) % 4];
                std::memcpy(element + c*4, &word, 4);
            }
        } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
            const std::uint64_t patterns[] = {0x7ff8123456789abcULL, 0x8000000000000000ULL,
                                              0x7ff0000000000000ULL, 0xfff0000000000000ULL};
            for (std::size_t c = 0; c < sizeof(T) / 8; ++c) {
                const auto word = patterns[(i + 2*j + c) % 4];
                std::memcpy(element + c*8, &word, 8);
            }
        }
    }
    const auto saved = src;
    auto expected = initial;
    reference<T>(expected, ldd, offset, src, lds, m, n, u, diagonal);
    auto host = initial;
    auto* hp = reinterpret_cast<T*>(host.data() + offset);
    const auto* sp = reinterpret_cast<const T*>(src.data() + offset);
    DDLA_CHECK(ddlaMemcpyTri(h, u == 'U' ? 'u' : 'l', m, n, hp, ldd, sp, lds, diagonal));
    check(host == expected && src == saved, "host triangle copy/padding/source");
    DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, hp, ldd, hp, ldd, diagonal));
    check(host == expected, "identical-layout self-copy");
    if (ddlaGetBackend(h) == DdlaBackend::CPU) {
        for (int value : {0, 0x1a5}) {
            host = initial;
            expected = initial;
            reference<T>(expected, ldd, offset, src, lds, m, n, u, diagonal, value);
            DDLA_CHECK(ddlaMemsetTri(h, u, m, n, reinterpret_cast<T*>(host.data() + offset),
                                    ldd, value, diagonal));
            check(host == expected, "host triangle byte fill/padding");
        }
    } else {
        DeviceBuffer<unsigned char> a(h, src.size()), b(h, initial.size());
        upload(h, a.ptr, src);
        upload(h, b.ptr, initial);
        const auto* ap = reinterpret_cast<const T*>(a.ptr + offset);
        auto* bp = reinterpret_cast<T*>(b.ptr + offset);
        // Device-to-device, followed by a same-queue download, needs no
        // intermediate synchronization to order the copy kernel.
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, bp, ldd, ap, lds, diagonal));
        check(download(h, b.ptr, initial.size()) == expected, "device triangle copy/padding");
        check(download(h, a.ptr, src.size()) == src, "device source immutable");
        upload(h, b.ptr, initial);
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, bp, ldd, sp, lds, diagonal));
        check(download(h, b.ptr, initial.size()) == expected, "host-to-device triangular copy");
        host = initial;
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, reinterpret_cast<T*>(host.data() + offset),
                                ldd, ap, lds, diagonal));
        DDLA_CHECK(ddlaSynchronize(h));
        check(host == expected, "device-to-host triangular copy");
        for (int value : {0, 0x1a5}) {
            expected = initial;
            reference<T>(expected, ldd, offset, src, lds, m, n, u, diagonal, value);
            upload(h, b.ptr, initial);
            DDLA_CHECK(ddlaMemsetTri(h, u, m, n, bp, ldd, value, diagonal));
            check(download(h, b.ptr, initial.size()) == expected, "device triangle byte fill/padding");
        }
    }
    ++cases;
}

void errors(DdlaHandle_t h)
{
    const auto ok = ddlaStatus_t::DDLA_STATUS_SUCCESS;
    const auto bad = ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;
    double* p = nullptr;
    check(ddlaMemsetTri<double>(nullptr, 'U', 0, 0, p, 0) == ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE, "null handle");
    check(ddlaMemcpyTri<double>(nullptr, 'L', 0, 0, p, 0, p, 0) == ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE, "null copy handle");
    check(ddlaMemsetTri(h, 'X', 0, 0, p, 0) == bad, "invalid uplo");
    check(ddlaMemsetTri(h, 'U', -1, 0, p, 0) == bad, "negative extent");
    check(ddlaMemcpyTri(h, 'L', 0, -1, p, 0, p, 0) == bad, "negative copy extent");
    check(ddlaMemsetTri(h, 'U', 0, 9, p, -1) == ok, "zero rows");
    check(ddlaMemcpyTri(h, 'L', 9, 0, p, -1, p, -1) == ok, "zero columns");
    check(ddlaMemsetTri(h, 'U', 9, 1, p, -1, 0, false) == ok, "empty strict upper");
    check(ddlaMemcpyTri(h, 'L', 1, 9, p, -1, p, -1, false) == ok, "empty strict lower");
    check(ddlaMemsetTri(h, 'U', 1, 1, p, 1) == bad, "nonnull fill selection requires pointer");
    double x = 0;
    check(ddlaMemsetTri(h, 'U', 2, 2, &x, 1) == bad, "short destination LD");
    check(ddlaMemcpyTri(h, 'L', 2, 2, &x, 2, &x, 1) == bad, "short source LD");
    check(ddlaMemsetTri(h, 'U', INT_MAX, INT_MAX, &x, INT_MAX) == bad, "byte span overflow");
    check(ddlaMemcpyTri(h, 'L', INT_MAX, INT_MAX, &x, INT_MAX, &x, INT_MAX) == bad, "copy span overflow");
}

template <class T>
void scalar_cases(DdlaHandle_t h)
{
    for (auto shape : {std::array<int, 2>{1, 1}, {5, 9}, {9, 5}, {33, 65}, {257, 129}})
        for (char u : {'U', 'L'})
            for (bool diagonal : {false, true})
                matrix_case<T>(h, shape[0], shape[1], u, diagonal);
}

// Exercise both capped-grid fallback directions without issuing hundreds
// of thousands of individual host/device triangular column transfers.
void grid_stride_cases(DdlaHandle_t h)
{
    for (auto shape : {std::array<int, 2>{2097121, 1}, {1, 524289}}) {
        const int m = shape[0], n = shape[1], ld = m + 3;
        const char u = m == 1 ? 'U' : 'L';
        std::vector<unsigned char> input(std::size_t(ld) * n, 1), initial(input.size(), 3);
        DeviceBuffer<unsigned char> src(h, input.size()), dst(h, input.size());
        upload(h, src.ptr, input);
        upload(h, dst.ptr, initial);
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, dst.ptr, ld, src.ptr, ld));
        auto got = download(h, dst.ptr, initial.size());
        for (int j = 0; j < n; ++j) for (int i = 0; i < ld; ++i)
            check(got[i + std::size_t(j)*ld] == (i < m ? 1 : 3), "grid stride copy/padding");
        DDLA_CHECK(ddlaMemsetTri(h, u, m, n, dst.ptr, ld));
        got = download(h, dst.ptr, initial.size());
        for (int j = 0; j < n; ++j) for (int i = 0; i < ld; ++i)
            check(got[i + std::size_t(j)*ld] == (i < m ? 0 : 3), "grid stride fill/padding");
        ++cases;
    }
}

#if DDLA_HAS_GPU
void pinned_and_managed(DdlaHandle_t h)
{
    constexpr int m = 13, n = 17, ld = 16;
    constexpr std::size_t bytes = std::size_t(ld) * n * sizeof(double);
    std::vector<unsigned char> pattern(bytes), expected(bytes, 0xcd);
    for (std::size_t i = 0; i < bytes; ++i) pattern[i] = (i * 13 + 7) % 251;
    reference<double>(expected, ld, 0, pattern, ld, m, n, 'L', false);
    void *ps = nullptr, *pd = nullptr, *ms = nullptr, *md = nullptr;
    DDLA_CHECK(ddlaMallocHost(h, &ps, bytes));
    DDLA_CHECK(ddlaMallocHost(h, &pd, bytes));
    GPU_CHECK(gpuMallocManaged(&ms, bytes));
    GPU_CHECK(gpuMallocManaged(&md, bytes));
    std::memcpy(ps, pattern.data(), bytes);
    std::memset(pd, 0xcd, bytes);
    std::memcpy(ms, pattern.data(), bytes);
    std::memset(md, 0xcd, bytes);
    DDLA_CHECK(ddlaMemcpyTri(h, 'L', m, n, static_cast<double*>(pd), ld,
                            static_cast<const double*>(ps), ld, false));
    check(std::memcmp(pd, expected.data(), bytes) == 0, "pinned immediate host copy");
    DDLA_CHECK(ddlaMemcpyTri(h, 'L', m, n, static_cast<double*>(md), ld,
                            static_cast<const double*>(ps), ld, false));
    DDLA_CHECK(ddlaSynchronize(h));
    check(std::memcmp(md, expected.data(), bytes) == 0, "pinned-to-managed copy");
    std::memset(pd, 0xcd, bytes);
    DDLA_CHECK(ddlaMemcpyTri(h, 'L', m, n, static_cast<double*>(pd), ld,
                            static_cast<const double*>(ms), ld, false));
    DDLA_CHECK(ddlaSynchronize(h));
    check(std::memcmp(pd, expected.data(), bytes) == 0, "managed-to-pinned copy");
    std::memset(md, 0xcd, bytes);
    DDLA_CHECK(ddlaMemcpyTri(h, 'L', m, n, static_cast<double*>(md), ld,
                            static_cast<const double*>(ms), ld, false));
    DDLA_CHECK(ddlaMemsetTri(h, 'U', m, n, static_cast<double*>(md), ld, 0, false));
    DDLA_CHECK(ddlaSynchronize(h));
    reference<double>(expected, ld, 0, pattern, ld, m, n, 'U', false, 0);
    check(std::memcmp(md, expected.data(), bytes) == 0, "managed queued copy/fill");
    GPU_CHECK(gpuFree(ms));
    GPU_CHECK(gpuFree(md));
    DDLA_CHECK(ddlaFreeHost(h, ps));
    DDLA_CHECK(ddlaFreeHost(h, pd));
}

// No initialization of ignored source/destination entries: initcheck must
// reject a kernel that reads the opposite triangle or reads old fill values.
template <class T>
void uninitialized_case(DdlaHandle_t h)
{
    constexpr int m = 37, n = 65, ld = 41;
    const std::size_t bytes = std::size_t(ld) * n * sizeof(T);
    std::vector<unsigned char> input(bytes), result(bytes, 0xcd), expected = result;
    for (std::size_t i = 0; i < bytes; ++i) input[i] = (i * 29 + 5) % 251;
    DeviceBuffer<unsigned char> dst(h, bytes);
    for (char u : {'U', 'L'}) for (bool diagonal : {false, true}) {
        // Reallocate each time so the ignored part does not inherit a prior
        // initialized triangle in the sanitizer's allocation state.
        DeviceBuffer<unsigned char> a(h, bytes), b(h, bytes);
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, reinterpret_cast<T*>(a.ptr), ld,
                                reinterpret_cast<const T*>(input.data()), ld, diagonal));
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, reinterpret_cast<T*>(b.ptr), ld,
                                reinterpret_cast<const T*>(a.ptr), ld, diagonal));
        result.assign(bytes, 0xcd);
        expected = result;
        reference<T>(expected, ld, 0, input, ld, m, n, u, diagonal);
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, reinterpret_cast<T*>(result.data()), ld,
                                reinterpret_cast<const T*>(b.ptr), ld, diagonal));
        DDLA_CHECK(ddlaSynchronize(h));
        check(result == expected, "uninitialized opposite triangle copy");
        DDLA_CHECK(ddlaMemsetTri(h, u, m, n, reinterpret_cast<T*>(dst.ptr), ld, 0, diagonal));
        result.assign(bytes, 0xcd);
        expected = result;
        reference<T>(expected, ld, 0, input, ld, m, n, u, diagonal, 0);
        DDLA_CHECK(ddlaMemcpyTri(h, u, m, n, reinterpret_cast<T*>(result.data()), ld,
                                reinterpret_cast<const T*>(dst.ptr), ld, diagonal));
        DDLA_CHECK(ddlaSynchronize(h));
        check(result == expected, "fill does not read old destination");
    }
}
#endif

void run(DdlaBackend backend, bool sanitizer)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, backend));
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, size, 'R'));
    cases = 0;
#if DDLA_HAS_GPU
    void* original_stream = nullptr;
    gpuStream_t alternate = nullptr;
    if (backend == DdlaBackend::GPU) {
        DDLA_CHECK(ddlaSynchronize(h));
        original_stream = ddlaGetStream(h);
        GPU_CHECK(gpuStreamCreateWithFlags(&alternate, gpuStreamNonBlocking));
        DDLA_CHECK(ddlaSetStream(h, alternate));
    }
#endif
    errors(h);
    scalar_cases<float>(h);
    scalar_cases<double>(h);
    scalar_cases<std::complex<float>>(h);
    scalar_cases<std::complex<double>>(h);
    scalar_cases<std::uint16_t>(h);
    scalar_cases<Odd3>(h);
    scalar_cases<Words3>(h);
    matrix_case<double>(h, 2048, 1025, 'L', true);
    grid_stride_cases(h);
#if DDLA_HAS_GPU
    if (backend == DdlaBackend::GPU) {
        pinned_and_managed(h);
        if (sanitizer) {
            uninitialized_case<float>(h);
            uninitialized_case<double>(h);
            uninitialized_case<std::complex<float>>(h);
            uninitialized_case<std::complex<double>>(h);
            uninitialized_case<std::uint16_t>(h);
            uninitialized_case<Odd3>(h);
            uninitialized_case<Words3>(h);
        }
    }
#else
    (void)sanitizer;
#endif
    DDLA_CHECK(ddlaSynchronize(h));
#if DDLA_HAS_GPU
    if (backend == DdlaBackend::GPU) {
        // Restore ownership of the original stream before releasing the
        // temporary nonblocking stream used for the queue-ordering checks.
        DDLA_CHECK(ddlaSetStream(h, original_stream));
        GPU_CHECK(gpuStreamDestroy(alternate));
    }
#endif
    if (!ddlaGetRank(h)) std::cout << "MEMORY_TRI_PASS backend=" << int(backend)
                                  << " cases=" << cases << " sanitizer=" << sanitizer << '\n';
    DDLA_CHECK(ddlaDestroy(h));
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        bool sanitizer = false;
        if (argc == 2 && std::string(argv[1]) == "--sanitizer") sanitizer = true;
        else if (argc != 1) throw std::runtime_error("usage: test_memory_tri [--sanitizer]");
        run(DdlaBackend::CPU, sanitizer);
#if DDLA_HAS_GPU
        run(DdlaBackend::GPU, sanitizer);
#endif
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MEMORY_TRI_FAILURE " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
