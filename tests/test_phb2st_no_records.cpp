// CPU regression for phb2st_node's eigenvalue-only fallback.
// The recorded and discarded paths must produce bitwise-identical results;
// the discarded path must not allocate the quadratic rotation-record array.
// C++ allocation maxima are measured only during phb2st_node; they are not
// process RSS or total working-set estimates. --baseline retains numerical
// assertions but reports the old allocation for diagnostic comparisons.
// With no --grid argument, parse_options derives every valid grid from the
// communicator size, including 1x4, 2x2 and 4x1 in the default CTest run.
// This test explicitly selects the CPU backend and is registered CPU-only.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <type_traits>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "Eigensolver/two_stage/phb2st_impl.h"

namespace allocation_probe {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> largest{0};

void record(std::size_t bytes) noexcept
{
    if (!enabled.load(std::memory_order_relaxed)) return;
    std::size_t previous = largest.load(std::memory_order_relaxed);
    while (previous < bytes &&
           !largest.compare_exchange_weak(previous, bytes,
                                          std::memory_order_relaxed)) {}
}

struct Scope {
    Scope()
    {
        largest.store(0, std::memory_order_relaxed);
        enabled.store(true, std::memory_order_relaxed);
    }
    ~Scope() { enabled.store(false, std::memory_order_relaxed); }
    std::size_t finish()
    {
        enabled.store(false, std::memory_order_relaxed);
        return largest.load(std::memory_order_relaxed);
    }
};
} // namespace allocation_probe

void* operator new(std::size_t bytes)
{
    void* p = std::malloc(bytes ? bytes : 1);
    if (!p) throw std::bad_alloc();
    allocation_probe::record(bytes);
    return p;
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t bytes, std::align_val_t alignment)
{
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), bytes ? bytes : 1))
        throw std::bad_alloc();
    allocation_probe::record(bytes);
    return p;
}
void* operator new[](std::size_t bytes, std::align_val_t alignment)
{ return ::operator new(bytes, alignment); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
using namespace api_grid_test;

template <typename T> const char* type_name();
template <> const char* type_name<float>() { return "float"; }
template <> const char* type_name<double>() { return "double"; }
template <> const char* type_name<std::complex<float>>() { return "complex_float"; }
template <> const char* type_name<std::complex<double>>() { return "complex_double"; }

void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

template <typename T>
bool equal_bits(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <typename T>
void require_finite(const std::vector<T>& a, const std::string& label)
{
    for (const T& x : a)
        require(std::isfinite(std::real(x)) && std::isfinite(std::imag(x)),
                label + ": non-finite result");
}

template <typename T>
void hash_append(std::uint64_t& hash, const std::vector<T>& a)
{
    const auto* p = reinterpret_cast<const unsigned char*>(a.data());
    for (std::size_t i = 0; i < a.size() * sizeof(T); ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
}

template <typename T>
T band_input(int i, int j, int kd)
{
    using rt = typename ddla::real_type<T>::type;
    if (i == j) return T(rt(4) + rt(i % 13) / rt(8));
    if (i > j || j - i > kd) return T(rt(-913)); // unread region sentinel
    const rt re = rt(((i + 1) * 7 + (j + 1) * 13) % 19 - 9) / rt(16);
    if constexpr (std::is_same_v<T, rt>) return re;
    else return T(re, rt(((i + 1) * 11 + (j + 1) * 3) % 17 - 8) / rt(32));
}

template <typename T>
struct Result {
    using rt = typename ddla::real_type<T>::type;
    std::vector<rt> diagonal, off_diagonal;
    std::vector<T> phases;
    std::size_t largest_allocation = 0;
    std::size_t records = 0;
};

template <typename T>
Result<T> run(const ddla::DdlaHandle_t& handle, int n, int kd, const int* desc,
              const std::vector<T>& input, bool keep_records)
{
    // phb2st_node chases a private gathered band; caller A must be unchanged.
    // The private mutated band is intentionally not exposed by this API.
    std::vector<T> a(input);
    Result<T> result;
    using rt = typename ddla::real_type<T>::type;
    result.diagonal.assign(static_cast<std::size_t>(n), std::numeric_limits<rt>::quiet_NaN());
    result.off_diagonal.assign(static_cast<std::size_t>(std::max(0, n - 1)),
                               std::numeric_limits<rt>::quiet_NaN());
    ddla::detail::DistributedHistory<T> stream;
    int info = -1;
    {
        allocation_probe::Scope allocations;
        ddla::phb2st_node<ddla::DdlaBackend::CPU, T>(
            handle, n, kd, a.data(), desc, result.diagonal.data(),
            result.off_diagonal.data(), stream, result.phases, keep_records, info);
        result.largest_allocation = allocations.finish();
    }
    result.records = stream.local_count;
    stream.clear();
    require(info == 0, "phb2st_node returned nonzero info");
    require(equal_bits(a, input), "phb2st_node changed caller A or padding");
    require(keep_records || result.records == 0, "eigenvalue-only call retained records");
    require(result.phases.size() == (std::is_same_v<T, rt> ? 0 : static_cast<std::size_t>(n - 1)),
            "unexpected phase count");
    require_finite(result.diagonal, "diagonal");
    require_finite(result.off_diagonal, "off_diagonal");
    require_finite(result.phases, "phases");
    return result;
}

template <typename T>
void check_type(const ddla::DdlaHandle_t& handle, bool baseline)
{
    // Include no-chase/small cases, dense bandwidth, partial descriptor blocks,
    // and both sides of the many-bulge thresholds (2*kd-1)*(kd+1).
    const std::pair<int, int> cases[] = {
        {1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {5, 6},
        {8, 2}, {9, 2}, {10, 2}, {19, 3}, {20, 3}, {21, 3},
        {34, 4}, {35, 4}, {36, 4}, {64, 8},
        {134, 8}, {135, 8}, {136, 8}, {193, 8}
    };
    for (int threads : {1, 2}) {
        setenv("DDLA_EIG_CHASE_THREADS", threads == 1 ? "1" : "2", 1);
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (const auto& test : cases) {
            const int n = test.first, kd = test.second;
            int desc[9];
            require(ddla::ddlaDescInit(desc, handle, n, n, 3, 3, 0, 0) ==
                        ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,
                    "descriptor initialization failed");
            desc[ddla::DDLA_LLD_] += 3;
            auto input = make_local<T>(handle, desc,
                [kd](int i, int j) { return band_input<T>(i, j, kd); });
            // Give zero-local-extent ranks a valid pointer as well.
            if (input.empty()) input.resize(1, T(-719));
            const int ranks=ddlaGetSize(handle);
            if(ranks>1 && n>2 && kd>ddla::eig_distributed_band_limit(n,ranks)) {
                bool rejected=false;
                try { (void)run<T>(handle,n,kd,desc,input,false); }
                catch(const std::invalid_argument& error) {
                    rejected=std::strstr(error.what(),"band is too wide")!=nullptr;
                }
                require(rejected,"eigenvalue-only chase accepted a replicated wide band");
                continue;
            }
            const auto recorded = run<T>(handle, n, kd, desc, input, true);
            const auto discarded = run<T>(handle, n, kd, desc, input, false);
            const std::string label = std::string(type_name<T>()) + " n=" +
                std::to_string(n) + " kd=" + std::to_string(kd) +
                " threads=" + std::to_string(threads);
            require(equal_bits(recorded.diagonal, discarded.diagonal), label + " d differs");
            require(equal_bits(recorded.off_diagonal, discarded.off_diagonal), label + " e differs");
            require(equal_bits(recorded.phases, discarded.phases), label + " phases differ");
            hash_append(hash, discarded.diagonal);
            hash_append(hash, discarded.off_diagonal);
            hash_append(hash, discarded.phases);
            if (n == 193 && kd == 8) {
                const std::size_t record_bytes =
                    (static_cast<std::size_t>(n) * n / 2 + 64) * sizeof(ddla::hb2st_rot<T>);
                unsigned long long retained=recorded.records;
                MPI_Allreduce(MPI_IN_PLACE,&retained,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,
                              ddlaGetCommunicator(handle));
                require(retained > 0, label + " distributed history has no records");
                // Exercise the intercepted allocator explicitly. The production
                // reduction no longer has a replicated-history positive control.
                std::size_t probe_bytes=0;
                {
                    allocation_probe::Scope allocations;
                    std::vector<ddla::hb2st_rot<T>> probe(
                        static_cast<std::size_t>(n)*n/2+64);
                    probe_bytes=allocations.finish();
                }
                require(probe_bytes >= record_bytes,
                        label + " allocation instrumentation missed explicit control");
                if (!baseline)
                    require(discarded.largest_allocation < record_bytes,
                            label + " eigenvalue-only path allocated quadratic records");
                unsigned long long local[2] = {
                    static_cast<unsigned long long>(recorded.largest_allocation),
                    static_cast<unsigned long long>(discarded.largest_allocation)
                }, maximum[2] = {};
                MPI_Allreduce(local, maximum, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              ddlaGetCommunicator(handle));
                if (ddlaGetRank(handle) == 0)
                    std::printf("NO_RECORD_ALLOC type=%s grid=%s threads=%d n=%d kd=%d "
                                "record_bytes=%zu keep_max_new=%llu discard_max_new=%llu baseline=%d\n",
                                type_name<T>(), grid_name(handle).c_str(), threads, n, kd,
                                record_bytes, maximum[0], maximum[1], baseline ? 1 : 0);
            }
        }
        unsigned long long local_hash = static_cast<unsigned long long>(hash);
        unsigned long long min_hash = 0, max_hash = 0;
        MPI_Allreduce(&local_hash, &min_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                      ddlaGetCommunicator(handle));
        MPI_Allreduce(&local_hash, &max_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                      ddlaGetCommunicator(handle));
        require(min_hash == max_hash, "chase outputs differ between ranks");
        if (ddlaGetRank(handle) == 0)
            std::printf("NO_RECORD_BITS type=%s grid=%s threads=%d cases=%zu hash=%016llx PASS\n",
                        type_name<T>(), grid_name(handle).c_str(), threads,
                        sizeof(cases) / sizeof(cases[0]), local_hash);
    }
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool baseline = false;
    std::vector<char*> forwarded;
    forwarded.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--baseline") == 0) baseline = true;
        else forwarded.push_back(argv[i]);
    }
    try {
        api_grid_test::TestOptions options;
        std::string error;
        require(api_grid_test::parse_options(static_cast<int>(forwarded.size()), forwarded.data(),
                                             size, options, error), error);
        setenv("DDLA_EIG_CHASE_TEAM", "0", 1);
        setenv("DDLA_EIG_NODE_STREAM", "0", 1);
        for (const auto& grid : options.grids) {
            ddla::DdlaHandle_t handle = nullptr;
            require(ddla::ddlaInit(handle, ddla::DdlaBackend::CPU) ==
                        ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,
                    "CPU handle initialization failed");
            require(ddla::ddlaSet(handle, MPI_COMM_WORLD, grid.first, grid.second) ==
                        ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,
                    "process grid initialization failed");
            check_type<float>(handle, baseline);
            check_type<double>(handle, baseline);
            check_type<std::complex<float>>(handle, baseline);
            check_type<std::complex<double>>(handle, baseline);
            ddla::ddlaDestroy(handle);
        }
        if (rank == 0) std::puts("test_phb2st_no_records passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_phb2st_no_records failed rank=%d %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
