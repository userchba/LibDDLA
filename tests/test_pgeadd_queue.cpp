#include "api_grid_test_common.h"
#include "PBLAS/pgeadd_impl.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include "pgeadd_queue_trace_api.h"
#include <cstring>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <dlfcn.h>
#include <iostream>
#include <string>

using namespace ddla;
using namespace api_grid_test;
namespace {
void require(bool okay, const char* message)
{
    int bad = !okay;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (bad) throw std::runtime_error(message);
}
template<class F> F symbol(const char* name)
{
    void* address = dlsym(RTLD_DEFAULT, name);
    if (!address) throw std::runtime_error("pgeadd queue preload is required for native GPU checks");
    F f;
    static_assert(sizeof(f) == sizeof(address), "dlsym ABI");
    std::memcpy(&f, &address, sizeof(f));
    return f;
}
struct Trace {
    decltype(&pgeadd_queue_trace_begin) begin = nullptr;
    decltype(&pgeadd_queue_trace_end) end = nullptr;
    decltype(&pgeadd_queue_trace_fail_create) fail_create = nullptr;
    decltype(&pgeadd_queue_trace_setup_snapshot) setup = nullptr;
    explicit Trace(bool enabled)
    {
        if (!enabled) return;
        begin = symbol<decltype(begin)>("pgeadd_queue_trace_begin");
        end = symbol<decltype(end)>("pgeadd_queue_trace_end");
        fail_create = symbol<decltype(fail_create)>("pgeadd_queue_trace_fail_create");
        setup = symbol<decltype(setup)>("pgeadd_queue_trace_setup_snapshot");
    }
};
template<class T> T value(int i, int j)
{
    if constexpr (std::is_arithmetic<T>::value)
        return T(0.3 + 0.017 * i - 0.023 * j);
    else return T(0.3 + 0.017 * i - 0.023 * j, -0.2 + 0.011 * i + 0.019 * j);
}
template<class T> T conjugate(T x) { return x; }
template<class R> std::complex<R> conjugate(std::complex<R> x) { return std::conj(x); }
void desc(const DdlaHandle_t& h, int* d, int m, int n, int mb, int nb)
{
    const int rows = num_loc(m, mb, h->myprow_, 0, h->nprows_);
    DDLA_CHECK(ddlaDescInit(d, h, m, n, mb, nb, 0, 0, std::max(1, rows) + 3));
}

#ifndef DDLA_PGEADD_QUEUE_BASELINE_CONTROL
#if DDLA_HAS_GPU
void partial_setup(int pr, int pc, const Trace& trace)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
    DDLA_CHECK(ddlaSetCommunication(h, DdlaCommunication::HostStagedMPI));
    trace.fail_create(rank == 0 ? 5 : 0);
    bool threw = false;
    try { DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc)); }
    catch (const std::exception&) { threw = true; }
    require(threw, "GPU preparation failure was not agreed on every rank");
    pgeadd_queue_snapshot counters{};
    trace.setup(&counters);
    require(counters.event_creates == counters.event_destroys &&
            counters.stream_creates == counters.stream_destroys &&
            counters.create_faults == std::uint64_t(rank == 0),
            "partial GPU setup leaked a prepared fence/stream or missed its fault");
    trace.fail_create(0);
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc));
    DDLA_CHECK(ddlaDestroy(h));
    trace.setup(&counters);
    require(counters.event_creates == counters.event_destroys &&
            counters.stream_creates == counters.stream_destroys,
            "GPU setup retry leaked prepared resources");
    if (!rank) std::cout << "PGEADD_PARTIAL_SETUP_PASS observed_fault=1 retry=1\n";
}
#endif

// Mismatched block boundaries create many individually serial GEAM copies.
// Compare the complete local storage across thread counts, including alias
// snapshots, offsets, signed zeros/nonfinites and untouched padding.
template<class T> void cpu_batch_threads(const DdlaHandle_t& h)
{
    constexpr int order = 1027, m = 769, n = 513;
    constexpr int ia = 2, ja = 3, ic = 3, jc = 5;
    const int budget = h->host_threads;
    int da[DDLA_DLEN_], dc[DDLA_DLEN_];
    desc(h, da, order, order, 37, 53);
    desc(h, dc, order, order, 61, 47);
    auto entry = [](int i, int j) {
        using R = typename real_type<T>::type;
        if ((i + 3 * j) % 101 == 0) return T(-R(0));
        if ((2 * i + j) % 103 == 0) return T(std::numeric_limits<R>::infinity());
        if ((i + j) % 107 == 0) return T(std::numeric_limits<R>::quiet_NaN());
        return value<T>(i, j);
    };
    const auto original = make_local<T>(h, da, entry);
    DeviceBuffer<T> a(h, std::max(std::size_t(1), original.size()));
    for (bool alias : {false, true}) {
        const int* target = alias ? da : dc;
        const auto initial = alias ? original : make_local<T>(h, dc, [](int, int) { return T(-7); });
        DeviceBuffer<T> c(h, std::max(std::size_t(1), initial.size()));
        T* output = alias ? a.ptr : c.ptr;
        for (char op : {'N', 'T', 'C'}) {
            std::size_t bytes = 0, host_bytes = 0;
            pgeadd_impl::pgeadd_workspace_size<T>(h, op, 'N', true, false, m, n,
                target, &bytes, &host_bytes, da, nullptr, true);
            ddla::vector<char> work(h, bytes);
            std::vector<char> host(host_bytes);
            std::vector<T> serial;
            for (int threads : {1, budget}) {
                h->host_threads = threads;
                upload(h, a.ptr, original);
                if (!alias) upload(h, c.ptr, initial);
                pgeadd_impl::pgeadd_copy_run<DdlaBackend::CPU,T>(h, op, m, n,
                    a.ptr, ia, ja, da, output, ic, jc, target,
                    work.data(), bytes, host.data(), host_bytes, nullptr, nullptr);
                const auto result = download(h, output, initial.size());
                bool correct = true;
                for (int j = 0; j < ddla_test::n_loc(h, target); ++j) {
                    const int gj = indx_l2g_c(target, h, j);
                    for (int i = 0; i < target[DDLA_LLD_]; ++i) {
                        const auto at = i + std::size_t(j) * target[DDLA_LLD_];
                        T expected = initial[at];
                        if (i < ddla_test::m_loc(h, target)) {
                            const int gi = indx_l2g_r(target, h, i);
                            if (gi >= ic - 1 && gi < ic - 1 + m &&
                                gj >= jc - 1 && gj < jc - 1 + n) {
                                const int row = gi - ic + 1, col = gj - jc + 1;
                                expected = op == 'N' ? entry(ia - 1 + row, ja - 1 + col)
                                                     : entry(ia - 1 + col, ja - 1 + row);
                                if (op == 'C') expected = conjugate(expected);
                            }
                        }
                        correct = correct && std::memcmp(&expected, &result[at], sizeof(T)) == 0;
                    }
                }
                require(correct, "CPU batch transpose/alias/padding mismatch");
                if (threads == 1) serial = result;
                require(result.empty() || std::memcmp(serial.data(), result.data(), result.size() * sizeof(T)) == 0,
                        "CPU batch changed bits across thread counts");
                if (!alias) {
                    const auto after = download(h, a.ptr, original.size());
                    require(original.empty() || std::memcmp(after.data(), original.data(), original.size() * sizeof(T)) == 0,
                            "CPU batch changed input");
                }
            }
        }
    }
    h->host_threads = budget;
    if (!h->myid) std::cout << "PGEADD_CPU_BATCH_THREADS_PASS threads=" << budget
                            << " scalar_bytes=" << sizeof(T) << "\n";
}

template<DdlaBackend Backend, class T>
void cases(const DdlaHandle_t& h, const Trace& trace)
{
    constexpr int m = 13, n = 11, ia = 2, ja = 3, ic = 3, jc = 2;
    constexpr std::size_t fence = 256;
    int da[DDLA_DLEN_], dc[DDLA_DLEN_];
    desc(h, da, 25, 27, 2, 3);
    desc(h, dc, 29, 31, 4, 5); // Force general redistribution even on1x1.
    const auto original = make_local<T>(h, da, [](int i, int j) { return value<T>(i, j); });
    const auto initial = make_local<T>(h, dc, [](int, int) { return T(-7); });
    DeviceBuffer<T> a(h, std::max(std::size_t(1), original.size()));
    DeviceBuffer<T> c(h, std::max(std::size_t(1), initial.size()));
    upload(h, a.ptr, original);
    std::size_t backend_bytes = 0, host_bytes = 0;
    for (char op : {'N', 'T', 'C'}) {
        std::size_t b = 0, v = 0;
        pgeadd_impl::pgeadd_workspace_size<T>(h, op, 'N', true, false, m, n, dc,
            &b, &v, da, nullptr, true);
        backend_bytes = std::max(backend_bytes, b);
        host_bytes = std::max(host_bytes, v);
    }
    // The two-operand case needs its own maximum, but shares the same host arena.
    std::size_t b = 0, v = 0;
    pgeadd_impl::pgeadd_workspace_size<T>(h, 'C', 'C', true, true, m, n, dc,
        &b, &v, da, da, true);
    backend_bytes = std::max(backend_bytes, b);
    host_bytes = std::max(host_bytes, v);
    ddla::vector<char> work(h, backend_bytes + 2 * fence);
    std::vector<char> host(host_bytes + 2 * fence, char(0x49));
    DDLA_CHECK(ddlaMemset(h, work.data(), 0x69, work.size()));
    DDLA_CHECK(ddlaSynchronize(h));
    void* const queue = Backend == DdlaBackend::GPU ? h->stream_data : nullptr;
    std::uint64_t observed_uploads = 0;
    if constexpr (Backend == DdlaBackend::GPU && std::is_same<T,std::complex<double>>::value) {
      if (h->communication() == DdlaCommunication::CCL) {
        upload(h, c.ptr, initial);
        DDLA_CHECK(ddlaSynchronize(h));
        if (!h->myid) commGroupStart<Backend>(h);
        bool checked = false;
        try {
            pgeadd_impl::pgeadd_check<Backend>(h, 'C', 'N', m, n, da, ia, ja,
                dc, 1, 1, dc, ic, jc, true, false);
        } catch (const std::exception&) { checked = true; }
        if (!h->myid) commGroupEnd<Backend>(h);
        require(checked, "PGEADD checker did not agree a rank-local active group");
        commGroupStart<Backend>(h);
        bool raw = false;
        try {
            pgeadd_impl::pgeadd_copy_run<Backend,T>(h, 'C', m, n,
                a.ptr, ia, ja, da, c.ptr, ic, jc, dc,
                work.data() + fence, backend_bytes, host.data() + fence, host_bytes,
                queue, h->summa_metadata[0]);
        } catch (const std::exception&) { raw = true; }
        commGroupEnd<Backend>(h);
        require(raw && download(h, c.ptr, initial.size()) == initial,
                "PGEADD raw group guard did not reject before output writes");
      }
    }
    for (int iteration = 0; iteration < 7; ++iteration) {
        const bool two_operands = iteration == 6;
        const char op = iteration % 3 == 0 ? 'N' : iteration % 3 == 1 ? 'T' : 'C';
        const char operation = two_operands ? 'C' : op;
        upload(h, c.ptr, initial);
        DDLA_CHECK(ddlaSynchronize(h));
        if constexpr (Backend == DdlaBackend::GPU) {
            DDLA_CHECK(detail::event_record_on(h, h->summa_input, h->stream));
            DDLA_CHECK(detail::event_wait_on(h, h->summa_input, queue));
            trace.begin(host.data() + fence, host_bytes, two_operands ? h->stream : queue,
                        h->summa_metadata[0], 0);
        }
        const bool marker_group = Backend == DdlaBackend::GPU && iteration == 0 &&
                                  h->communication() != DdlaCommunication::CCL;
        if (marker_group) commGroupStart<Backend>(h);
        if (two_operands)
            pgeadd_impl::pgeadd_run<Backend,T>(h, 'C', 'C', m, n,
                T(2), a.ptr, ia, ja, da, T(-0.5), a.ptr, ia, ja, da,
                c.ptr, ic, jc, dc, work.data() + fence, backend_bytes,
                host.data() + fence, host_bytes);
        else
            pgeadd_impl::pgeadd_copy_run<Backend,T>(h, operation, m, n,
                a.ptr, ia, ja, da, c.ptr, ic, jc, dc,
                work.data() + fence, backend_bytes, host.data() + fence, host_bytes,
                queue, h->summa_metadata[0]);
        if (marker_group) commGroupEnd<Backend>(h);
        if constexpr (Backend == DdlaBackend::GPU) {
            pgeadd_queue_snapshot counters{};
            trace.end(&counters);
            require(!counters.wrong_queue && !counters.unretired &&
                    (!counters.uploads || (counters.records && counters.waits)),
                    "metadata source was not explicitly retired on its selected queue");
            observed_uploads += counters.uploads;
            // Refill immediately: only device schedule copies may remain live.
            std::fill(host.begin() + fence, host.begin() + fence + host_bytes, char(0x27));
            DDLA_CHECK(detail::synchronize_on(h, queue));
        }
        const auto result = download(h, c.ptr, initial.size());
        double error = 0;
        for (int j = 0; j < ddla_test::n_loc(h, dc); ++j) {
            const int gj = indx_l2g_c(dc, h, j);
            for (int i = 0; i < ddla_test::m_loc(h, dc); ++i) {
                const int gi = indx_l2g_r(dc, h, i);
                const auto at = i + std::size_t(j) * dc[DDLA_LLD_];
                T expected = initial[at];
                if (gi >= ic - 1 && gi < ic - 1 + m && gj >= jc - 1 && gj < jc - 1 + n) {
                    const int row = gi - ic + 1, column = gj - jc + 1;
                    expected = operation == 'N' ? value<T>(ia - 1 + row, ja - 1 + column)
                                               : value<T>(ia - 1 + column, ja - 1 + row);
                    if (operation == 'C') expected = conjugate(expected);
                    if (two_operands) expected *= T(1.5);
                }
                error = std::max(error, double(std::abs(result[at] - expected)));
            }
            for (int i = ddla_test::m_loc(h, dc); i < dc[DDLA_LLD_]; ++i)
                error = std::max(error, double(std::abs(result[i + std::size_t(j) * dc[DDLA_LLD_]] - initial[i + std::size_t(j) * dc[DDLA_LLD_]])));
        }
        require_close(h, "PGEADD selected queue/metadata reuse", error,
            20 * std::numeric_limits<typename real_type<T>::type>::epsilon());
        const auto after = download(h, a.ptr, original.size());
        require(after == original, "PGEADD unit copy modified input");
        const auto scratch = download(h, work.data(), work.size());
        bool guarded = true;
        for (std::size_t q = 0; q < fence; ++q)
            guarded = guarded && scratch[q] == char(0x69) && scratch[backend_bytes + fence + q] == char(0x69)
                      && host[q] == char(0x49) && host[host_bytes + fence + q] == char(0x49);
        require(guarded, "PGEADD exact arena guard changed");
    }
    if constexpr (Backend == DdlaBackend::GPU) {
        unsigned long long total = observed_uploads;
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        require(total > 0, "metadata upload observer did not fire");
        if (h->nprocs == 1) {
            trace.begin(host.data() + fence, host_bytes, queue, h->summa_metadata[1], 1);
            bool threw = false;
            try {
                pgeadd_impl::pgeadd_copy_run<Backend,T>(h, 'C', m, n,
                    a.ptr, ia, ja, da, c.ptr, ic, jc, dc,
                    work.data() + fence, backend_bytes, host.data() + fence, host_bytes,
                    queue, h->summa_metadata[1]);
            } catch (const std::exception&) { threw = true; }
            pgeadd_queue_snapshot counters{};
            trace.end(&counters);
            require(threw && counters.uploads && counters.record_faults == 1 &&
                    counters.drains && !counters.unretired && !counters.wrong_queue,
                    "failed metadata event record did not drain its selected queue");
        }
    }
}
#else
// Compile this control with the PRE-FIX private headers and library. It uses
// only the unchanged pgeadd_run signature; no new handle field is inspected.
int baseline_control(const DdlaHandle_t& h, const Trace& trace)
{
    if (h->backend != DdlaBackend::GPU) throw std::runtime_error("baseline control requires a native GPU");
    DDLA_CHECK(ddlaSetCommunication(h, DdlaCommunication::HostStagedMPI));
    int da[DDLA_DLEN_], dc[DDLA_DLEN_];
    desc(h, da, 25, 27, 2, 3); desc(h, dc, 29, 31, 4, 5);
    const auto input = make_local<double>(h, da, [](int i, int j) { return value<double>(i, j); });
    const auto initial = make_local<double>(h, dc, [](int, int) { return -7.; });
    DeviceBuffer<double> a(h, std::max(std::size_t(1), input.size()));
    DeviceBuffer<double> c(h, std::max(std::size_t(1), initial.size()));
    upload(h, a.ptr, input); upload(h, c.ptr, initial);
    std::size_t bb = 0, hb = 0;
    pgeadd_impl::pgeadd_workspace_size<double>(h, 'C', 'N', true, false, 13, 11, dc,
        &bb, &hb, da, nullptr, true);
    ddla::vector<char> work(h, bb);
    std::vector<char> host(hb);
    DDLA_CHECK(ddlaSynchronize(h));
    trace.begin(host.data(), hb, h->stream, nullptr, 0);
    pgeadd_impl::pgeadd_run<DdlaBackend::GPU,double>(h, 'C', 'N', 13, 11,
        1., a.ptr, 2, 3, da, 0., nullptr, 1, 1, dc, c.ptr, 3, 2, dc,
        work.data(), bb, host.data(), hb);
    pgeadd_queue_snapshot counters{};
    trace.end(&counters);
    // Retire all work before freeing anything, even for the negative control.
    DDLA_CHECK(ddlaSynchronize(h));
    unsigned long long unretired = counters.unretired, uploads = counters.uploads;
    MPI_Allreduce(MPI_IN_PLACE, &unretired, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &uploads, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    require(uploads > 0, "baseline metadata hook did not fire");
    if (!h->myid) std::cout << (unretired ? "PGEADD_OLD_RETURN_UNFENCED" : "PGEADD_RETURN_RETIRED")
        << " uploads=" << uploads << " unretired=" << unretired
        << " observed_protocol_only=1 no_hardware_corruption_claim=1\n";
    return unretired ? 42 : 0;
}
#endif
}
int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    try {
        int ranks = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        int pr = 1, pc = ranks;
        DdlaBackend backend = DdlaBackend::CPU;
        DdlaCommunication transport = DdlaCommunication::HostStagedMPI;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--gpu") backend = DdlaBackend::GPU;
            else if (arg == "--cpu") backend = DdlaBackend::CPU;
            else if (arg == "--grid" && i + 1 < argc) {
                const std::string grid = argv[++i]; const auto split = grid.find('x');
                pr = std::stoi(grid.substr(0, split)); pc = std::stoi(grid.substr(split + 1));
            } else if (arg == "--transport" && i + 1 < argc) {
                const std::string mode = argv[++i];
                if (mode == "mpi") transport = DdlaCommunication::MPI;
                else if (mode == "staged") transport = DdlaCommunication::HostStagedMPI;
                else if (mode == "ccl") transport = DdlaCommunication::CCL;
                else throw std::invalid_argument("invalid transport");
            } else throw std::invalid_argument("test_pgeadd_queue: invalid argument");
        }
        require(pr > 0 && pc > 0 && pr * pc == ranks, "grid must equal launched ranks");
        Trace trace(backend == DdlaBackend::GPU);
#ifndef DDLA_PGEADD_QUEUE_BASELINE_CONTROL
#if DDLA_HAS_GPU
        if (backend == DdlaBackend::GPU) partial_setup(pr, pc, trace);
#endif
#endif
        DdlaHandle_t h = nullptr;
        DDLA_CHECK(ddlaInit(h, backend));
        if (backend == DdlaBackend::GPU) DDLA_CHECK(ddlaSetCommunication(h, transport));
        DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc));
        int status = 0;
#ifdef DDLA_PGEADD_QUEUE_BASELINE_CONTROL
        status = baseline_control(h, trace);
#else
        if (backend == DdlaBackend::CPU) {
            cases<DdlaBackend::CPU,float>(h, trace); cases<DdlaBackend::CPU,double>(h, trace);
            cases<DdlaBackend::CPU,std::complex<float>>(h, trace);
            cases<DdlaBackend::CPU,std::complex<double>>(h, trace);
            cpu_batch_threads<float>(h); cpu_batch_threads<double>(h);
            cpu_batch_threads<std::complex<float>>(h);
            cpu_batch_threads<std::complex<double>>(h);
        } else {
#if DDLA_HAS_GPU
            cases<DdlaBackend::GPU,float>(h, trace); cases<DdlaBackend::GPU,double>(h, trace);
            cases<DdlaBackend::GPU,std::complex<float>>(h, trace);
            cases<DdlaBackend::GPU,std::complex<double>>(h, trace);
            DDLA_CHECK(ddlaSetStream(h, nullptr));
            cases<DdlaBackend::GPU,std::complex<double>>(h, trace);
#endif
        }
        if (!h->myid) std::cout << "PGEADD_QUEUE_PASS explicit_host_retirement=1 no_performance_claim=1\n";
#endif
        DDLA_CHECK(ddlaDestroy(h));
        MPI_Finalize();
        return status;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
}
