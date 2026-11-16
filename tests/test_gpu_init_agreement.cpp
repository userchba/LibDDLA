#include "mpi_test_init.h"
#include "ddla_stream.h"
#include "stream_ownership_trace_api.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

using namespace ddla;
namespace {
template<class F> F symbol(const char* name)
{
    void* address = dlsym(RTLD_DEFAULT, name);
    if (!address) throw std::runtime_error("stream fault preload is missing");
    F result;
    static_assert(sizeof(result) == sizeof(address), "function pointer ABI");
    std::memcpy(&result, &address, sizeof(result));
    return result;
}
void require(bool condition, const char* message)
{
    int good = condition ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &good, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!good) throw std::runtime_error(message);
}
void recovery(DdlaHandle_t h, int ranks)
{
    const int n = 2 * ranks + 1;
    const int cols = num_loc(n, 2, ddlaGetRank(h), 0, ranks);
    int descriptor[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descriptor, h, n, n, 2, 2, 0, 0, n));
    const std::size_t count = std::size_t(n) * cols;
    std::vector<double> input(count, 1.), output(count, 3.);
    ddla::vector<double> a(h, count), b(h, count), c(h, count);
    if (count) {
        DDLA_CHECK(ddlaMemcpy(h, a.data(), input.data(), count * sizeof(double)));
        std::fill(input.begin(), input.end(), 2.);
        DDLA_CHECK(ddlaMemcpy(h, b.data(), input.data(), count * sizeof(double)));
        DDLA_CHECK(ddlaMemcpy(h, c.data(), output.data(), count * sizeof(double)));
    }
    pgemm<double>(h, 'N', 'N', n, n, n, 1., a.data(), descriptor,
                  b.data(), descriptor, 1., c.data(), descriptor);
    if (count) DDLA_CHECK(ddlaMemcpy(h, output.data(), c.data(), count * sizeof(double)));
    DDLA_CHECK(ddlaSynchronize(h));
    bool good = true;
    for (double value : output) good = good && value == 2. * n + 3.;
    require(good, "GPU setup recovery GEMM failed");
}
}
int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks < 2) { MPI_Finalize(); return 77; }
    try {
        const auto reset = symbol<decltype(&stream_ownership_trace_reset)>("stream_ownership_trace_reset");
        const auto snapshot = symbol<decltype(&stream_ownership_trace_snapshot)>("stream_ownership_trace_snapshot");
        const auto arm = symbol<decltype(&stream_ownership_trace_fail_once)>("stream_ownership_trace_fail_once");
        int cases = 0;
        for (DdlaCommunication mode : {DdlaCommunication::CCL, DdlaCommunication::HostStagedMPI})
            for (int fault : {StreamSolverFault, StreamBlasFault})
                for (int target : {0, ranks - 1}) {
                    DdlaHandle_t h = nullptr;
                    DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
                    if (!ddlaCommunicationAvailable(h, mode)) { DDLA_CHECK(ddlaDestroy(h)); continue; }
                    DDLA_CHECK(ddlaSetCommunication(h, mode));
                    reset();
                    if (rank == target) arm(fault);
                    bool caught = false;
                    try { DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, ranks)); }
                    catch (const std::exception&) { caught = true; }
                    stream_ownership_snapshot state{};
                    snapshot(&state);
                    const auto fired = state.blas_faults + state.solver_faults;
                    std::fprintf(stderr, "GPU_INIT_FAILURE_RETURN rank=%d target=%d fault=%d caught=%d fired=%llu\n",
                        rank, target, fault, int(caught), static_cast<unsigned long long>(fired));
                    require(caught && fired == unsigned(rank == target), "GPU setup failure did not agree");
                    require(!h->initialized && h->comm == MPI_COMM_NULL && h->row_comm == MPI_COMM_NULL &&
                            h->col_comm == MPI_COMM_NULL && !h->stream && !h->stream_data && !h->blasH && !h->solverH,
                            "failed GPU setup retained partial resources");
                    require(state.creates == state.destroys && state.default_destroys == 0,
                            "failed GPU setup leaked/destroyed an unowned stream");
                    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, ranks));
                    recovery(h, ranks);
                    DDLA_CHECK(ddlaDestroy(h));
                    snapshot(&state);
                    require(state.creates == state.destroys && state.default_destroys == 0,
                            "recovered GPU handle leaked streams");
                    ++cases;
                }
        if (!rank) std::printf("GPU_INIT_AGREEMENT_PASS cases=%d\n", cases);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
