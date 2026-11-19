// A GPU handle uses the device of its process -- the node-local rank among all
// ranks of the job modulo the visible devices -- whatever communicator it is
// set on. Before, a handle on a sub-communicator picked the device from its
// rank in that communicator, so the first rank of every k-point pool or
// sub-grid on a node landed on device 0.
#include <ddla/ddla.h>
#include "test_backend.h"
#include "ddla_stream.h"   // handle->local_device -- internal test only

#include <mpi.h>

#include <cstdio>

namespace {

int failures = 0;

void expect_device(const char* what, ddla::DdlaHandle_t handle, int expected, int world_rank)
{
    if (handle->local_device != expected) {
        std::printf("rank %d: %s handle uses device %d, expected %d\n", world_rank, what,
                    handle->local_device, expected);
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (ddla_test::execution_backend != ddla::DdlaBackend::GPU || world_size < 2) {
        if (world_rank == 0) std::printf("test_device_selection skipped: needs a GPU build and 2+ ranks\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Comm node = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int node_rank = 0;
    MPI_Comm_rank(node, &node_rank);
    MPI_Comm_free(&node);

    const bool first_half = world_rank < world_size / 2;
    MPI_Comm halves = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, first_half ? 0 : 1, world_rank, &halves);
    MPI_Comm interleaved = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank, &interleaved);

    ddla::DdlaHandle_t half = nullptr;
    ddla::ddlaInit(half, ddla::DdlaBackend::GPU);
    int count = 0;
    DDLA_CHECK(ddla::ddlaGetDeviceCount(half, &count));
    const int expected = node_rank % count;

    // A sub-communicator handle before any handle on MPI_COMM_WORLD.
    DDLA_CHECK(ddla::ddlaSet(half, halves));
    expect_device("half", half, expected, world_rank);

    ddla::DdlaHandle_t world = nullptr;
    ddla::ddlaInit(world, ddla::DdlaBackend::GPU);
    DDLA_CHECK(ddla::ddlaSet(world, MPI_COMM_WORLD));
    expect_device("world", world, expected, world_rank);

    ddla::DdlaHandle_t strided = nullptr;
    ddla::ddlaInit(strided, ddla::DdlaBackend::GPU);
    DDLA_CHECK(ddla::ddlaSet(strided, interleaved));
    expect_device("interleaved", strided, expected, world_rank);

    // Only part of the job sets this handle: no collective call may reach
    // the ranks outside its communicator.
    if (first_half) {
        ddla::DdlaHandle_t part = nullptr;
        ddla::ddlaInit(part, ddla::DdlaBackend::GPU);
        DDLA_CHECK(ddla::ddlaSet(part, halves));
        expect_device("part-of-job", part, expected, world_rank);
        DDLA_CHECK(ddla::ddlaDestroy(part));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    DDLA_CHECK(ddla::ddlaDestroy(strided));
    DDLA_CHECK(ddla::ddlaDestroy(world));
    DDLA_CHECK(ddla::ddlaDestroy(half));
    MPI_Comm_free(&interleaved);
    MPI_Comm_free(&halves);

    int total = failures;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::printf("test_device_selection: %s (%d wrong device choices, %d ranks, %d devices)\n",
                    total == 0 ? "PASS" : "FAIL", total, world_size, count);
    }
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
