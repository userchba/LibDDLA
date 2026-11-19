#include "ddla_stream.h"
#include "ddla_connector.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace ddla {
namespace {

#if DDLA_HAS_GPU
static_assert(std::is_pointer_v<gpuStream_t>);
static_assert(std::is_pointer_v<gpublasHandle_t>);
static_assert(std::is_pointer_v<gpusolverHandle_t>);
void create_stream(void*& opaque, bool nonblocking = false)
{
    gpuStream_t native = nullptr;
    const auto status = nonblocking ? gpuStreamCreateWithFlags(&native, gpuStreamNonBlocking)
                                    : gpuStreamCreate(&native);
    opaque = native;
    GPU_CHECK(status);
}
void create_fence(ddlaEvent_t& opaque)
{
    gpuEvent_t native = nullptr;
    const auto status = gpuEventCreateWithFlags(&native, gpuEventDisableTiming);
    opaque = native;
    GPU_CHECK(status);
}
void destroy_fence(ddlaEvent_t& opaque)
{
    if (opaque) GPU_CHECK(gpuEventDestroy(static_cast<gpuEvent_t>(opaque)));
    opaque = nullptr;
}
#endif

// Both backend specializations are generated from this translation unit.
// No CPU handle enters a vendor resource call, including in a dual build.
template <DdlaBackend Backend>
void init_resources(DdlaStream& s)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        s.stream = s.stream_data = s.stream_level = nullptr;
        s.blasH = s.solverH = nullptr;
        s.nccl_comm = s.nccl_row_comm = s.nccl_col_comm = nullptr;
    } else {
#if DDLA_HAS_GPU
        create_stream(s.stream);
        create_stream(s.stream_data, true);
        for (auto& event : s.summa_ready) create_fence(event);
        for (auto& event : s.summa_reuse) create_fence(event);
        create_fence(s.summa_input);
        for (auto& event : s.summa_metadata) create_fence(event);
        gpusolverHandle_t solver = nullptr;
        const auto solver_status = gpusolverCreate(&solver);
        s.solverH = solver;
        SOLVER_CHECK(solver_status);
        SOLVER_CHECK(gpusolverSetStream(solver, static_cast<gpuStream_t>(s.stream)));
        gpublasHandle_t blas = nullptr;
        const auto blas_status = gpublasCreate(&blas);
        s.blasH = blas;
        BLAS_CHECK(blas_status);
        BLAS_CHECK(gpublasSetStream(blas, static_cast<gpuStream_t>(s.stream)));
#else
        (void)s;
        throw std::runtime_error("ddla: GPU resources unavailable in this build");
#endif
    }
}

template <DdlaBackend Backend>
void clean_resources(DdlaStream& s)
{
    if constexpr (Backend == DdlaBackend::GPU) {
#if DDLA_HAS_GPU
        if (s.local_device >= 0) GPU_CHECK(gpuSetDevice(s.local_device));
        // CCL work may still reference the queues. Finish it before releasing
        // communicators, then destroy queues and compute-library resources.
        if (s.stream || s.blasH || s.solverH)
            GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(s.stream)));
        if (s.stream_data) GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(s.stream_data)));
        if (s.stream_level) GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(s.stream_level)));
        s.communicator->release(&s, false);
        for (auto& event : s.summa_ready) destroy_fence(event);
        for (auto& event : s.summa_reuse) destroy_fence(event);
        destroy_fence(s.summa_input);
        for (auto& event : s.summa_metadata) destroy_fence(event);
        if (s.stream) GPU_CHECK(gpuStreamDestroy(static_cast<gpuStream_t>(s.stream)));
        s.stream = nullptr;
        if (s.stream_data) GPU_CHECK(gpuStreamDestroy(static_cast<gpuStream_t>(s.stream_data)));
        s.stream_data = nullptr;
        if (s.stream_level) GPU_CHECK(gpuStreamDestroy(static_cast<gpuStream_t>(s.stream_level)));
        s.stream_level = nullptr;
        if (s.solverH) SOLVER_CHECK(gpusolverDestroy(static_cast<gpusolverHandle_t>(s.solverH)));
        s.solverH = nullptr;
        if (s.blasH) BLAS_CHECK(gpublasDestroy(static_cast<gpublasHandle_t>(s.blasH)));
        s.blasH = nullptr;
#endif
    }
    s.stream = s.stream_data = s.stream_level = nullptr;
    s.blasH = s.solverH = nullptr;
    s.nccl_comm = s.nccl_row_comm = s.nccl_col_comm = nullptr;
    for (auto& event : s.summa_ready) event = nullptr;
    for (auto& event : s.summa_reuse) event = nullptr;
    s.summa_input = nullptr;
    for (auto& event : s.summa_metadata) event = nullptr;
}

} // namespace

void DdlaStream::select_communication(DdlaCommunication mode)
{
    const auto* next = Communicator::get(backend, mode);
    if (!next) throw std::invalid_argument("ddla: communication mode unavailable");
    if (next == communicator) return;
    if (initialized) {
        next->initialize(this);
        communicator->release(this, true);
    }
    communicator = next;
}

void* DdlaStream::level_queue()
{
    if (backend == DdlaBackend::CPU) return nullptr;
    if (!stream_level) {
#if DDLA_HAS_GPU
        create_stream(stream_level);
#else
        throw std::runtime_error("ddla: GPU resources unavailable in this build");
#endif
    }
    return stream_level;
}

#if DDLA_HAS_GPU
namespace {

// Rank of this process among the job's processes on its node, as exported by
// the MPI launcher; -1 when the launcher exports none of these variables.
int launcher_node_rank()
{
    for (const char* name : {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK",
                             "PALS_LOCAL_RANKID"}) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') continue;
        char* end = nullptr;
        const long rank = std::strtol(value, &end, 10);
        if (*end == '\0' && rank >= 0) return static_cast<int>(rank);
    }
    return -1;
}

// Node-local rank in comm, ordered by key.
int node_rank_in(MPI_Comm comm, int key)
{
    int rank = 0;
    MPI_Comm node = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, &node));
    MPI_CHECK(MPI_Comm_rank(node, &rank));
    MPI_CHECK(MPI_Comm_free(&node));
    return rank;
}

// The GPU of this process, shared by all of its GPU handles once chosen.
std::mutex process_device_mutex;
int process_device = -1;

} // namespace
#endif

// The GPU belongs to the process, not to a handle: node-local rank among ALL
// processes of the job (MPI_COMM_WORLD) modulo the visible devices. A handle
// on a sub-communicator, such as one k-point pool, keeps the device that the
// process already uses instead of picking one by its rank in the pool, which
// would put the first rank of every pool on device 0. The node-local rank in
// MPI_COMM_WORLD is computed collectively only when local_comm spans it;
// otherwise it comes from the launcher, so a handle created by part of the job
// needs no collective call outside local_comm. Without either, the node-local
// rank in local_comm is used and the choice is not kept.
int DdlaStream::getLocalDevice(MPI_Comm local_comm) const
{
    if (backend == DdlaBackend::CPU) return 0;
#if DDLA_HAS_GPU
    int count = 0;
    GPU_CHECK(gpuGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("ddla: no GPU devices available");
    int world_compare = MPI_UNEQUAL;
    MPI_CHECK(MPI_Comm_compare(local_comm, MPI_COMM_WORLD, &world_compare));
    const bool spans_world = world_compare != MPI_UNEQUAL;
    std::lock_guard<std::mutex> lock(process_device_mutex);
    if (spans_world) {
        // Every rank of the job is here, so the collective call is safe even
        // on ranks that already chose a device. Ordering by the rank in
        // MPI_COMM_WORLD gives the same answer for a reordered communicator.
        int world_rank = 0;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        const int world_device = node_rank_in(local_comm, world_rank) % count;
        if (process_device < 0) process_device = world_device;
        return process_device;
    }
    if (process_device >= 0) return process_device;
    const int launcher_rank = launcher_node_rank();
    if (launcher_rank >= 0) {
        process_device = launcher_rank % count;
        return process_device;
    }
    return node_rank_in(local_comm, 0) % count;
#else
    (void)local_comm;
    throw std::runtime_error("ddla: GPU resources unavailable in this build");
#endif
}

void DdlaStream::set_local_device(int dev)
{
    if (backend == DdlaBackend::CPU) {
        local_device = 0;
        return;
    }
#if DDLA_HAS_GPU
    GPU_CHECK(gpuSetDevice(dev));
    local_device = dev;
#else
    (void)dev;
    throw std::runtime_error("ddla: GPU resources unavailable in this build");
#endif
}

void DdlaStream::init(MPI_Comm comm_group, char order)
{
    int count = 0;
    MPI_CHECK(MPI_Comm_size(comm_group, &count));
    int rows = static_cast<int>(std::ceil(std::sqrt(count)));
    while (count % rows != 0) --rows;
    init(rows, count / rows, comm_group, order);
}

void DdlaStream::init(int rows, int cols, MPI_Comm comm_group, char order)
{
    const DdlaBackend saved_backend = backend;
    clean();
    backend = saved_backend;
    if (backend == DdlaBackend::GPU) set_local_device(getLocalDevice(comm_group));
    else local_device = 0;
    major = order;
    MPI_CHECK(MPI_Comm_dup(comm_group, &comm));
    MPI_CHECK(MPI_Comm_rank(comm, &myid));
    MPI_CHECK(MPI_Comm_size(comm, &nprocs));
    nprows_ = rows;
    npcols_ = cols;
    if (nprows_ * npcols_ != nprocs)
        throw std::runtime_error("ddla: nprows * npcols != nprocs");
    if (major == 'R') {
        myprow_ = myid / cols;
        mypcol_ = myid % cols;
    } else if (major == 'C') {
        myprow_ = myid % rows;
        mypcol_ = myid / rows;
    } else {
        throw std::runtime_error("ddla: major must be 'R' or 'C'");
    }
    MPI_CHECK(MPI_Comm_split(comm, myprow_, myid, &row_comm));
    MPI_CHECK(MPI_Comm_split(comm, mypcol_, myid, &col_comm));
    if (backend == DdlaBackend::CPU) {
        init_resources<DdlaBackend::CPU>(*this);
    } else {
        // No rank may enter CCL initialization after another failed local
        // setup. CPU setup above owns no vendor resources and cannot fail.
        std::exception_ptr setup_error;
        try { init_resources<DdlaBackend::GPU>(*this); }
        catch (...) { setup_error = std::current_exception(); }
        int failed = setup_error ? 1 : 0;
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm));
        if (failed) {
            // Every rank owns the same communicator topology, including a rank
            // whose vendor setup stopped partway through resource creation.
            try { clean(); } catch (...) {}
            if (setup_error) std::rethrow_exception(setup_error);
            throw std::runtime_error("ddla: GPU resource preparation failed on another rank");
        }
    }
    communicator->initialize(this);
    initialized = true;
}

void DdlaStream::clean()
{
    if (backend == DdlaBackend::CPU) clean_resources<DdlaBackend::CPU>(*this);
    else clean_resources<DdlaBackend::GPU>(*this);
    if (row_comm != MPI_COMM_NULL) MPI_CHECK(MPI_Comm_free(&row_comm));
    if (col_comm != MPI_COMM_NULL) MPI_CHECK(MPI_Comm_free(&col_comm));
    if (comm != MPI_COMM_NULL) MPI_CHECK(MPI_Comm_free(&comm));
    row_comm = col_comm = comm = MPI_COMM_NULL;
    myid = -1;
    nprocs = 0;
    node_ranks = 0;
    myprow_ = mypcol_ = -1;
    nprows_ = npcols_ = 0;
    local_device = -1;
    initialized = false;
    communication_group_depth = 0;
    communication_group_queue = nullptr;
}

DdlaStream::~DdlaStream()
{
    if (!destroyed) clean();
}

void DdlaStream::check_memory()
{
    if (backend == DdlaBackend::CPU) {
        std::printf("myid:%d, CPU backend (no device memory to check)\n", myid);
    } else {
#if DDLA_HAS_GPU
        std::size_t free_mem = 0, total_mem = 0;
        GPU_CHECK(gpuMemGetInfo(&free_mem, &total_mem));
        std::printf("myid:%d, local_device:%d, free_mem:%lf GB, total_mem:%lf GB\n",
                    myid, local_device, free_mem / 1024. / 1024 / 1024,
                    total_mem / 1024. / 1024 / 1024);
#else
        std::printf("myid:%d, GPU handle but no device API compiled\n", myid);
#endif
    }
}

} // namespace ddla
