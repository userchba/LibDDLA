#include "ddla_stream.h"
#include "ddla_connector.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace ddla {
namespace {

#if DDLA_HAS_GPU
static_assert(std::is_pointer_v<gpuStream_t>);
static_assert(std::is_pointer_v<gpublasHandle_t>);
static_assert(std::is_pointer_v<gpusolverHandle_t>);
#ifdef DDLA_USE_CCL
static_assert(std::is_pointer_v<gpucclComm_t>);

bool create_comm(void*& opaque, MPI_Comm group)
{
    int rank = 0, size = 0;
    MPI_CHECK(MPI_Comm_rank(group, &rank));
    MPI_CHECK(MPI_Comm_size(group, &size));
    gpucclUniqueId id;
    int failed = rank == 0 && gpucclGetUniqueId(&id) != gpucclSuccess;
    MPI_CHECK(MPI_Bcast(&failed, 1, MPI_INT, 0, group));
    if (failed) return false;
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, group));
    gpucclComm_t native = nullptr;
    const auto status = gpucclCommInitRank(&native, size, id, rank);
    failed = status != gpucclSuccess;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, group));
    if (failed) {
        if (native) (void)gpucclCommAbort(native);
        return false;
    }
    opaque = native;
    return true;
}

bool release_comm(void*& opaque, bool abort = false)
{
    if (!opaque) return true;
    const auto native = static_cast<gpucclComm_t>(opaque);
    const auto status = abort ? gpucclCommAbort(native) : gpucclCommDestroy(native);
    opaque = nullptr;
    return status == gpucclSuccess;
}
#endif

void create_stream(void*& opaque)
{
    gpuStream_t native = nullptr;
    const auto status = gpuStreamCreate(&native);
    opaque = native;
    GPU_CHECK(status);
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
        create_stream(s.stream_data);
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
        if (s.stream) GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(s.stream)));
        if (s.stream_data) GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(s.stream_data)));
        if (s.stream_level) GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(s.stream_level)));
#ifdef DDLA_USE_CCL
        bool released = release_comm(s.nccl_comm);
        released = release_comm(s.nccl_row_comm) && released;
        released = release_comm(s.nccl_col_comm) && released;
        if (!released) throw std::runtime_error("ddla: CCL communicator cleanup failed");
#endif
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
}

} // namespace

void DdlaStream::set_communication_resources(DdlaCommunication mode)
{
#if DDLA_HAS_GPU && defined(DDLA_USE_CCL)
    if (backend != DdlaBackend::GPU) return;
    if (mode == DdlaCommunication::CCL) {
        if (nccl_comm) return;
        // Commit only a complete grid/row/column set. Agree globally after
        // each subgroup setup, so a failed row cannot strand another row in
        // the subsequent column setup. Vendor fatal errors remain fatal.
        void* temporary[3] = {};
        MPI_Comm groups[] = {comm, row_comm, col_comm};
        for (int i = 0; i < 3; ++i) {
            int failed = !create_comm(temporary[i], groups[i]);
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm));
            if (failed) {
                for (auto& resource : temporary) (void)release_comm(resource, true);
                throw std::runtime_error("ddla: CCL communicator initialization failed");
            }
        }
        nccl_comm = temporary[0]; nccl_row_comm = temporary[1]; nccl_col_comm = temporary[2];
    } else {
        if (!nccl_comm && !nccl_row_comm && !nccl_col_comm) return;
        int failed = !release_comm(nccl_comm);
        failed = !release_comm(nccl_row_comm) || failed;
        failed = !release_comm(nccl_col_comm) || failed;
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm));
        if (failed) throw std::runtime_error("ddla: CCL communicator cleanup failed");
    }
#else
    (void)mode;
#endif
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

int DdlaStream::getLocalDevice(MPI_Comm local_comm) const
{
    if (backend == DdlaBackend::CPU) return 0;
#if DDLA_HAS_GPU
    int local_rank = 0;
    MPI_Comm local = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(local_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local));
    MPI_CHECK(MPI_Comm_rank(local, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local));
    int count = 0;
    GPU_CHECK(gpuGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("ddla: no GPU devices available");
    return local_rank % count;
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
    if (backend == DdlaBackend::CPU) init_resources<DdlaBackend::CPU>(*this);
    else init_resources<DdlaBackend::GPU>(*this);
    if (communication == DdlaCommunication::CCL) set_communication_resources(communication);
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
