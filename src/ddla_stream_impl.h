#ifndef DDLA_STREAM_IMPL_H
#define DDLA_STREAM_IMPL_H

#include <ddla/ddla_connector.h>
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_config.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <cmath>

namespace ddla {

class DdlaStream {
public:
    // -- per-handle device selection (replaces static local_device) ---------
    int local_device = -1;

    // -- MPI communicators --------------------------------------------------
    MPI_Comm comm      = MPI_COMM_NULL;   // duplicated from user, owned
    MPI_Comm row_comm  = MPI_COMM_NULL;
    MPI_Comm col_comm  = MPI_COMM_NULL;
    int myid  = -1, nprocs = 0;
    int myprow_ = -1, nprows_ = 0, mypcol_ = -1, npcols_ = 0;

#ifdef DDLA_USE_CCL
    ncclComm_t nccl_comm     = nullptr;
    ncclComm_t nccl_row_comm = nullptr;
    ncclComm_t nccl_col_comm = nullptr;
#endif

    runtimeStream_t  stream =
#ifdef DDLA_USE_CPU
        0;
#else
        nullptr;
#endif
    runtimeStream_t  stream_data =
#ifdef DDLA_USE_CPU
        0;
#else
        nullptr;
#endif
    desolverHandle_t solverH = nullptr;
    deblasHandle_t   blasH   = nullptr;

    char major = 'R';

    // -- backend identity (set during ddla_init) -----------------------------
    // Placeholder until ddla_init overwrites it; never consulted before that.
    DdlaBackend backend = DdlaBackend::CPU;

    // -- lifecycle guards ---------------------------------------------------
    bool initialized = false;
    bool destroyed   = false;

    // -- host staging for the GPU_CPU_TUNNEL path ---------------------------
    // Grow-on-demand host buffers reused across comm* calls instead of
    // allocating a fresh std::vector per collective.  Two buffers are needed
    // because commAlltoallv stages both send and receive data simultaneously.
    // ::operator new guarantees alignment suitable for any fundamental type, so
    // reinterpret_cast<T*> is safe for T in
    // {float, double, std::complex<float>, std::complex<double>}.
    std::vector<unsigned char> tunnel_host_staging_a;
    std::vector<unsigned char> tunnel_host_staging_b;

    // -----------------------------------------------------------------------
    // Device selection (per-handle)
    // -----------------------------------------------------------------------
    int getLocalDevice(MPI_Comm local_comm) const
    {
#ifdef DDLA_USE_CPU
        (void)local_comm;
        return 0;
#else
        int localRank;
        MPI_Comm localCommInner;
        MPI_CHECK(MPI_Comm_split_type(local_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCommInner));
        MPI_CHECK(MPI_Comm_rank(localCommInner, &localRank));
        MPI_CHECK(MPI_Comm_free(&localCommInner));

        int deviceCount = 0;
        RUNTIME_CHECK(runtimeGetDeviceCount(&deviceCount));
        return localRank % deviceCount;
#endif
    }

    void set_local_device(int dev)
    {
        local_device = dev;
#ifdef DDLA_USE_CUDA
        RUNTIME_CHECK(cudaSetDevice(local_device));
#endif
#ifdef DDLA_USE_HIP
        RUNTIME_CHECK(hipSetDevice(local_device));
#endif
#if DDLA_HAS_CPU
        (void)dev;
#endif
    }

    // -----------------------------------------------------------------------
    // NCCL helper (moved from public header)
    // -----------------------------------------------------------------------
#ifdef DDLA_USE_CCL
    static inline void nccl_comm_create(ncclComm_t &c, const MPI_Comm &comm_group)
    {
        int rank, size;
        MPI_CHECK(MPI_Comm_rank(comm_group, &rank));
        MPI_CHECK(MPI_Comm_size(comm_group, &size));
        ncclUniqueId id;
        if (rank == 0) { CCL_CHECK(ncclGetUniqueId(&id)); }
        MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm_group));
        CCL_CHECK(ncclCommInitRank(&c, size, id, rank));
    }
#endif

    // -----------------------------------------------------------------------
    // init(comm, major) — grid auto-detect
    // -----------------------------------------------------------------------
    void init(MPI_Comm comm_group, const char& major)
    {
        MPI_CHECK(MPI_Comm_rank(comm_group, &myid));
        MPI_CHECK(MPI_Comm_size(comm_group, &nprocs));
        nprows_ = static_cast<int>(std::ceil(std::sqrt(nprocs)));
        while (nprocs % nprows_ != 0) { nprows_--; }
        npcols_ = nprocs / nprows_;
        init(nprows_, npcols_, comm_group, major);
    }

    // -----------------------------------------------------------------------
    // init(nprows, npcols, comm, major)
    // -----------------------------------------------------------------------
    void init(int nprows, int npcols, MPI_Comm comm_group, const char& major)
    {
        // Preserve immutable backend across clean/reinit
        DdlaBackend saved_backend = backend;
        clean();

        // Device selection (GPU handles only). `saved_backend` is already
        // concrete here: ddla_init stores the requested backend directly.
        if (saved_backend == DdlaBackend::GPU) {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
            set_local_device(getLocalDevice(comm_group));
#endif
        }
#if DDLA_HAS_CPU
        else {
            local_device = 0;
        }
#endif

        backend = saved_backend;
        this->major = major;

        // Duplicate communicator
        MPI_CHECK(MPI_Comm_dup(comm_group, &comm));
        MPI_CHECK(MPI_Comm_rank(comm, &myid));
        MPI_CHECK(MPI_Comm_size(comm, &nprocs));

        nprows_ = nprows;
        npcols_ = npcols;
        if (nprows_ * npcols_ != nprocs) {
            throw std::runtime_error("ddla: nprows * npcols != nprocs");
        }

        if (major == 'R') {
            myprow_ = myid / npcols;
            mypcol_ = myid % npcols;
        } else if (major == 'C') {
            mypcol_ = myid / nprows;
            myprow_ = myid % nprows;
        } else {
            throw std::runtime_error("ddla: major must be 'R' or 'C'");
        }

        MPI_CHECK(MPI_Comm_split(comm, myprow_, myid, &row_comm));
        MPI_CHECK(MPI_Comm_split(comm, mypcol_, myid, &col_comm));

        // CCL / stream / BLAS / solver: GPU handles only. `backend` is
        // already concrete here (set by ddla_init).
        if (backend == DdlaBackend::GPU) {
#ifdef DDLA_USE_CCL
            DdlaStream::nccl_comm_create(nccl_comm, comm);
            DdlaStream::nccl_comm_create(nccl_row_comm, row_comm);
            DdlaStream::nccl_comm_create(nccl_col_comm, col_comm);
#endif

#ifdef DDLA_USE_CUDA
            RUNTIME_CHECK(cudaStreamCreate(&stream));
            RUNTIME_CHECK(cudaStreamCreate(&stream_data));
            BLAS_CHECK(cublasCreate(&blasH));
            BLAS_CHECK(cublasSetStream(blasH, stream));
            SOLVER_CHECK(cusolverDnCreate(&solverH));
            SOLVER_CHECK(cusolverDnSetStream(solverH, stream));
#endif
#ifdef DDLA_USE_HIP
            RUNTIME_CHECK(hipStreamCreate(&stream));
            RUNTIME_CHECK(hipStreamCreate(&stream_data));
            SOLVER_CHECK(hipsolverCreate(&solverH));
            SOLVER_CHECK(hipsolverSetStream(solverH, stream));
            BLAS_CHECK(hipblasCreate(&blasH));
            BLAS_CHECK(hipblasSetStream(blasH, stream));
#endif
        }
#if DDLA_HAS_CPU
        else {
            // CPU handle: no device resources
            stream = 0;
            stream_data = 0;
            blasH = nullptr;
            solverH = nullptr;
        }
#endif
        initialized = true;
    }

    // -----------------------------------------------------------------------
    // clean — release all backend resources, then communicators
    // -----------------------------------------------------------------------
    void clean()
    {
        // Backend resources first
        if (stream != 0) {
#ifdef DDLA_USE_CUDA
            RUNTIME_CHECK(cudaStreamDestroy(stream));
#endif
#ifdef DDLA_USE_HIP
            RUNTIME_CHECK(hipStreamDestroy(stream));
#endif
            stream = 0;
        }
        if (stream_data != 0) {
#ifdef DDLA_USE_CUDA
            RUNTIME_CHECK(cudaStreamDestroy(stream_data));
#endif
#ifdef DDLA_USE_HIP
            RUNTIME_CHECK(hipStreamDestroy(stream_data));
#endif
            stream_data = 0;
        }
        if (solverH != nullptr) {
#ifdef DDLA_USE_CUDA
            SOLVER_CHECK(cusolverDnDestroy(solverH));
#endif
#ifdef DDLA_USE_HIP
            SOLVER_CHECK(hipsolverDestroy(solverH));
#endif
            solverH = nullptr;
        }
        if (blasH != nullptr) {
#ifdef DDLA_USE_CUDA
            BLAS_CHECK(cublasDestroy(blasH));
#endif
#ifdef DDLA_USE_HIP
            BLAS_CHECK(hipblasDestroy(blasH));
#endif
            blasH = nullptr;
        }

#ifdef DDLA_USE_CCL
        if (nccl_comm != nullptr) {
            CCL_CHECK(ncclCommDestroy(nccl_comm));
            nccl_comm = nullptr;
        }
        if (nccl_row_comm != nullptr) {
            CCL_CHECK(ncclCommDestroy(nccl_row_comm));
            nccl_row_comm = nullptr;
        }
        if (nccl_col_comm != nullptr) {
            CCL_CHECK(ncclCommDestroy(nccl_col_comm));
            nccl_col_comm = nullptr;
        }
#endif

        // Communicators last
        if (row_comm != MPI_COMM_NULL) {
            MPI_CHECK(MPI_Comm_free(&row_comm));
            row_comm = MPI_COMM_NULL;
        }
        if (col_comm != MPI_COMM_NULL) {
            MPI_CHECK(MPI_Comm_free(&col_comm));
            col_comm = MPI_COMM_NULL;
        }
        if (comm != MPI_COMM_NULL) {
            MPI_CHECK(MPI_Comm_free(&comm));
            comm = MPI_COMM_NULL;
        }

        // Reset grid / lifecycle state (backend is preserved by caller)
        myid = -1;   nprocs = 0;
        myprow_ = -1; nprows_ = 0;
        mypcol_ = -1; npcols_ = 0;
        local_device = -1;
        initialized = false;
    }

    ~DdlaStream()
    {
        if (!destroyed) clean();
    }

    // -----------------------------------------------------------------------
    // Utility: check memory
    // -----------------------------------------------------------------------
    void check_memory()
    {
        if (backend == DdlaBackend::CPU) {
            printf("myid:%d, CPU backend (no device memory to check)\n", myid);
        } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
            size_t free_mem, total_mem;
            RUNTIME_CHECK(runtimeMemGetInfo(&free_mem, &total_mem));
            printf("myid:%d, local_device:%d, free_mem:%lf GB, total_mem:%lf GB\n",
                   myid, local_device,
                   free_mem / 1024. / 1024 / 1024,
                   total_mem / 1024. / 1024 / 1024);
#else
            printf("myid:%d, GPU handle but no device API compiled\n", myid);
#endif
        }
    }

    // -----------------------------------------------------------------------
    // Rank / grid coordinate conversion
    // -----------------------------------------------------------------------
    void rank_to_rc(int rank, int& row, int& col) const
    {
        if (major == 'R') {
            row = rank / npcols_;
            col = rank % npcols_;
        } else if (major == 'C') {
            row = rank % nprows_;
            col = rank / nprows_;
        } else {
            throw std::runtime_error("major should be 'R' or 'C'\n");
        }
    }

    int rc_to_rank(int row, int col) const
    {
        if (major == 'R') {
            return row * npcols_ + col;
        } else if (major == 'C') {
            return col * nprows_ + row;
        } else {
            throw std::runtime_error("major should be 'R' or 'C'\n");
        }
    }
};

// ---------------------------------------------------------------------------
// Opaque handle type (defined here; forward-declared in public header)
// ---------------------------------------------------------------------------
// DdlaHandle_t = DdlaStream*  remains source-compatible internally
// because DdlaStream now carries all state that DdlaHandle_t pointed to.

} // namespace ddla

#endif // DDLA_STREAM_IMPL_H
