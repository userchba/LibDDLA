#ifndef DDLA_CONNECTOR_H
#define DDLA_CONNECTOR_H

#include <mpi.h>
#include <iostream>
#include "gpu/gpu.h"     // vendor headers + the gpu* names (CUDA, HIP or the CPU stand-ins)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>
#include <complex>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <ddla/ddla_handle_t.h>

namespace ddla{

#ifdef DDLA_USE_CCL
using cclOp=gpucclRedOp_t;
const auto cclSum=gpucclSum;
#else
using cclOp=MPI_Op;
const auto cclSum=MPI_SUM;
#endif

// ---------------------------------------------------------------------------
// The backend this translation unit was compiled for.  A TU compiled with a GPU vendor active is a GPU TU (a dual
// CPU+GPU build compiles every TU that way and picks CPU or GPU per call from the handle); otherwise it is a CPU
// TU.  The vendor runtime itself is reached through the handle-first ddla* functions of <ddla/ddla_handle_t.h>
// (ddlaMalloc, ddlaMemcpy, ddlaSynchronize, ddlaEventRecord, ...); there is no per-TU runtime wrapper layer.
// ---------------------------------------------------------------------------
namespace detail {

#if DDLA_GPU_VENDOR
inline constexpr DdlaBackend local_backend_v = DdlaBackend::GPU;
#elif defined(DDLA_USE_CPU)
inline constexpr DdlaBackend local_backend_v = DdlaBackend::CPU;
#endif

} // namespace detail

// ---------------------------------------------------------------------------
// Private work-queue access (library code only: the public API has no stream).  A GPU handle owns more than one
// vendor queue: the one every ddla* function runs on (`stream`), the data queue (`stream_data`) and the level queue
// of the eigensolver back-transform (`stream_level`, DdlaStream::level_queue()).  These four calls are the ddla* operations the library needs on one
// of the extra queues.  Defined in ddla_handle_t.cpp; same validation and CPU behaviour as the public functions.
// ---------------------------------------------------------------------------
namespace detail {

ddlaStatus_t memcpy_on(const DdlaHandle_t& handle, void* dst, const void* src, std::size_t bytes, gpuStream_t queue);
ddlaStatus_t synchronize_on(const DdlaHandle_t& handle, gpuStream_t queue);
ddlaStatus_t event_record_on(const DdlaHandle_t& handle, ddlaEvent_t event, gpuStream_t queue);
ddlaStatus_t event_wait_on(const DdlaHandle_t& handle, ddlaEvent_t event, gpuStream_t queue);

} // namespace detail

// ---------------------------------------------------------------------------
// The gpublas* / gpusolver* / gpurand* handle, status and enum names, their CPU-only stand-ins and the
// vendor function aliases (gpuMalloc, gpublasZgemm, ...) are defined by gpu/gpu.h, the only place
// that spells CUDA / HIP API names.  The public <ddla/ddla.h> stays vendor-free.
// ---------------------------------------------------------------------------

static inline void MPI_CHECK(int status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != MPI_SUCCESS)
    {
        fprintf(stderr, "mpi error at %s:%d : %d\n", file, line, status);
        int mpi_initialized = 0;
        int mpi_finalized = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized);
        // Abort the whole MPI job when possible so the other ranks do not
        // hang waiting in collectives.  MPI_Abort is only valid between
        // MPI_Init and MPI_Finalize; after finalization fall through to the
        // throw below.
        if (mpi_initialized && !mpi_finalized) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        throw std::runtime_error("ddla: MPI error at " + std::string(file) +
                                 ":" + std::to_string(line));
    }
}

// Error-propagation model for the CHECK family below: on failure these throw
// std::runtime_error (previously they called exit(EXIT_FAILURE)).  An
// uncaught exception still terminates the rank, so uncaught it behaves like
// the old exit(); but if a caller catches the exception on one rank only,
// the remaining ranks can hang in MPI/NCCL/RCCL collectives.  Callers that
// catch must therefore abort the whole job themselves (e.g. MPI_Abort) when
// the error could leave peers blocked in a collective.  MPI_CHECK already
// aborts the job directly (see above); the other CHECK macros cannot, since
// they have no communicator available.
#if DDLA_GPU_VENDOR
// Check a vendor runtime call made directly through a gpu* name.  Only code that has no DdlaHandle_t to go through
// uses this (the stream-ordered solver workspace in getrf.cpp / potrf.cpp, write_matrix.cpp, the handle
// implementation itself); everything else calls the ddla* functions and checks them with DDLA_CHECK.
static inline void GPU_CHECK(gpuError_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != gpuSuccess)
    {
        fprintf(stderr, "gpu error at %s:%d : %s\n", file, line, gpuGetErrorString(status));
        throw std::runtime_error("ddla: gpu error at " + std::string(file) +
                                 ":" + std::to_string(line) + " : " +
                                 gpuGetErrorString(status));
    }
}
#endif

static inline void BLAS_CHECK(gpublasStatus_t err_, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (err_ != GPUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "gpublas error %d at %s:%d\n", err_, file, line);
        throw std::runtime_error("ddla: gpublas error " + std::to_string(err_) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}

static inline void SOLVER_CHECK(gpusolverStatus_t err_, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (err_ != GPUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "solver error %d at %s:%d\n", err_, file, line);
        throw std::runtime_error("ddla: solver error " + std::to_string(err_) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}

#ifdef DDLA_USE_CCL
static inline void CCL_CHECK(gpucclResult_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != gpucclSuccess)
    {
        fprintf(stderr, "nccl error at %s:%d : %d\n", file, line, status);
        throw std::runtime_error("ddla: nccl error " + std::to_string(status) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}
#else
static inline void CCL_CHECK(int status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    MPI_CHECK(status, file, line);
}
#endif

static inline void GPURAND_CHECK(gpurandStatus_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != GPURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "gpurand error at %s:%d : %d\n", file, line, status);
        throw std::runtime_error("ddla: gpurand error " + std::to_string(status) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}


} // namespace ddla

#endif // DDLA_CONNECTOR_H
