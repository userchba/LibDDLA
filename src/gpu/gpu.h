#ifndef DDLA_GPU_GPU_H
#define DDLA_GPU_GPU_H

// Vendor-neutral GPU layer.  Everything below the umbrella is chosen at compile time by exactly one of
//   DDLA_USE_CUDA -> gpu_cuda.h   (cuda* / cublas* / cusolverDn* / curand* / nccl*  ->  gpu*)
//   DDLA_USE_HIP  -> gpu_hip.h    (hip*  / hipblas* / hipsolver*  / hiprand* / rccl ->  gpu*)
//   neither       -> gpu_cpu.h    (CPU-only stand-ins for the types, enums and no-op macros)
// and the rest of the library spells only gpu* names: gpu<Runtime>, gpublas<..>, gpusolver<..>, gpurand<..>,
// gpuccl<..>.  DDLA_GPU_VENDOR is 1 when a GPU vendor is active in this translation unit.
//
// This is a hand-written alias layer (the MAGMA approach of writing CUDA once and running hipify at build
// time is deliberately not used: it only works through a Makefile and leaves HIP code spelled in CUDA).
// gpu_alias.h holds the aliases whose vendor names differ only by the family prefix (cuda/hip, cublas/hipblas,
// cusolverDn/hipsolver, curand/hiprand); each vendor header defines the prefixes and lists the rest itself.

#include <cstddef>
#include <stdexcept>

#if defined(DDLA_USE_CUDA) && defined(DDLA_USE_HIP)
#error "LibDDLA: DDLA_USE_CUDA and DDLA_USE_HIP are mutually exclusive"
#endif

#if defined(DDLA_USE_CUDA)
#define DDLA_GPU_VENDOR 1
#include "gpu_cuda.h"
#elif defined(DDLA_USE_HIP)
#define DDLA_GPU_VENDOR 1
#include "gpu_hip.h"
#else
#define DDLA_GPU_VENDOR 0
#include "gpu_cpu.h"
#endif

#if DDLA_GPU_VENDOR
namespace ddla {

/// Where a pointer lives, as the GPU runtime sees it.
enum class GpuMemType {
    Host,     ///< pageable, pinned or registered host memory
    Device,   ///< device memory (managed memory counts as device: it is reachable from both sides)
    Unknown   ///< the runtime could not classify the pointer
};

/// Classify @p p with the runtime's pointer query (the equivalent of MAGMA's magma_is_devptr).
/// The sticky runtime error left behind by a failed query is cleared, so a later ddlaGetLastError()
/// does not report it.
inline GpuMemType gpuPointerMemType(const void* p)
{
    gpuPointerAttributes attr;
    const gpuError_t err = gpuPointerGetAttributes(&attr, p);
    if (err == gpuSuccess) {
        const auto t = GPU_POINTER_MEMTYPE(attr);
        return (t == gpuMemoryTypeDevice || t == gpuMemoryTypeManaged) ? GpuMemType::Device
                                                                        : GpuMemType::Host;
    }
    (void)gpuGetLastError();
    // Older runtimes report ordinary pageable host memory as an invalid value.
    return err == gpuErrorInvalidValue ? GpuMemType::Host : GpuMemType::Unknown;
}

} // namespace ddla
#endif // DDLA_GPU_VENDOR

#endif // DDLA_GPU_GPU_H
