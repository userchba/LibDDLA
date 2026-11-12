#ifndef DDLA_BLAS_POINTER_MODE_H
#define DDLA_BLAS_POINTER_MODE_H

// ---------------------------------------------------------------------------
// Scalar-result pointer mode for the vendor BLAS handles.
//
// cuBLAS and hipBLAS default to HOST pointer mode, in which the scalar-
// returning Level-1 routines (`xdot*`, `xnrm2`) dereference their result
// argument as a HOST address.  Passing a device buffer in that mode either
// writes through an incidental host store (UVA-mapped allocations) or faults
// -- the latter is the SEGV_ACCERR seen inside `cublasDnrm2_v2` on the V100 /
// CUDA 11.8 stack when the result comes from the stream-ordered
// ddlaMallocAsync pool, which is not host-mapped.
//
// The eigensolver wrapper headers (`src/dot.h`, `src/nrm2.h`) document a
// DEVICE result pointer for GPU handles.  These helpers make that contract
// true by switching the handle into DEVICE mode for the duration of the call
// and restoring the previous mode afterwards.  The handle is process-local
// state and its BLAS handle is not shared between threads, so the
// save/restore is safe here.
//
// The CPU arms of the wrappers keep a HOST result pointer and do not use
// this header.
// ---------------------------------------------------------------------------

#include "ddla_connector.h"

namespace ddla {
namespace detail {

#if DDLA_HAS_GPU

using blas_pointer_mode_t = gpublasPointerMode_t;

inline blas_pointer_mode_t blas_pointer_mode_device() { return GPUBLAS_POINTER_MODE_DEVICE; }
inline void blas_get_pointer_mode(gpublasHandle_t h, blas_pointer_mode_t& m)
{
    BLAS_CHECK(gpublasGetPointerMode(h, &m));
}
inline void blas_set_pointer_mode(gpublasHandle_t h, blas_pointer_mode_t m)
{
    BLAS_CHECK(gpublasSetPointerMode(h, m));
}

/// RAII: put @p handle into DEVICE pointer mode, restoring the previous mode
/// on scope exit. Only GPU execution uses this scope.
struct blas_device_pointer_mode_scope {
    gpublasHandle_t h_;
    blas_pointer_mode_t saved_;

    explicit blas_device_pointer_mode_scope(gpublasHandle_t h) : h_(h), saved_()
    {
        blas_get_pointer_mode(h_, saved_);
        if (saved_ != blas_pointer_mode_device()) blas_set_pointer_mode(h_, blas_pointer_mode_device());
    }
    ~blas_device_pointer_mode_scope()
    {
        if (saved_ != blas_pointer_mode_device()) blas_set_pointer_mode(h_, saved_);
    }

    blas_device_pointer_mode_scope(const blas_device_pointer_mode_scope&) = delete;
    blas_device_pointer_mode_scope& operator=(const blas_device_pointer_mode_scope&) = delete;
};

#endif // DDLA_HAS_GPU

} // namespace detail
} // namespace ddla

#endif // DDLA_BLAS_POINTER_MODE_H
