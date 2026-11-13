"""LibDDLA -- Distributed Dense Linear Algebra Python Bindings.

A high-performance library for ScaLAPACK-style 2D block-cyclic distributed dense
linear algebra with CPU and GPU backends, designed with native NumPy array
interoperability and mpi4py integration (inspired by ELPA/pyelpa in LibRPA).
"""

from ._c_api import (
    DdlaError,
    DdlaInvalidHandleError,
    DdlaInvalidValueError,
    DdlaAllocFailedError,
    DdlaBackendUnavailableError,
    DdlaInternalError,
    DdlaExceptionError,
    STATUS_SUCCESS,
    STATUS_INVALID_HANDLE,
    STATUS_INVALID_VALUE,
    STATUS_ALLOC_FAILED,
    STATUS_BACKEND_UNAVAILABLE,
    STATUS_INTERNAL_ERROR,
    STATUS_EXCEPTION,
    BACKEND_CPU,
    BACKEND_GPU,
    COMMUNICATION_MPI,
    COMMUNICATION_HOST_STAGED_MPI,
    COMMUNICATION_CCL,
    DTYPE_,
    CTXT_,
    M_,
    N_,
    MB_,
    NB_,
    RSRC_,
    CSRC_,
    LLD_,
    DLEN_
)

from .backend import Backend, Communication, Handle
from .layout import ProcessorLayout
from .matrix import DistributedMatrix
from .ops import (
    pgemm,
    pgetrf,
    pgetrs,
    pgesv,
    ppotrf,
    ppotrs,
    pposv,
    pgeqrf,
    pgeqlf,
    ptrtrs,
    ptrmm,
    pgeadd,
    pdam,
    gather,
    scatter
)

try:
    from ._c_api import ddla_c_finalize as finalize
except (ImportError, AttributeError):
    def finalize():
        pass

__version__ = "0.1.0"

__all__ = [
    # Lifecycle
    "finalize",
    # Classes
    "Backend",
    "Communication",
    "Handle",
    "ProcessorLayout",
    "DistributedMatrix",
    # Distributed Linear Algebra Operations
    "pgemm",
    "pgetrf",
    "pgetrs",
    "pgesv",
    "ppotrf",
    "ppotrs",
    "pposv",
    "pgeqrf",
    "pgeqlf",
    "ptrtrs",
    "ptrmm",
    "pgeadd",
    "pdam",
    "gather",
    "scatter",
    # Exceptions
    "DdlaError",
    "DdlaInvalidHandleError",
    "DdlaInvalidValueError",
    "DdlaAllocFailedError",
    "DdlaBackendUnavailableError",
    "DdlaInternalError",
    "DdlaExceptionError",
    # Constants
    "STATUS_SUCCESS",
    "BACKEND_CPU",
    "COMMUNICATION_MPI",
    "COMMUNICATION_HOST_STAGED_MPI",
    "COMMUNICATION_CCL",
    "BACKEND_GPU",
]
