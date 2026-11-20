"""High-level mathematical routines for LibDDLA distributed matrices."""
import numpy as np
import ctypes
from typing import Optional, Tuple, Union
from .matrix import DistributedMatrix
from .backend import Backend
from .pivots import PivotVector
from ._c_api import (
    lib, check_status, to_scalar,
    ddla_c_float_complex, ddla_c_double_complex
)

def _to_char_bytes(c: Union[str, bytes]) -> bytes:
    if isinstance(c, bytes):
        return c.upper()
    return c.upper().encode('ascii')

def pgemm(A: DistributedMatrix, B: DistributedMatrix, C: Optional[DistributedMatrix] = None,
          alpha: float = 1.0, beta: float = 0.0,
          transa: str = 'N', transb: str = 'N') -> DistributedMatrix:
    """Distributed matrix-matrix multiplication: C := alpha * op(A) * op(B) + beta * C.
    
    If C is None, a new DistributedMatrix of appropriate dimensions and matching dtype is allocated.
    """
    if A.type_char != B.type_char:
        raise TypeError(f"A and B must have identical dtype, got {A.dtype} and {B.dtype}")
    
    ta = _to_char_bytes(transa)
    tb = _to_char_bytes(transb)

    m = A.n if ta in (b'T', b'C') else A.m
    k_a = A.m if ta in (b'T', b'C') else A.n
    k_b = B.n if tb in (b'T', b'C') else B.m
    n = B.m if tb in (b'T', b'C') else B.n

    if k_a != k_b:
        raise ValueError(f"Incompatible inner dimensions: op(A) has {k_a} cols, op(B) has {k_b} rows")

    if C is None:
        C = DistributedMatrix(A.handle, m, n, mb=A.mb, nb=B.nb, dtype=A.dtype)
    else:
        if C.m != m or C.n != n:
            raise ValueError(f"C shape {C.shape} incompatible with product ({m}, {n})")
        if C.type_char != A.type_char:
            raise TypeError("C dtype must match A and B")

    func = getattr(lib, f"ddla_{A.type_char}_pgemm")
    al_val = to_scalar(alpha, A.type_char)
    be_val = to_scalar(beta, A.type_char)

    st = func(
        A.handle.ptr, ta, tb, m, n, k_a,
        ctypes.byref(al_val), ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data),
        ctypes.byref(be_val), ctypes.c_void_p(C.data_ptr), ctypes.c_void_p(C.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_pgemm")
    return C

def _pivot_storage(A, ipiv):
    if ipiv is None:
        ipiv = (np.zeros(A.m_loc + A.mb, dtype=np.int32)
                if A.handle.backend == Backend.CPU else PivotVector(A.handle, A.m_loc + A.mb))
    if isinstance(ipiv, PivotVector):
        if ipiv.handle is not A.handle or len(ipiv) < A.m_loc or ipiv._closed:
            raise ValueError("pivots need matching handle, sufficient local rows and live storage")
        return ipiv, ipiv.data_ptr
    if A.handle.backend != Backend.CPU:
        raise TypeError("GPU LU pivots must use ddla.PivotVector(handle, local_rows + mb)")
    if (not isinstance(ipiv, np.ndarray) or ipiv.dtype != np.int32 or ipiv.ndim != 1
            or not ipiv.flags.c_contiguous or not ipiv.flags.writeable or ipiv.size < A.m_loc):
        raise ValueError("CPU pivots need a contiguous, writable int32 vector covering local rows")
    return ipiv, ipiv.ctypes.data


def pgetrf(A: DistributedMatrix, ipiv=None):
    """Factor A in place; pivots are NumPy on CPU and PivotVector on GPU."""
    ipiv, pivot_ptr = _pivot_storage(A, ipiv)
    info = ctypes.c_int(0)
    func = getattr(lib, f"ddla_{A.type_char}_pgetrf")
    st = func(
        A.handle.ptr, A.m, A.n, ctypes.c_void_p(A.data_ptr), 1, 1,
        ctypes.c_void_p(A.desc.ctypes.data), ctypes.c_void_p(pivot_ptr), ctypes.byref(info)
    )
    check_status(st, f"ddla_{A.type_char}_pgetrf")
    return A, ipiv, info.value


def pgetrs(A: DistributedMatrix, B: DistributedMatrix, ipiv, trans: str = 'N') -> DistributedMatrix:
    """Solve using the backend pivot vector returned by pgetrf."""
    if ipiv is None:
        raise TypeError("pgetrs requires the pivots returned by pgetrf")
    ipiv, pivot_ptr = _pivot_storage(A, ipiv)
    t = _to_char_bytes(trans)
    func = getattr(lib, f"ddla_{A.type_char}_pgetrs")
    st = func(
        A.handle.ptr, b'L', t, A.n, B.n,
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(pivot_ptr), ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_pgetrs")
    return B


def pgesv(A: DistributedMatrix, B: DistributedMatrix, ipiv=None):
    """Factor and solve; retain pivots in the selected backend."""
    A, ipiv, info = pgetrf(A, ipiv=ipiv)
    if info == 0:
        pgetrs(A, B, ipiv, trans='N')
    return B, ipiv, info


def ppotrf(A: DistributedMatrix, uplo: str = 'L') -> Tuple[DistributedMatrix, int]:
    """Distributed Cholesky factorization of a symmetric/Hermitian positive-definite matrix."""
    u = _to_char_bytes(uplo)
    info = ctypes.c_int(0)
    func = getattr(lib, f"ddla_{A.type_char}_ppotrf")
    st = func(
        A.handle.ptr, u, A.n, ctypes.c_void_p(A.data_ptr), 1, 1,
        ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.byref(info)
    )
    check_status(st, f"ddla_{A.type_char}_ppotrf")
    return A, info.value

def ppotrs(A: DistributedMatrix, B: DistributedMatrix, uplo: str = 'L') -> DistributedMatrix:
    """Solve distributed linear system A * X = B using Cholesky factor from ppotrf."""
    u = _to_char_bytes(uplo)
    func = getattr(lib, f"ddla_{A.type_char}_ppotrs")
    st = func(
        A.handle.ptr, b'L', u, b'N', A.n, B.n,
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_ppotrs")
    return B

def pposv(A: DistributedMatrix, B: DistributedMatrix, uplo: str = 'L') -> Tuple[DistributedMatrix, int]:
    """Distributed Cholesky driver: factors A and solves A * X = B."""
    A, info = ppotrf(A, uplo=uplo)
    if info == 0:
        ppotrs(A, B, uplo=uplo)
    return B, info

def _qr_factor(A: DistributedMatrix, tau: Optional[np.ndarray], routine: str,
               default_tau_size: int) -> Tuple[DistributedMatrix, np.ndarray, int]:
    if tau is None:
        tau = np.zeros(default_tau_size, dtype=A.dtype)
    info = ctypes.c_int(0)
    func = getattr(lib, f"ddla_{A.type_char}_{routine}")
    st = func(
        A.handle.ptr, A.m, A.n,
        ctypes.c_void_p(A.data_ptr),
        ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(tau.ctypes.data),
        ctypes.byref(info)
    )
    check_status(st, f"ddla_{A.type_char}_{routine}")
    return A, tau, info.value


def pgeqrf(A: DistributedMatrix, tau: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Distributed Householder QR factorization, A = Q * R, overwriting A."""
    return _qr_factor(A, tau, "pgeqrf", min(A.m, A.n))


def pgeqlf(A: DistributedMatrix, tau: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Distributed Householder QL factorization, A = Q * L, overwriting A.

    tau uses A.n_loc entries indexed by local column. Wide matrices store
    their reflector scalars in the final columns.
    """
    return _qr_factor(A, tau, "pgeqlf", A.n_loc)

def pgelqf(A: DistributedMatrix, tau: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Distributed LQ, A=L*Q; host tau uses A.m_loc local-row entries."""
    return _qr_factor(A, tau, "pgelqf", A.m_loc)


def pgerqf(A: DistributedMatrix, tau: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Distributed RQ, A=R*Q; host tau uses A.m_loc local-row entries.

    Only the final min(A.m,A.n) rows carry scalars; earlier entries are preserved.
    """
    return _qr_factor(A, tau, "pgerqf", A.m_loc)


def ptrtrs(A: DistributedMatrix, B: DistributedMatrix, side: str = 'L',
           uplo: str = 'U', trans: str = 'N', diag: str = 'N') -> DistributedMatrix:
    """Solve triangular distributed system with multiple right-hand sides."""
    s = _to_char_bytes(side)
    u = _to_char_bytes(uplo)
    t = _to_char_bytes(trans)
    d = _to_char_bytes(diag)
    func = getattr(lib, f"ddla_{A.type_char}_ptrtrs")
    st = func(
        A.handle.ptr, s, u, t, d, B.m, B.n,
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_ptrtrs")
    return B

def ptrmm(A: DistributedMatrix, B: DistributedMatrix, alpha: float = 1.0,
          side: str = 'L', uplo: str = 'U', trans: str = 'N', diag: str = 'N') -> DistributedMatrix:
    """Distributed triangular matrix multiplication: B := alpha * op(A) * B (or B * op(A))."""
    s = _to_char_bytes(side)
    u = _to_char_bytes(uplo)
    t = _to_char_bytes(trans)
    d = _to_char_bytes(diag)
    al_val = to_scalar(alpha, A.type_char)
    func = getattr(lib, f"ddla_{A.type_char}_ptrmm")
    st = func(
        A.handle.ptr, s, u, t, d, B.m, B.n, ctypes.byref(al_val),
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_ptrmm")
    return B

def pgeadd(A: DistributedMatrix, B: DistributedMatrix, C: Optional[DistributedMatrix] = None,
           alpha: float = 1.0, beta: float = 1.0,
           transa: str = 'N', transb: str = 'N') -> DistributedMatrix:
    """Scaled distributed matrix addition: C := alpha * op(A) + beta * op(B).
    
    If C is None, a new DistributedMatrix is allocated.
    """
    if A.type_char != B.type_char:
        raise TypeError(f"A and B must have identical dtype, got {A.dtype} and {B.dtype}")
    ta = _to_char_bytes(transa)
    tb = _to_char_bytes(transb)
    m = A.n if ta in (b'T', b'C') else A.m
    n = A.m if ta in (b'T', b'C') else A.n
    if C is None:
        C = DistributedMatrix(A.handle, m, n, mb=A.mb, nb=A.nb, dtype=A.dtype)
    al_val = to_scalar(alpha, A.type_char)
    be_val = to_scalar(beta, B.type_char)
    func = getattr(lib, f"ddla_{A.type_char}_pgeadd")
    st = func(
        A.handle.ptr, ta, tb, m, n,
        ctypes.byref(al_val), ctypes.c_void_p(A.data_ptr), 1, 1, ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.byref(be_val), ctypes.c_void_p(B.data_ptr), 1, 1, ctypes.c_void_p(B.desc.ctypes.data),
        ctypes.c_void_p(C.data_ptr), 1, 1, ctypes.c_void_p(C.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_pgeadd")
    return C

def pdam(A: DistributedMatrix, alpha: float = 1.0, n: int = -1) -> DistributedMatrix:
    """Add alpha to each diagonal element: A(i,i) := A(i,i) + alpha."""
    al_val = to_scalar(alpha, A.type_char)
    func = getattr(lib, f"ddla_{A.type_char}_pdam")
    st = func(
        A.handle.ptr, ctypes.byref(al_val),
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data), int(n)
    )
    check_status(st, f"ddla_{A.type_char}_pdam")
    return A

def gather(A: DistributedMatrix, root: int = 0) -> Optional[np.ndarray]:
    """Gather distributed matrix to a global NumPy array on rank `root`."""
    return A.to_global_matrix(root=root)

def scatter(handle, global_arr: Optional[np.ndarray], mb: int = 64, nb: int = 64, root: int = 0) -> DistributedMatrix:
    """Scatter global NumPy array from rank `root` across the process grid."""
    return DistributedMatrix.from_global_matrix(handle, global_arr, mb=mb, nb=nb, root=root)
