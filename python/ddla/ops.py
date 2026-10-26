"""High-level mathematical routines for LibDDLA distributed matrices."""
import numpy as np
import ctypes
from typing import Optional, Tuple, Union
from .matrix import DistributedMatrix
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

def pgetrf(A: DistributedMatrix, ipiv: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Distributed LU factorization with partial pivoting: A = P * L * U."""
    if ipiv is None:
        ipiv = np.zeros(A.m_loc + A.mb, dtype=np.int32)
    info = ctypes.c_int(0)
    func = getattr(lib, f"ddla_{A.type_char}_pgetrf")
    st = func(
        A.handle.ptr, A.m, A.n, ctypes.c_void_p(A.data_ptr), 1, 1,
        ctypes.c_void_p(A.desc.ctypes.data), ctypes.c_void_p(ipiv.ctypes.data),
        ctypes.byref(info)
    )
    check_status(st, f"ddla_{A.type_char}_pgetrf")
    return A, ipiv, info.value

def pgetrs(A: DistributedMatrix, B: DistributedMatrix, ipiv: np.ndarray,
           trans: str = 'N') -> DistributedMatrix:
    """Solve distributed linear system op(A) * X = B using LU factors from pgetrf."""
    t = _to_char_bytes(trans)
    func = getattr(lib, f"ddla_{A.type_char}_pgetrs")
    st = func(
        A.handle.ptr, b'L', t, A.n, B.n,
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(ipiv.ctypes.data),
        ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data)
    )
    check_status(st, f"ddla_{A.type_char}_pgetrs")
    return B

def pgesv(A: DistributedMatrix, B: DistributedMatrix,
          ipiv: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Solve distributed linear system A * X = B with partial pivoting (computes LU then solves)."""
    A, ipiv, info = pgetrf(A, ipiv=ipiv)
    if info == 0:
        pgetrs(A, B, ipiv, trans='N')
    return B, ipiv, info

def ppotrf(A: DistributedMatrix, uplo: str = 'L') -> Tuple[DistributedMatrix, int, bool]:
    """Distributed Cholesky factorization of a symmetric/Hermitian positive-definite matrix."""
    u = _to_char_bytes(uplo)
    info = ctypes.c_int(0)
    is_nega = ctypes.c_int(0)
    func = getattr(lib, f"ddla_{A.type_char}_ppotrf")
    st = func(
        A.handle.ptr, u, A.n, ctypes.c_void_p(A.data_ptr), 1, 1,
        ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.byref(info), ctypes.byref(is_nega)
    )
    check_status(st, f"ddla_{A.type_char}_ppotrf")
    return A, info.value, bool(is_nega.value)

def ppotrs(A: DistributedMatrix, B: DistributedMatrix, uplo: str = 'L',
           is_nega: bool = False, location: int = -1) -> DistributedMatrix:
    """Solve distributed linear system A * X = B using Cholesky factor from ppotrf."""
    u = _to_char_bytes(uplo)
    func = getattr(lib, f"ddla_{A.type_char}_ppotrs")
    st = func(
        A.handle.ptr, b'L', u, b'N', A.n, B.n,
        ctypes.c_void_p(A.data_ptr), ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(B.data_ptr), ctypes.c_void_p(B.desc.ctypes.data),
        int(is_nega), int(location)
    )
    check_status(st, f"ddla_{A.type_char}_ppotrs")
    return B

def pposv(A: DistributedMatrix, B: DistributedMatrix, uplo: str = 'L') -> Tuple[DistributedMatrix, int, bool]:
    """Distributed Cholesky driver: factors A and solves A * X = B."""
    A, info, is_nega = ppotrf(A, uplo=uplo)
    if info == 0:
        ppotrs(A, B, uplo=uplo, is_nega=is_nega, location=-1)
    return B, info, is_nega

def pgeqrf(A: DistributedMatrix, tau: Optional[np.ndarray] = None) -> Tuple[DistributedMatrix, np.ndarray, int]:
    """Distributed Householder QR factorization: A = Q * R."""
    min_dim = min(A.m, A.n)
    if tau is None:
        tau = np.zeros(min_dim, dtype=A.dtype)
    info = ctypes.c_int(0)
    func = getattr(lib, f"ddla_{A.type_char}_pgeqrf")
    st = func(
        A.handle.ptr, b'U', A.m, A.n,
        ctypes.c_void_p(A.data_ptr),
        ctypes.c_void_p(A.desc.ctypes.data),
        ctypes.c_void_p(tau.ctypes.data),
        ctypes.byref(info)
    )
    check_status(st, f"ddla_{A.type_char}_pgeqrf")
    return A, tau, info.value

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
