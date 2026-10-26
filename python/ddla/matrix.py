"""Distributed block-cyclic matrix class for LibDDLA."""
import numpy as np
import ctypes
from typing import Optional, Tuple, Union
from .backend import Handle, Backend
from ._c_api import lib, check_status, DTYPE_, M_, N_, MB_, NB_, RSRC_, CSRC_, LLD_

DTYPE_TO_CHAR = {
    np.dtype(np.float32): 's',
    np.dtype(np.float64): 'd',
    np.dtype(np.complex64): 'c',
    np.dtype(np.complex128): 'z',
}

CHAR_TO_DTYPE = {
    's': np.float32,
    'd': np.float64,
    'c': np.complex64,
    'z': np.complex128,
}

class DistributedMatrix:
    """A 2D block-cyclic distributed matrix conforming to ScaLAPACK and ELPA conventions.
    
    Provides NumPy array interoperability for local partitions on CPU handles,
    device memory management on GPU handles, and native gather/scatter routines.
    """
    def __init__(self, handle: Handle, m: int, n: int, mb: int = 64, nb: int = 64,
                 dtype = np.float64, irsrc: int = 0, icsrc: int = 0, lld: int = -1,
                 data: Optional[np.ndarray] = None):
        self.handle = handle
        self.m = int(m)
        self.n = int(n)
        self.mb = int(mb)
        self.nb = int(nb)
        self.irsrc = int(irsrc)
        self.icsrc = int(icsrc)
        
        self.dtype = np.dtype(dtype)
        if self.dtype not in DTYPE_TO_CHAR:
            raise TypeError(f"Unsupported dtype: {self.dtype}. Supported: float32, float64, complex64, complex128")
        self.type_char = DTYPE_TO_CHAR[self.dtype]

        nprows, npcols = handle.grid_dims
        myprow, mypcol = handle.grid_coords
        self.m_loc = handle.numloc(self.m, self.mb, myprow, self.irsrc, nprows)
        self.n_loc = handle.numloc(self.n, self.nb, mypcol, self.icsrc, npcols)

        self.desc = np.zeros(9, dtype=np.int32)
        handle.descinit(self.desc, self.m, self.n, self.mb, self.nb, self.irsrc, self.icsrc, lld)

        self.lld = int(self.desc[LLD_])
        self._device_ptr = None

        if handle.backend == Backend.CPU:
            if data is not None:
                if data.shape != (self.m_loc, self.n_loc):
                    raise ValueError(f"Provided data shape {data.shape} does not match expected local shape ({self.m_loc}, {self.n_loc})")
                self.data = np.asfortranarray(data, dtype=self.dtype)
            else:
                self.data = np.zeros((self.m_loc, self.n_loc), dtype=self.dtype, order='F')
            self.data_ptr = self.data.ctypes.data
        else:
            nbytes = self.lld * max(1, self.n_loc) * self.dtype.itemsize
            self._device_ptr = handle.malloc(nbytes)
            self.data_ptr = self._device_ptr
            self.data = None
            if data is not None:
                self.from_host(data)

    def __del__(self):
        if self._device_ptr is not None:
            try:
                self.handle.free(self._device_ptr)
                self._device_ptr = None
            except Exception:
                pass

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.m, self.n)

    @property
    def local_shape(self) -> Tuple[int, int]:
        return (self.m_loc, self.n_loc)

    @property
    def block_size(self) -> Tuple[int, int]:
        return (self.mb, self.nb)

    def to_host(self) -> np.ndarray:
        """Download local partition from GPU device to a NumPy array (or copy CPU partition)."""
        if self.handle.backend == Backend.CPU:
            return self.data.copy()
        host_arr = np.empty((self.m_loc, self.n_loc), dtype=self.dtype, order='F')
        elem_size = self.dtype.itemsize
        dpitch = self.m_loc * elem_size
        spitch = self.lld * elem_size
        width = self.m_loc * elem_size
        height = self.n_loc
        if width > 0 and height > 0:
            st = lib.ddla_c_memcpy2d(self.handle.ptr, ctypes.c_void_p(host_arr.ctypes.data),
                                     dpitch, ctypes.c_void_p(self._device_ptr), spitch,
                                     width, height)
            check_status(st, "ddla_c_memcpy2d (to_host)")
        return host_arr

    def from_host(self, host_arr: np.ndarray):
        """Upload local partition from a NumPy array to GPU device (or copy to CPU buffer)."""
        if host_arr.shape != (self.m_loc, self.n_loc):
            raise ValueError(f"Shape {host_arr.shape} != local shape ({self.m_loc}, {self.n_loc})")
        arr = np.asfortranarray(host_arr, dtype=self.dtype)
        if self.handle.backend == Backend.CPU:
            self.data[:, :] = arr
        else:
            elem_size = self.dtype.itemsize
            dpitch = self.lld * elem_size
            spitch = self.m_loc * elem_size
            width = self.m_loc * elem_size
            height = self.n_loc
            if width > 0 and height > 0:
                st = lib.ddla_c_memcpy2d(self.handle.ptr, ctypes.c_void_p(self._device_ptr),
                                         dpitch, ctypes.c_void_p(arr.ctypes.data), spitch,
                                         width, height)
                check_status(st, "ddla_c_memcpy2d (from_host)")

    def to_global_matrix(self, root: int = 0) -> Optional[np.ndarray]:
        """Gather distributed matrix to a global NumPy array on rank `root`."""
        rank = self.handle.rank
        out = None
        dst_ptr = ctypes.c_void_p(None)
        lddst = 1
        if rank == root:
            out = np.zeros((self.m, self.n), dtype=self.dtype, order='F')
            dst_ptr = ctypes.c_void_p(out.ctypes.data)
            lddst = max(1, self.m)

        gather_func = getattr(lib, f"ddla_{self.type_char}_gather")
        st = gather_func(
            self.handle.ptr, self.m, self.n,
            ctypes.c_void_p(self.data_ptr), 1, 1,
            ctypes.c_void_p(self.desc.ctypes.data),
            root, dst_ptr, lddst
        )
        check_status(st, f"ddla_{self.type_char}_gather")
        return out

    @classmethod
    def from_global_matrix(cls, handle: Handle, global_arr: Optional[np.ndarray],
                           mb: int = 64, nb: int = 64, root: int = 0):
        """Scatter global NumPy array from rank `root` across the process grid."""
        rank = handle.rank
        m, n, dtype = 0, 0, np.float64
        if rank == root:
            if global_arr is None:
                raise ValueError("global_arr must be provided on root rank")
            m, n = global_arr.shape
            dtype = global_arr.dtype

        if hasattr(handle.layout, 'comm') and handle.layout.comm is not None:
            m, n, dtype = handle.layout.comm.bcast((m, n, dtype), root=root)
        else:
            from ._c_api import ddla_c_bcast_dims
            cm = ctypes.c_int(m)
            cn = ctypes.c_int(n)
            tc = DTYPE_TO_CHAR.get(np.dtype(dtype), 'd')
            ctc = ctypes.c_int(ord(tc))
            ddla_c_bcast_dims(handle.layout.comm_f, ctypes.byref(cm), ctypes.byref(cn), ctypes.byref(ctc), int(root))
            m, n = cm.value, cn.value
            dtype = CHAR_TO_DTYPE[chr(ctc.value)]

        A = cls(handle, m, n, mb=mb, nb=nb, dtype=dtype)
        src_ptr = ctypes.c_void_p(None)
        ldsrc = 1
        arr = None
        if rank == root:
            arr = np.asfortranarray(global_arr, dtype=A.dtype)
            src_ptr = ctypes.c_void_p(arr.ctypes.data)
            ldsrc = max(1, m)

        scatter_func = getattr(lib, f"ddla_{A.type_char}_scatter")
        st = scatter_func(
            handle.ptr, m, n,
            ctypes.c_void_p(A.data_ptr), 1, 1,
            ctypes.c_void_p(A.desc.ctypes.data),
            root, src_ptr, ldsrc
        )
        check_status(st, f"ddla_{A.type_char}_scatter")
        return A

    def is_local_row(self, global_r: int) -> bool:
        nprows, _ = self.handle.grid_dims
        myprow, _ = self.handle.grid_coords
        return ((global_r // self.mb) % nprows) == myprow

    def is_local_col(self, global_c: int) -> bool:
        _, npcols = self.handle.grid_dims
        _, mypcol = self.handle.grid_coords
        return ((global_c // self.nb) % npcols) == mypcol

    def get_local_index(self, global_r: int, global_c: int) -> Tuple[int, int]:
        nprows, npcols = self.handle.grid_dims
        myprow, mypcol = self.handle.grid_coords
        lr = (global_r // (nprows * self.mb)) * self.mb + (global_r % self.mb)
        lc = (global_c // (npcols * self.nb)) * self.nb + (global_c % self.nb)
        return (lr, lc)

    def get_global_index(self, local_r: int, local_c: int) -> Tuple[int, int]:
        nprows, npcols = self.handle.grid_dims
        myprow, mypcol = self.handle.grid_coords
        gr = ((local_r // self.mb) * nprows + myprow) * self.mb + (local_r % self.mb)
        gc = ((local_c // self.nb) * npcols + mypcol) * self.nb + (local_c % self.nb)
        return (gr, gc)

    def __repr__(self) -> str:
        return f"<DistributedMatrix shape={self.shape} local={self.local_shape} block={self.block_size} dtype={self.dtype} backend={self.handle.backend.name}>"
