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
        self._device_ptr = None
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
        if handle.backend == Backend.CPU:
            if data is not None and data.shape != self.local_shape:
                raise ValueError(f"Provided data shape {data.shape} does not match expected local shape {self.local_shape}")
            if data is not None and self.lld == self.m_loc:
                self.data = np.require(data, dtype=self.dtype, requirements=['F', 'A', 'W'])
            else:
                # Native calls use LLD as the distance between column starts.
                rows = self.lld if self.m_loc else 0
                storage = np.zeros((rows, self.n_loc), dtype=self.dtype, order='F')
                self.data = storage[:self.m_loc, :]
                if data is not None:
                    self.data[...] = data
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
        if self.handle.backend == Backend.CPU:
            self.data[...] = host_arr
        else:
            arr = np.asfortranarray(host_arr, dtype=self.dtype)
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

    def _global_transfer(self, root, operation, array=None):
        # Native gather/scatter access root storage in the handle's backend.
        # Keep both the NumPy array and any device staging alive until completion.
        on_root = self.handle.rank == root
        upload = operation == 'scatter'
        device = None
        ptr = None
        preparation_error = None
        try:
            if on_root:
                try:
                    if upload:
                        array = np.asfortranarray(array)
                    else:
                        array = np.empty(self.shape, dtype=self.dtype, order='F')
                    if self.handle.backend == Backend.GPU and array.nbytes:
                        device = self.handle.malloc(array.nbytes)
                        ptr = device
                        if upload:
                            self.handle.memcpy(device, array, array.nbytes)
                    else:
                        ptr = array.ctypes.data
                except Exception as error:
                    # A null root pointer makes native validation fail on all
                    # ranks. Still enter the collective so peers can return.
                    preparation_error = error
                    ptr = None

            name = f"ddla_{self.type_char}_{operation}"
            st = getattr(lib, name)(
                self.handle.ptr, self.m, self.n,
                ctypes.c_void_p(self.data_ptr), 1, 1,
                ctypes.c_void_p(self.desc.ctypes.data),
                root, ctypes.c_void_p(ptr), max(1, self.m)
            )
            if preparation_error is not None:
                raise preparation_error
            check_status(st, name)
            if device is not None and not upload:
                self.handle.memcpy(array, device, array.nbytes)
            return array if on_root and not upload else None
        finally:
            if device is not None:
                try:
                    self.handle.synchronize()
                finally:
                    self.handle.free(device)

    def to_global_matrix(self, root: int = 0) -> Optional[np.ndarray]:
        """Gather distributed matrix to a global NumPy array on rank `root`."""
        return self._global_transfer(root, 'gather')

    @classmethod
    def from_global_matrix(cls, handle: Handle, global_arr: Optional[np.ndarray],
                           mb: int = 64, nb: int = 64, root: int = 0):
        """Scatter global NumPy array from rank `root` across the process grid."""
        if root < 0 or root >= handle.size:
            raise ValueError("root must be a rank in the handle communicator")
        m, n, type_code = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if handle.rank == root and isinstance(global_arr, np.ndarray) and global_arr.ndim == 2:
            rows, cols = global_arr.shape
            if max(rows, cols) <= np.iinfo(np.int32).max:
                m.value, n.value = rows, cols
                type_char = DTYPE_TO_CHAR.get(global_arr.dtype)
                type_code.value = ord(type_char) if type_char else -1

        # Every layout supplies a Fortran communicator, including mpi4py
        # layouts. Broadcast invalid root input along with valid metadata.
        st = lib.ddla_c_bcast_dims(handle.layout.comm_f, ctypes.byref(m),
                                   ctypes.byref(n), ctypes.byref(type_code), root)
        check_status(st, "ddla_c_bcast_dims")
        if type_code.value == 0:
            raise ValueError("root must provide a 2D NumPy array with dimensions representable as int32")
        if type_code.value == -1:
            raise TypeError("unsupported dtype for the root matrix")
        A = cls(handle, m.value, n.value, mb=mb, nb=nb, dtype=CHAR_TO_DTYPE[chr(type_code.value)])
        A._global_transfer(root, 'scatter', global_arr)
        return A

    def is_local_row(self, global_r: int) -> bool:
        nprows, _ = self.handle.grid_dims
        myprow, _ = self.handle.grid_coords
        return (self.irsrc + global_r // self.mb) % nprows == myprow

    def is_local_col(self, global_c: int) -> bool:
        _, npcols = self.handle.grid_dims
        _, mypcol = self.handle.grid_coords
        return (self.icsrc + global_c // self.nb) % npcols == mypcol

    def get_local_index(self, global_r: int, global_c: int) -> Tuple[int, int]:
        nprows, npcols = self.handle.grid_dims
        lr = (global_r // (nprows * self.mb)) * self.mb + (global_r % self.mb)
        lc = (global_c // (npcols * self.nb)) * self.nb + (global_c % self.nb)
        return (lr, lc)

    def get_global_index(self, local_r: int, local_c: int) -> Tuple[int, int]:
        nprows, npcols = self.handle.grid_dims
        myprow, mypcol = self.handle.grid_coords
        row_offset = (myprow - self.irsrc) % nprows
        col_offset = (mypcol - self.icsrc) % npcols
        gr = ((local_r // self.mb) * nprows + row_offset) * self.mb + (local_r % self.mb)
        gc = ((local_c // self.nb) * npcols + col_offset) * self.nb + (local_c % self.nb)
        return (gr, gc)

    def __repr__(self) -> str:
        return f"<DistributedMatrix shape={self.shape} local={self.local_shape} block={self.block_size} dtype={self.dtype} backend={self.handle.backend.name}>"
