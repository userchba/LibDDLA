"""LibDDLA Backend and Handle management."""
import enum
import ctypes
from typing import Optional, Tuple
from ._c_api import (
    lib, check_status,
    STATUS_SUCCESS, BACKEND_CPU, BACKEND_GPU,
    ddla_c_init, ddla_c_init_backend, ddla_c_set, ddla_c_destroy,
    ddla_c_backend_available, ddla_c_get_backend,
    ddla_c_grid_dims, ddla_c_grid_coords, ddla_c_rank, ddla_c_size,
    ddla_c_descinit, ddla_c_numloc, ddla_c_malloc, ddla_c_free,
    ddla_c_memcpy, ddla_c_memcpy2d, ddla_c_memset, ddla_c_synchronize
)
from .layout import ProcessorLayout

class Backend(enum.IntEnum):
    CPU = BACKEND_CPU
    GPU = BACKEND_GPU

class Handle:
    """Manages the LibDDLA handle lifecycle, process grid, backend execution, and device streams."""
    def __init__(self, backend: Optional[Backend] = None, layout: Optional[ProcessorLayout] = None,
                 comm=None, nprows: Optional[int] = None, npcols: Optional[int] = None, major: str = 'R'):
        self._handle = ctypes.c_void_p()
        if backend is not None:
            st = ddla_c_init_backend(ctypes.byref(self._handle), int(backend))
            check_status(st, "ddla_c_init_backend")
        else:
            st = ddla_c_init(ctypes.byref(self._handle))
            check_status(st, "ddla_c_init")

        if layout is None:
            layout = ProcessorLayout(comm=comm, np_rows=nprows, np_cols=npcols, major=major)
        self.layout = layout
        st = ddla_c_set(self._handle, layout.comm_f, layout.np_rows, layout.np_cols, ord(layout.major))
        check_status(st, "ddla_c_set")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    def destroy(self):
        if self._handle and self._handle.value is not None:
            ddla_c_destroy(ctypes.byref(self._handle))
            self._handle = ctypes.c_void_p()

    def __del__(self):
        self.destroy()

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._handle

    @property
    def backend(self) -> Backend:
        be = ddla_c_get_backend(self._handle)
        return Backend(be)

    @staticmethod
    def is_backend_available(backend: Backend) -> bool:
        return bool(ddla_c_backend_available(int(backend)))

    @property
    def grid_dims(self) -> Tuple[int, int]:
        r, c = ctypes.c_int(), ctypes.c_int()
        st = ddla_c_grid_dims(self._handle, ctypes.byref(r), ctypes.byref(c))
        check_status(st, "ddla_c_grid_dims")
        return (r.value, c.value)

    @property
    def grid_coords(self) -> Tuple[int, int]:
        r, c = ctypes.c_int(), ctypes.c_int()
        st = ddla_c_grid_coords(self._handle, ctypes.byref(r), ctypes.byref(c))
        check_status(st, "ddla_c_grid_coords")
        return (r.value, c.value)

    @property
    def rank(self) -> int:
        return ddla_c_rank(self._handle)

    @property
    def size(self) -> int:
        return ddla_c_size(self._handle)

    def synchronize(self):
        st = ddla_c_synchronize(self._handle)
        check_status(st, "ddla_c_synchronize")

    def numloc(self, n: int, nb: int, iproc: int, srcproc: int = 0, nprocs: Optional[int] = None) -> int:
        if nprocs is None:
            nprocs = self.size
        return ddla_c_numloc(int(n), int(nb), int(iproc), int(srcproc), int(nprocs))

    def descinit(self, desc, m: int, n: int, mb: int, nb: int, irsrc: int = 0, icsrc: int = 0, lld: int = -1):
        if hasattr(desc, "ctypes"):
            desc_ptr = ctypes.c_void_p(desc.ctypes.data)
        else:
            desc_ptr = ctypes.byref(desc)
        st = ddla_c_descinit(desc_ptr, self._handle, int(m), int(n), int(mb), int(nb), int(irsrc), int(icsrc), int(lld))
        check_status(st, "ddla_c_descinit")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        st = ddla_c_malloc(self._handle, ctypes.byref(ptr), int(nbytes))
        check_status(st, "ddla_c_malloc")
        return ptr.value

    def free(self, ptr: int):
        if ptr is not None:
            st = ddla_c_free(self._handle, ctypes.c_void_p(ptr))
            check_status(st, "ddla_c_free")

    def memcpy(self, dst, src, nbytes: int):
        d_ptr = dst if isinstance(dst, int) else dst.ctypes.data
        s_ptr = src if isinstance(src, int) else src.ctypes.data
        st = ddla_c_memcpy(self._handle, ctypes.c_void_p(d_ptr), ctypes.c_void_p(s_ptr), int(nbytes))
        check_status(st, "ddla_c_memcpy")

    def memcpy2d(self, dst, dpitch: int, src, spitch: int, width: int, height: int):
        d_ptr = dst if isinstance(dst, int) else dst.ctypes.data
        s_ptr = src if isinstance(src, int) else src.ctypes.data
        st = ddla_c_memcpy2d(self._handle, ctypes.c_void_p(d_ptr), int(dpitch),
                             ctypes.c_void_p(s_ptr), int(spitch), int(width), int(height))
        check_status(st, "ddla_c_memcpy2d")

    def memset(self, ptr, value: int, nbytes: int):
        p = ptr if isinstance(ptr, int) else ptr.ctypes.data
        st = ddla_c_memset(self._handle, ctypes.c_void_p(p), int(value), int(nbytes))
        check_status(st, "ddla_c_memset")

    def __repr__(self) -> str:
        return f"<Handle backend={self.backend.name} rank={self.rank}/{self.size} grid={self.grid_dims}>"
