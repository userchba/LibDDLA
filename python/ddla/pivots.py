"""Backend-resident integer pivots for distributed LU."""
import numpy as np
from .backend import Backend


class PivotVector:
    """Own an int32 pivot vector on the handle's CPU or GPU backend.

    Entries use the matrix descriptor's local row indexing. Keep this object
    alive through factorization and solves, and close it before the handle.
    """
    def __init__(self, handle, size):
        self.handle = handle
        self.size = int(size)
        self._closed = False
        if self.size < 0:
            raise ValueError("pivot vector size must be nonnegative")
        self._device_ptr = None
        self._host = None
        if handle.backend == Backend.CPU:
            self._host = np.zeros(self.size, dtype=np.int32)
            self.data_ptr = self._host.ctypes.data
        else:
            self._device_ptr = handle.malloc(self.size * 4) if self.size else None
            self.data_ptr = self._device_ptr or 0
            if self.size:
                handle.memset(self.data_ptr, 0, self.size * 4)

    def __len__(self):
        return self.size

    def to_host(self):
        if self._closed:
            raise ValueError("pivot vector is closed")
        if self._host is not None:
            return self._host.copy()
        result = np.empty(self.size, dtype=np.int32)
        if self.size:
            self.handle.memcpy(result, self.data_ptr, result.nbytes)
            self.handle.synchronize()
        return result

    def from_host(self, values):
        if self._closed:
            raise ValueError("pivot vector is closed")
        source = np.ascontiguousarray(values, dtype=np.int32)
        if source.shape != (self.size,):
            raise ValueError("pivot vector shape mismatch")
        if self._host is not None:
            self._host[:] = source
        elif self.size:
            self.handle.memcpy(self.data_ptr, source, source.nbytes)
            self.handle.synchronize()  # The temporary host source may now be released.

    def close(self):
        if self._device_ptr is not None:
            self.handle.free(self._device_ptr)
            self._device_ptr = None
        self._host = None
        self.data_ptr = 0
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
