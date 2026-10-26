"""Processor layout management for 2D process grids."""
import math
from typing import Optional, Tuple

class ProcessorLayout:
    """Rectangular 2D processor grid layout for distributed linear algebra.
    
    Inspired by ELPA (pyelpa in LibRPA) and ScaLAPACK grid conventions.
    Wraps an MPI communicator (mpi4py or native LibDDLA C-ABI) and factors the
    process count into np_rows x np_cols with rank coordinates (my_prow, my_pcol).
    """
    def __init__(self, comm=None, np_rows: Optional[int] = None, np_cols: Optional[int] = None, major: str = 'R'):
        self.major = major.upper()
        if self.major not in ('R', 'C'):
            raise ValueError(f"Major must be 'R' (row-major) or 'C' (column-major), got {major}")
            
        self.comm = comm
        self._comm_f = None
        
        if comm is not None and hasattr(comm, 'py2f'):
            # mpi4py communicator
            self._comm_f = comm.py2f()
            self.nprocs = comm.Get_size()
            self.rank = comm.Get_rank()
        elif isinstance(comm, int):
            self._comm_f = comm
            from ._c_api import ddla_c_comm_rank, ddla_c_comm_size
            self.rank = ddla_c_comm_rank(self._comm_f)
            self.nprocs = ddla_c_comm_size(self._comm_f)
        else:
            from ._c_api import ddla_c_default_comm, ddla_c_comm_rank, ddla_c_comm_size
            self._comm_f = ddla_c_default_comm()
            self.rank = ddla_c_comm_rank(self._comm_f)
            self.nprocs = ddla_c_comm_size(self._comm_f)

        if np_rows is not None and np_cols is not None:
            if np_rows * np_cols > self.nprocs:
                raise ValueError(f"Grid size {np_rows}x{np_cols}={np_rows*np_cols} exceeds communicator size {self.nprocs}")
            self.np_rows = int(np_rows)
            self.np_cols = int(np_cols)
        elif np_rows is not None:
            self.np_rows = int(np_rows)
            self.np_cols = int(self.nprocs // self.np_rows)
        elif np_cols is not None:
            self.np_cols = int(np_cols)
            self.np_rows = int(self.nprocs // self.np_cols)
        else:
            best_r = 1
            for r in range(int(math.sqrt(self.nprocs)), 0, -1):
                if self.nprocs % r == 0:
                    best_r = r
                    break
            self.np_rows = best_r
            self.np_cols = self.nprocs // best_r

        if self.major == 'R':
            self.my_prow = self.rank // self.np_cols if self.rank < self.np_rows * self.np_cols else -1
            self.my_pcol = self.rank % self.np_cols if self.rank < self.np_rows * self.np_cols else -1
        else:
            self.my_prow = self.rank % self.np_rows if self.rank < self.np_rows * self.np_cols else -1
            self.my_pcol = self.rank // self.np_rows if self.rank < self.np_rows * self.np_cols else -1

    @property
    def comm_f(self) -> int:
        return self._comm_f

    def __repr__(self) -> str:
        return f"<ProcessorLayout grid={self.np_rows}x{self.np_cols} rank={self.rank}/{self.nprocs} coord=({self.my_prow},{self.my_pcol})>"
