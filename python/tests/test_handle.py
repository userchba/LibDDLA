import unittest
import numpy as np
import ddla

class TestHandle(unittest.TestCase):
    def test_handle_lifecycle(self):
        h = ddla.Handle(backend=ddla.Backend.CPU)
        self.assertEqual(h.backend, ddla.Backend.CPU)
        dims = h.grid_dims
        self.assertGreaterEqual(dims[0], 1)
        self.assertGreaterEqual(dims[1], 1)
        coords = h.grid_coords
        self.assertGreaterEqual(coords[0], 0)
        self.assertGreaterEqual(coords[1], 0)
        self.assertGreaterEqual(h.rank, 0)
        self.assertGreaterEqual(h.size, 1)
        h.destroy()

    def test_context_manager(self):
        with ddla.Handle(backend=ddla.Backend.CPU) as h:
            self.assertEqual(h.backend, ddla.Backend.CPU)

    def test_numloc(self):
        with ddla.Handle(backend=ddla.Backend.CPU) as h:
            # numloc(n=100, nb=16, iproc=0, srcproc=0, nprocs=2) -> 52
            loc = h.numloc(100, 16, 0, 0, 2)
            self.assertEqual(loc, 52)
            loc1 = h.numloc(100, 16, 1, 0, 2)
            self.assertEqual(loc1, 48)

    def test_descinit(self):
        with ddla.Handle(backend=ddla.Backend.CPU) as h:
            desc = np.zeros(9, dtype=np.int32)
            h.descinit(desc, 100, 100, 16, 16, 0, 0, -1)
            self.assertEqual(desc[ddla.DTYPE_], 1)
            self.assertEqual(desc[ddla.M_], 100)
            self.assertEqual(desc[ddla.N_], 100)
            self.assertEqual(desc[ddla.MB_], 16)
            self.assertEqual(desc[ddla.NB_], 16)
            self.assertGreaterEqual(desc[ddla.LLD_], 1)

if __name__ == '__main__':
    unittest.main()
