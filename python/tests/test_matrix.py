import unittest
import numpy as np
import ddla

class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.handle = ddla.Handle(backend=ddla.Backend.CPU)

    def tearDown(self):
        self.handle.destroy()

    def test_matrix_creation_and_dtypes(self):
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            A = ddla.DistributedMatrix(self.handle, 50, 40, mb=16, nb=16, dtype=dtype)
            self.assertEqual(A.shape, (50, 40))
            self.assertEqual(A.block_size, (16, 16))
            self.assertEqual(A.dtype, np.dtype(dtype))
            self.assertEqual(A.data.shape, (A.m_loc, A.n_loc))
            self.assertTrue(A.data.flags['F_CONTIGUOUS'])

    def test_index_mapping(self):
        A = ddla.DistributedMatrix(self.handle, 100, 100, mb=16, nb=16, dtype=np.float64)
        for r in range(0, 100, 23):
            for c in range(0, 100, 29):
                if A.is_local_row(r) and A.is_local_col(c):
                    lr, lc = A.get_local_index(r, c)
                    gr, gc = A.get_global_index(lr, lc)
                    self.assertEqual((gr, gc), (r, c))

    def test_gather_scatter_roundtrip(self):
        np.random.seed(101)
        orig = np.random.randn(64, 48)
        A = ddla.DistributedMatrix.from_global_matrix(self.handle, orig, mb=16, nb=16, root=0)
        gathered = A.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(gathered - orig))
            self.assertAlmostEqual(diff, 0.0, places=12)

if __name__ == '__main__':
    unittest.main()
