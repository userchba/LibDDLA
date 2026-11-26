import os
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

    def test_padded_shifted_partition(self):
        m, n, mb, nb = 13, 11, 4, 3
        pr, pc = self.handle.grid_dims
        r, c = self.handle.grid_coords
        rs, cs = pr - 1, pc - 1
        rows = [i for i in range(m) if (rs + i // mb) % pr == r]
        cols = [j for j in range(n) if (cs + j // nb) % pc == c]
        global_data = np.arange(m * n, dtype=np.int32).reshape(m, n)
        local_data = global_data[np.ix_(rows, cols)]
        lld = len(rows) + 5

        A = ddla.DistributedMatrix(self.handle, m, n, mb, nb, np.float64,
                                   rs, cs, lld, data=local_data)
        self.assertEqual(A.local_shape, local_data.shape)
        self.assertEqual(A.data.strides[1], lld * A.dtype.itemsize)
        np.testing.assert_array_equal(A.to_host(), local_data)
        A.from_host(local_data + 1)
        result = A.to_global_matrix()
        if self.handle.rank == 0:
            np.testing.assert_array_equal(result, global_data + 1)
        for i in range(m):
            self.assertEqual(A.is_local_row(i), i in rows)
        for j in range(n):
            self.assertEqual(A.is_local_col(j), j in cols)
        for lr, i in enumerate(rows):
            for lc, j in enumerate(cols):
                self.assertEqual(A.get_local_index(i, j), (lr, lc))
                self.assertEqual(A.get_global_index(lr, lc), (i, j))

    def test_gather_scatter_roundtrip(self):
        np.random.seed(101)
        orig = np.random.randn(64, 48)
        A = ddla.DistributedMatrix.from_global_matrix(self.handle, orig, mb=16, nb=16, root=0)
        gathered = A.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(gathered - orig))
            self.assertAlmostEqual(diff, 0.0, places=12)

    def _check_scatter(self, handle):
        root = handle.size - 1
        original = np.arange(63, dtype=np.float64).reshape(7, 9) if handle.rank == root else None
        A = ddla.scatter(handle, original, mb=3, nb=2, root=root)
        gathered = ddla.gather(A, root=root)
        if handle.rank == root:
            np.testing.assert_array_equal(gathered, original)
        else:
            self.assertIsNone(gathered)

    def test_integer_communicator_scatter(self):
        with ddla.Handle(backend=ddla.Backend.CPU, comm=self.handle.layout.comm_f) as handle:
            self._check_scatter(handle)

    def test_invalid_root_input(self):
        root = self.handle.size - 1
        for array, error in ((None, ValueError), (np.zeros(3), ValueError),
                             (np.zeros((2, 2), dtype=np.int32), TypeError)):
            with self.subTest(array=array), self.assertRaises(error):
                ddla.scatter(self.handle, array if self.handle.rank == root else None, root=root)

    @unittest.skipUnless(os.getenv("DDLA_PYTHON_TEST_GPU") == "1", "requires allocated GPUs")
    def test_gpu_numpy_roundtrip(self):
        with ddla.Handle(backend=ddla.Backend.GPU) as handle:
            self._check_scatter(handle)

    def test_packed_input_storage(self):
        base = ddla.DistributedMatrix(self.handle, 13, 11, mb=3, nb=2)
        data = np.ones(base.local_shape, dtype=base.dtype, order='F')
        borrowed = ddla.DistributedMatrix(self.handle, 13, 11, 3, 2, data=data)
        if data.size:
            self.assertTrue(np.shares_memory(borrowed.data, data))

        data.setflags(write=False)
        writable = ddla.DistributedMatrix(self.handle, 13, 11, 3, 2, data=data)
        writable.from_host(np.zeros_like(data))
        np.testing.assert_array_equal(data, np.ones_like(data))
        np.testing.assert_array_equal(writable.to_host(), np.zeros_like(data))

if __name__ == '__main__':
    unittest.main()
