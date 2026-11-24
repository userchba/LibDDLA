import gc
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

    def assert_cpu_layout(self, A):
        self.assertEqual(A.data.shape, A.local_shape)
        self.assertEqual(A.data.dtype, A.dtype)
        self.assertEqual(A.data_ptr, A.data.ctypes.data)
        self.assertEqual(A.lld, A.desc[ddla.LLD_])
        self.assertEqual(A.data.strides, (A.dtype.itemsize, A.lld * A.dtype.itemsize))
        # Check the allocation before accessing descriptor-sized storage. These
        # regressions must fail safely without calling native matrix operators.
        owner = A.data
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        self.assertGreaterEqual(owner.nbytes, A.lld * A.n_loc * A.dtype.itemsize)
        storage = np.ndarray((A.lld, A.n_loc), dtype=A.dtype, buffer=owner, order='F')
        self.assertEqual(storage.ctypes.data, A.data_ptr)
        return storage

    def test_cpu_padded_storage_and_host_copies(self):
        nprows, npcols = self.handle.grid_dims
        m, n = 5 * nprows, 4 * npcols
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            with self.subTest(dtype=dtype):
                A = ddla.DistributedMatrix(self.handle, m, n, mb=2, nb=2,
                                           dtype=dtype, lld=m + 3)
                self.assertEqual(A.shape, (m, n))
                self.assertEqual(A.lld, m + 3)
                storage = self.assert_cpu_layout(A)
                self.assertFalse(A.data.flags['F_CONTIGUOUS'])
                np.testing.assert_array_equal(A.data, np.zeros(A.local_shape))
                storage[A.m_loc:, :] = -97

                values = np.arange(A.m_loc * A.n_loc).reshape(A.local_shape)
                if np.issubdtype(dtype, np.complexfloating):
                    values = values + 1j * (values + 1)
                values = values[:, ::-1]  # Noncontiguous input, with dtype conversion.
                pointer = A.data_ptr
                A.from_host(values)
                self.assertEqual(A.data_ptr, pointer)
                np.testing.assert_array_equal(A.data, values)
                np.testing.assert_array_equal(A.to_host(), values)

                lr, lc = A.m_loc - 1, A.n_loc - 1
                gr, gc = A.get_global_index(lr, lc)
                self.assertTrue(A.is_local_row(gr) and A.is_local_col(gc))
                self.assertEqual(A.get_local_index(gr, gc), (lr, lc))
                A.data[lr, lc] = 42
                self.assertEqual(storage[lr, lc], 42)
                storage[0, 1] = 17
                self.assertEqual(A.data[0, 1], 17)
                copied = A.to_host()
                self.assertEqual(copied.shape, A.local_shape)
                self.assertFalse(np.shares_memory(copied, A.data))
                copied[:, :] = -1
                self.assertEqual(A.data[lr, lc], 42)
                np.testing.assert_array_equal(storage[A.m_loc:, :], -97)

    def test_cpu_padded_storage_with_data(self):
        nprows, npcols = self.handle.grid_dims
        m, n = 5 * nprows, 4 * npcols
        compact = ddla.DistributedMatrix(self.handle, m, n, mb=2, nb=2)
        values = np.arange(compact.m_loc * compact.n_loc).reshape(compact.local_shape)
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            source = values.astype(np.float64)
            if np.issubdtype(dtype, np.complexfloating):
                source = source + 1j * (values + 1)
            expected = source.astype(dtype)
            for layout, data in [
                ('C', np.array(source, order='C')),
                ('F', np.array(source, order='F')),
                ('strided', np.repeat(source, 2, axis=1)[:, ::2]),
            ]:
                with self.subTest(dtype=dtype, layout=layout):
                    A = ddla.DistributedMatrix(self.handle, m, n, mb=2, nb=2,
                                               dtype=dtype, lld=compact.lld + 3, data=data)
                    storage = self.assert_cpu_layout(A)
                    np.testing.assert_array_equal(A.data, expected)
                    np.testing.assert_array_equal(A.to_host(), expected)
                    np.testing.assert_array_equal(storage[A.m_loc:, :], 0)
                    self.assertFalse(np.shares_memory(A.data, data))
                    view = A.data
                    del A, storage
                    gc.collect()
                    np.testing.assert_array_equal(view, expected)
                    view[-1, -1] = -1
                    np.testing.assert_array_equal(data, expected)

    def test_cpu_compact_data_compatibility(self):
        nprows, npcols = self.handle.grid_dims
        m, n = 5 * nprows, 4 * npcols
        compact = ddla.DistributedMatrix(self.handle, m, n, mb=2, nb=2)
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for lld in [-1, 0, compact.m_loc]:
                with self.subTest(dtype=dtype, lld=lld):
                    data = np.ones(compact.local_shape, dtype=dtype, order='F')
                    A = ddla.DistributedMatrix(self.handle, m, n, mb=2, nb=2,
                                               dtype=dtype, lld=lld, data=data)
                    self.assertEqual(A.lld, compact.m_loc)
                    self.assert_cpu_layout(A)
                    self.assertTrue(A.data.flags['F_CONTIGUOUS'])
                    self.assertTrue(np.shares_memory(A.data, data))
                    A.from_host(np.full(A.local_shape, 3))
                    np.testing.assert_array_equal(data, 3)
                    np.testing.assert_array_equal(A.to_host(), data)

    def test_cpu_empty_local_storage(self):
        for m, n in [(0, 0), (0, 4), (4, 0), (1, 1)]:
            for lld in [-1, 7]:
                with self.subTest(shape=(m, n), lld=lld):
                    A = ddla.DistributedMatrix(self.handle, m, n, mb=1, nb=1, lld=lld)
                    self.assertEqual(A.data.shape, A.local_shape)
                    values = np.ones(A.local_shape)
                    A.from_host(values)
                    np.testing.assert_array_equal(A.to_host(), values)
                    B = ddla.DistributedMatrix(self.handle, m, n, mb=1, nb=1,
                                               lld=lld, data=values)
                    self.assertEqual(B.data.shape, B.local_shape)
                    np.testing.assert_array_equal(B.to_host(), values)

    def test_cpu_local_data_shape_validation(self):
        for lld in [-1, 20]:
            with self.subTest(lld=lld):
                A = ddla.DistributedMatrix(self.handle, 5, 4, mb=2, nb=2, lld=lld)
                wrong_shape = np.zeros((A.m_loc + 1, A.n_loc))
                with self.assertRaises(ValueError):
                    ddla.DistributedMatrix(self.handle, 5, 4, mb=2, nb=2,
                                           lld=lld, data=wrong_shape)
                with self.assertRaises(ValueError):
                    A.from_host(wrong_shape)

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
