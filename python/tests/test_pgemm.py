import unittest
import numpy as np
import ddla

class TestPgemm(unittest.TestCase):
    def setUp(self):
        self.handle = ddla.Handle(backend=ddla.Backend.CPU)

    def tearDown(self):
        self.handle.destroy()

    def test_pgemm_double(self):
        np.random.seed(42)
        m, n, k = 32, 24, 40
        nb = 16
        A_g = np.random.randn(m, k)
        B_g = np.random.randn(k, n)
        C_expected = A_g @ B_g

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_g, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_g, mb=nb, nb=nb)
        C = ddla.pgemm(A, B)

        C_g = C.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(C_g - C_expected))
            self.assertLess(diff, 1e-12)

    def test_new_output_ignores_beta(self):
        source = np.eye(8)
        A = ddla.scatter(self.handle, source, mb=2, nb=2)
        result = ddla.gather(ddla.pgemm(A, A, beta=np.nan))
        if self.handle.rank == 0:
            np.testing.assert_array_equal(result, source)

    def test_operand_validation_precedes_factorization(self):
        matrix = np.diag([4., 9., 16., 25.])
        A = ddla.scatter(self.handle, matrix, mb=2, nb=2)
        B = ddla.scatter(self.handle, matrix.astype(np.float32), mb=2, nb=2)
        initial = A.to_host()
        with self.assertRaises(TypeError):
            ddla.pposv(A, B if self.handle.rank == 0 else A)
        for solve in (ddla.pgesv, ddla.pposv, ddla.ppotrs, ddla.ptrtrs, ddla.ptrmm):
            with self.subTest(solve=solve.__name__), self.assertRaises(TypeError):
                solve(A, B)
        np.testing.assert_array_equal(A.to_host(), initial)
        for operation in (ddla.pgemm, ddla.pgeadd):
            with self.subTest(operation=operation.__name__), self.assertRaises(TypeError):
                operation(A, A, C=B)
        with ddla.Handle(backend=ddla.Backend.CPU) as other_handle:
            other = ddla.scatter(other_handle, matrix, mb=2, nb=2)
            with self.assertRaisesRegex(ValueError, "same handle"):
                ddla.pgemm(A, other)
            with self.assertRaisesRegex(ValueError, "same handle"):
                ddla.pgeadd(A, A, C=other)

    def test_pgemm_complex(self):
        np.random.seed(43)
        m, n, k = 24, 24, 24
        nb = 12
        A_g = np.random.randn(m, k) + 1j * np.random.randn(m, k)
        B_g = np.random.randn(k, n) + 1j * np.random.randn(k, n)
        C_expected = 2.0 * (A_g @ B_g)

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_g, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_g, mb=nb, nb=nb)
        C = ddla.pgemm(A, B, alpha=2.0)

        C_g = C.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(C_g - C_expected))
            self.assertLess(diff, 1e-12)

    def test_pgemm_transposed(self):
        np.random.seed(44)
        m, n, k = 20, 25, 30
        nb = 10
        A_g = np.random.randn(k, m)
        B_g = np.random.randn(k, n)
        C_expected = A_g.T @ B_g

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_g, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_g, mb=nb, nb=nb)
        C = ddla.pgemm(A, B, transa='T', transb='N')

        C_g = C.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(C_g - C_expected))
            self.assertLess(diff, 1e-12)

if __name__ == '__main__':
    unittest.main()
