import unittest
import numpy as np
import ddla

class TestCholesky(unittest.TestCase):
    def setUp(self):
        self.handle = ddla.Handle(backend=ddla.Backend.CPU)

    def tearDown(self):
        self.handle.destroy()

    def test_ppotrf_and_ppotrs(self):
        np.random.seed(55)
        n = 32
        nb = 16
        M = np.random.randn(n, n)
        A_orig = M @ M.T + np.eye(n) * 10.0
        X_exact = np.random.randn(n, 4)
        B_orig = A_orig @ X_exact

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_orig, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_orig, mb=nb, nb=nb)

        # Cholesky factorization
        A, info = ddla.ppotrf(A, uplo='L')
        self.assertEqual(info, 0)

        # Cholesky solve
        X = ddla.ppotrs(A, B, uplo='L')
        X_g = X.to_global_matrix(root=0)

        if self.handle.rank == 0:
            diff = np.max(np.abs(X_g - X_exact))
            self.assertLess(diff, 1e-11)

    def test_pposv_driver(self):
        np.random.seed(56)
        n = 24
        nb = 12
        M = np.random.randn(n, n)
        A_orig = M @ M.T + np.eye(n) * 5.0
        X_exact = np.random.randn(n, 2)
        B_orig = A_orig @ X_exact

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_orig, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_orig, mb=nb, nb=nb)

        X, info = ddla.pposv(A, B, uplo='L')
        self.assertEqual(info, 0)

        X_g = X.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(X_g - X_exact))
            self.assertLess(diff, 1e-11)

    def test_all_types_and_triangles(self):
        rng = np.random.RandomState(57)
        n = 15
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for uplo in ('L', 'U'):
                M = rng.randn(n, n)
                X_exact = rng.randn(n, 2)
                if np.issubdtype(dtype, np.complexfloating):
                    M = M + 1j * rng.randn(n, n)
                    X_exact = X_exact + 1j * rng.randn(n, 2)
                M = M.astype(dtype)
                X_exact = X_exact.astype(dtype)
                original = (M @ M.conj().T + np.eye(n, dtype=dtype) * 10).astype(dtype)
                rhs = original @ X_exact
                A = ddla.DistributedMatrix.from_global_matrix(self.handle, original, mb=3, nb=3)
                B = ddla.DistributedMatrix.from_global_matrix(self.handle, rhs, mb=3, nb=3)
                A, info = ddla.ppotrf(A, uplo=uplo)
                self.assertEqual(info, 0)
                X = ddla.ppotrs(A, B, uplo=uplo).to_global_matrix(root=0)
                if self.handle.rank == 0:
                    tol = 2e-4 if dtype in (np.float32, np.complex64) else 1e-11
                    np.testing.assert_allclose(X, X_exact, atol=tol, rtol=tol)

    def test_driver_rejects_final_negative_pivot(self):
        n = 9
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for uplo in ('L', 'U'):
                original = np.eye(n, dtype=dtype)
                original[-1, -1] = -1
                rhs = np.ones((n, 2), dtype=dtype)
                A = ddla.DistributedMatrix.from_global_matrix(self.handle, original, mb=3, nb=3)
                B = ddla.DistributedMatrix.from_global_matrix(self.handle, rhs, mb=3, nb=3)
                X, info = ddla.pposv(A, B, uplo=uplo)
                self.assertEqual(info, n)
                unchanged = X.to_global_matrix(root=0)
                if self.handle.rank == 0:
                    np.testing.assert_array_equal(unchanged, rhs)

if __name__ == '__main__':
    unittest.main()
