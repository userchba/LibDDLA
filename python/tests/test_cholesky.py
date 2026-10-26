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
        A, info, is_nega = ddla.ppotrf(A, uplo='L')
        self.assertEqual(info, 0)
        self.assertFalse(is_nega)

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

        X, info, is_nega = ddla.pposv(A, B, uplo='L')
        self.assertEqual(info, 0)
        self.assertFalse(is_nega)

        X_g = X.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(X_g - X_exact))
            self.assertLess(diff, 1e-11)

if __name__ == '__main__':
    unittest.main()
