import unittest
import numpy as np
import ddla

class TestLuQr(unittest.TestCase):
    def setUp(self):
        self.handle = ddla.Handle(backend=ddla.Backend.CPU)

    def tearDown(self):
        self.handle.destroy()

    def test_pgesv(self):
        np.random.seed(67)
        n = 32
        nb = 16
        A_orig = np.random.randn(n, n) + np.eye(n) * 4.0
        X_exact = np.random.randn(n, 2)
        B_orig = A_orig @ X_exact

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_orig, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_orig, mb=nb, nb=nb)

        X, ipiv, info = ddla.pgesv(A, B)
        self.assertEqual(info, 0)

        X_g = X.to_global_matrix(root=0)
        if self.handle.rank == 0:
            diff = np.max(np.abs(X_g - X_exact))
            self.assertLess(diff, 1e-11)

    def test_pgeqrf(self):
        np.random.seed(68)
        m, n = 32, 24
        nb = 16
        A_orig = np.random.randn(m, n)
        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_orig, mb=nb, nb=nb)

        A_fact, tau, info = ddla.pgeqrf(A)
        self.assertEqual(info, 0)
        self.assertEqual(len(tau), min(m, n))
        self.assertTrue(np.all(np.isfinite(tau)))

    def test_pgeadd_and_pdam(self):
        np.random.seed(69)
        n = 20
        nb = 10
        A_orig = np.random.randn(n, n)
        B_orig = np.random.randn(n, n)

        A = ddla.DistributedMatrix.from_global_matrix(self.handle, A_orig, mb=nb, nb=nb)
        B = ddla.DistributedMatrix.from_global_matrix(self.handle, B_orig, mb=nb, nb=nb)

        # C := 2*A + 3*B
        C = ddla.pgeadd(A, B, alpha=2.0, beta=3.0)
        C_g = C.to_global_matrix(root=0)

        if self.handle.rank == 0:
            expected = 2.0 * A_orig + 3.0 * B_orig
            diff = np.max(np.abs(C_g - expected))
            self.assertLess(diff, 1e-12)

        # Add 0.5 to diagonal of C
        ddla.pdam(C, alpha=0.5)
        C_scaled = C.to_global_matrix(root=0)
        if self.handle.rank == 0:
            np.fill_diagonal(expected, expected.diagonal() + 0.5)
            diff = np.max(np.abs(C_scaled - expected))
            self.assertLess(diff, 1e-12)

if __name__ == '__main__':
    unittest.main()
