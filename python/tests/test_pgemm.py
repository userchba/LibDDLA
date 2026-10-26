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
