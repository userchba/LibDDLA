import unittest
import os
import numpy as np
import ddla

class TestHandle(unittest.TestCase):
    def test_communication_cpu(self):
        with ddla.Handle(backend=ddla.Backend.CPU) as h:
            self.assertEqual(h.communication, ddla.Communication.MPI)
            self.assertTrue(h.is_communication_available(ddla.Communication.MPI))
            self.assertFalse(h.is_communication_available(ddla.Communication.HOST_STAGED_MPI))
            self.assertFalse(h.is_communication_available(ddla.Communication.CCL))
            h.set_communication(ddla.Communication.MPI)
            with self.assertRaises(ddla.DdlaBackendUnavailableError):
                h.set_communication(ddla.Communication.CCL)
            with self.assertRaises(ddla.DdlaInvalidValueError):
                h.set_communication(17)
            self.assertEqual(h.communication, ddla.Communication.MPI)

    def test_communication_mismatched_invalid_request(self):
        with ddla.Handle(backend=ddla.Backend.CPU) as h:
            # Do not prevalidate the enum in Python: peers must all reach the
            # collective rejection even when only one rank requests an invalid id.
            request = 17 if h.size == 1 or h.rank == 0 else ddla.Communication.MPI
            with self.assertRaises(ddla.DdlaInvalidValueError):
                h.set_communication(request)
            self.assertEqual(h.communication, ddla.Communication.MPI)

    @unittest.skipUnless(os.environ.get("DDLA_TEST_GPU_COMMUNICATION") == "1",
                         "GPU communication test requires an allocated GPU per rank")
    def test_communication_gpu(self):
        with ddla.Handle(backend=ddla.Backend.GPU) as h:
            original = h.communication
            self.assertTrue(h.is_communication_available(ddla.Communication.MPI))
            self.assertTrue(h.is_communication_available(ddla.Communication.HOST_STAGED_MPI))
            h.set_communication(ddla.Communication.HOST_STAGED_MPI)
            self.assertEqual(h.communication, ddla.Communication.HOST_STAGED_MPI)
            if h.is_communication_available(ddla.Communication.CCL):
                h.set_communication(ddla.Communication.CCL)
                self.assertEqual(h.communication, ddla.Communication.CCL)
            h.set_communication(original)
            self.assertEqual(h.communication, original)

    def test_backend_is_required(self):
        with self.assertRaises(TypeError):
            ddla.Handle()
        with self.assertRaises(TypeError):
            ddla.Handle(backend=None)

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
