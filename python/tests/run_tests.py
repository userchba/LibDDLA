#!/usr/bin/env python3
import sys
import os
import unittest
import ddla

def run_all():
    here = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=here, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    try:
        ddla.finalize()
    except Exception:
        pass
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    run_all()
