#!/usr/bin/env python3
"""Development-time generator regression checks; no compiler/GPU required."""
from __future__ import print_function

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/generate_public_api.py'
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location('generate_public_api', str(MODULE_PATH))
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)

HEADER = '''namespace ddla {
template <typename T>
void example(int n, T* a, const DdlaHandle_t& h, int stride = -1);
template <typename T, typename Alpha>
typename std::enable_if<std::is_same<T, std::complex<Alpha>>::value, void>::type
scal(const DdlaHandle_t& h, const Alpha& alpha, T* a);
template <typename T>
void gemmVbatched(const DdlaHandle_t& h, T* const arrays[], int count);
template <typename T>
void write_matrix(const DdlaHandle_t& h, const T* a, int rows, int cols, const char* file);
template <typename T> void pstedc(const DdlaHandle_t& h, T* a);
}'''


def policy(name, scalar, route='direct_backend_forward'):
    return dict(name=name, header_overload=0,
                supported_scalar_arguments=scalar, policy=route)


class PublicApiGenerationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for path in (GEN.HEADER, GEN.MANIFEST, GEN.BACKEND, GEN.DISPATCH):
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / GEN.HEADER).write_text(HEADER)
        self.manifest = dict(manual_public_templates=['pstedc'], overloads=[
            policy('example', ['float', 'double']),
            policy('scal', ['std::complex<float>, float']),
            policy('gemmVbatched', ['double'], 'cpu_only_rejection_without_backend_symbol'),
            policy('write_matrix', ['double'], 'handle_stream_writer')])
        self.write_manifest()

    def write_manifest(self):
        (self.root / GEN.MANIFEST).write_text(json.dumps(self.manifest))

    def run_generator(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = GEN.main(['--root', str(self.root)] + list(args))
        return result, out.getvalue(), err.getvalue()

    def test_nonvoid_workspace_query_preserves_return_in_instantiation(self):
        header = (self.root / GEN.HEADER).read_text()
        header = header.replace('template <typename T> void pstedc',
            'template <typename T> std::size_t scratch(const DdlaHandle_t& h, T* a);\n'
            'template <typename T> void pstedc')
        (self.root / GEN.HEADER).write_text(header)
        self.manifest['overloads'].append(policy('scratch', ['double']))
        self.write_manifest()
        self.assertEqual(self.run_generator()[0], 0)
        dispatcher = (self.root / GEN.DISPATCH).read_text()
        self.assertIn('template std::size_t scratch<double>', dispatcher)
        self.assertEqual(self.run_generator('--check')[0], 0)

    def test_round_trip_check_is_read_only_and_detects_stale_output(self):
        self.assertEqual(self.run_generator()[0], 0)
        paths = [self.root / p for p in (GEN.HEADER, GEN.MANIFEST, GEN.BACKEND, GEN.DISPATCH)]
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
        self.assertEqual(self.run_generator('--check')[0], 0)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])
        dispatch = self.root / GEN.DISPATCH
        dispatch.write_text(dispatch.read_text() + '// stale\n')
        stale = dispatch.read_bytes()
        result, out, err = self.run_generator('--check')
        self.assertEqual(result, 1)
        self.assertIn(str(GEN.DISPATCH), err)
        self.assertEqual(stale, dispatch.read_bytes())

    def test_header_signature_is_authoritative_defaults_and_constraints_survive(self):
        self.run_generator()
        header = (self.root / GEN.HEADER).read_text().replace('int stride = -1', 'int stride = -2')
        (self.root / GEN.HEADER).write_text(header)
        self.assertEqual(self.run_generator('--check')[0], 1)
        self.assertEqual(self.run_generator()[0], 0)
        backend = (self.root / GEN.BACKEND).read_text()
        dispatch = (self.root / GEN.DISPATCH).read_text()
        self.assertIn('int stride = -2', backend)
        self.assertNotIn('= -2', dispatch)
        self.assertIn('std::enable_if<std::is_same<T, std::complex<Alpha>>::value, void>::type scal', backend)
        self.assertIn('template void scal<std::complex<float>, float>', dispatch)
        self.assertIn('if (h && h->backend == DdlaBackend::GPU)', dispatch)
        self.assertIn('example<DdlaBackend::CPU, T>(n, a, h, stride)', dispatch)

    def test_explicit_exceptions_and_scalar_support_are_preserved(self):
        entries, manifest = GEN.read_contract(self.root)
        self.assertEqual(manifest['public_backend_overloads'], 4)
        self.assertEqual(manifest['public_concrete_specializations'], 5)
        self.assertEqual([e['supported_scalar_arguments'] for d, e in entries],
                         [e['supported_scalar_arguments'] for e in self.manifest['overloads']])
        backend = GEN.backend_text(entries)
        dispatch = GEN.dispatch_text(entries)
        self.assertIn('void write_matrix(const T* a, int rows, int cols, const char* file);', backend)
        self.assertNotIn('template void write_matrix', dispatch)
        self.assertNotIn('void pstedc', backend)
        self.assertIn('gemmVbatched<DdlaBackend::GPU, T>(h, arrays, count)', dispatch)
        cpu_only = dispatch.split('#else', 1)[1].split('#endif', 1)[0]
        self.assertNotIn('gemmVbatched<DdlaBackend::', cpu_only)
        self.assertIn('gemmVbatched: invalid DDLA handle', cpu_only)
        self.assertIn('gemmVbatched: only the GPU backend is implemented', cpu_only)

    def test_undeclared_or_missing_policy_fails_before_writing(self):
        self.manifest['overloads'].pop()
        self.write_manifest()
        result, out, err = self.run_generator()
        self.assertEqual(result, 2)
        self.assertIn('missing scalar/routing policy', err)
        self.assertFalse((self.root / GEN.BACKEND).exists())
        self.manifest['overloads'].append(policy('not_declared', ['float']))
        self.write_manifest()
        with self.assertRaises(ValueError):
            GEN.read_contract(self.root)

    def test_unsupported_new_return_type_requires_generator_review(self):
        path = self.root / GEN.HEADER
        path.write_text(path.read_text().replace('void example(', 'int example('))
        result, out, err = self.run_generator()
        self.assertEqual(result, 2)
        self.assertIn('unsupported public return type: int', err)
        self.assertFalse((self.root / GEN.BACKEND).exists())

    def test_duplicate_policy_and_wrong_scalar_arity_are_rejected(self):
        self.manifest['overloads'].append(self.manifest['overloads'][0])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate scalar policy'):
            GEN.read_contract(self.root)
        self.manifest['overloads'].pop()
        self.manifest['overloads'][1]['supported_scalar_arguments'] = ['float']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'wrong number of scalar arguments'):
            GEN.read_contract(self.root)

    def test_duplicate_specializations_unknown_routes_and_stale_manual_names_fail(self):
        original = self.manifest['overloads'][0]['supported_scalar_arguments']
        self.manifest['overloads'][0]['supported_scalar_arguments'] = ['float', 'float']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate scalar specializations'):
            GEN.read_contract(self.root)
        self.manifest['overloads'][0]['supported_scalar_arguments'] = original
        self.manifest['overloads'][0]['policy'] = 'new_unimplemented_route'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'unknown routing policy'):
            GEN.read_contract(self.root)
        self.manifest['overloads'][0]['policy'] = 'direct_backend_forward'
        self.manifest['manual_public_templates'].append('missing_manual')
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'manual public templates missing'):
            GEN.read_contract(self.root)


if __name__ == '__main__':
    unittest.main()
