import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL))
import endpoint
import relay
import report


class ProtocolTests(unittest.TestCase):
    def test_rejects_commands_outside_protocol(self):
        for command in ['sh -c id', 'libddla-ci status ../../home',
                        'libddla-ci cancel 123;id', 'libddla-ci submit gh-123-1 gpu8',
                        'libddla-ci status gh-123-1 extra']:
            with self.subTest(command=command), self.assertRaises(ValueError):
                endpoint.parse_command(command)

    def test_input_size_limit_and_digest(self):
        out = io.BytesIO()
        self.assertEqual(endpoint.bounded_copy(io.BytesIO(b'abc'), out, 3), hashlib.sha256(b'abc').hexdigest())
        with self.assertRaises(ValueError):
            endpoint.bounded_copy(io.BytesIO(b'abcd'), io.BytesIO(), 3)

    def test_slurm_environment_cannot_request_memory(self):
        with patch.dict(os.environ, {'SBATCH_MEM_PER_NODE': '1T', 'SBATCH_PARTITION': 'other',
                                     'LD_PRELOAD': '/tmp/untrusted.so', 'GIT_CONFIG_COUNT': '2'}):
            env = endpoint.environment()
        self.assertFalse(any(key.startswith('SBATCH_') for key in env))
        self.assertNotIn('LD_PRELOAD', env)
        self.assertNotIn('GIT_CONFIG_COUNT', env)

    def test_submit_receipt_is_reused_without_sbatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            receipt = {'job_id': '123', 'job_name': 'libddla-gh-1-1-gpu4'}
            endpoint.write_json(run / 'gpu4-job.json', receipt)
            with patch.object(endpoint, 'capture') as capture:
                self.assertEqual(endpoint.submit(run, run, 'gpu4'), receipt)
                capture.assert_not_called()

    def test_lost_submit_reply_is_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            endpoint.write_json(run / 'gpu4-intent.json', {'job_name': 'libddla-gh-1-1-gpu4'})
            with patch.object(endpoint, 'capture') as capture, self.assertRaises(RuntimeError):
                endpoint.submit(run, run, 'gpu4')
            capture.assert_not_called()

    def test_cancel_refuses_changed_owner_or_name(self):
        job = {'job_id': '123', 'job_name': 'libddla-gh-1-1-gpu4', 'terminal': False}
        with patch.object(endpoint, 'state', return_value={'jobs': {'gpu4': job}}):
            with patch.object(endpoint, 'capture', return_value=f'JobId=123 JobName=someone-else UserId=test({os.getuid()})') as call:
                with self.assertRaises(ValueError):
                    endpoint.cancel(Path('/unused'))
                self.assertEqual(call.call_count, 1)

    def test_cancel_targets_only_receipted_job(self):
        job = {'job_id': '123', 'job_name': 'libddla-gh-1-1-gpu4', 'terminal': False}
        with patch.object(endpoint, 'state', return_value={'jobs': {'gpu4': job}}):
            with patch.object(endpoint, 'capture', side_effect=[f'JobName={job["job_name"]} UserId=test({os.getuid()})', '']) as call:
                self.assertEqual(endpoint.cancel(Path('/unused'))['cancel_requested'], ['123'])
                self.assertEqual(call.call_args.args[0], ['scancel', '123'])


class ArchiveTests(unittest.TestCase):
    def archive(self, name, kind=tarfile.REGTYPE):
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode='w') as tar:
            item = tarfile.TarInfo(name)
            item.type = kind
            if kind == tarfile.REGTYPE:
                item.size = 1
                tar.addfile(item, io.BytesIO(b'x'))
            else:
                item.linkname = '/etc/passwd'
                tar.addfile(item)
        out.seek(0)
        return out

    def test_source_rejects_traversal_and_links(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/tmp/escape', tarfile.REGTYPE),
                           ('CMakeLists.txt', tarfile.SYMTYPE), ('.git/config', tarfile.REGTYPE)]:
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                endpoint.extract_source(self.archive(name, kind), Path(temporary))

    def test_result_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'results.tar'
            archive.write_bytes(self.archive('escape', tarfile.SYMTYPE).read())
            with self.assertRaises(ValueError):
                relay.unpack_results(archive, root / 'out')


class GitProvenanceTests(unittest.TestCase):
    def test_real_endpoint_accepts_exact_commit_and_rejects_forged_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'git'
            subprocess.run(['git', 'init', '--quiet', str(source)], check=True)
            (source / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\n')
            git = ['git', '-C', str(source)]
            subprocess.run(git + ['add', '.'], check=True)
            subprocess.run(git + ['-c', 'user.name=CI test', '-c', 'user.email=ci@example.invalid',
                                  'commit', '--quiet', '-m', 'fixture'], check=True)
            revision = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
            subprocess.run(git + ['update-ref', 'refs/heads/selected', revision], check=True)
            bundle = root / 'source.bundle'
            subprocess.run(git + ['bundle', 'create', str(bundle), 'refs/heads/selected'], check=True)
            data = bundle.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            work = root / 'ci'
            work.mkdir()
            config = root / 'policy.json'
            config.write_text(json.dumps({'root': str(work), 'approved_source_shas': [revision, 'a' * 40]}))
            def call(run_id, sha):
                env = dict(os.environ, SSH_ORIGINAL_COMMAND=f'libddla-ci prepare {run_id} {sha} {digest}')
                return subprocess.run([sys.executable, '-I', str(CONTROL / 'endpoint.py'), '--config', str(config)],
                                      input=data, env=env, capture_output=True)
            result = call('gh-1-1', revision)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(json.loads(result.stdout)['source_sha'], revision)
            self.assertTrue((work / 'runs/gh-1-1/source/CMakeLists.txt').is_file())
            self.assertEqual(call('gh-1-1', revision).returncode, 0)
            self.assertNotEqual(call('gh-2-1', 'a' * 40).returncode, 0)
            self.assertNotEqual(call('gh-3-1', 'b' * 40).returncode, 0)
            self.assertFalse((work / 'runs/gh-2-1').exists())


class ReportingTests(unittest.TestCase):
    def result(self, xml, inventory=('one',), ctest=0, state='COMPLETED'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'ctest-inventory.json').write_text(json.dumps({'tests': [{'name': name} for name in inventory]}))
            (root / 'ctest.xml').write_text(xml)
            (root / 'exitcode.txt').write_text(str(ctest))
            (root / 'ctest-exitcode.txt').write_text(str(ctest))
            return report.stage_result(root, {'state': state, 'exit_code': '0:0'})

    def test_complete_passing_evidence(self):
        self.assertTrue(self.result('<testsuite><testcase name="one" time="1"/></testsuite>')['success'])

    def test_failure_is_never_allowlisted_away(self):
        result = self.result('<testsuite><testcase name="one"><failure/></testcase></testsuite>', ctest=8)
        self.assertEqual(result['failed'], 1)
        self.assertFalse(result['success'])

    def test_skipped_or_missing_test_is_not_green(self):
        for xml in ['<testsuite><testcase name="one"><skipped/></testcase></testsuite>', '<testsuite/>']:
            self.assertFalse(self.result(xml)['success'])

    def test_duplicate_test_is_not_green(self):
        self.assertFalse(self.result('<testsuite><testcase name="one"/><testcase name="one"/></testsuite>')['success'])

    def test_inconsistent_junit_counts_are_not_green(self):
        self.assertFalse(self.result('<testsuite tests="2"><testcase name="one"/></testsuite>')['success'])

    def test_nonzero_process_exit_or_slurm_timeout_is_not_green(self):
        xml = '<testsuite><testcase name="one"/></testsuite>'
        self.assertFalse(self.result(xml, ctest=1)['success'])
        self.assertFalse(self.result(xml, state='TIMEOUT')['success'])


class RelayLifecycleTests(unittest.TestCase):
    def exercise(self, temporary, mode):
        root = Path(temporary)
        bundle = root / 'source.bundle'
        bundle.write_bytes(b'fixture bundle; remote is mocked for lifecycle checks')
        args = SimpleNamespace(run_id='gh-123-1', source_sha='a' * 40,
                               work=root / 'work', results=root / 'results',
                               ssh_config=root / 'ssh-config', cancel_only=False,
                               timeout_minutes=0 if mode == 'deadline' else 1)
        calls = []
        jobs = {}
        request = {'run_id': args.run_id, 'source_sha': args.source_sha}

        class MockRemote:
            def call(self, action, arguments, stdin=None, output=None):
                calls.append((action, arguments))
                if action == 'prepare':
                    return request
                if action == 'submit':
                    kind = arguments[1]
                    if mode == 'lost_reply':
                        raise subprocess.CalledProcessError(255, ['ssh'])
                    jobs[kind] = {'job_id': str(123 + len(jobs)),
                                  'state': 'FAILED' if kind == 'gpu4' else 'COMPLETED',
                                  'exit_code': '8:0' if kind == 'gpu4' else '0:0', 'terminal': True}
                    return jobs[kind]
                if action == 'cancel':
                    if mode == 'lost_reply':
                        raise RuntimeError('Uncertain submission needs administrator reconciliation')
                    for job in jobs.values():
                        job.update(state='CANCELLED', exit_code='0:15', terminal=True)
                    return {'cancel_requested': ['123']}
                state = {'request': request, 'jobs': dict(jobs), 'build_ready': True}
                if action == 'status':
                    return state
                if action == 'results':
                    if mode == 'lost_reply':
                        raise RuntimeError('Submission state remains uncertain')
                    entries = {'state.json': json.dumps(state)}
                    for kind, job in jobs.items():
                        failed = job['state'] != 'COMPLETED'
                        entries[kind + '/ctest-inventory.json'] = json.dumps({'tests': [{'name': 'one'}]})
                        entries[kind + '/ctest.xml'] = ('<testsuite><testcase name="one">' +
                                                       ('<failure/>' if failed else '') + '</testcase></testsuite>')
                        for file in ('exitcode.txt', 'ctest-exitcode.txt'):
                            entries[kind + '/' + file] = '8' if failed else '0'
                    with tarfile.open(fileobj=output, mode='w:gz') as archive:
                        for name, value in entries.items():
                            data = value.encode()
                            info = tarfile.TarInfo(name)
                            info.size = len(data)
                            archive.addfile(info, io.BytesIO(data))
                    return None
                raise AssertionError(action)

        original_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with patch.object(relay, 'Remote', return_value=MockRemote()), \
                    patch.object(relay, 'bundle_source', return_value=bundle), \
                    patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': ''}):
                rc = relay.run(args)
        finally:
            for signum, handler in original_signals.items():
                signal.signal(signum, handler)
        return rc, calls, args.results

    def test_test_failure_still_runs_six_ranks_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            rc, calls, results = self.exercise(temporary, 'test_failure')
            self.assertEqual(rc, 1)
            self.assertEqual([args[1] for action, args in calls if action == 'submit'], ['gpu4', 'gpu6'])
            self.assertFalse(any(action == 'cancel' for action, _ in calls))
            summary = json.loads((results / 'remote/summary.json').read_text())
            self.assertEqual(summary['stages']['gpu4']['failed'], 1)
            self.assertTrue(summary['stages']['gpu6']['success'])

    def test_deadline_cancels_this_run_and_collects_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            rc, calls, results = self.exercise(temporary, 'deadline')
            self.assertEqual(rc, 1)
            self.assertIn(('cancel', ['gh-123-1']), calls)
            self.assertTrue((results / 'remote/state.json').exists())

    def test_lost_submission_reply_never_retries_submit(self):
        with tempfile.TemporaryDirectory() as temporary:
            rc, calls, results = self.exercise(temporary, 'lost_reply')
            self.assertEqual(rc, 1)
            self.assertEqual(sum(action == 'submit' for action, _ in calls), 1)
            self.assertIn(('cancel', ['gh-123-1']), calls)
            errors = json.loads((results / 'relay-errors.json').read_text())
            self.assertTrue(any('Cancellation unconfirmed' in message for message in errors))


if __name__ == '__main__':
    unittest.main()
