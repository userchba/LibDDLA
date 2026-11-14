#!/usr/bin/env python3
"""Prepare approved Git objects on the VM and monitor fixed SAI jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from endpoint import MAX_RESULTS, parse_command
from report import generate

REPOSITORY = 'https://github.com/userchba/LibDDLA.git'


class Interrupted(Exception):
    pass


def interrupt(signum, frame):
    raise Interrupted(f'Relay received signal {signum}')


class Remote:
    def __init__(self, config):
        self.config = config

    def call(self, action, arguments, *, stdin=None, output=None):
        command = shlex.join(['libddla-ci', action, *arguments])
        parse_command(command)
        response = subprocess.run(
            ['ssh', '-F', str(self.config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
             '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', 'libddla-sai', command],
            stdin=stdin, stdout=output or subprocess.PIPE, check=True, timeout=300 if stdin else 90)
        if output is None:
            return json.loads(response.stdout)


def bundle_source(root, revision):
    bare = root / 'source.git'
    bundle = root / 'source.bundle'
    if bundle.exists():
        raise ValueError('Relay working directory must be fresh for this run attempt')
    git = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'pack.threads=1', '-C', str(bare)]
    subprocess.run(['git', 'init', '--quiet', '--bare', str(bare)], check=True)
    subprocess.run(git + ['fetch', '--quiet', '--no-tags', REPOSITORY, revision], check=True, timeout=300)
    actual = subprocess.check_output(git + ['rev-parse', 'FETCH_HEAD^{commit}'], text=True).strip()
    if actual != revision:
        raise ValueError('Source SHA did not resolve exactly')
    subprocess.run(git + ['update-ref', 'refs/heads/selected', revision], check=True)
    subprocess.run(git + ['bundle', 'create', str(bundle.resolve()), 'refs/heads/selected'],
                   check=True, timeout=300)
    if bundle.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Git history exceeds the endpoint bundle limit')
    return bundle


def unpack_results(archive, results):
    total = 0
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise ValueError('Unsafe result path')
            if member.name in seen or not member.isfile():
                raise ValueError('Result archive contains duplicate, linked or special entries')
            seen.add(member.name)
            if len(seen) > 20000:
                raise ValueError('Result archive contains too many files')
            total += member.size
            if total > MAX_RESULTS:
                raise ValueError('Results exceed their transfer limit')
            target = results / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                data = tar.extractfile(member)
                shutil.copyfileobj(data, stream)


def run(args):
    parse_command(shlex.join(['libddla-ci', 'prepare', args.run_id, args.source_sha, '0' * 64]))
    args.work.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    remote = Remote(args.ssh_config)
    if args.cancel_only:
        print(json.dumps(remote.call('cancel', [args.run_id])))
        return 0
    deadline = time.monotonic() + args.timeout_minutes * 60
    submitted = False
    complete = False
    errors = []
    current = {'request': {'source_sha': args.source_sha}, 'jobs': {}}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    try:
        bundle = bundle_source(args.work, args.source_sha)
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        with bundle.open('rb') as stream:
            remote.call('prepare', [args.run_id, args.source_sha, digest], stdin=stream)
        for kind in ('gpu4', 'gpu6'):
            if kind == 'gpu6' and not current.get('build_ready'):
                errors.append('gpu4 did not produce a verified build; gpu6 was not started')
                break
            # Mark intent before SSH: losing its reply must still invoke cancellation/reconciliation.
            submitted = True
            receipt = remote.call('submit', [args.run_id, kind])
            print(json.dumps({'submitted': receipt}), flush=True)
            unavailable = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Relay deadline reached, including queue time')
                try:
                    current = remote.call('status', [args.run_id])
                    unavailable = 0
                except (subprocess.SubprocessError, ValueError) as error:
                    unavailable += 1
                    if unavailable >= 3:
                        raise RuntimeError('Three consecutive status requests failed') from error
                    time.sleep(15)
                    continue
                (args.results / 'relay-state.json').write_text(json.dumps(current, indent=2) + '\n')
                job = current['jobs'].get(kind, {})
                print(json.dumps({'stage': kind, 'job': job}), flush=True)
                if job.get('state') == 'SUBMISSION_UNCERTAIN':
                    raise RuntimeError('Submission needs administrator reconciliation')
                if job.get('terminal'):
                    break
                time.sleep(20)
        complete = True
    except (Exception, KeyboardInterrupt) as error:
        errors.append(str(error))
    finally:
        # Allow a second TERM to terminate cleanup; fixed Slurm wall times still bound compute.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_DFL)
        if submitted and not complete:
            try:
                answer = remote.call('cancel', [args.run_id])
                print(json.dumps(answer), flush=True)
                for _ in range(6):
                    current = remote.call('status', [args.run_id])
                    if all(job.get('terminal') for job in current['jobs'].values()):
                        break
                    time.sleep(5)
            except Exception as error:
                errors.append(f'Cancellation unconfirmed: {error}')
        if submitted:
            archive = args.work / 'results.tar.gz'
            try:
                with archive.open('wb') as stream:
                    remote.call('results', [args.run_id], output=stream)
                # The endpoint emits flat regular-file entries; no remote archive is blindly extracted.
                unpack_results(archive, args.results / 'remote')
                current = json.loads((args.results / 'remote/state.json').read_text())
                if (current['request']['run_id'] != args.run_id
                        or current['request']['source_sha'] != args.source_sha):
                    raise ValueError('Collected evidence identifies a different run or source')
            except Exception as error:
                errors.append(f'Result collection failed: {error}')
        (args.results / 'relay-errors.json').write_text(json.dumps(errors, indent=2) + '\n')
        result_dir = args.results / 'remote'
        result_dir.mkdir(exist_ok=True)
        summary = generate(result_dir, current)
        if errors:
            with (result_dir / 'summary.md').open('a') as stream:
                stream.write('\nRelay errors:\n\n' + '\n'.join('- ' + error for error in errors) + '\n')
        github_summary = os.environ.get('GITHUB_STEP_SUMMARY')
        if github_summary:
            with open(github_summary, 'a') as stream:
                stream.write((result_dir / 'summary.md').read_text())
    return 0 if summary['success'] and not errors else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--ssh-config', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--timeout-minutes', type=int, default=165)
    parser.add_argument('--cancel-only', action='store_true')
    raise SystemExit(run(parser.parse_args()))
