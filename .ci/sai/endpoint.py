#!/usr/bin/env python3
"""Administrator-installed SSH endpoint for approved LibDDLA commits only.

The controller is not a sandbox for CMake or tests. Native execution requires
trusted source and a dedicated SAI account. See deployment/README.md.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
MAX_BUNDLE = 64 * 1024 * 1024
MAX_SOURCE = 512 * 1024 * 1024
MAX_RESULTS = 512 * 1024 * 1024
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
COMMANDS = {'prepare': 3, 'submit': 2, 'status': 1, 'cancel': 1, 'results': 1}


def parse_command(command):
    words = shlex.split(command)
    if len(words) < 3 or words[0] != 'libddla-ci' or words[1] not in COMMANDS:
        raise ValueError('Only the libddla-ci protocol is allowed')
    action, arguments = words[1], words[2:]
    if len(arguments) != COMMANDS[action]:
        raise ValueError('Wrong number of protocol arguments')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,62}', arguments[0]):
        raise ValueError('Invalid run identifier')
    if action == 'prepare':
        if not re.fullmatch(r'[0-9a-f]{40}', arguments[1]):
            raise ValueError('A full source SHA is required')
        if not re.fullmatch(r'[0-9a-f]{64}', arguments[2]):
            raise ValueError('A SHA-256 bundle digest is required')
    if action == 'submit' and arguments[1] not in ('gpu4', 'gpu6'):
        raise ValueError('Unknown stage')
    return action, arguments


def environment():
    # In particular, do not inherit SBATCH_* memory/resource overrides or Git hooks/config.
    result = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME') if key in os.environ}
    result.update(PATH='/usr/bin:/bin', LANG='C.UTF-8', GIT_CONFIG_NOSYSTEM='1',
                  GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0')
    return result


def capture(arguments, timeout=60):
    return subprocess.check_output(arguments, text=True, env=environment(), timeout=timeout).strip()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.new')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def locked(path):
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def bounded_copy(source, destination, maximum):
    digest = hashlib.sha256()
    size = 0
    while True:
        block = source.read(min(1024 * 1024, maximum - size + 1))
        if not block:
            break
        size += len(block)
        if size > maximum:
            raise ValueError('Input exceeds its size limit')
        digest.update(block)
        destination.write(block)
    return digest.hexdigest()


def extract_source(archive, destination):
    total = 0
    seen = set()
    with tarfile.open(fileobj=archive, mode='r|*') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or '.git' in path.parts:
                raise ValueError('Unsafe source path')
            if member.name in seen or len(seen) >= 20000:
                raise ValueError('Duplicate or excessive source entries')
            seen.add(member.name)
            target = destination / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                if total > MAX_SOURCE:
                    raise ValueError('Source exceeds size limit')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as out:
                    shutil.copyfileobj(tar.extractfile(member), out)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise ValueError('Source links and special files are not supported')
    if not (destination / 'CMakeLists.txt').is_file():
        raise ValueError('Missing CMakeLists.txt')


def prepare(root, policy, run_id, source_sha, bundle_digest, stream):
    if source_sha not in policy['approved_source_shas']:
        raise ValueError('Commit is not in the administrator source allowlist')
    run = root / 'runs' / run_id
    if run.exists():
        old = json.loads((run / 'request.json').read_text())
        if old['source_sha'] != source_sha or old['bundle_sha256'] != bundle_digest:
            raise ValueError('Run identifier already has different inputs')
        # Drain and verify repeated uploads; do not extract or overwrite a running source tree.
        with open(os.devnull, 'wb') as sink:
            if bounded_copy(stream, sink, MAX_BUNDLE) != bundle_digest:
                raise ValueError('Bundle checksum mismatch')
        return old
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=root) as temporary:
        staging = Path(temporary)
        bundle = staging / 'source.bundle'
        with bundle.open('wb') as out:
            actual = bounded_copy(stream, out, MAX_BUNDLE)
            out.flush()
            os.fsync(out.fileno())
        if actual != bundle_digest:
            raise ValueError('Bundle checksum mismatch')
        bare = staging / 'source.git'
        capture(['git', 'init', '--quiet', '--bare', str(bare)])
        git = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'fetch.fsckObjects=true', '-C', str(bare)]
        capture(git + ['fetch', '--quiet', '--no-tags', str(bundle), 'refs/heads/selected'], timeout=180)
        if capture(git + ['rev-parse', 'FETCH_HEAD^{commit}']) != source_sha:
            raise ValueError('Git objects do not match the approved source commit')
        if any(line.startswith('160000 ') for line in capture(git + ['ls-tree', '-r', source_sha]).splitlines()):
            raise ValueError('Submodules are not supported')
        archive = staging / 'source.tar'
        with archive.open('wb') as out:
            subprocess.run(git + ['archive', '--format=tar', source_sha], stdout=out,
                           env=environment(), timeout=120, check=True)
        if archive.stat().st_size > MAX_SOURCE:
            raise ValueError('Source archive exceeds size limit')
        source = staging / 'source'
        source.mkdir()
        with archive.open('rb') as data:
            extract_source(data, source)
        control = staging / 'control'
        shutil.copytree(HERE, control, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        manifest = {p.relative_to(control).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(control.rglob('*')) if p.is_file()}
        request = {'run_id': run_id, 'source_sha': source_sha, 'bundle_sha256': bundle_digest,
                   'controller_sha256': manifest, 'created_at': int(time.time()),
                   'repository': 'userchba/LibDDLA', 'toolchain': 'cuda12.9.1-gcc13.3-nvhpc26.3'}
        write_json(staging / 'request.json', request)
        (staging / 'results').mkdir()
        for path in (bare,):
            shutil.rmtree(path)
        archive.unlink()
        bundle.unlink()
        staging.rename(run)
    return request


def job_state(receipt):
    job_id = str(receipt['job_id'])
    if not re.fullmatch(r'[1-9][0-9]*', job_id):
        raise ValueError('Invalid stored Slurm job ID')
    text = capture(['sacct', '-X', '-n', '-P', '-j', job_id,
                    '--format=JobIDRaw,JobName%100,State,ExitCode'])
    for line in text.splitlines():
        fields = line.split('|')
        if fields[0] == job_id:
            if fields[1] != receipt['job_name']:
                raise ValueError('Slurm job identity does not match the receipt')
            state = fields[2].split()[0].rstrip('+')
            return {**receipt, 'state': state, 'exit_code': fields[3], 'terminal': state in TERMINAL}
    queued = capture(['squeue', '-h', '-j', job_id, '-o', '%i|%j|%T'])
    for line in queued.splitlines():
        fields = line.split('|')
        if fields[0] == job_id and fields[1] == receipt['job_name']:
            return {**receipt, 'state': fields[2], 'exit_code': None, 'terminal': False}
    return {**receipt, 'state': 'UNKNOWN', 'exit_code': None, 'terminal': False}


def state(run):
    request = json.loads((run / 'request.json').read_text())
    jobs = {}
    for kind in ('gpu4', 'gpu6'):
        receipt = run / (kind + '-job.json')
        if receipt.exists():
            jobs[kind] = job_state(json.loads(receipt.read_text()))
        elif (run / (kind + '-intent.json')).exists():
            jobs[kind] = {'state': 'SUBMISSION_UNCERTAIN', 'terminal': False}
    return {'request': request, 'jobs': jobs, 'build_ready': (run / 'build-complete.txt').is_file()}


def submit(root, run, kind):
    receipt = run / (kind + '-job.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    intent = run / (kind + '-intent.json')
    if intent.exists():
        raise RuntimeError('Previous submission has no receipt; administrator must reconcile it before retrying')
    if kind == 'gpu6':
        previous = state(run)
        if not previous['build_ready'] or not previous['jobs'].get('gpu4', {}).get('terminal'):
            raise ValueError('gpu6 requires a completed gpu4 attempt and its verified build')
    # The account is dedicated: at most one allocation is active for this controller.
    active = capture(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i'])
    if active:
        raise RuntimeError('Dedicated SAI account already has an active or queued job')
    run_id = run.name
    job_name = f'libddla-{run_id}-{kind}'
    args = ['sbatch', '--parsable', '--job-name=' + job_name,
            '--output=' + str(run / 'results' / (kind + '-slurm-%j.log')),
            str(run / 'control' / (kind + '.sbatch')), str(run), str(run / 'control')]
    write_json(intent, {'command': args, 'job_name': job_name, 'created_at': int(time.time())})
    # Never retry sbatch automatically: a lost reply can still mean a live allocation.
    answer = capture(args, timeout=60)
    if not re.fullmatch(r'[1-9][0-9]*(;[A-Za-z0-9_.-]+)?', answer):
        raise RuntimeError('Unrecognized sbatch receipt; reconcile the submission intent')
    result = {'job_id': answer.split(';')[0], 'job_name': job_name, 'stage': kind}
    write_json(receipt, result)
    return result


def cancel(run):
    current = state(run)
    cancelled = []
    for job in current['jobs'].values():
        if job.get('terminal'):
            continue
        if 'job_id' not in job:
            raise RuntimeError('Uncertain submission needs administrator reconciliation')
        # Check owner and name from scontrol, not a caller-supplied numeric ID.
        text = capture(['scontrol', 'show', 'job', '-o', str(job['job_id'])])
        name = re.search(r'\bJobName=(\S+)', text)
        uid = re.search(r'\bUserId=\S+\(([0-9]+)\)', text)
        if not name or not uid or name.group(1) != job['job_name'] or int(uid.group(1)) != os.getuid():
            raise ValueError('Refusing to cancel a job with a different owner or name')
        capture(['scancel', str(job['job_id'])])
        cancelled.append(job['job_id'])
    return {'cancel_requested': cancelled}


def archive_results(run, output):
    current = state(run)
    if any(not job.get('terminal') for job in current['jobs'].values()):
        raise RuntimeError('Results are still changing; wait for terminal Slurm states')
    result_root = run / 'results'
    write_json(result_root / 'state.json', current)
    shutil.copyfile(run / 'request.json', result_root / 'request.json')
    total = 0
    with tarfile.open(fileobj=output, mode='w|gz') as archive:
        for path in sorted(result_root.rglob('*')):
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError('Results contain a link or special file')
            total += metadata.st_size
            if total > MAX_RESULTS:
                raise ValueError('Results exceed transfer limit; collect them administratively')
            name = path.relative_to(result_root).as_posix()
            info = archive.gettarinfo(str(path), arcname=name)
            with path.open('rb') as data:
                archive.addfile(info, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    action, words = parse_command(os.environ.get('SSH_ORIGINAL_COMMAND', ''))
    policy = json.loads(args.config.read_text())
    configured_root = Path(policy['root'])
    if not configured_root.is_absolute():
        raise ValueError('Controller root must be absolute')
    root = configured_root.resolve()
    if not root.is_dir() or root == Path('/'):
        raise ValueError('Administrator must provision the controller root')
    os.umask(0o077)
    (root / 'runs').mkdir(exist_ok=True)
    with locked(root / 'controller.lock'):
        run = root / 'runs' / words[0]
        if action == 'prepare':
            answer = prepare(root, policy, *words, sys.stdin.buffer)
        else:
            if not run.is_dir() or run.is_symlink():
                raise ValueError('Unknown run identifier')
            if action == 'submit':
                request = json.loads((run / 'request.json').read_text())
                if request['source_sha'] not in policy['approved_source_shas']:
                    raise ValueError('Source approval has been revoked')
                answer = submit(root, run, words[1])
            elif action == 'status':
                answer = state(run)
            elif action == 'cancel':
                answer = cancel(run)
            else:
                archive_results(run, sys.stdout.buffer)
                return
        print(json.dumps(answer))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'libddla-ci: {error}', file=sys.stderr)
        raise SystemExit(1)
