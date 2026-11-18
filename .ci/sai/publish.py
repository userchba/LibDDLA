"""Poll once from the trusted site account. Never install this on the Actions runner."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys


REPO = "userchba/LibDDLA"
WORKFLOW = ".github/workflows/sai-validation.yml"
BRANCHES = {"main", "develop"}
REVIEW_ENV = "sai-libddla"
RESULT_LABEL = "sai-libddla-results"


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", "--hostname", "github.com", f"repos/{REPO}/{path}"],
                                              timeout=45))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def admission(run, reviewers):
    event = run["event"]
    match = re.fullmatch(r"SAI CUDA / (0|[1-9][0-9]*) / ([a-f0-9]{40})", run["display_title"])
    require(run["path"] == WORKFLOW and run["repository"]["full_name"] == REPO and match,
            "Unknown workflow or invalid request")
    pr, sha = match.groups()
    branch = run["head_branch"]
    if event == "pull_request":
        pull = api(f"pulls/{pr}")
        require(pull["state"] == "open" and pull["base"]["repo"]["full_name"] == REPO
                and pull["head"]["sha"] == sha == run["head_sha"]
                and pull["head"]["repo"]["id"] == run["head_repository"]["id"],
                "This is not the current PR head")
        branch = pull["base"]["ref"]
    else:
        require(event in {"push", "workflow_dispatch"} and pr == "0"
                and run["head_repository"]["full_name"] == REPO, "Unexpected event or repository")
    require(branch in BRANCHES, "Unexpected base branch")
    if event == "push":
        require(sha == run["head_sha"] and
                api(f"compare/{sha}...{branch}")["status"] in {"ahead", "identical"},
                "Push SHA is not on the trusted branch")
    else:
        decisions = [a for a in api(f"actions/runs/{run['id']}/approvals")
                     if any(e["name"] == REVIEW_ENV for e in a["environments"])]
        require(not any(a["state"] == "rejected" for a in decisions) and
                any(a["state"] == "approved" and a["user"]["login"].lower() in reviewers
                    for a in decisions),
                "No trusted Environment approval for this run")
    return branch, sha, pr


def publish(run, reviewers):
    number, attempt = int(run["id"]), int(run["run_attempt"])
    key = f"gh-{number}-{attempt}"
    result = Path(os.environ["SAI_RESULTS"]) / key
    if result.exists():
        return
    jobs = api(f"actions/runs/{number}/attempts/{attempt}/jobs")["jobs"]
    if not any(j["status"] == "in_progress" and RESULT_LABEL in j["labels"] for j in jobs):
        return  # A queued job can still be waiting for its Environment reviewer.
    # Never replay a partially published attempt after a service restart.
    try:
        result.mkdir()
    except FileExistsError:
        return
    work = Path(os.environ["SAI_WORK"]) / key
    sha, code = None, 1
    try:
        current = api(f"actions/runs/{number}")
        require(current["run_attempt"] == attempt and current["status"] == "in_progress",
                "Attempt is no longer active")
        branch, sha, pr = admission(current, reviewers)
        # Site policy may authorize this exact SHA; the endpoint still checks it on prepare and submit.
        if hook := os.environ.get("SAI_AUTHORIZE"):
            subprocess.run([hook, current["event"], branch, sha, key], check=True, timeout=60)
        cmd = [sys.executable, os.environ["SAI_RELAY"], "--source-sha", sha, "--run-id", key,
               "--ssh-config", os.environ["SAI_SSH_CONFIG"], "--work", str(work),
               "--results", str(result)]
        with subprocess.Popen(cmd) as relay:
            try:
                while True:
                    try:
                        code = relay.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        active = api(f"actions/runs/{number}")
                        require(active["run_attempt"] == attempt and active["status"] == "in_progress",
                                "Attempt ended or was superseded")
                        if pr != "0":
                            pull = api(f"pulls/{pr}")
                            require(pull["state"] == "open" and pull["head"]["sha"] == sha
                                    and pull["base"]["ref"] == branch, "PR changed or closed")
            finally:
                if relay.poll() is None:
                    relay.send_signal(signal.SIGINT)  # The installed relay cancels its own Slurm jobs.
                    relay.wait()
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"{key}: {error}", file=sys.stderr, flush=True)
    finally:
        receipt = result / "completion.tmp"
        receipt.write_text(json.dumps({"run_id": key, "source_sha": sha, "exit_code": code}) + "\n")
        receipt.replace(result / "completion.json")
        if work.is_dir():
            shutil.rmtree(work)


def main():
    private, public = (Path(os.environ[k]).resolve(strict=True) for k in ("SAI_WORK", "SAI_RESULTS"))
    require(private.is_dir() and public.is_dir() and not private.is_relative_to(public)
            and not public.is_relative_to(private), "Private work and public results must be separate directories")
    reviewers = {login.strip().lower() for login in os.environ["SAI_REVIEWERS"].split(",") if login.strip()}
    for request in sorted(Path(os.environ["SAI_REQUESTS"]).glob("gh-*-*.token")):
        match = re.fullmatch(r"gh-([1-9][0-9]*)-([1-9][0-9]*)\.token", request.name)
        if not match or (public / request.stem).exists():
            continue
        descriptor = None
        try:
            # This mailbox is writable by jobs. Never follow links or read special files.
            descriptor = os.open(request, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Token is not a regular file")
            token = os.read(descriptor, 257).decode("ascii")
            require(re.fullmatch(r"[A-Za-z0-9_]{20,256}", token), "Invalid token format")
            os.environ["GH_TOKEN"] = token
            run = api(f"actions/runs/{match[1]}")
            if run["run_attempt"] != int(match[2]) or run["status"] == "completed":
                request.unlink(missing_ok=True)
                continue
            require(run["status"] == "in_progress", "Request is not active yet")
            if run["path"] == WORKFLOW:
                publish(run, reviewers)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"{request.name}: {type(error).__name__}", file=sys.stderr, flush=True)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.environ.pop("GH_TOKEN", None)
            if (public / request.stem / "completion.json").is_file():
                request.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
