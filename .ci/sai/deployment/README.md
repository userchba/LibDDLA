# Administrator deployment

This is an installation draft. Installing the runner, changing authorized keys
and starting the service are separate administrator actions. The controller
supports trusted, explicitly approved source only. An SSH command restriction
and Slurm limits do not isolate arbitrary CMake or test code from the SAI account.

## GitHub repository owner

- Register a repository runner for `userchba/LibDDLA`, label `sai-libddla`,
  name `sai-libddla-vm`. Supply a short-lived registration token to the installer;
  do not give the VM a persistent owner PAT. The installer reads the token on
  stdin and leaves service startup to the administrator.
- Create Environment `sai-libddla`, limit deployment branches to `main` and
  `develop`, require trusted reviewers, and prevent self-review where available.
  Review workflow changes as executable infrastructure code. A runner label
  is a scheduling label, not a permission boundary.
- Merge the manual workflow into the default branch. Initially do not make its
  check mandatory and do not add PR or `pull_request_target` triggers. The job
  token has only `contents: read`; no repository-write token is needed on SAI.

## VM administrator

Create a dedicated `libddla-runner` Linux user without sudo, Docker, disk or
other privileged group membership. Do not reuse another runner's UID or home.
The intended layout is:

```text
/home/libddla-runner/ci/                 root-owned parent
  runner/ work/ tmp/ cache/ state/       owned by libddla-runner
  work/home/                            isolated runtime home when needed
  bin/ deployment/                      administrator-controlled, read-only
  ssh/config known_hosts id_ed25519      only this project's SSH identity
```

Install `sandbox.py`, `preflight.py` and `install-runner.sh` in `deployment/`.
Install `ssh_config` as `ssh/config` and the pinned public `known_hosts` file;
verify that host key with the SAI administrator. Provision the dedicated private
key through a private channel, mode 0600, readable only by this user. Never put
the private key in the repository, artifacts or issue text.

Install ordinary runtime dependencies for the official Actions runner plus
Python 3.10+, Git, SSH, curl and tar. As `libddla-runner`, run
`bash /home/libddla-runner/ci/deployment/install-runner.sh` and supply the
repository registration token on stdin. Runner 2.338.2's official archive and
SHA-256 are pinned in the script; normal runner auto-update remains available.

Install `libddla-runner.service` into `/etc/systemd/system/`, run
`systemctl daemon-reload`, then inspect it before enabling. This system service
runs under the dedicated UID with no capabilities, no new privileges, a 2 GiB
memory cap and a two-CPU quota. It invokes a Landlock launcher that requires
Linux x86_64 and Landlock ABI 4+. Missing support is a startup failure, not an
unrestricted fallback. Reads/writes are limited to system runtime and this CI
tree; TCP connections are limited to ports 443 and 12022. These port rules do
not restrict the destination IP and do not make the runner a hostile-code VM.

Before `systemctl enable --now libddla-runner.service`, run the launcher under
the dedicated account with a small diagnostic command. Verify it can read its
own SSH configuration and reach GitHub/SAI but cannot read another user's key,
write outside its CI tree, open Docker's socket or read files on `/mnt/c` and
`/mnt/d`. Confirm `NoNewPrivileges`, cgroup limits and runner online state.
These checks must be performed on the installed account; a local unit-file
check does not establish isolation. On WSL, Linux service enablement is not
proof of Windows boot recovery; verify the existing WSL startup/keepalive path.

## SAI administrator

Keep `libddla-ci` dedicated to this controller, with only the required project
storage and Slurm account access. Review `saicommongrp01` / `org-open-sai` group
access to other project data. No sudo or software/module installation permission
is required. Native approved builds run as this UID and inherit its filesystem
and network rights; remove unrelated data and credentials from that boundary.

Install the reviewed contents of `.ci/sai/` in `/opt/libddla-ci/controller`,
owned by the administrator and not writable by `libddla-ci`. Shell wrappers
must be executable. Install `site-policy.example.json` as
`/etc/libddla-ci/policy.json`, also administrator-owned, and create
`/org/org-open-sai/libddla-ci/ci` writable by `libddla-ci`. Set the policy's
`approved_source_shas` to the exact reviewed commits; the default empty list
blocks submissions. Python 3.10+ and the Slurm/Git executables must be available.

Use a dedicated authorized-key entry with a forced command, for example:

```text
restrict,command="/usr/bin/python3 -I /opt/libddla-ci/controller/endpoint.py --config /etc/libddla-ci/policy.json" ssh-ed25519 <PUBLIC-KEY> libddla-ci
```

Enforce this through an administrator-controlled authorized-key file or SSH
configuration. Merely placing a restricted entry in a user-writable
`authorized_keys` is not a durable boundary if another login method permits
the user to edit it. Remove unrestricted duplicates of this CI key after
acceptance; keep any administrator recovery access separate.

The accepted protocol is only:

```text
libddla-ci prepare RUN_ID APPROVED_SHA BUNDLE_SHA256   # bundle on stdin
libddla-ci submit RUN_ID gpu4|gpu6
libddla-ci status RUN_ID
libddla-ci cancel RUN_ID
libddla-ci results RUN_ID                            # tar.gz on stdout
```

No caller chooses a shell command, account, partition, resource size, script
path or job ID. The site fixes `16V100`, `org-open-sai`, `rush-gpu`, GPU counts
4/8 and wall times 90/25 minutes. Submission uses a sanitized environment with
no `SBATCH_*` overrides. The partition supplies memory automatically: do not
add `--mem`, `--mem-per-cpu` or `--mem-per-gpu` in site wrappers.

Set a site-enforced limit of one active/queued job for this account and at most
eight GPUs, plus the agreed storage and usage budgets. The controller also
checks for active jobs under a lock, but site quotas are the authoritative
limit if an approved native test calls Slurm itself. Arrange retention cleanup
for finished runs only, for example after 30 days; failed runs retain raw logs.

## Activation acceptance

1. The restricted key must reject `id`, shell metacharacters, arbitrary paths,
   unapproved SHAs and cancellation of an unrelated job. Verify it cannot open
   a shell, forward ports or request a PTY. Confirm the administrator's recovery
   path before changing any live key.
2. Upload one approved Git bundle and check that the extracted source and
   request SHA agree. Repeating the same request must not replace its source;
   submitting the same stage again must reuse its receipt. Reconcile any
   uncertain submission intent with Slurm instead of deleting it blindly.
3. Run the manual GitHub workflow and retain its artifact, including actual
   known failures. Confirm four/six unique GPU mappings, complete JUnit and
   final accounting. This is the first real Actions lifecycle acceptance;
   submitting the same recipe manually over SSH is a separate check.
4. Cancel a controlled run and confirm only its Slurm jobs terminate. Confirm
   idle runner restart after a service crash. Record remaining gaps before
   enabling trusted-branch automation or making the check required.

Supporting unreviewed public PR code requires a separately validated compute
sandbox and credential boundary. Approval of this manual workflow alone does
not establish that capability.
