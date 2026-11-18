# SAI GPU validation

`SAI CUDA validation` tests an approved LibDDLA commit on
SAI `16V100` with CUDA 12.9.1, GCC 13.3.0, Open MPI 5.0.10 and NCCL 2.29.3.
It builds with `sm_70` and runs the full CUDA-enabled CTest suite with four MPI
ranks, then reuses the same binaries for six-rank `pgeadd_indexed` and
`pgemm_branches` tests on a `1x6` process grid. The partition assigns host memory;
neither stage requests explicit Slurm memory.

## What gets tested

| Event | Source sent to SAI | Approval |
| --- | --- | --- |
| Push or merge to protected `develop` or `main` | Pushed commit | Automatic once enabled |
| PR opened, reopened or updated against either branch | Current PR head, including for forks | A maintainer approves that run; a new commit needs new approval |
| Manual run on either protected branch | Supplied full commit SHA | A maintainer approves that run |

Push and PR checks work on `develop` as soon as it contains the workflow. Bring
the workflow to `main` to cover pushes there and expose **Run workflow** in
Actions. A skipped job means no GPU test was run. Failed or incomplete suites
remain failures; the job uploads available JUnit, logs and Slurm evidence.

## Repository owner

1. Protect `develop` and `main` with required review, including workflow changes;
   restrict direct and force pushes. Under **Settings > Actions > General**,
   require approval for fork workflows from **all outside collaborators**.
2. Under **Settings > Environments**, keep required maintainer reviewers on
   `sai-libddla` and allow the branch patterns `main`, `develop` and
   `refs/pull/*/merge`. Review the source SHA in the run title before approving:
   native builds run as the SAI account. Create `sai-libddla-push` with only
   `main` and `develop` allowed and no required reviewer. Neither needs secrets.
3. Under **Settings > Actions > Runners**, privately give SAI a short-lived
   **repository runner registration token** for the initial setup. Agree on
   renewal for disposable runners and reviewer logins with SAI. Provide the site
   publisher read access to Actions, Contents, Pull requests and Variables for
   this repository. No repository-write token or SAI SSH key is needed by a job.
4. Once SAI confirms isolation, set the repository Actions variable
   `SAI_CI_ENABLED=true` for a supervised first run on `develop`. Keep it on
   after verifying reporting and cancellation; removing it disables new
   automatic runs. Leave this check out of required merge checks initially.

## SAI administrators

- Register clean, disposable Linux x64 runners labeled `sai-libddla-results`,
  with Bash and jq and one concurrent job to match site capacity. Give jobs
  **only read-only access to published results** at
  `/srv/sai-libddla/results`: no site credentials, SSH agent, controller socket,
  private storage or cluster network access. Retire any credential-bearing
  runner accessible to this public repository before accepting PR workflows.
  A label or an Environment cannot stop a fork from changing workflow steps.
- On the trusted side, install a reviewed copy of [publish.py](publish.py) and the
  existing [site controller](https://github.com/Stardust0831/LibDDLA/tree/66b2c419de961618671c9fbd99298dc6c4b2d255/.ci/sai).
  Keep SSH, the restricted endpoint and Slurm quotas there. Run one publisher
  at a time on a short timer (e.g. every 20 seconds) with Python 3.10+ and an
  authenticated `gh` bound to github.com. It polls GitHub directly; it does not
  accept commands or source files from the runner.
- Configure the publisher's `SAI_RELAY` and `SAI_SSH_CONFIG` paths, a private
  `SAI_WORK` directory, the site-writable `SAI_RESULTS` directory exported
  read-only to jobs, and comma-separated trusted `SAI_REVIEWERS` GitHub logins.
  The existing endpoint requires the exact SHA during both prepare and submit.
  For automatic admission, provide a site-owned `SAI_AUTHORIZE` executable:
  it receives `event branch sha run_id` after verification and must authorize
  only that SHA or exit nonzero. Without it only already-allowlisted SHAs run.
- Each attempt has its own results directory. On cancellation, supersession or
  PR closure the publisher interrupts that attempt's relay to cancel its Slurm
  jobs. An interrupted publisher does not replay an old attempt; reconcile it
  with Slurm, then re-run with a new attempt. Retain public evidence per site
  policy and keep partial or absent results from counting as a pass.

Before enabling routine runs, verify the credential boundary from an actual job,
check that an unapproved SHA is refused, and exercise artifact upload and
cancellation through Actions. Keep the check informational while deployment and
the two CUDA 12.9 Cholesky error-code failures are outstanding.
