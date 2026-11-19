# Mac to SAI GPU validation

This manual workflow adapts the controller used by
[reference run 44370955906](https://github.com/Stardust0831/LibDDLA/actions/runs/44370955906)
at controller commit `33f6a66598ee78ed77283d0d33e8b494824f0ffa`.

The native macOS ARM64 GitHub runner is named `libddla-sai-macos`, registered
to `userchba/LibDDLA` from
`/Users/chenhaobo/Downloads/sai-libddla-runner`. Its custom label is
`libddla-sai-macos`. This is the user's macOS account, not an isolated Linux
runner. It uses the account's existing SSH configuration for `libddla-ci`.

The workflow is manual only, runs from repository `main`, and requires the
existing `sai-libddla` Environment review. It adds no schedule or PR trigger.
Run `runner_check=true` first; that checks the native runner and SAI connection
without submitting Slurm jobs. Run `runner_check=false` for the GPU suite.

A full run resolves `userchba/LibDDLA:develop` once to a full commit SHA. It
retains the reference controller's build, 1/2/4/6-rank GPU groups, GPU mapping
checks, deadlines, failure reporting, cancellation and remote cleanup. The
only controller configuration changes are the SSH alias and project root:
`/org/org-open-sai/libddla-ci/mac-validation`. Existing `daily` runs are
separate. Do not add MPI oversubscription or request more ranks than allocated.

The Mac workflow commit, reference-controller commit and tested source commit
are recorded separately in `mac-adapter.json`. The remote request records hashes
of the actual controller files, including the adapted configuration.
SSH configuration and credentials are passed by path and are never copied to
the source/controller package or uploaded artifact.

The Actions summary and `libddla-macos-sai-<run>-<attempt>` artifact contain
source/controller inputs, per-case logs/JUnit and Slurm accounting. Any failed,
skipped or infrastructure case keeps the suite unsuccessful. The reference run
reported 177 PASS and two Cholesky FAIL results; do not suppress those failures.

Completed remote build/source directories and local work are removed after
collection. Raw remote evidence is retained by the controller for 30 days.
The Mac must remain awake and connected throughout a full run.

## Starting the native runner

macOS may deny a LaunchAgent access to files inside Downloads. On this Mac the
service was removed after that denial; use the prepared foreground launcher:

```sh
/bin/bash /Users/chenhaobo/Downloads/sai-libddla-runner/mac-setup/macos-sai-20261120/start-macos.command
```

Keep that Terminal session open. The launcher prevents idle sleep only while
the runner is active. Registration remains saved when the listener stops, but
this setup does not provide automatic startup after login or reboot.

Local setup checks, run identifiers and collected evidence are kept in
`/Users/chenhaobo/Downloads/sai-libddla-runner/mac-setup/macos-sai-20261120`.
No registration token is stored in those setup files.
