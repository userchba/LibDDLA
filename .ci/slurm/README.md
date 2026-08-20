# SAI GPU validation

The `GPU validation` workflow sends the selected committed LibDDLA source to
the SAI cluster, builds the CUDA backend, and runs every CTest entry through
Slurm. The GitHub job summary contains one table row per registered test. Raw
CTest, JUnit, build, module, GPU, and Slurm logs are retained as an artifact.

The build uses one V100. After it produces the CTest manifest, ordinary tests
are grouped by their `PROCESSORS` property and submitted as Slurm arrays. Each
array element runs exactly one test on one node with one GPU per MPI rank, so
the common four-rank tests request four GPUs and `test_ptran_mpi` requests six.
The array concurrency limit is public configuration and Slurm schedules the
elements independently. Each element executes the exact command and working
directory recorded by CTest and emits its own JUnit file; concurrent elements
do not write CTest's shared `Testing/Temporary` state.

The SAI `flood-gpu` QoS has a four-GPU minimum, while `flood-1o2gpu` permits
one or two GPUs. Rank groups at or below that small-job limit use
`4V100/flood-1o2gpu`; larger groups use `16V100/flood-gpu`. This keeps the two
one-rank cases at one GPU without violating either QoS or the site's mapping
requirement that ranks be divisible by allocated GPUs. Larger rank counts that
are not multiples of four, such as `test_ptran_mpi` at six ranks, use the
site's `improper-gpu` QoS because the ordinary GPU QoS enforces `4N` GPUs per
node.

`test_pzgemm_mpi` is the one cross-node validation: two nodes, one V100 and one
MPI rank per node under `flood-1o2gpu`. The test supports this two-rank grid;
the runner overrides its normal four-rank registration with the allocation's
actual rank count, which is also recorded in the report. Slurm arrays cannot
contain heterogeneous allocations, so this case is a separate job.

## GitHub setup

The repository has two Actions environments:

- `gpu-ci-manual` for `workflow_dispatch` runs
- `gpu-ci-scheduled` for the daily scheduled run

Each environment requires `REMOTE_USER` and `REMOTE_SSH_PRIVATE_KEY` secrets.
Host verification uses the committed `known_hosts` entry; do not disable it.

Manual and scheduled runs default to the `develop` branch. A manual run may
provide another branch name or full commit SHA through `source_sha`.

## Remote layout

Each run is created below:

```text
<project_root>/runs/github/<github_run_id>-<attempt>/
```

The `results/` directory contains `result.json`, `summary.md`, per-case JUnit
and diagnostic logs, the case-to-job mapping, build logs, and Slurm output.
The workflow downloads this directory even when tests fail.

The build and test jobs use the locally maintained tiny rootfs in `config.ini`.
It only contains a shell and loader; CMake, CUDA, and MPI tools come from the
host `/opt` bind. The host supplies the NVIDIA driver through `--nv`; `/opt`,
`/usr`, `/lib`, and `/lib64` are explicitly bind-mounted read-only,
MPI installation, source, control, and mapping directories are explicitly
bind-mounted read-only, while only build and results are writable. The image
is never taken from the checked-out source, and `--containall --cleanenv`
prevents implicit host-home, temporary-directory, and environment access.
The trusted runner invokes the host MPI launcher, and MPI wraps every test
rank in `apptainer exec`; this preserves Slurm's native multi-node allocation
while keeping all test executables inside the container. Rank containers use
`--contain --cleanenv` rather than `--containall` so same-node MPI transports
can use the allocation's IPC namespace; their filesystem, home, temporary
directories, and environment remain contained. `--cleanenv`
is retained, with only PMIx/OpenMPI runtime variables, Slurm allocation
metadata, the site mapping's NCCL/UCX settings, and CUDA device visibility
passed into each rank so it can join the host launcher's communicator. The
mapping runs from the bound results directory, making its generated NCCL
topology file available at the same path inside every rank container.
For cross-node NCCL, the rank wrapper also binds the Slurm-visible node-local
`/dev/infiniband` device directory; unlike ordinary filesystem binds this is
writable because RDMA character devices must accept client operations.

Create or refresh the rootfs once as the remote workflow user:

```bash
.ci/slurm/create-rootfs.sh ~/libddla_gpu_ci/rootfs
```

The generated sandbox is about 2 MB on the current SAI login image. It is not
downloaded from a registry and contains no compiler, MPI, or CUDA installation.
NVIDIA utility/configuration mount points are pre-created so `--nv` can inject
the GPU node's local driver tools. When the site mapping enables CUDA MPS, its
job-specific pipe and log directories are bind-mounted into the container.
The OpenMPI root is resolved from the loaded module instead of being pinned in
the repository configuration.
