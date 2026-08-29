#!/usr/bin/env bash
# Wrapper used as MPIEXEC_EXECUTABLE by CMake/CTest for the GPU job.
# CTest launches: <MPIEXEC_EXECUTABLE> <MPIEXEC_NUMPROC_FLAG> <n> <MPIEXEC_PREFLAGS> <test>
# We strip the wrapper-flags, run mpirun -n <n> --oversubscribe, and pin each
# rank to its own GPU (rank % 8) inside the 8-V100 allocation.
set -euo pipefail
n=4; exe=; args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -n) n=$2; shift 2;;
    --oversubscribe) shift;;
    --bind-to) shift 2;;
    "/data/openmpi"*) shift;;
    *) exe=$1; shift; args=("$@"); break;;
  esac
done
: "${exe:?no executable}"
exec mpirun -n "$n" --oversubscribe -x CUDA_VISIBLE_DEVICES -x NCCL_DEBUG=WARN \
  bash -c 'd=$((OMPI_COMM_WORLD_RANK % ${NVIDIA_VISIBLE_DEVICES:-8})); CUDA_VISIBLE_DEVICES=$d exec "$0" "$@"' "$exe" "${args[@]}"
