#!/usr/bin/env bash
set -euo pipefail
ddla_n=4
case "${1:-}" in
  -n|-np) ddla_n=${2:?}; shift 2 ;;
  *) echo "Expected -n/-np and a rank count" >&2; exit 2 ;;
esac
[[ "$ddla_n" =~ ^[1-9][0-9]*$ ]]
(( ddla_n <= ${SLURM_NTASKS:?} ))
ddla_control=$(cd -- "$(dirname -- "$0")" && pwd)
exec mpirun -n "$ddla_n" --map-by slot:PE=4 --bind-to core \
  -x DDLA_ALLOCATED_GPU_MASK \
  "$ddla_control/rank.sh" "$@"
