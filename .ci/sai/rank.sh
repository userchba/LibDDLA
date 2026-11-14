#!/usr/bin/env bash
set -euo pipefail
: "${DDLA_ALLOCATED_GPU_MASK:?}"
IFS=, read -r -a ddla_devices <<< "$DDLA_ALLOCATED_GPU_MASK"
ddla_local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:?}
if (( ddla_local_rank >= ${#ddla_devices[@]} )); then
  echo "More local ranks than allocated GPUs" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=${ddla_devices[$ddla_local_rank]}
exec "$@"
