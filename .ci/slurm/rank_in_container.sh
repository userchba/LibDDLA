#!/usr/bin/env bash
set -euo pipefail

: "${CI_CONTAINER_IMAGE:?}"
: "${CI_MPI_ROOT:?}"
: "${CI_SOURCE:?}"
: "${CI_BUILD:?}"
: "${CI_RESULTS:?}"
: "${CI_CONTROL:?}"
: "${CI_MAPPING_ROOT:?}"

container=(apptainer exec --nv --contain --cleanenv
    --bind /opt:/opt:ro --bind /usr:/usr:ro --bind /lib:/lib:ro
    --bind /lib64:/lib64:ro --bind "$CI_MPI_ROOT:$CI_MPI_ROOT:ro"
    --bind "$CI_SOURCE:$CI_SOURCE:ro" --bind "$CI_BUILD:$CI_BUILD"
    --bind "$CI_RESULTS:$CI_RESULTS" --bind "$CI_CONTROL:$CI_CONTROL:ro"
    --bind "$CI_MAPPING_ROOT:$CI_MAPPING_ROOT:ro"
    --env "PATH=$PATH" --env "TMPDIR=$TMPDIR"
    --env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    --env "CPATH=${CPATH:-}" --env "LIBRARY_PATH=${LIBRARY_PATH:-}"
    --env "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}")
if [[ -d /dev/infiniband ]]; then
    container+=(--bind /dev/infiniband:/dev/infiniband)
fi
if [[ -n ${CUDA_MPS_PIPE_DIRECTORY:-} ]]; then
    container+=(--bind "$CUDA_MPS_PIPE_DIRECTORY:/run/nvidia-mps"
        --env CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps)
fi
if [[ -n ${CUDA_MPS_LOG_DIRECTORY:-} ]]; then
    container+=(--bind "$CUDA_MPS_LOG_DIRECTORY:/run/nvidia-mps-log"
        --env CUDA_MPS_LOG_DIRECTORY=/run/nvidia-mps-log)
fi
while IFS= read -r name; do
    case "$name" in
        PMIX_*|OMPI_*|PRTE_*|OPAL_*|SLURM_*|NCCL_*|UCX_*|CUDA_VISIBLE_DEVICES)
            container+=(--env "$name=${!name}")
            ;;
    esac
done < <(compgen -e)
exec "${container[@]}" "$CI_CONTAINER_IMAGE" "$@"
