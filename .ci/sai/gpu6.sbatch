#!/usr/bin/env bash
#SBATCH --job-name=libddla-sai-gpu6
#SBATCH --partition=16V100
#SBATCH --account=org-open-sai
#SBATCH --qos=rush-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --gpus-per-node=8
#SBATCH --time=00:25:00
#SBATCH --export=ALL
# rush-gpu requires multiples of four GPUs; six distinct GPUs are used.
# The partition assigns host memory automatically.
set -euo pipefail
exec bash "${2:?trusted controller}/run.sh" "${1:?run directory}" gpu6
