#!/usr/bin/env bash
#SBATCH --job-name=libddla-sai-gpu4
#SBATCH --partition=16V100
#SBATCH --account=org-open-sai
#SBATCH --qos=rush-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --export=ALL
# The partition assigns host memory automatically.
set -euo pipefail
exec bash "${2:?trusted controller}/run.sh" "${1:?run directory}" gpu4
