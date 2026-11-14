#!/usr/bin/env bash
# Load the SAI toolchain inside the compute allocation.
set -eo pipefail
source /etc/profile.d/lmod.sh
module purge
module use /opt/modules/modulefiles/devtools
module load cmake/3.31.6 openmpi/5.0.10-nvhpc26.3-gnu-cuda12-auto
# The NVHPC dependency of Open MPI already supplies SAI NCCL 2.29.3.
# Loading the standalone nccl module together with nvhpc is forbidden by Lmod.
module load gcc/13.3.0 saiblas/2603-gnu-auto cuda/12.9.1
export CMAKE_LIBRARY_PATH=${LIBRARY_PATH:-}
export CMAKE_INCLUDE_PATH=${CPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
export OMPI_MCA_plm_slurm_args=--external-launcher
export PRTE_MCA_plm_slurm_args=--external-launcher
export PYTHONDONTWRITEBYTECODE=1
test "${SLURM_JOB_PARTITION:?}" = 16V100
test "${CUDA_HOME:?}" = /opt/devtools/nvidia/cuda-12.9.1
test "${NCCL_ROOT:?}" = /opt/devtools/nvidia/nccl_2.29.3_cuda12.9_sai_v2.29.3-1-sai.2
test -r "${OPENBLAS_ROOT:?}/lib/libopenblas.so"
case "${MPI_HOME:?}:${OPENBLAS_ROOT:?}" in
  *-avx2:*-avx2) ;;
  *) echo "Unexpected 16V100 dependency architecture" >&2; return 1 ;;
esac
