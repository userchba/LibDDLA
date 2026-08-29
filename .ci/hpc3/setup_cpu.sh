#!/usr/bin/env bash
# CPU-side toolchain for HPC3 validation jobs (must NOT be mixed with the
# openmpi/4.1.8-cuda profile used by the GPU job).
set -euo pipefail

source /usr/share/Modules/init/bash
export MODULEPATH=${MODULEPATH:-/usr/share/Modules/modulefiles:/etc/modulefiles:/usr/share/modulefiles}
[ -f /etc/profile.d/nmodules.sh ] && source /etc/profile.d/nmodules.sh
module purge
module load gcc/11.4.0
module load openmpi/4.1.1
module load cmake/3.25.3
export OPENBLAS=/data/home/renxg/app/toolchain/260328/toolchain/install/openblas-0.3.29/lib/libopenblas.so
