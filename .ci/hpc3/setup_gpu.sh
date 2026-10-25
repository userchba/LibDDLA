#!/usr/bin/env bash
# GPU-side toolchain for HPC3 validation jobs (must NOT be mixed with the
# openmpi/4.1.1 profile used by the CPU job).
set -euo pipefail

source /usr/share/Modules/init/bash
export MODULEPATH=${MODULEPATH:-/usr/share/Modules/modulefiles:/etc/modulefiles:/usr/share/modulefiles}
[ -f /etc/profile.d/nmodules.sh ] && source /etc/profile.d/nmodules.sh
module purge
module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3
source /data/home/renxg/app/nvhpc/setup_nvhpc
export OPENBLAS=/data/home/renxg/app/toolchain/260328/toolchain/install/openblas-0.3.29/lib/libopenblas.so

# stage_gpu_libs <dir>: copy the NVHPC runtime libraries every test loads
# (2.5 GB) to <dir> on the node-local disk; put <dir> first in
# LD_LIBRARY_PATH once it returns. From GPFS the loader pages them in at
# ~17 MB/s: on a node that had not loaded them lately the first test took
# 152 s instead of 2.4 s (measured 2026-10-24). Run it in the background
# while CMake configures and builds. A library that fails to copy is
# skipped (its .part file is never loaded).
stage_gpu_libs() {
    local dir=$1 lib d
    mkdir -p "$dir"
    for lib in libcublasLt.so.11 libcublas.so.11 libcusolver.so.11 libcurand.so.10 libnccl.so.2; do
        for d in ${LD_LIBRARY_PATH//:/ }; do
            if [ -e "$d/$lib" ]; then
                cp -L "$d/$lib" "$dir/$lib.part" && mv "$dir/$lib.part" "$dir/$lib"
                break
            fi
        done
    done
}
