#!/usr/bin/env bash
# Toolchain for KSSG validation jobs (HIP/DCU GPU stage). Sourced by the sbatch.
set -euo pipefail

source /usr/share/Modules/init/bash
# TRAP: default-loaded compiler/rocm/dtk-22.10.1 declares "conflict compiler",
# so module loads silently no-op until ALL compiler/* modules unload. Also its
# PATH puts dtk-22.10.1 first — it must go before DTK 25.04.4 is used.
module unload compiler/rocm/dtk-22.10.1 mpi/hpcx/2.11.0/gcc-7.3.1 compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/11.2.0
module load compiler/cmake/3.25.0
source /public/home/ac5g1n561p/app/hpcx/hpcx-2.13.1-gcc-11.2.1-wangxh/env.sh
module load compiler/dtk/25.04.4
export ROCM_PATH=/public/software/compiler/rocm/dtk-25.04.4
# TRAP: HIP_PATH must be the DTK ROOT (no /hip suffix) or the 25.04.4 clang
# picks dtk-22.10.1 from PATH and enable_language(HIP) cannot find amd_comgr.
export HIP_PATH=$ROCM_PATH
# TRAP: try_compile inside enable_language(HIP) inherits ENV only — a
# ;-joined multi-entry value is treated as ONE literal path. Single entry:
export CMAKE_PREFIX_PATH=$ROCM_PATH/lib64/cmake
# nccl.h and thrust/ ship under $ROCM_PATH/include (the bundled CUDA include
# tree CONFLICTS with hip's vector types — never add it). Needed for
# DDLA_USE_CCL=ON builds.
export NCCL_INCLUDE_DIR=$ROCM_PATH/include
export OPENBLAS=/public/home/ac5g1n561p/app/openblas/0.3.29/lib/libopenblas.so
