#!/usr/bin/env bash
# HPC3 runner for GitHub Actions: stages the pushed commit, submits the CPU
# and GPU Slurm jobs (in parallel, independent build dirs), waits for all,
# and writes per-stage result/summary files. The GPU side is split into two
# jobs/partitions to avoid the congested v100g32 partition: gpu4 (4 GPUs,
# the whole 4-rank suite, v100g32fat) and gpu6 (6 GPUs, the 6-rank
# test_ptran, v100m3).
#
# Usage: submit.sh <run_root>
#   <run_root> must contain source/ (the commit), control/ (rendered from
#   .ci/hpc3/*.in), build/ and results/ (created here).
set -euo pipefail

run_root=${1:?run root is required}
control=$run_root/control
source_dir=$run_root/source
build=$run_root/build
results=$run_root/results
mkdir -p "$control" "$build" "$results"

cfg() {
    python3 - "$control/config.ini" "$1" "$2" <<'PY'
import configparser
import sys
config = configparser.ConfigParser()
config.read(sys.argv[1])
print(config[sys.argv[2]][sys.argv[3]].strip())
PY
}

render() {
    python3 - "$@" <<'PY'
from pathlib import Path
import sys
template = Path(sys.argv[1]).read_text()
values = {
    "@HOME@": sys.argv[3], "@CONTROL@": sys.argv[4],
    "@SOURCE@": sys.argv[5], "@RESULTS@": sys.argv[6],
    "@BUILD_CPU@": sys.argv[7], "@BUILD_GPU@": sys.argv[8],
    "@CPU_PARTITION@": sys.argv[9], "@CPU_CPUS@": sys.argv[10],
    "@CPU_MEM@": sys.argv[11], "@CPU_TIME@": sys.argv[12],
    "@GPU4_PARTITION@": sys.argv[13], "@GPU4_CPUS@": sys.argv[14],
    "@GPU4_MEM@": sys.argv[15], "@GPU4_TIME@": sys.argv[16],
    "@GPU4_GPUS@": sys.argv[17], "@BUILD_GPU6@": sys.argv[18],
    "@GPU6_PARTITION@": sys.argv[19], "@GPU6_CPUS@": sys.argv[20],
    "@GPU6_MEM@": sys.argv[21], "@GPU6_TIME@": sys.argv[22],
    "@GPU6_GPUS@": sys.argv[23],
}
for key, value in values.items():
    template = template.replace(key, value)
Path(sys.argv[2]).write_text(template)
PY
}

cpu_partition=$(cfg cpu partition)
cpu_cpus=$(cfg cpu cpus)
cpu_mem=$(cfg cpu mem)
cpu_time=$(cfg cpu time)
gpu4_partition=$(cfg gpu4 partition)
gpu4_cpus=$(cfg gpu4 cpus)
gpu4_mem=$(cfg gpu4 mem)
gpu4_time=$(cfg gpu4 time)
gpu4_gpus=$(cfg gpu4 gpus)
gpu6_partition=$(cfg gpu6 partition)
gpu6_cpus=$(cfg gpu6 cpus)
gpu6_mem=$(cfg gpu6 mem)
gpu6_time=$(cfg gpu6 time)
gpu6_gpus=$(cfg gpu6 gpus)
home_dir=$(getent passwd "$USER" | cut -d: -f6)
build_cpu=$build/cpu
build_gpu=$build/gpu
build_gpu6=$build/gpu6

render "$control/validate_cpu.sbatch.in" "$control/validate_cpu.sbatch" \
    "$home_dir" "$control" "$source_dir" "$results" "$build_cpu" "$build_gpu" \
    "$cpu_partition" "$cpu_cpus" "$cpu_mem" "$cpu_time" \
    "$gpu4_partition" "$gpu4_cpus" "$gpu4_mem" "$gpu4_time" "$gpu4_gpus" \
    "$build_gpu6" \
    "$gpu6_partition" "$gpu6_cpus" "$gpu6_mem" "$gpu6_time" "$gpu6_gpus"
render "$control/validate_gpu4.sbatch.in" "$control/validate_gpu4.sbatch" \
    "$home_dir" "$control" "$source_dir" "$results" "$build_cpu" "$build_gpu" \
    "$cpu_partition" "$cpu_cpus" "$cpu_mem" "$cpu_time" \
    "$gpu4_partition" "$gpu4_cpus" "$gpu4_mem" "$gpu4_time" "$gpu4_gpus" \
    "$build_gpu6" \
    "$gpu6_partition" "$gpu6_cpus" "$gpu6_mem" "$gpu6_time" "$gpu6_gpus"
render "$control/validate_gpu6.sbatch.in" "$control/validate_gpu6.sbatch" \
    "$home_dir" "$control" "$source_dir" "$results" "$build_cpu" "$build_gpu" \
    "$cpu_partition" "$cpu_cpus" "$cpu_mem" "$cpu_time" \
    "$gpu4_partition" "$gpu4_cpus" "$gpu4_mem" "$gpu4_time" "$gpu4_gpus" \
    "$build_gpu6" \
    "$gpu6_partition" "$gpu6_cpus" "$gpu6_mem" "$gpu6_time" "$gpu6_gpus"
chmod 700 "$control/validate_cpu.sbatch" "$control/validate_gpu4.sbatch" \
    "$control/validate_gpu6.sbatch" "$control/gpurank_wrapper.sh"

wait_job() {
    local job=$1
    while squeue -h -j "$job" | grep -q .; do
        squeue -h -j "$job" -o "%i %T %M %R" >&2 || true
        sleep 30
    done
    sacct -X -j "$job" --format=State,ExitCode -n -P | head -n 1
}

cpu_job=$(sbatch --parsable "$control/validate_cpu.sbatch")
echo "cpu_job=$cpu_job" | tee "$results/jobs.txt"
gpu4_job=$(sbatch --parsable "$control/validate_gpu4.sbatch")
echo "gpu4_job=$gpu4_job" | tee -a "$results/jobs.txt"
gpu6_job=$(sbatch --parsable "$control/validate_gpu6.sbatch")
echo "gpu6_job=$gpu6_job" | tee -a "$results/jobs.txt"

overall=0
for stage in cpu gpu4 gpu6; do
    eval job="\$${stage}_job"
    accounting=$(wait_job "$job")
    echo "${stage}_accounting=$accounting" | tee -a "$results/jobs.txt"
    if [[ "$accounting" != COMPLETED\|0:0 ]]; then
        echo "$stage job failed: $accounting" >&2
        overall=1
        continue
    fi
    test -s "$results/result_$stage.json" || { echo "missing result_$stage.json" >&2; overall=1; }
done
exit "$overall"
