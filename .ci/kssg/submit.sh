#!/usr/bin/env bash
# KSSG runner for GitHub Actions: stages the pushed commit, submits the
# HIP/DCU validation sbatch job, waits for it, and checks the result file.
# (CPU validation is owned by the hpc3 workflow; KSSG tests HIP only.)
# No python3 exists on KSSG, so this is pure bash; the summary table is
# rendered on the GitHub runner from the JUnit XML file.
#
# Usage: submit.sh <run_root>
#   <run_root> must contain source/ (the commit), control/ (rendered from
#   .ci/kssg/*.in), build/ and results/ (created here).
set -euo pipefail

run_root=${1:?run root is required}
control=$run_root/control
source_dir=$run_root/source
results=$run_root/results
build_root=$run_root/build
mkdir -p "$control" "$build_root" "$results"

cfg() {
    # config.ini section/key reader; bash-only (no python3 on KSSG).
    # Usage: cfg <section> <key>
    awk -v section="[$1]" -v key="$2" '
        /^[ \t]*$/ { next }
        { sub(/^[ \t]+/, ""); sub(/[ \t]+$/, "") }
        /^\[/ { insec = ($0 == section); next }
        insec {
            split($0, p, "=")
            k = p[1]; sub(/^[ \t]+/, "", k); sub(/[ \t]+$/, "", k)
            if (k == key) { v = p[2]; sub(/^[ \t]+/, "", v); sub(/[ \t]+$/, "", v); print v; exit }
        }
    ' "$control/config.ini"
}

render() {
    # Plain @TOKEN@ replacement (bash-only, no python3 on KSSG).
    local template=$1 out=$2
    shift 2
    local content
    content=$(cat "$template")
    while [ $# -ge 2 ]; do
        content=${content//$1/$2}
        shift 2
    done
    printf '%s' "$content" > "$out"
}

gpu_dcus=$(cfg gpu dcus)
gpu_cpus=$(cfg gpu cpus)
gpu_mem=$(cfg gpu mem)
gpu_time=$(cfg gpu time)
home_dir=$(getent passwd "$USER" 2>/dev/null | cut -d: -f6 || echo "$HOME")
[ -n "$home_dir" ] || home_dir=$HOME

render "$control/validate.sbatch.in" "$control/validate.sbatch" \
    "@HOME@" "$home_dir" \
    "@CONTROL@" "$control" \
    "@SOURCE@" "$source_dir" \
    "@RESULTS@" "$results" \
    "@BUILD_ROOT@" "$build_root" \
    "@BUILD_GPU@" "$build_root/gpu" \
    "@PARTITION@" "kshdnormal" \
    "@CPUS@" "$gpu_cpus" \
    "@DCUS@" "$gpu_dcus" \
    "@MEM@" "$gpu_mem" \
    "@TIME@" "$gpu_time"
chmod 700 "$control/validate.sbatch"

job=$(sbatch --parsable "$control/validate.sbatch")
echo "job=$job" | tee "$results/jobs.txt"

# Wait for the job to leave the queue (bash-only; sacct -X for final state).
while squeue -h -j "$job" | grep -q .; do
    squeue -h -j "$job" -o "%i %T %M %R" >&2 || true
    sleep 30
done
accounting=$(sacct -X -j "$job" --format=State,ExitCode -n -P | head -n 1)
echo "accounting=$accounting" | tee -a "$results/jobs.txt"

if [[ "$accounting" != COMPLETED\|0:0 ]]; then
    echo "validation job failed: $accounting" >&2
    exit 1
fi
for result in "$results/result_gpu.json"; do
    test -s "$result" || { echo "missing $(basename "$result")" >&2; exit 1; }
done
exit 0
