#!/usr/bin/env bash
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
    "@BUILD_PARTITION@": sys.argv[3], "@BUILD_QOS@": sys.argv[4],
    "@BUILD_TIME@": sys.argv[5], "@TEST_PARTITION@": sys.argv[6],
    "@TEST_QOS@": sys.argv[7], "@TEST_TIME@": sys.argv[8],
    "@HOME@": sys.argv[9], "@CONTROL@": sys.argv[10],
    "@SOURCE@": sys.argv[11], "@BUILD@": sys.argv[12],
    "@RESULTS@": sys.argv[13], "@MAPPING_ROOT@": sys.argv[14],
    "@DISABLE_NCCL_IB@": sys.argv[15], "@BUILD_JOB@": sys.argv[16],
    "@CONTAINER_IMAGE@": sys.argv[17],
    "@CASE_FILE@": sys.argv[18],
}
for key, value in values.items():
    template = template.replace(key, value)
Path(sys.argv[2]).write_text(template)
PY
}

build_partition=$(cfg build partition)
build_qos=$(cfg build qos)
build_time=$(cfg build time)
test_partition=$(cfg test partition)
test_qos=$(cfg test qos)
test_time=$(cfg test time)
array_concurrency=$(cfg test array_concurrency)
small_test_partition=$(cfg small_test partition)
small_test_qos=$(cfg small_test qos)
small_test_max_gpus=$(cfg small_test max_gpus)
improper_test_partition=$(cfg improper_test partition)
improper_test_qos=$(cfg improper_test qos)
mapping_root=$(cfg cluster mapping_root)
disable_nccl_ib=$(cfg cluster disable_nccl_ib)
home_dir=$(getent passwd "$USER" | cut -d: -f6)
container_image=$(cfg container rootfs)
if [[ "$container_image" == "~/"* ]]; then
    container_image="$home_dir/${container_image#\~/}"
fi

build_script=$control/build.sbatch
render "$control/build.sbatch.in" "$build_script" \
    "$build_partition" "$build_qos" "$build_time" "$test_partition" \
    "$test_qos" "$test_time" "$home_dir" "$control" "$source_dir" \
    "$build" "$results" "$mapping_root" "$disable_nccl_ib" "0" \
    "$container_image" "/dev/null"
chmod 700 "$build_script"
build_job=$(sbatch --parsable "$build_script")
echo "build_job=$build_job" | tee "$results/jobs.txt"

wait_job() {
    local job=$1
    while squeue -h -j "$job" | grep -q .; do
        squeue -h -j "$job" -o "%i %T %M %R" >&2 || true
        sleep 30
    done
    sacct -X -j "$job" --format=State,ExitCode -n -P | head -n 1
}

build_accounting=$(wait_job "$build_job")
echo "build_accounting=$build_accounting" | tee -a "$results/jobs.txt"
if [[ "$build_accounting" != COMPLETED\|0:0 ]]; then
    echo "build failed" >&2
    exit 1
fi

cross_case=$(cfg cross_node case)
python3 "$control/case_plan.py" --manifest "$results/manifest.json" \
    --output-dir "$results/plan" --cross-node-case "$cross_case" \
    --build "$build" --mpi-wrapper "$control/mpirun_with_mapping.sh" \
    --rank-wrapper "$control/rank_in_container.sh"

job_map=$results/job-map.json
printf '{}\n' > "$job_map"
test_jobs=()
submit_cases() {
    local label=$1 case_file=$2 count=$3 ranks=$4
    shift 4
    local test_script=$control/test-${label}.sbatch
    render "$control/test.sbatch.in" "$test_script" \
        "$build_partition" "$build_qos" "$build_time" "$test_partition" \
        "$test_qos" "$test_time" "$home_dir" "$control" "$source_dir" \
        "$build" "$results" "$mapping_root" "$disable_nccl_ib" "$build_job" \
        "$container_image" "$case_file"
    chmod 700 "$test_script"
    local job
    job=$(sbatch --parsable "$@" "$test_script")
    test_jobs+=("$job")
    echo "test_job_${label}=$job" | tee -a "$results/jobs.txt"
    python3 - "$job_map" "$case_file" "$job" "$count" "$ranks" <<'PY'
import json
from pathlib import Path
import sys
path, cases_path, job = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
count, ranks = int(sys.argv[4]), int(sys.argv[5])
data = json.loads(path.read_text())
cases = cases_path.read_text().splitlines()
if len(cases) != count:
    raise SystemExit(f"case count mismatch: expected {count}, found {len(cases)}")
for index, name in enumerate(cases, 1):
    data[name] = {
        "job_id": f"{job}_{index}" if count > 1 else job,
        "ranks": ranks,
    }
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

while read -r ranks count case_file; do
    group_partition=$test_partition
    group_qos=$test_qos
    if (( ranks <= small_test_max_gpus )); then
        group_partition=$small_test_partition
        group_qos=$small_test_qos
    elif (( ranks % 4 != 0 )); then
        group_partition=$improper_test_partition
        group_qos=$improper_test_qos
    fi
    submit_cases "normal-${ranks}" "$case_file" "$count" "$ranks" \
        --partition="$group_partition" --qos="$group_qos" --nodes=1 \
        --ntasks="$ranks" --ntasks-per-node="$ranks" --gpus-per-node="$ranks" \
        --time="$test_time" --array="1-${count}%${array_concurrency}" \
        --output="$results/test-normal-${ranks}-%A_%a.log"
done < <(python3 - "$results/plan/plan.json" <<'PY'
import json, sys
for group in json.load(open(sys.argv[1]))["normal"]:
    print(group["ranks"], group["count"], group["file"])
PY
)

cross_partition=$(cfg cross_node partition)
cross_qos=$(cfg cross_node qos)
cross_nodes=$(cfg cross_node nodes)
cross_gpus=$(cfg cross_node gpus_per_node)
cross_tasks=$(cfg cross_node tasks_per_node)
cross_time=$(cfg cross_node time)
cross_total_tasks=$((cross_nodes * cross_tasks))
submit_cases cross-node "$results/plan/cross-node.txt" 1 "$cross_total_tasks" \
    --partition="$cross_partition" --qos="$cross_qos" --nodes="$cross_nodes" \
    --ntasks="$cross_total_tasks" --ntasks-per-node="$cross_tasks" \
    --gpus-per-node="$cross_gpus" \
    --time="$cross_time" --output="$results/test-cross-node-%j.log"

for job in "${test_jobs[@]}"; do
    accounting=$(wait_job "$job")
    echo "test_accounting_${job}=$accounting" | tee -a "$results/jobs.txt"
done

set +e
python3 "$control/report.py" --manifest "$results/manifest.json" \
    --junit-dir "$results/junit" --job-map "$job_map" --results "$results"
report_rc=$?
set -e
test -s "$results/result.json"
test -s "$results/summary.md"
exit "$report_rc"
