#!/usr/bin/env bash
set -eo pipefail
ddla_root=${1:?run directory}; ddla_kind=${2:?gpu4 or gpu6}
[[ "$ddla_kind" == gpu4 || "$ddla_kind" == gpu6 ]]
ddla_control=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ddla_out=$ddla_root/results/$ddla_kind
mkdir -p "$ddla_out"
ddla_tmp=
finish() {
    ddla_rc=$?
    trap - EXIT
    printf '%s\n' "$ddla_rc" > "$ddla_out/exitcode.txt"
    date -Is > "$ddla_out/finish.txt"
    if [[ -n "$ddla_tmp" ]]; then rm -rf -- "$ddla_tmp"; fi
    exit "$ddla_rc"
}
trap finish EXIT
cp -a "$ddla_control" "$ddla_out/control"
source "$ddla_control/modules.sh"
module -t list > "$ddla_out/modules.txt" 2>&1
unset UCX_TLS OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_coll LD_PRELOAD
ddla_tmp=$(mktemp -d "/tmp/libddla-sai-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR=$ddla_tmp DDLA_ALLOCATED_GPU_MASK=${CUDA_VISIBLE_DEVICES:?}
set -u
date -Is > "$ddla_out/start.txt"
scontrol show job -dd "$SLURM_JOB_ID" > "$ddla_out/allocation.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader > "$ddla_out/gpus.txt"
printf '%s\n' "$MPI_HOME" "$CUDA_HOME" "$NCCL_ROOT" "$OPENBLAS_ROOT" > "$ddla_out/dependency-roots.txt"
cp "$ddla_root/request.json" "$ddla_out/request.json"
if [[ "$ddla_kind" == gpu4 ]]; then
    ddla_np=4
    printf '%s\n' configure > "$ddla_out/phase.txt"
    cmake -S "$ddla_root/source" -B "$ddla_root/build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DDDLA_USE_CUDA=ON -DDDLA_USE_HIP=OFF -DDDLA_USE_CCL=ON \
        -DDDLA_ENABLE_FORTRAN=OFF -DDDLA_ENABLE_PYTHON=OFF \
        -DCMAKE_CXX_COMPILER=/opt/devtools/gcc/13.3.0/bin/g++ \
        -DCMAKE_CUDA_HOST_COMPILER=/opt/devtools/gcc/13.3.0/bin/g++ \
        -DCMAKE_CUDA_COMPILER=/opt/devtools/nvidia/cuda-12.9.1/bin/nvcc \
        -DCUDAToolkit_ROOT=/opt/devtools/nvidia/cuda-12.9.1 -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DDDLA_CPU_BLAS_LIBRARY="$OPENBLAS_ROOT/lib/libopenblas.so" -DDDLA_TEST_NP=4 \
        -DMPIEXEC_EXECUTABLE="$ddla_control/mpiexec.sh" -DMPIEXEC_NUMPROC_FLAG=-n -DMPIEXEC_PREFLAGS= \
        > "$ddla_out/configure.log" 2>&1
    cp "$ddla_root/build/CMakeCache.txt" "$ddla_out/CMakeCache.txt"
    printf '%s\n' build > "$ddla_out/phase.txt"
    cmake --build "$ddla_root/build" --parallel 16 > "$ddla_out/build.log" 2>&1
    nvcc -arch=sm_70 "$ddla_control/gpu_probe.cu" -o "$ddla_root/build/gpu_probe" > "$ddla_out/probe-build.log" 2>&1
    ddla_ctest=$ddla_root/build
else
    ddla_np=6
    test -s "$ddla_root/build-complete.txt"
    ddla_ctest=$ddla_root/ctest-gpu6
    mkdir -p "$ddla_ctest"
    python3 - "$ddla_root" "$ddla_control" "$ddla_ctest" <<'REGISTER'
from pathlib import Path
import json,sys
root,control,out=map(Path,sys.argv[1:]); lines=[]
for test in ['test_pgeadd_indexed','test_pgemm_branches']:
    name=test+'_mpi6_1x6'
    command=[name,str(control/'mpiexec.sh'),'-n','6',str(root/'build/tests'/test),'1','6']
    lines.append('add_test('+' '.join(json.dumps(v) for v in command)+')')
    lines.append(f'set_tests_properties({name} PROPERTIES PROCESSORS 6 LABELS gpu ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")')
(out/'CTestTestfile.cmake').write_text('\n'.join(lines)+'\n')
REGISTER
fi
ldd "$ddla_root/build/src/libddla.so" > "$ddla_out/ldd.txt"
python3 - "$ddla_out/ldd.txt" <<'AUDIT'
from pathlib import Path
import sys
text=Path(sys.argv[1]).read_text()
assert 'not found' not in text
for library in ['libcudart.so.12','libcublas.so.12','libcusolver.so.11']:
    line=next(line for line in text.splitlines() if line.lstrip().startswith(library+' '))
    assert '/cuda-12.9.1/' in line,line
line=next(line for line in text.splitlines() if line.lstrip().startswith('libnccl.so.2 '))
assert '/nccl_2.29.3_cuda12.9_sai_v2.29.3-1-sai.2/' in line,line
assert 'libcudart.so.11' not in text and 'libcudart.so.13' not in text
AUDIT
"$ddla_control/mpiexec.sh" -n "$ddla_np" "$ddla_root/build/gpu_probe" > "$ddla_out/rank-gpu-mapping.txt" 2>&1
python3 - "$ddla_out/rank-gpu-mapping.txt" "$ddla_np" <<'MAPPING'
from pathlib import Path
import re,sys
text=Path(sys.argv[1]).read_text(); expected=int(sys.argv[2])
uuids=re.findall(r'\buuid=([0-9a-f]{32})\b',text)
ranks=re.findall(r'\brank=(\d+)\b',text)
assert len(uuids)==len(set(uuids))==expected,(uuids,expected)
assert sorted(map(int,ranks))==list(range(expected)),ranks
MAPPING
sha256sum "$ddla_root/build/src/libddla.so" "$ddla_root/build/tests/test_pgeadd_indexed" \
    "$ddla_root/build/tests/test_pgemm_branches" > "$ddla_out/binary-sha256.txt"
if [[ "$ddla_kind" == gpu4 ]]; then cp "$ddla_out/binary-sha256.txt" "$ddla_root/build-complete.txt"; fi
ctest --test-dir "$ddla_ctest" --show-only=json-v1 > "$ddla_out/ctest-inventory.json"
printf '%s\n' ctest > "$ddla_out/phase.txt"
set +e
ctest --test-dir "$ddla_ctest" --output-on-failure --parallel 1 --timeout 600 --no-tests=error \
    --output-junit "$ddla_out/ctest.xml" > "$ddla_out/ctest.log" 2>&1
ddla_ctest_rc=$?
printf '%s\n' "$ddla_ctest_rc" > "$ddla_out/ctest-exitcode.txt"
set -e
cp "$ddla_ctest/Testing/Temporary/LastTest.log" "$ddla_out/LastTest.log"
if [[ -f "$ddla_ctest/Testing/Temporary/LastTestsFailed.log" ]]; then
    cp "$ddla_ctest/Testing/Temporary/LastTestsFailed.log" "$ddla_out/LastTestsFailed.log"
fi
printf '%s\n' finished > "$ddla_out/phase.txt"
exit "$ddla_ctest_rc"
