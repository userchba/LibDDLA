#!/usr/bin/env bash
# Run inside the CPU/CUDA/HIP validation allocation after building ddla_lib.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: $0 <existing-build-dir> <new-empty-scratch-dir>" >&2
    exit 2
fi
build_dir=$(cd "$1" && pwd)
fixture_dir=$(cd "$(dirname "$0")" && pwd)
source_dir=$(cd "$fixture_dir/../.." && pwd)
scratch_dir=$2
if [[ -e "$scratch_dir" ]]; then
    echo "scratch directory must not exist: $scratch_dir" >&2
    exit 2
fi
mkdir -p "$scratch_dir"
scratch_dir=$(cd "$scratch_dir" && pwd)
cmake --install "$build_dir" --prefix "$scratch_dir/original"
mv "$scratch_dir/original" "$scratch_dir/relocated"
cmake -S "$fixture_dir" -B "$scratch_dir/consumer" \
    -DCMAKE_PREFIX_PATH="$scratch_dir/relocated${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}" \
    -DDDLA_FORBIDDEN_PREFIXES="$source_dir;$build_dir;$scratch_dir/original"
cmake --build "$scratch_dir/consumer" --parallel 2
(cd "$scratch_dir/consumer" && ctest --output-on-failure)
cat "$scratch_dir/consumer/installed-target.txt"
