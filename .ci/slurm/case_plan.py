#!/usr/bin/env python3
"""Create Slurm array case lists from a CTest JSON manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cross-node-case", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--mpi-wrapper", type=Path, required=True)
    parser.add_argument("--rank-wrapper", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups: dict[int, list[str]] = {}
    cross_node: list[str] = []
    for test in manifest.get("tests", []):
        name = test["name"]
        properties = {
            item.get("name"): item.get("value", "")
            for item in test.get("properties", [])
        }
        ranks = int(properties.get("PROCESSORS", 1))
        command = test.get("command", [])
        expected_prefix = [str(args.mpi_wrapper), "-n", str(ranks), str(args.rank_wrapper)]
        if command[:4] != expected_prefix or len(command) < 5:
            raise SystemExit(f"test {name} does not use the trusted MPI/container wrappers")
        executable = Path(command[4])
        try:
            executable.resolve().relative_to(args.build.resolve())
        except ValueError:
            raise SystemExit(f"test {name} executable is outside the build directory")
        if ":" in command[5:]:
            raise SystemExit(f"test {name} contains an unsupported MPMD separator")
        if name == args.cross_node_case:
            cross_node.append(name)
        else:
            groups.setdefault(ranks, []).append(name)

    if len(cross_node) != 1:
        raise SystemExit(
            f"expected exactly one {args.cross_node_case}, found {len(cross_node)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"normal": [], "cross_node": args.cross_node_case}
    for ranks, names in sorted(groups.items()):
        path = args.output_dir / f"normal-{ranks}.txt"
        path.write_text("\n".join(names) + "\n", encoding="utf-8")
        plan["normal"].append({"ranks": ranks, "count": len(names), "file": str(path)})

    cross_path = args.output_dir / "cross-node.txt"
    cross_path.write_text(args.cross_node_case + "\n", encoding="utf-8")
    plan["cross_node_file"] = str(cross_path)
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
