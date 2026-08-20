#!/usr/bin/env python3
"""Turn CTest's manifest and JUnit output into one row per registered test."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--junit-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--job-map", type=Path, required=True)
    return parser.parse_args()


def manifest_tests(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for test in data.get("tests", []):
        properties = {
            item.get("name"): item.get("value", "")
            for item in test.get("properties", [])
        }
        rows.append({
            "name": test["name"],
            "processors": properties.get("PROCESSORS", "1"),
        })
    return rows


def junit_tests(directory: Path) -> dict[str, dict[str, str]]:
    result = {}
    for path in sorted(directory.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for case in root.iter("testcase"):
            name = case.attrib.get("name", "")
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            state = "PASS"
            if failure is not None or error is not None:
                state = "FAIL"
            elif skipped is not None:
                state = "INFRA"
            result[name] = {
                "state": state,
                "duration": case.attrib.get("time", "0"),
            }
    return result


def duration(value: str) -> str:
    try:
        return f"{float(value):.2f}s"
    except ValueError:
        return "-"


def main() -> int:
    args = parse_args()
    expected = manifest_tests(args.manifest)
    actual = junit_tests(args.junit_dir)
    job_map = json.loads(args.job_map.read_text(encoding="utf-8"))
    rows = []
    for index, test in enumerate(expected, 1):
        result = actual.get(test["name"], {"state": "INFRA", "duration": "0"})
        allocation = job_map.get(test["name"], {})
        if isinstance(allocation, str):
            allocation = {"job_id": allocation, "ranks": test["processors"]}
        rows.append({
            "index": index,
            "name": test["name"],
            "processors": allocation.get("ranks", test["processors"]),
            "state": result["state"],
            "duration": result["duration"],
            "job_id": allocation.get("job_id", "-"),
        })

    passed = sum(row["state"] == "PASS" for row in rows)
    failed = sum(row["state"] == "FAIL" for row in rows)
    infrastructure = len(rows) - passed - failed
    result = {
        "protocol": 1,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "infrastructure": infrastructure,
        "cases": rows,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# LibDDLA GPU validation",
        "",
        f"**{passed}/{len(rows)} passed**; {failed} failed; {infrastructure} infrastructure.",
        "",
        "| # | Test case | MPI ranks | State | Duration | Slurm job |",
        "|---:|---|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | `{row['name']}` | {row['processors'] or '-'} | "
            f"{row['state']} | {duration(row['duration'])} | `{row['job_id']}` |"
        )
    (args.results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if rows and passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
