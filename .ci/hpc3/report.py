#!/usr/bin/env python3
"""Turn CTest's JUnit output into result.json + summary.md for one stage.

One stage = the CPU job or the GPU job. ``--kind`` is a short label used in
the summary heading. Rows keep their order from the JUnit file.

Compatible with Python 3.6 (the login-node interpreter on HPC3).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, help="cpu or gpu")
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def junit_tests(path):
    root = ET.parse(str(path)).getroot()
    rows = []
    for case in root.iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        state = "PASS"
        if failure is not None or error is not None:
            state = "FAIL"
        elif skipped is not None:
            state = "INFRA"
        rows.append({
            "name": case.attrib.get("name", ""),
            "classname": case.attrib.get("classname", ""),
            "state": state,
            "duration": case.attrib.get("time", "0"),
        })
    return rows


def duration(value):
    try:
        return "%.2fs" % float(value)
    except ValueError:
        return "-"


def main():
    args = parse_args()
    rows = junit_tests(args.junit)
    if not rows:
        return 2  # no tests parsed -> infra failure
    passed = sum(1 for row in rows if row["state"] == "PASS")
    failed = sum(1 for row in rows if row["state"] == "FAIL")
    infra = len(rows) - passed - failed

    args.results.mkdir(parents=True, exist_ok=True)
    result_path = args.results / ("result_%s.json" % args.kind)
    summary_path = args.results / ("summary_%s.md" % args.kind)
    result_path.write_text(json.dumps({
        "kind": args.kind,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "infrastructure": infra,
        "cases": rows,
    }, indent=2) + "\n", encoding="utf-8")

    label = {"cpu": "CPU", "gpu": "GPU", "gpu4": "GPU (4-rank)", "gpu6": "GPU (6-rank)"}.get(args.kind, "GPU")
    lines = [
        "# LibDDLA HPC3 %s validation" % label,
        "",
        "**%d/%d passed**; %d failed; %d infrastructure." % (
            passed, len(rows), failed, infra),
        "",
        "| # | Test case | State | Duration | Slurm job |",
        "|---:|---|---:|---:|---|",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            "| %d | `%s` | %s | %s | `%s` |" % (
                i, row["name"], row["state"], duration(row["duration"]),
                args.job_id)
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
