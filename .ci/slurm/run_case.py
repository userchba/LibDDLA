#!/usr/bin/env python3
"""Run one CTest-manifest command and emit its JUnit result."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def xml_text(value: str) -> str:
    return "".join(
        char
        for char in value
        if char in "\t\n\r"
        or "\u0020" <= char <= "\ud7ff"
        or "\ue000" <= char <= "\ufffd"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--ranks", type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = [test for test in manifest.get("tests", []) if test["name"] == args.case]
    if len(matches) != 1:
        raise SystemExit(f"expected one manifest entry for {args.case}, found {len(matches)}")
    test = matches[0]
    command = list(test["command"])
    if args.ranks is not None:
        if len(command) < 3 or command[1] != "-n":
            raise SystemExit(f"cannot override ranks for command: {command}")
        command[2] = str(args.ranks)
    properties = {item.get("name"): item.get("value") for item in test.get("properties", [])}
    environment = os.environ.copy()
    for assignment in properties.get("ENVIRONMENT", []) or []:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise SystemExit(f"invalid ENVIRONMENT assignment: {assignment}")
        environment[key] = value

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=properties.get("WORKING_DIRECTORY") or None,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    elapsed = time.monotonic() - started
    sys.stdout.write(completed.stdout)

    suite = ET.Element(
        "testsuite",
        name="LibDDLA",
        tests="1",
        failures=str(int(completed.returncode != 0)),
        time=f"{elapsed:.6f}",
    )
    case = ET.SubElement(
        suite, "testcase", name=args.case, time=f"{elapsed:.6f}"
    )
    if completed.returncode:
        failure = ET.SubElement(
            case, "failure", message=f"process exited with {completed.returncode}"
        )
        failure.text = xml_text(completed.stdout)
    else:
        output = ET.SubElement(case, "system-out")
        output.text = xml_text(completed.stdout)
    args.junit.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(args.junit, encoding="utf-8", xml_declaration=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
