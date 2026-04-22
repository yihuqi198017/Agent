#!/usr/bin/env python
"""Run focused medical triage checks for this repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FOCUSED_TESTS = [
    "tests/test_medical_policy.py",
    "tests/test_medical_red_flags.py",
    "tests/test_medical_safety_guard.py",
]


def build_pytest_cmd(full: bool, quiet: bool, maxfail: int) -> list[str]:
    targets = ["tests"] if full else FOCUSED_TESTS
    cmd = [sys.executable, "-m", "pytest", *targets, "--maxfail", str(maxfail)]
    if quiet:
        cmd.append("-q")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run triage safety checks.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full test suite instead of focused medical tests.",
    )
    parser.add_argument(
        "--no-quiet",
        action="store_true",
        help="Show full pytest output.",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=1,
        help="Stop after this many failures (default: 1).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    cmd = build_pytest_cmd(full=args.full, quiet=not args.no_quiet, maxfail=max(1, args.maxfail))

    print(f"[triage-check] cwd={repo_root}")
    print(f"[triage-check] command={' '.join(cmd)}")

    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
