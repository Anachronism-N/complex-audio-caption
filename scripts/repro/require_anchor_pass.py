#!/usr/bin/env python3
"""Fail unless the TAG 2021 reproduction summary passed its acceptance gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: require_anchor_pass.py REPRODUCTION_SUMMARY.json", file=sys.stderr)
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"TAG anchor summary missing: {path}", file=sys.stderr)
        print("complete scripts/repro/tag2021/00_bootstrap.sh through 06_summarize.sh", file=sys.stderr)
        return 2
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read TAG anchor summary {path}: {exc}", file=sys.stderr)
        return 2
    if payload.get("pass") is not True:
        print(f"TAG anchor has not passed: {path}", file=sys.stderr)
        print(json.dumps(payload.get("acceptance", {}), ensure_ascii=False), file=sys.stderr)
        return 3
    print(f"TAG anchor gate passed: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
