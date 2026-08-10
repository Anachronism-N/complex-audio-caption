#!/usr/bin/env python3
"""Fail unless a B3 data reproduction summary passed every acceptance check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--dataset-id")
    args = parser.parse_args(argv)
    path = Path(args.summary)
    if not path.is_file():
        print(f"B3 data summary missing: {path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read B3 data summary {path}: {exc}", file=sys.stderr)
        return 2
    failed = payload.get("failed_checks", [])
    if payload.get("pass") is not True or failed:
        print(f"B3 data reproduction has not passed: {path}", file=sys.stderr)
        print(json.dumps(failed, ensure_ascii=False), file=sys.stderr)
        return 3
    if not payload.get("dataset_id"):
        print(f"B3 data summary has no dataset_id: {path}", file=sys.stderr)
        return 3
    if args.dataset_id and payload.get("dataset_id") != args.dataset_id:
        print(
            f"B3 dataset ID mismatch: {payload.get('dataset_id')} != {args.dataset_id}",
            file=sys.stderr,
        )
        return 4
    print(f"B3 data gate passed: {payload.get('dataset_id')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
