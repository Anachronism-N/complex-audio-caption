#!/usr/bin/env python3
"""Fail unless a source-pool readiness report passed the expected profile."""

from __future__ import annotations

import argparse
import sys

from sceneledger.data.source_readiness import require_source_readiness_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    try:
        payload = require_source_readiness_summary(
            args.report, expected_profile=args.profile
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"source readiness gate passed: {payload['source_pool_id']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
