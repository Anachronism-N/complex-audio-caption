"""CLI for the fail-closed manifest complexity audit."""

from __future__ import annotations

import argparse
import json

from sceneledger.data.complexity_audit import (
    audit_manifest_complexity,
    load_complexity_profile,
    write_complexity_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-audit-complexity")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_manifest_complexity(
        args.manifest, load_complexity_profile(args.config, args.profile)
    )
    write_complexity_report(args.output, report)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "summary": report["summary"],
                "failed_checks": [
                    check["name"] for check in report["checks"] if not check["pass"]
                ],
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
