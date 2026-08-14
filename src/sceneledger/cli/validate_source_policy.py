"""Validate source-bank policy and bind prepared catalogs to a profile."""

from __future__ import annotations

import argparse
import json

from sceneledger.data.source_policy import (
    load_source_bank_policy,
    validate_source_bank_policy,
    write_policy_report,
)


def _catalog_argument(value: str) -> tuple[str, str]:
    dataset, separator, path = value.partition("=")
    if not separator or not dataset.strip() or not path.strip():
        raise argparse.ArgumentTypeError("catalog must be DATASET_ID=/path/to/catalog.jsonl")
    return dataset.strip(), path.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-validate-source-policy")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--catalog", action="append", type=_catalog_argument, default=[])
    parser.add_argument("--require-catalogs", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    catalogs = dict(args.catalog)
    if len(catalogs) != len(args.catalog):
        parser.error("each --catalog DATASET_ID may be supplied only once")
    policy = load_source_bank_policy(args.policy)
    report = validate_source_bank_policy(
        policy,
        profile_name=args.profile,
        catalogs=catalogs,
        require_catalogs=args.require_catalogs,
    )
    if args.output:
        write_policy_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
