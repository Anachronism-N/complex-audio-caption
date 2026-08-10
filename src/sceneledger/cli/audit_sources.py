"""Probe and freeze a real-audio source inventory before B3 rendering."""

from __future__ import annotations

import argparse
import json

from sceneledger.data.source_readiness import (
    audit_source_pool,
    load_readiness_profile,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-audit-sources")
    parser.add_argument("--catalog", required=True, help="canonical source JSONL")
    parser.add_argument("--config", required=True, help="versioned readiness YAML")
    parser.add_argument("--profile", required=True, help="profile name, e.g. smoke/release")
    parser.add_argument("--inventory", required=True, help="frozen per-file JSONL")
    parser.add_argument("--report", required=True, help="machine-readable acceptance JSON")
    args = parser.parse_args(argv)

    profile, config_hash = load_readiness_profile(args.config, args.profile)
    summary = audit_source_pool(
        catalog_path=args.catalog,
        inventory_path=args.inventory,
        report_path=args.report,
        profile_name=args.profile,
        profile_config=profile,
        config_sha256=config_hash,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
