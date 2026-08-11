"""Audit one rendered manifest against a versioned mixture-quality profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.experiment_data import (
    audit_mixture_distribution,
    load_quality_profile,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-audit-mixtures")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quality-config", required=True)
    parser.add_argument("--profile", default="release")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    profile, config_hash = load_quality_profile(args.quality_config, args.profile)
    report = audit_mixture_distribution(
        args.manifest,
        profile_name=args.profile,
        profile=profile,
        config_sha256=config_hash,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
