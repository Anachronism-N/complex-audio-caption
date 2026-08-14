"""Build an auditable speech catalog from a local Common Voice release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.common_voice import (
    COMMON_VOICE_LICENSE,
    convert_common_voice_records,
)
from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-common-voice")
    parser.add_argument("--root", required=True, help="locale root containing TSVs and clips/")
    parser.add_argument("--release", required=True, help="exact MDC release, e.g. cv-corpus-22.0")
    parser.add_argument("--locale", required=True, help="BCP-47/Common Voice locale, e.g. zh-CN")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-up-votes", type=int, default=2)
    parser.add_argument("--max-down-votes", type=int, default=0)
    parser.add_argument("--max-per-speaker", type=int, default=None)
    parser.add_argument("--audit-per-kind", type=int, default=30)
    parser.add_argument("--min-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = convert_common_voice_records(
        args.root,
        release=args.release,
        locale=args.locale,
        min_up_votes=args.min_up_votes,
        max_down_votes=args.max_down_votes,
        max_per_speaker=args.max_per_speaker,
    )
    raw_catalog = output / "common_voice_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=args.root,
        allowed_licenses={COMMON_VOICE_LICENSE},
        seed=args.seed,
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=args.min_per_split,
        min_groups_per_kind_per_split=args.min_per_split,
        min_caption_unique_fraction=0.95,
        required_kinds={"speech"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "eligible_utterances": len(records),
                "raw_catalog": str(raw_catalog),
                "report": str(prepared / "source_catalog_report.json"),
            },
            ensure_ascii=False,
        )
    )
    if not report["pass"]:
        for check in report.get("checks", []):
            if not check["pass"]:
                print(f"FAIL {check['name']}: {check['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
