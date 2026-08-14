"""Build an auditable UrbanSound8K strict-foreground SFX catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.source_catalog import prepare_source_catalog, write_source_catalog
from sceneledger.data.urbansound8k import (
    URBANSOUND8K_LICENSE,
    convert_urbansound8k_records,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-urbansound8k")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-per-kind", type=int, default=30)
    parser.add_argument("--min-per-split", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_root, records = convert_urbansound8k_records(args.root)
    raw_catalog = output / "urbansound8k_raw.jsonl"
    write_source_catalog(raw_catalog, records)
    prepared = output / "prepared"
    report = prepare_source_catalog(
        raw_catalog,
        prepared,
        audio_root=audio_root,
        allowed_licenses={URBANSOUND8K_LICENSE},
        seed=args.seed,
        audit_per_kind=args.audit_per_kind,
        min_records_per_kind_per_split=args.min_per_split,
        min_groups_per_kind_per_split=args.min_per_split,
        min_caption_unique_fraction=0.005,
        required_kinds={"sfx"},
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "eligible_clips": len(records),
                "audio_root": str(audio_root),
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
