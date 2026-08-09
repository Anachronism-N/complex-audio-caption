#!/usr/bin/env python3
"""Collect S1a run summaries into a paper-table-friendly JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "run",
    "git_commit",
    "config_sha256",
    "n_validation",
    "eventness_threshold",
    "activity_threshold",
    "tiou_threshold",
    "macro_event_f1",
    "macro_event_precision",
    "macro_event_recall",
    "micro_event_f1",
    "micro_event_precision",
    "micro_event_recall",
    "macro_seg_f1_100ms",
    "mean_onset_mae",
    "mean_offset_mae",
    "total_hallucination",
    "total_omission",
    "matched_boundary_count",
    "boundary_reference_coverage",
    "matched_onset_mae",
    "matched_offset_mae",
)


def collect(root: str | Path) -> list[dict]:
    root_path = Path(root).resolve()
    rows = []
    for summary_path in sorted(root_path.glob("*/model/run_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = summary_path.with_name("run_manifest.json")
        split_path = summary_path.with_name("split.json")
        if not manifest_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(f"incomplete S1 run evidence near {summary_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("source_leakage_count") != 0:
            raise ValueError(f"source leakage recorded by {split_path}")
        if summary.get("config_sha256") != manifest.get("config_sha256"):
            raise ValueError(f"config hash mismatch near {summary_path}")
        metrics = summary["metrics"]
        rows.append(
            {
                "run": summary_path.parents[1].name,
                "git_commit": summary.get("git_commit"),
                "config_sha256": summary["config_sha256"],
                "n_validation": summary["n_validation"],
                "eventness_threshold": summary["eventness_threshold"],
                "activity_threshold": summary["activity_threshold"],
                "tiou_threshold": summary["tiou_threshold"],
                **{field: metrics[field] for field in FIELDS[7:]},
            }
        )
    if not rows:
        raise FileNotFoundError(f"no */model/run_summary.json files under {root_path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collect-s1-results")
    parser.add_argument("root")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args(argv)
    rows = collect(args.root)

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"n_runs": len(rows), "output_json": str(json_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
