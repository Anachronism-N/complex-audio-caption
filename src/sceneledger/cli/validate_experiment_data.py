"""Create the fail-closed data contract required before a GPU experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sceneledger.data.experiment_data import (
    SPLIT_NAMES,
    audit_mixture_distribution,
    build_split_contract,
    file_sha256,
    load_quality_profile,
    scene_plan_sha256,
    write_references,
    write_split_contract,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sceneledger-validate-experiment-data",
        description=(
            "Freeze explicit train/val/test manifests, reject leakage, audit "
            "temporal density, and export held-out references."
        ),
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--quality-config", required=True)
    parser.add_argument("--profile", default="release")
    parser.add_argument(
        "--scene-plan-preflight",
        required=True,
        help="passed scene_plan_preflight.json produced before rendering",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args(argv)

    manifests = {
        "train": Path(args.train_manifest).resolve(),
        "val": Path(args.val_manifest).resolve(),
        "test": Path(args.test_manifest).resolve(),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile, config_hash = load_quality_profile(args.quality_config, args.profile)
    preflight_path = Path(args.scene_plan_preflight).resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("schema_version") != "sceneledger-data-preflight-v1"
        or preflight.get("pass") is not True
        or preflight.get("failed_checks")
    ):
        raise ValueError("scene-plan preflight is missing, unsupported, or failed")
    if preflight.get("quality_config_sha256") != config_hash:
        raise ValueError("scene-plan preflight used a different quality config")
    contract = build_split_contract(
        train_manifest=manifests["train"],
        val_manifest=manifests["val"],
        test_manifest=manifests["test"],
        seed=args.seed,
    )
    contract_path = output_dir / "split_contract.json"
    write_split_contract(contract_path, contract)

    from sceneledger.data.manifests import read_manifest

    for split in SPLIT_NAMES:
        fold = preflight.get("folds", {}).get(split)
        if not isinstance(fold, dict) or fold.get("pass") is not True:
            raise ValueError(f"scene-plan preflight for {split} is missing or failed")
        entries = read_manifest(manifests[split])
        if scene_plan_sha256([entry.scene for entry in entries]) != fold.get(
            "scene_plan_sha256"
        ):
            raise ValueError(f"rendered {split} scenes differ from preflight plan")

    quality_reports: dict[str, dict] = {}
    references: dict[str, dict] = {}
    for split in SPLIT_NAMES:
        report = audit_mixture_distribution(
            manifests[split],
            profile_name=args.profile,
            profile=profile,
            config_sha256=config_hash,
        )
        report_path = output_dir / f"{split}_mixture_quality.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        quality_reports[split] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
            "pass": report["pass"],
            "failed_checks": report["failed_checks"],
        }
        reference_path = output_dir / f"{split}_references.jsonl"
        reference_count = write_references(manifests[split], reference_path)
        references[split] = {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "n_samples": reference_count,
        }

    passed = contract["pass"] is True and all(
        report["pass"] is True for report in quality_reports.values()
    )
    summary = {
        "schema_version": "sceneledger-experiment-data-gate-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "dataset_id": contract["dataset_id"],
        "quality_profile": args.profile,
        "quality_config_path": str(Path(args.quality_config).resolve()),
        "quality_config_sha256": config_hash,
        "split_contract_path": str(contract_path),
        "split_contract_sha256": file_sha256(contract_path),
        "split_contract_pass": contract["pass"],
        "split_failed_checks": contract["failed_checks"],
        "scene_plan_preflight": {
            "path": str(preflight_path),
            "sha256": file_sha256(preflight_path),
            "pass": True,
        },
        "quality_reports": quality_reports,
        "references": references,
        "failed_checks": [
            *[f"split:{name}" for name in contract["failed_checks"]],
            *[
                f"quality:{split}:{name}"
                for split, report in quality_reports.items()
                for name in report["failed_checks"]
            ],
        ],
    }
    summary_path = output_dir / "experiment_data_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
