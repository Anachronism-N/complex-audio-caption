"""Create the fail-closed data contract required before a GPU experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sceneledger.data.complexity_audit import (
    audit_manifest_complexity,
    load_complexity_profile,
)
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
        "--complexity-config",
        default=None,
        help="optional complexity profile config; must be paired with --complexity-profile",
    )
    parser.add_argument("--complexity-profile", default=None)
    parser.add_argument(
        "--recipe-review-dir",
        default=None,
        help=(
            "optional directory containing passed train/val/test_recipe_review.json; "
            "each report is bound to the rendered recipe plan hash"
        ),
    )
    parser.add_argument(
        "--scene-plan-preflight",
        required=True,
        help="passed scene_plan_preflight.json produced before rendering",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args(argv)
    if bool(args.complexity_config) != bool(args.complexity_profile):
        parser.error(
            "--complexity-config and --complexity-profile must be provided together"
        )

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

    recipe_review_reports: dict[str, dict] = {}
    if args.recipe_review_dir:
        review_dir = Path(args.recipe_review_dir).resolve()
        for split in SPLIT_NAMES:
            review_path = review_dir / f"{split}_recipe_review.json"
            review = json.loads(review_path.read_text(encoding="utf-8"))
            entries = read_manifest(manifests[split])
            observed_plan_hashes = {
                str(
                    entry.scene.get("recipe_metadata", {}).get(
                        "recipe_plan_sha256"
                    )
                    or ""
                )
                for entry in entries
            }
            observed_plan_hashes.discard("")
            expected_hash = str(review.get("recipe_plan_sha256") or "")
            review_pass = (
                review.get("schema_version")
                == "sceneledger.recipe_human_review.v1"
                and review.get("pass") is True
                and int(review.get("n_expected", -1)) == len(entries)
                and observed_plan_hashes == {expected_hash}
            )
            recipe_review_reports[split] = {
                "path": str(review_path),
                "sha256": file_sha256(review_path),
                "pass": review_pass,
                "recipe_plan_sha256": expected_hash,
                "observed_manifest_recipe_plan_sha256": sorted(
                    observed_plan_hashes
                ),
            }

    quality_reports: dict[str, dict] = {}
    complexity_reports: dict[str, dict] = {}
    references: dict[str, dict] = {}
    complexity_profile = (
        load_complexity_profile(args.complexity_config, args.complexity_profile)
        if args.complexity_config
        else None
    )
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
        if complexity_profile is not None:
            complexity = audit_manifest_complexity(
                manifests[split], complexity_profile
            )
            complexity_path = output_dir / f"{split}_complexity.json"
            complexity_path.write_text(
                json.dumps(complexity, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            complexity_reports[split] = {
                "path": str(complexity_path),
                "sha256": file_sha256(complexity_path),
                "pass": complexity["pass"],
                "failed_checks": [
                    check["name"]
                    for check in complexity["checks"]
                    if not check["pass"]
                ],
            }
        reference_path = output_dir / f"{split}_references.jsonl"
        reference_count = write_references(manifests[split], reference_path)
        references[split] = {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "n_samples": reference_count,
        }

    passed = (
        contract["pass"] is True
        and all(report["pass"] is True for report in quality_reports.values())
        and all(report["pass"] is True for report in complexity_reports.values())
        and all(
            report["pass"] is True for report in recipe_review_reports.values()
        )
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
        "complexity_profile": args.complexity_profile,
        "complexity_config_path": (
            str(Path(args.complexity_config).resolve())
            if args.complexity_config
            else None
        ),
        "complexity_config_sha256": (
            file_sha256(args.complexity_config) if args.complexity_config else None
        ),
        "complexity_reports": complexity_reports,
        "recipe_review_reports": recipe_review_reports,
        "references": references,
        "failed_checks": [
            *[f"split:{name}" for name in contract["failed_checks"]],
            *[
                f"quality:{split}:{name}"
                for split, report in quality_reports.items()
                for name in report["failed_checks"]
            ],
            *[
                f"complexity:{split}:{name}"
                for split, report in complexity_reports.items()
                for name in report["failed_checks"]
            ],
            *[
                f"recipe_review:{split}"
                for split, report in recipe_review_reports.items()
                if not report["pass"]
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
