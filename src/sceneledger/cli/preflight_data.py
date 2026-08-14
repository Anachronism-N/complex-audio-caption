"""Preflight complete scene plans before spending time rendering waveforms."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sceneledger.cli.render import sample_scene_plan
from sceneledger.data.experiment_data import (
    audit_scene_plan_distribution,
    file_sha256,
    load_quality_profile,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-preflight-data")
    parser.add_argument(
        "--config", action="append", required=True, help="data YAML; repeat per fold"
    )
    parser.add_argument("--quality-config", required=True)
    parser.add_argument("--profile", default="release")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    profile, quality_hash = load_quality_profile(args.quality_config, args.profile)
    fold_reports: dict[str, dict] = {}
    for config_value in args.config:
        config_path = Path(config_value).resolve()
        scenes = sample_scene_plan(str(config_path))
        report = audit_scene_plan_distribution(scenes, profile)
        fold_name = str(scenes[0].scene_id).split("_", 1)[0] if scenes else config_path.stem
        if fold_name in fold_reports:
            raise ValueError(f"duplicate preflight fold name: {fold_name}")
        report["config_path"] = str(config_path)
        report["config_sha256"] = file_sha256(config_path)
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        render_config = config_payload.get("render", {})
        recipe_artifacts: dict[str, dict[str, str]] = {}
        for field in ("recipe_plan_path", "recipe_inventory_path"):
            if render_config.get(field):
                artifact = Path(str(render_config[field])).expanduser()
                if not artifact.is_absolute():
                    artifact = (config_path.parent / artifact).resolve()
                recipe_artifacts[field] = {
                    "path": str(artifact),
                    "sha256": file_sha256(artifact),
                }
        if recipe_artifacts:
            report["recipe_artifacts"] = recipe_artifacts
        fold_reports[fold_name] = report

    passed = bool(fold_reports) and all(report["pass"] for report in fold_reports.values())
    payload = {
        "schema_version": "sceneledger-data-preflight-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "status": "passed" if passed else "failed",
        "profile": args.profile,
        "quality_config_path": str(Path(args.quality_config).resolve()),
        "quality_config_sha256": quality_hash,
        "folds": fold_reports,
        "failed_checks": [
            f"{fold}:{check}"
            for fold, report in fold_reports.items()
            for check in report["failed_checks"]
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
