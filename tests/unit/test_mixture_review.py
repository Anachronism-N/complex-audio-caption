from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import sceneledger.eval.mixture_review as review
from sceneledger.data.experiment_data import file_sha256
from sceneledger.data.manifests import ManifestEntry, read_manifest, write_manifest
from sceneledger.data.scene_recipes import (
    SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION,
    SceneRecipe,
    SourceTimelineSelection,
    write_recipes,
)


def _recipe(index: int, arm: str) -> SceneRecipe:
    return SceneRecipe(
        schema_version=SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION,
        recipe_id=f"recipe_{index:03d}",
        seed=index,
        template="speech_with_sfx",
        context="home",
        difficulty="medium",
        label_preferences_by_kind={"sfx": [f"{arm}_effect_{index}"]},
        relations=["overlap"],
        rationale=f"A valid {arm} source and timeline plan.",
        proposal_source=(
            "rules:source-timeline-keyword_v1" if arm == "rule" else "llm-source-timeline:fixture"
        ),
        scene_duration_sec=4.0,
        source_plan=[
            SourceTimelineSelection(
                slot_id="speech_1",
                kind="speech",
                catalog_source_id=f"{arm}:speech:{index}",
                onset_sec=0.2,
            ),
            SourceTimelineSelection(
                slot_id="sfx_1",
                kind="sfx",
                catalog_source_id=f"{arm}:sfx:{index}",
                onset_sec=1.1,
            ),
        ],
        proposal_metadata={"task_sha256": f"task-{index}"},
    )


def _ledger(sample_id: str, arm: str) -> dict:
    return {
        "schema_version": "0.2.0",
        "sample_id": sample_id,
        "duration_sec": 4.0,
        "time_resolution_sec": 0.1,
        "tracks": [
            {
                "id": "T1",
                "kind": "speech",
                "spans": [{"start_sec": 0.2, "end_sec": 1.2}],
                "confidence": 1.0,
            },
            {
                "id": "T2",
                "kind": "sfx",
                "spans": [{"start_sec": 1.1, "end_sec": 2.1}],
                "confidence": 1.0,
            },
        ],
        "events": [
            {
                "id": "E001",
                "type": "speech",
                "track_id": "T1",
                "spans": [{"start_sec": 0.2, "end_sec": 1.2}],
                "text": f"{arm} speech",
                "confidence": 1.0,
            },
            {
                "id": "E002",
                "type": "sfx",
                "track_id": "T2",
                "spans": [{"start_sec": 1.1, "end_sec": 2.1}],
                "text": f"{arm} effect",
                "confidence": 1.0,
            },
        ],
    }


def _manifest(
    tmp_path: Path,
    recipes: list[SceneRecipe],
    recipe_path: Path,
    arm: str,
) -> tuple[Path, Path]:
    audio_root = tmp_path / arm
    entries = []
    recipe_hash = file_sha256(recipe_path)
    for index, recipe in enumerate(recipes, 1):
        sample_id = f"{arm}_{index:03d}"
        mixture = audio_root / "audio" / f"{sample_id}.wav"
        speech_stem = audio_root / "audio" / "stems" / f"{sample_id}_SP01.wav"
        sfx_stem = audio_root / "audio" / "stems" / f"{sample_id}_SF02.wav"
        for path, payload in (
            (mixture, f"{arm}-mix-{index}".encode()),
            (speech_stem, f"{arm}-speech-{index}".encode()),
            (sfx_stem, f"{arm}-sfx-{index}".encode()),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        entries.append(
            ManifestEntry(
                scene={
                    "scene_id": sample_id,
                    "seed": recipe.seed,
                    "duration": 4.0,
                    "template": recipe.template,
                    "sources": [
                        {
                            "source_id": "SP01",
                            "kind": "speech",
                            "path": recipe.source_plan[0].catalog_source_id,
                            "onset": 0.2,
                            "gain_db": 0.0,
                        },
                        {
                            "source_id": "SF02",
                            "kind": "sfx",
                            "path": recipe.source_plan[1].catalog_source_id,
                            "onset": 1.1,
                            "gain_db": -3.0,
                        },
                    ],
                    "recipe_metadata": {
                        "recipe_id": recipe.recipe_id,
                        "recipe_plan_sha256": recipe_hash,
                        "source_plan": [item.model_dump() for item in recipe.source_plan],
                    },
                },
                mixture_path=f"audio/{sample_id}.wav",
                stem_paths={
                    "SP01": f"audio/stems/{sample_id}_SP01.wav",
                    "SF02": f"audio/stems/{sample_id}_SF02.wav",
                },
                mixture_hash=f"mix-{arm}-{index}",
                dry_mixture_hash=f"dry-{arm}-{index}",
                stem_hashes={"SP01": "speech", "SF02": "sfx"},
                activity_hashes={"SP01": "speech-mask", "SF02": "sfx-mask"},
                target_ledger=_ledger(sample_id, arm),
                sample_rate=16000,
            )
        )
    manifest_path = audio_root / "manifest.jsonl"
    write_manifest(manifest_path, entries)
    return manifest_path, audio_root


def _quality_report(tmp_path: Path, manifest_path: Path, arm: str) -> Path:
    report_path = tmp_path / f"{arm}_mixture_quality.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "sceneledger-mixture-quality-v1",
                "pass": True,
                "failed_checks": [],
                "manifest_sha256": file_sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_mixture_review_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    rule_recipes = [_recipe(index, "rule") for index in range(1, 5)]
    llm_recipes = [_recipe(index, "llm") for index in range(1, 5)]
    rule_recipe_path = tmp_path / "rule_recipes.jsonl"
    llm_recipe_path = tmp_path / "llm_recipes.jsonl"
    write_recipes(rule_recipe_path, rule_recipes)
    write_recipes(llm_recipe_path, llm_recipes)
    rule_manifest, rule_root = _manifest(tmp_path, rule_recipes, rule_recipe_path, "rule")
    llm_manifest, llm_root = _manifest(tmp_path, llm_recipes, llm_recipe_path, "llm")
    rule_quality = _quality_report(tmp_path, rule_manifest, "rule")
    llm_quality = _quality_report(tmp_path, llm_manifest, "llm")

    rows, metadata, key = review.prepare_mixture_review(
        rule_recipes_path=rule_recipe_path,
        rule_manifest_path=rule_manifest,
        rule_quality_report_path=rule_quality,
        rule_audio_base=rule_root,
        llm_recipes_path=llm_recipe_path,
        llm_manifest_path=llm_manifest,
        llm_quality_report_path=llm_quality,
        llm_audio_base=llm_root,
        package_dir=tmp_path / "blind_package",
        sample_count=4,
        seed="unit-test",
        blinding_salt="unit-test-secret-salt",
    )
    assignments = {item["task_id"]: item for item in key["assignments"]}
    assert len(metadata["package_files"]) == 24
    assert sum(item["arm_a"] == "llm" for item in key["assignments"]) == 2
    assert all("rule" not in row["audio_a_path"] for row in rows)
    assert all("llm" not in row["audio_b_path"] for row in rows)
    for row in rows:
        row["reviewer"] = "reviewer-1"
        row["reviewed_at_utc"] = "2026-08-17T00:00:00Z"
        llm_side = "a" if assignments[row["task_id"]]["arm_a"] == "llm" else "b"
        rule_side = "b" if llm_side == "a" else "a"
        for side, score, errors in ((llm_side, "5", "0"), (rule_side, "3", "1")):
            for name in review.RATING_NAMES:
                row[f"{side}_{name}_1_5"] = score
            row[f"{side}_inaudible_sources_count"] = errors
            row[f"{side}_unsupported_labels_count"] = errors
        row["preference_a_b_tie"] = llm_side
        row["notes"] = ""

    csv_path = tmp_path / "review.csv"
    metadata_path = tmp_path / "review.metadata.json"
    key_path = tmp_path / "private.key.json"
    review.write_mixture_review(
        rows,
        metadata,
        key,
        csv_path=csv_path,
        metadata_path=metadata_path,
        key_path=key_path,
    )
    summary = review.summarize_mixture_review(
        review_csv_path=csv_path,
        metadata_path=metadata_path,
        key_path=key_path,
        max_sign_p=0.2,
    )
    assert summary["pass"] is True
    assert summary["go_for_scale"] is True
    assert summary["arms"]["llm"]["mean_scene_plausibility"] == 5.0
    assert summary["paired_delta_llm_minus_rule"]["mean_temporal_plausibility"] == 2.0
    assert summary["preference_sample_consensus"]["llm"] == 4

    rows[0]["audio_a_path"] = "tampered.wav"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="immutable mixture review fields changed"):
        review.summarize_mixture_review(
            review_csv_path=csv_path,
            metadata_path=metadata_path,
            key_path=key_path,
        )


def test_mixture_review_rejects_missing_stem_evidence(tmp_path: Path) -> None:
    rule_recipes = [_recipe(1, "rule")]
    llm_recipes = [_recipe(1, "llm")]
    rule_recipe_path = tmp_path / "rule_recipes.jsonl"
    llm_recipe_path = tmp_path / "llm_recipes.jsonl"
    write_recipes(rule_recipe_path, rule_recipes)
    write_recipes(llm_recipe_path, llm_recipes)
    rule_manifest, rule_root = _manifest(tmp_path, rule_recipes, rule_recipe_path, "rule")
    llm_manifest, llm_root = _manifest(tmp_path, llm_recipes, llm_recipe_path, "llm")
    entry = read_manifest(rule_manifest)[0]
    entry.stem_paths = {}
    write_manifest(rule_manifest, [entry])
    rule_quality = _quality_report(tmp_path, rule_manifest, "rule")
    llm_quality = _quality_report(tmp_path, llm_manifest, "llm")

    with pytest.raises(ValueError, match="one isolated stem per source"):
        review.prepare_mixture_review(
            rule_recipes_path=rule_recipe_path,
            rule_manifest_path=rule_manifest,
            rule_quality_report_path=rule_quality,
            rule_audio_base=rule_root,
            llm_recipes_path=llm_recipe_path,
            llm_manifest_path=llm_manifest,
            llm_quality_report_path=llm_quality,
            llm_audio_base=llm_root,
            package_dir=tmp_path / "blind_package",
            sample_count=1,
        )


def test_mixture_review_rejects_quality_report_for_other_manifest(tmp_path: Path) -> None:
    rule_recipes = [_recipe(1, "rule")]
    llm_recipes = [_recipe(1, "llm")]
    rule_recipe_path = tmp_path / "rule_recipes.jsonl"
    llm_recipe_path = tmp_path / "llm_recipes.jsonl"
    write_recipes(rule_recipe_path, rule_recipes)
    write_recipes(llm_recipe_path, llm_recipes)
    rule_manifest, rule_root = _manifest(tmp_path, rule_recipes, rule_recipe_path, "rule")
    llm_manifest, llm_root = _manifest(tmp_path, llm_recipes, llm_recipe_path, "llm")
    rule_quality = _quality_report(tmp_path, rule_manifest, "rule")
    llm_quality = _quality_report(tmp_path, llm_manifest, "llm")
    payload = json.loads(rule_quality.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    rule_quality.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bound to another manifest"):
        review.prepare_mixture_review(
            rule_recipes_path=rule_recipe_path,
            rule_manifest_path=rule_manifest,
            rule_quality_report_path=rule_quality,
            rule_audio_base=rule_root,
            llm_recipes_path=llm_recipe_path,
            llm_manifest_path=llm_manifest,
            llm_quality_report_path=llm_quality,
            llm_audio_base=llm_root,
            package_dir=tmp_path / "blind_package",
            sample_count=1,
        )
