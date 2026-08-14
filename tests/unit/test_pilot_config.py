from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sceneledger.data.scene_recipes import read_recipes
from sceneledger.data.source_catalog import SourceRecord, write_source_catalog
from scripts.make_complex_speech_sfx_config import main as complex_main
from scripts.make_real_speech_sfx_pilot_config import (
    _passed_audit,
    _rms_ready_catalog,
    main,
)


def _write_audit(path: Path, *, required_splits: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "pass": True,
                "required_splits": required_splits,
                "counts_by_split_kind": {
                    "test": {"speech": 10, "sfx": 3, "ambience": 3}
                },
            }
        ),
        encoding="utf-8",
    )


def test_pilot_config_requires_test_fold_source_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_audit(audit, required_splits=[])

    with pytest.raises(ValueError, match="required-split test"):
        _passed_audit(
            audit,
            "source audit",
            required_test_kinds={"speech"},
            minimum_test_per_kind=10,
        )


def test_pilot_config_accepts_sufficient_test_fold_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_audit(audit, required_splits=["test"])

    assert _passed_audit(
        audit,
        "source audit",
        required_test_kinds={"sfx", "ambience"},
        minimum_test_per_kind=3,
    ) == audit.resolve()


def test_pilot_config_rejects_catalog_without_active_rms(tmp_path: Path) -> None:
    catalog = tmp_path / "test.jsonl"
    catalog.write_text(
        json.dumps({"source_id": "old", "rms_dbfs": -20.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rerun source prepare"):
        _rms_ready_catalog(catalog, "old catalog")


def test_pilot_config_accepts_active_rms_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "test.jsonl"
    catalog.write_text(
        json.dumps(
            {"source_id": "new", "rms_dbfs": -24.0, "active_rms_dbfs": -20.0}
        )
        + "\n",
        encoding="utf-8",
    )
    assert _rms_ready_catalog(catalog, "new catalog") == catalog.resolve()


def test_pilot_config_can_add_fsd50k_and_urbansound8k_banks(
    tmp_path: Path,
) -> None:
    roots: dict[str, Path] = {}
    prepared: dict[str, Path] = {}
    for name in ("librispeech", "esc50", "fsd50k", "urbansound8k"):
        roots[name] = tmp_path / name / "audio"
        roots[name].mkdir(parents=True)
        prepared[name] = tmp_path / name / "prepared"
        prepared[name].mkdir()
        (prepared[name] / "test.jsonl").write_text(
            json.dumps(
                {
                    "source_id": name,
                    "rms_dbfs": -24.0,
                    "active_rms_dbfs": -20.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        kinds = (
            {"speech": 10}
            if name == "librispeech"
            else {"sfx": 3}
            if name == "urbansound8k"
            else {"sfx": 3, "ambience": 3}
        )
        (prepared[name] / "source_audit_report.json").write_text(
            json.dumps(
                {
                    "pass": True,
                    "required_splits": ["test"],
                    "counts_by_split_kind": {"test": kinds},
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "pilot.yaml"

    assert (
        main(
            [
                "--librispeech-root",
                str(roots["librispeech"]),
                "--librispeech-prepared",
                str(prepared["librispeech"]),
                "--esc50-audio-root",
                str(roots["esc50"]),
                "--esc50-prepared",
                str(prepared["esc50"]),
                "--fsd50k-root",
                str(roots["fsd50k"]),
                "--fsd50k-prepared",
                str(prepared["fsd50k"]),
                "--urbansound8k-root",
                str(roots["urbansound8k"]),
                "--urbansound8k-prepared",
                str(prepared["urbansound8k"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert len(config["pool"]["catalogs"]) == 4
    assert config["render"]["sample_count"] == 60


def test_complex_config_binds_rule_recipes_inventory_and_review(
    tmp_path: Path,
) -> None:
    bank_specs = {
        "librispeech": [("speech", "read_english_speech")],
        "esc50": [("ambience", "rain"), ("sfx", "dog_bark")],
        "fsd50k": [
            ("ambience", "wind"),
            ("sfx", "bird_call"),
            ("sfx", "water_drop"),
        ],
    }
    roots: dict[str, Path] = {}
    prepared: dict[str, Path] = {}
    for bank, specs in bank_specs.items():
        roots[bank] = tmp_path / bank / "audio"
        roots[bank].mkdir(parents=True)
        prepared[bank] = tmp_path / bank / "prepared"
        prepared[bank].mkdir()
        records = [
            SourceRecord(
                source_id=f"{bank}:{index}",
                kind=kind,  # type: ignore[arg-type]
                audio_path=f"{bank}_{index}.wav",
                source_group=f"{bank}:group:{index}",
                labels=[label],
                caption=f"{bank} {label}",
                dataset=bank,
                license="CC0-1.0",
                annotation_origin="dataset",
                split="test",
                rms_dbfs=-24.0,
                active_rms_dbfs=-20.0,
            )
            for index, (kind, label) in enumerate(specs)
        ]
        write_source_catalog(prepared[bank] / "test.jsonl", records)
        kinds = {"speech": 10} if bank == "librispeech" else {
            "sfx": 10,
            "ambience": 10,
        }
        (prepared[bank] / "source_audit_report.json").write_text(
            json.dumps(
                {
                    "pass": True,
                    "required_splits": ["test"],
                    "counts_by_split_kind": {"test": kinds},
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "complex.yaml"

    assert (
        complex_main(
            [
                "--librispeech-root",
                str(roots["librispeech"]),
                "--librispeech-prepared",
                str(prepared["librispeech"]),
                "--esc50-audio-root",
                str(roots["esc50"]),
                "--esc50-prepared",
                str(prepared["esc50"]),
                "--fsd50k-root",
                str(roots["fsd50k"]),
                "--fsd50k-prepared",
                str(prepared["fsd50k"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    inventory = output.with_suffix(".inventory.json")
    recipes = output.with_suffix(".rule_recipes.jsonl")
    review = output.with_suffix(".rule_recipe_review.csv")
    assert inventory.is_file()
    assert review.is_file()
    assert len(read_recipes(recipes)) == 120
    assert config["render"]["recipe_plan_path"] == str(recipes)
    assert config["recipe_experiment"]["human_review_path"] == str(review)
