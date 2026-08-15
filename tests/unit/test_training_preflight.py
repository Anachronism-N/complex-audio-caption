from __future__ import annotations

import json
from pathlib import Path

import yaml

from sceneledger.cli.train_preflight import main as preflight_main
from sceneledger.data import training_preflight
from sceneledger.data.manifests import ManifestEntry, write_manifest


def _entry(sample_id: str = "train-1") -> ManifestEntry:
    source = {
        "source_id": "S1",
        "kind": "sfx",
        "path": "sources/esc50/1.wav",
        "source_group": "esc50:1",
        "onset": 0.0,
        "gain_db": 0.0,
        "text": "a dog barks",
    }
    return ManifestEntry(
        scene={
            "scene_id": sample_id,
            "seed": 1,
            "duration": 2.0,
            "template": "isolated_sfx",
            "sources": [source],
        },
        mixture_path=f"audio/{sample_id}.wav",
        stem_paths={"S1": f"audio/stems/{sample_id}_S1.wav"},
        mixture_hash="mixture-hash",
        dry_mixture_hash="dry-mixture-hash",
        stem_hashes={"S1": "stem-hash"},
        activity_hashes={"S1": "activity-hash"},
        target_ledger={
            "schema_version": "0.2.0",
            "sample_id": sample_id,
            "duration_sec": 2.0,
            "time_resolution_sec": 0.1,
            "tracks": [],
            "events": [],
            "provenance": {
                "label_level": "human",
                "source_dataset": "ESC-50",
                "license_status": "test",
            },
        },
        sample_rate=16000,
    )


def _write_config(tmp_path: Path, data: dict, *, dataset_id: str | None = None) -> Path:
    config = {
        "data": data,
        "train": {"seed": 7},
    }
    if dataset_id is not None:
        config["experiment_contract"] = {"dataset_id": dataset_id}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_current_v6k_is_rejected_before_training(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "v6k_preflight.json"

    exit_code = preflight_main(
        [
            "--config",
            "configs/model/b3_real_v6k_3k.yaml",
            "--repo-root",
            str(repo),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["status"] == "rejected_uncontracted_or_invalid"
    assert report["manifest"]["n_samples"] == 1000
    assert report["manifest"]["n_actual_training_samples"] == 702
    assert report["manifest"]["n_internal_validation_samples"] == 298
    assert report["split_diagnostics"]["internal_validation_template_counts"] == {
        "concert_outdoor": 148,
        "restaurant_busy": 150,
    }
    assert report["source_diagnostics"]["n_unique_paths"] == 4
    assert report["source_diagnostics"]["n_placeholder_sources"] == 2740
    assert report["source_diagnostics"]["n_nonempty_mixture_hashes"] == 0
    assert report["source_diagnostics"]["n_nonempty_stem_maps"] == 0


def test_explicit_legacy_override_is_non_publication(tmp_path: Path) -> None:
    manifest = tmp_path / "legacy.jsonl"
    write_manifest(manifest, [_entry()])
    config = _write_config(
        tmp_path,
        {
            "manifest_path": str(manifest),
            "val_fraction": 0.1,
            "group_key": "source_id",
        },
    )

    report = training_preflight.audit_training_config(
        config,
        repo_root=tmp_path,
        allow_exploratory_uncontracted=True,
    )

    assert report["authorized_to_train"] is True
    assert report["publication_eligible"] is False
    assert report["status"] == "exploratory_uncontracted"
    assert "no_valid_frozen_training_contract" in report["publication_blockers"]
    assert "temporary_in_trainer_split" in report["publication_blockers"]


def test_complete_valid_contract_is_authorized(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "train.jsonl"
    write_manifest(manifest, [_entry()])
    split_contract = tmp_path / "split_contract.json"
    data_summary = tmp_path / "experiment_data_summary.json"
    human_summary = tmp_path / "human_audit_summary.json"
    for path in (split_contract, data_summary, human_summary):
        path.write_text("{}\n", encoding="utf-8")
    dataset_id = "frozen-dataset-1"
    monkeypatch.setattr(
        training_preflight,
        "require_experiment_data_summary",
        lambda *_args: {"dataset_id": dataset_id},
    )
    monkeypatch.setattr(
        training_preflight,
        "require_split_manifest",
        lambda *_args: {"dataset_id": dataset_id},
    )
    monkeypatch.setattr(
        training_preflight,
        "require_human_audit_summary",
        lambda *_args, **_kwargs: {"dataset_id": dataset_id},
    )
    config = _write_config(
        tmp_path,
        {
            "manifest_path": str(manifest),
            "pre_split": True,
            "expected_split": "train",
            "split_contract_path": str(split_contract),
            "data_gate_summary_path": str(data_summary),
            "require_human_audit": True,
            "human_audit_summary_path": str(human_summary),
        },
        dataset_id=dataset_id,
    )

    report = training_preflight.audit_training_config(config, repo_root=tmp_path)

    assert report["authorized_to_train"] is True
    assert report["publication_eligible"] is True
    assert report["status"] == "contracted_paper_eligible"
    assert report["publication_blockers"] == []
    assert report["failed_checks"] == []


def test_partial_contract_cannot_be_overridden(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    write_manifest(manifest, [_entry()])
    config = _write_config(
        tmp_path,
        {"manifest_path": str(manifest), "pre_split": True},
    )

    report = training_preflight.audit_training_config(
        config,
        repo_root=tmp_path,
        allow_exploratory_uncontracted=True,
    )

    assert report["authorized_to_train"] is False
    assert report["status"] == "rejected_uncontracted_or_invalid"
    contract_check = next(
        check for check in report["checks"] if check["name"] == "frozen_training_contract_valid"
    )
    assert contract_check["detail"]["partial_contract"] is True
