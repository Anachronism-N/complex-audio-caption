from __future__ import annotations

import json
from pathlib import Path

import pytest

from sceneledger.cli.evaluate import main as evaluate_main
from sceneledger.cli.infer import main as infer_main
from sceneledger.cli.validate_experiment_data import main as validate_data_main
from sceneledger.data.experiment_data import (
    audit_mixture_distribution,
    build_split_contract,
    file_sha256,
    require_experiment_data_summary,
    require_ledger_split,
    require_split_manifest,
    write_references,
    write_split_contract,
)
from sceneledger.data.manifests import ManifestEntry, write_manifest
from sceneledger.data.scene_graph_sampler import (
    SceneGraphSampler,
    SceneSamplerConfig,
    SyntheticSourcePool,
)


def _entry(
    sample_id: str,
    source_paths: list[str],
    *,
    template: str = "speech_music_sfx",
    duration: float = 10.0,
    event_spans: list[list[tuple[float, float]]] | None = None,
    track_spans: list[list[tuple[float, float]]] | None = None,
) -> ManifestEntry:
    event_spans = event_spans or [[(0.0, duration)], [(2.0, 3.0)]]
    track_spans = track_spans or event_spans
    sources = [
        {
            "source_id": f"SRC{index:02d}",
            "kind": "music" if index == 1 else "sfx",
            "path": path,
            "onset": 0.0,
            "gain_db": 0.0,
            "text": "event",
        }
        for index, path in enumerate(source_paths, 1)
    ]
    events = [
        {
            "id": f"E{index:03d}",
            "type": "music" if index == 1 else "sfx",
            "track_id": f"T{index}",
            "spans": [
                {"start_sec": start, "end_sec": end} for start, end in spans
            ],
            "text": "event",
            "confidence": 1.0,
        }
        for index, spans in enumerate(event_spans, 1)
    ]
    tracks = [
        {
            "id": f"T{index}",
            "kind": "music" if index == 1 else "sfx",
            "spans": [
                {"start_sec": start, "end_sec": end} for start, end in spans
            ],
            "confidence": 1.0,
        }
        for index, spans in enumerate(track_spans, 1)
    ]
    return ManifestEntry(
        scene={
            "scene_id": sample_id,
            "seed": 1,
            "duration": duration,
            "template": template,
            "sources": sources,
        },
        mixture_path=f"audio/{sample_id}.wav",
        stem_paths={},
        mixture_hash="mixture",
        dry_mixture_hash="dry",
        stem_hashes={},
        activity_hashes={},
        target_ledger={
            "schema_version": "0.2.0",
            "sample_id": sample_id,
            "duration_sec": duration,
            "time_resolution_sec": 0.1,
            "conditions": {"overlap_ratio": None},
            "tracks": tracks,
            "events": events,
            "provenance": {
                "label_level": "human",
                "source_dataset": "fixture",
                "license_status": "test",
            },
        },
        sample_rate=24000,
    )


def _write_three_splits(tmp_path: Path, *, leak: bool = False) -> dict[str, Path]:
    paths = {
        "train": tmp_path / "train.jsonl",
        "val": tmp_path / "val.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    write_manifest(paths["train"], [_entry("train-1", ["train-a.wav", "train-b.wav"])])
    write_manifest(
        paths["val"],
        [_entry("val-1", ["train-a.wav" if leak else "val-a.wav", "val-b.wav"])],
    )
    write_manifest(paths["test"], [_entry("test-1", ["test-a.wav", "test-b.wav"])])
    return paths


def _profile() -> dict:
    return {
        "global": {
            "min_active_ratio": 0.3,
            "max_single_event_fraction": 0.05,
            "max_low_active_fraction": 0.1,
            "long_trailing_silence_sec": 5.0,
            "max_long_trailing_silence_fraction": 0.1,
            "long_silence_sec": 5.0,
            "max_long_silence_fraction": 0.1,
            "max_duplicate_source_id_fraction": 0.0,
        },
        "sparse_templates": {"names": ["isolated_sfx"], "max_fraction": 0.05},
        "repeated_event": {
            "template": "repeated_event",
            "min_sfx_spans": 2,
            "max_violation_fraction": 0.0,
        },
        "overlapping_speakers": {
            "template": "overlapping_speakers",
            "min_overlap_ratio": 0.1,
            "max_violation_fraction": 0.1,
        },
    }


def test_split_contract_rejects_raw_source_leakage(tmp_path: Path) -> None:
    paths = _write_three_splits(tmp_path, leak=True)
    contract = build_split_contract(
        train_manifest=paths["train"],
        val_manifest=paths["val"],
        test_manifest=paths["test"],
    )
    assert contract["pass"] is False
    assert "train_val_sources_disjoint" in contract["failed_checks"]


def test_split_contract_freezes_manifest_and_ledger_ids(tmp_path: Path) -> None:
    paths = _write_three_splits(tmp_path)
    contract = build_split_contract(
        train_manifest=paths["train"],
        val_manifest=paths["val"],
        test_manifest=paths["test"],
        seed=7,
    )
    assert contract["pass"] is True
    contract_path = tmp_path / "split_contract.json"
    write_split_contract(contract_path, contract)
    assert require_split_manifest(contract_path, "test", paths["test"])["dataset_id"]

    references = tmp_path / "test_references.jsonl"
    assert write_references(paths["test"], references) == 1
    require_ledger_split(contract_path, "test", references, role="reference")

    references.write_text(
        json.dumps({"sample_id": "wrong-id"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="do not match"):
        require_ledger_split(contract_path, "test", references, role="reference")

    paths["test"].write_text(paths["test"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        require_split_manifest(contract_path, "test", paths["test"])


def test_complete_data_gate_rechecks_quality_artifact_hashes(tmp_path: Path) -> None:
    paths = _write_three_splits(tmp_path)
    contract = build_split_contract(
        train_manifest=paths["train"],
        val_manifest=paths["val"],
        test_manifest=paths["test"],
    )
    contract_path = tmp_path / "split_contract.json"
    write_split_contract(contract_path, contract)
    quality_config = tmp_path / "quality.yaml"
    quality_config.write_text("profiles: {}\n", encoding="utf-8")

    reports = {}
    references = {}
    for split in ("train", "val", "test"):
        report_path = tmp_path / f"{split}_quality.json"
        report_path.write_text(
            json.dumps(
                {
                    "pass": True,
                    "failed_checks": [],
                    "manifest_sha256": contract["splits"][split]["manifest_sha256"],
                }
            ),
            encoding="utf-8",
        )
        reports[split] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
            "pass": True,
            "failed_checks": [],
        }
        reference_path = tmp_path / f"{split}_references.jsonl"
        write_references(paths[split], reference_path)
        references[split] = {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "n_samples": 1,
        }

    summary_path = tmp_path / "experiment_data_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "sceneledger-experiment-data-gate-v1",
                "pass": True,
                "failed_checks": [],
                "dataset_id": contract["dataset_id"],
                "split_contract_sha256": file_sha256(contract_path),
                "quality_config_path": str(quality_config),
                "quality_config_sha256": file_sha256(quality_config),
                "quality_reports": reports,
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    assert require_experiment_data_summary(summary_path, contract_path)["pass"] is True

    Path(reports["test"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after data gate"):
        require_experiment_data_summary(summary_path, contract_path)


def test_distribution_gate_rejects_long_sparse_scene(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    write_manifest(
        manifest,
        [
            _entry(
                "sparse",
                ["sfx.wav"],
                template="isolated_sfx",
                duration=15.0,
                event_spans=[[(0.5, 1.0)]],
                track_spans=[[(0.5, 1.0)]],
            )
        ],
    )
    report = audit_mixture_distribution(
        manifest, profile_name="test", profile=_profile()
    )
    assert report["pass"] is False
    assert "single_event_fraction" in report["failed_checks"]
    assert "low_active_fraction" in report["failed_checks"]
    assert "long_trailing_silence_fraction" in report["failed_checks"]


def test_distribution_gate_reports_invalid_ledger_without_crashing(tmp_path: Path) -> None:
    entry = _entry("invalid", ["a.wav", "b.wav"])
    del entry.target_ledger["events"][0]["spans"][0]["end_sec"]
    manifest = tmp_path / "invalid.jsonl"
    write_manifest(manifest, [entry])
    report = audit_mixture_distribution(
        manifest, profile_name="test", profile=_profile()
    )
    assert report["pass"] is False
    assert "all_ledgers_schema_valid" in report["failed_checks"]


def test_distribution_gate_accepts_complex_repeated_and_overlap_scenes(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "good.jsonl"
    entries = [
        _entry("complex", ["music.wav", "sfx.wav"]),
        _entry(
            "repeat",
            ["ambience.wav", "repeat.wav"],
            template="repeated_event",
            event_spans=[[(0.0, 10.0)], [(1.0, 1.5), (5.0, 5.5)]],
        ),
        _entry(
            "overlap",
            ["speaker-a.wav", "speaker-b.wav"],
            template="overlapping_speakers",
            event_spans=[[(0.5, 5.0)], [(1.0, 5.5)]],
            track_spans=[[(0.5, 5.0)], [(1.0, 5.5)]],
        ),
    ]
    write_manifest(manifest, entries)
    report = audit_mixture_distribution(
        manifest, profile_name="test", profile=_profile()
    )
    assert report["pass"] is True
    assert report["metrics"]["repeated_event_violation_fraction"] == 0.0
    assert report["metrics"]["overlap_violation_fraction"] == 0.0


def test_validate_experiment_data_cli_builds_complete_gate(tmp_path: Path) -> None:
    manifests = {}
    for split in ("train", "val", "test"):
        manifest = tmp_path / f"{split}.jsonl"
        write_manifest(
            manifest,
            [
                _entry(
                    f"{split}-complex",
                    [f"{split}-music.wav", f"{split}-sfx.wav"],
                ),
                _entry(
                    f"{split}-repeat",
                    [f"{split}-ambience.wav", f"{split}-repeat.wav"],
                    template="repeated_event",
                    event_spans=[[(0.0, 10.0)], [(1.0, 1.5), (5.0, 5.5)]],
                ),
                _entry(
                    f"{split}-overlap",
                    [f"{split}-speaker-a.wav", f"{split}-speaker-b.wav"],
                    template="overlapping_speakers",
                    event_spans=[[(0.5, 5.0)], [(1.0, 5.5)]],
                    track_spans=[[(0.5, 5.0)], [(1.0, 5.5)]],
                ),
            ],
        )
        manifests[split] = manifest

    quality_config = Path(__file__).resolve().parents[2] / "configs/data/mixture_quality.yaml"
    output_dir = tmp_path / "gate"
    assert (
        validate_data_main(
            [
                "--train-manifest",
                str(manifests["train"]),
                "--val-manifest",
                str(manifests["val"]),
                "--test-manifest",
                str(manifests["test"]),
                "--quality-config",
                str(quality_config),
                "--profile",
                "release",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    summary_path = output_dir / "experiment_data_summary.json"
    summary = require_experiment_data_summary(
        summary_path, output_dir / "split_contract.json"
    )
    assert summary["pass"] is True
    assert (output_dir / "test_references.jsonl").exists()

    predictions = tmp_path / "predictions.jsonl"
    infer_report = tmp_path / "infer_report.json"
    assert (
        infer_main(
            [
                "--manifest",
                str(manifests["test"]),
                "--backend",
                "mock",
                "--split-contract",
                str(output_dir / "split_contract.json"),
                "--data-gate-summary",
                str(summary_path),
                "--expected-split",
                "test",
                "--output",
                str(predictions),
                "--report",
                str(infer_report),
            ]
        )
        == 0
    )
    metrics = tmp_path / "metrics.json"
    assert (
        evaluate_main(
            [
                "--prediction",
                str(predictions),
                "--reference",
                str(output_dir / "test_references.jsonl"),
                "--split-contract",
                str(output_dir / "split_contract.json"),
                "--data-gate-summary",
                str(summary_path),
                "--expected-split",
                "test",
                "--output",
                str(metrics),
            ]
        )
        == 0
    )
    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert metrics_payload["n_samples"] == 3
    assert metrics_payload["experiment_contract"]["split"] == "test"


def test_sampler_v2_creates_dense_template_primitives() -> None:
    pool = SyntheticSourcePool(index_range=(900, 999))
    config = SceneSamplerConfig(
        template_duration_ranges={
            "isolated_sfx": (2.0, 5.0),
            "overlapping_speakers": (5.0, 10.0),
        },
        repeat_range=(2, 5),
        loop_background_to_scene=True,
        enforce_speaker_overlap=True,
        dense_repeated_event=True,
        spread_repeated_event=True,
        stable_unique_source_ids=True,
    )
    sampler = SceneGraphSampler(pool, config)

    isolated = sampler.sample("isolated", 1, "isolated_sfx")
    assert 2.0 <= isolated.duration <= 5.0

    repeated = sampler.sample("repeated", 2, "repeated_event")
    assert [source.kind for source in repeated.sources] == ["ambience", "sfx"]
    assert repeated.sources[0].loop_to_scene is True
    assert repeated.sources[1].repeat >= 2

    overlap = sampler.sample("overlap", 3, "overlapping_speakers")
    assert len({source.source_id for source in overlap.sources}) == 2
    assert abs(overlap.sources[0].onset - overlap.sources[1].onset) <= 0.5

    all_types = sampler.sample("all", 4, "speech_music_lyrics_sfx")
    assert len({source.source_id for source in all_types.sources}) == 4
