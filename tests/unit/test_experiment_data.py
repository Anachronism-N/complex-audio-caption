from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sceneledger.cli.evaluate import main as evaluate_main
from sceneledger.cli.human_audit import main as human_audit_main
from sceneledger.cli.infer import main as infer_main
from sceneledger.cli.validate_experiment_data import main as validate_data_main
from sceneledger.data.experiment_data import (
    audit_mixture_distribution,
    audit_scene_plan_distribution,
    build_split_contract,
    file_sha256,
    require_experiment_data_summary,
    require_ledger_split,
    require_split_manifest,
    scene_plan_sha256,
    write_references,
    write_split_contract,
)
from sceneledger.data.manifests import ManifestEntry, write_manifest
from sceneledger.data.scene_graph_sampler import (
    PlacedSource,
    Scene,
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
    source_kinds: list[str] | None = None,
) -> ManifestEntry:
    event_spans = event_spans or [[(0.0, duration)], [(2.0, 3.0)]]
    track_spans = track_spans or event_spans
    source_kinds = source_kinds or [
        "music" if index == 1 else "sfx"
        for index in range(1, len(source_paths) + 1)
    ]
    sources = [
        {
            "source_id": f"SRC{index:02d}",
            "kind": source_kinds[index - 1],
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
            "type": source_kinds[index - 1],
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
            "kind": source_kinds[index - 1],
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
            "max_duplicate_source_path_fraction": 0.0,
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


def _write_preflight(
    path: Path,
    manifests: dict[str, Path],
    quality_config: Path,
    *,
    profile: str = "test",
) -> Path:
    from sceneledger.data.manifests import read_manifest

    payload = {
        "schema_version": "sceneledger-data-preflight-v1",
        "pass": True,
        "failed_checks": [],
        "profile": profile,
        "quality_config_sha256": file_sha256(quality_config),
        "folds": {
            split: {
                "pass": True,
                "failed_checks": [],
                "scene_plan_sha256": scene_plan_sha256(
                    [entry.scene for entry in read_manifest(manifest)]
                ),
            }
            for split, manifest in manifests.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
    preflight_path = _write_preflight(
        tmp_path / "scene_plan_preflight.json", paths, quality_config
    )

    reports = {}
    complexity_config = tmp_path / "complexity.yaml"
    complexity_config.write_text("profiles: {}\n", encoding="utf-8")
    complexity_reports = {}
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
        complexity_path = tmp_path / f"{split}_complexity.json"
        complexity_path.write_text(
            json.dumps(
                {
                    "schema_version": "sceneledger.complexity_audit.v1",
                    "pass": True,
                    "manifest_sha256": contract["splits"][split]["manifest_sha256"],
                }
            ),
            encoding="utf-8",
        )
        complexity_reports[split] = {
            "path": str(complexity_path),
            "sha256": file_sha256(complexity_path),
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
                "scene_plan_preflight": {
                    "path": str(preflight_path),
                    "sha256": file_sha256(preflight_path),
                    "pass": True,
                },
                "quality_reports": reports,
                "complexity_config_path": str(complexity_config),
                "complexity_config_sha256": file_sha256(complexity_config),
                "complexity_reports": complexity_reports,
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    assert require_experiment_data_summary(summary_path, contract_path)["pass"] is True

    Path(reports["test"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after data gate"):
        require_experiment_data_summary(summary_path, contract_path)

    # Restore the quality artifact and prove complexity evidence is also bound.
    quality_payload = {
        "pass": True,
        "failed_checks": [],
        "manifest_sha256": contract["splits"]["test"]["manifest_sha256"],
    }
    Path(reports["test"]["path"]).write_text(
        json.dumps(quality_payload), encoding="utf-8"
    )
    reports["test"]["sha256"] = file_sha256(reports["test"]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["quality_reports"] = reports
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    Path(complexity_reports["test"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="complexity report.*changed"):
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


def test_track_supervision_gate_requires_identifiable_event_grouping(
    tmp_path: Path,
) -> None:
    entry = _entry(
        "trackable",
        ["speaker-a-1.wav", "speaker-a-2.wav"],
        template="overlapping_speakers",
        duration=3.0,
        event_spans=[[(0.0, 1.0)], [(1.5, 2.5)]],
        track_spans=[[(0.0, 1.0), (1.5, 2.5)], [(0.0, 0.1)]],
        source_kinds=["speech", "speech"],
    )
    entry.target_ledger["events"][1]["track_id"] = "T1"
    manifest = tmp_path / "trackable.jsonl"
    write_manifest(manifest, [entry])
    profile = {
        "global": {
            "min_active_ratio": 0.0,
            "max_single_event_fraction": 1.0,
            "max_low_active_fraction": 1.0,
            "long_trailing_silence_sec": 99.0,
            "max_long_trailing_silence_fraction": 1.0,
            "long_silence_sec": 99.0,
            "max_long_silence_fraction": 1.0,
        },
        "sparse_templates": {"names": [], "max_fraction": 1.0},
        "repeated_event": {"required": False},
        "overlapping_speakers": {
            "required": False,
            "min_overlap_ratio": 0.0,
            "max_violation_fraction": 1.0,
        },
        "track_supervision": {
            "min_pointer_complete_scene_fraction": 1.0,
            "min_multi_event_track_scene_fraction": 1.0,
            "required_source_count": 2,
            "required_event_count": 2,
            "required_track_count": 2,
        },
    }

    passed = audit_mixture_distribution(
        manifest, profile_name="trackable", profile=profile
    )
    assert passed["pass"] is True
    assert passed["metrics"]["multi_event_track_scene_fraction"] == 1.0
    assert passed["metrics"]["track_structure_match_scene_fraction"] == 1.0

    profile["track_supervision"]["required_track_count"] = 3
    wrong_structure = audit_mixture_distribution(
        manifest, profile_name="trackable", profile=profile
    )
    assert wrong_structure["pass"] is False
    assert "track_structure_match_scene_fraction" in wrong_structure["failed_checks"]
    profile["track_supervision"]["required_track_count"] = 2

    entry.target_ledger["events"][1]["track_id"] = "T2"
    write_manifest(manifest, [entry])
    failed = audit_mixture_distribution(
        manifest, profile_name="trackable", profile=profile
    )
    assert failed["pass"] is False
    assert "multi_event_track_scene_fraction" in failed["failed_checks"]


@pytest.mark.parametrize(
    ("speech_amplitude", "sfx_amplitude", "expected_pass"),
    [(0.20, 0.04, True), (0.08, 0.20, False)],
)
def test_stem_audibility_gate_measures_persisted_audio(
    tmp_path: Path,
    speech_amplitude: float,
    sfx_amplitude: float,
    expected_pass: bool,
) -> None:
    sample_rate = 8000
    duration = 1.0
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    stems_dir = tmp_path / "audio" / "stems"
    stems_dir.mkdir(parents=True)
    sf.write(
        stems_dir / "speech.wav",
        speech_amplitude * np.sin(2 * np.pi * 190.0 * time),
        sample_rate,
        subtype="PCM_16",
    )
    sf.write(
        stems_dir / "sfx.wav",
        sfx_amplitude * np.sin(2 * np.pi * 510.0 * time),
        sample_rate,
        subtype="PCM_16",
    )
    entry = _entry(
        "audibility",
        ["speech.wav", "sfx.wav"],
        template="speech_with_sfx",
        duration=duration,
        event_spans=[[(0.0, duration)], [(0.0, duration)]],
        source_kinds=["speech", "sfx"],
    )
    entry.sample_rate = sample_rate
    entry.stem_paths = {
        "SRC01": "audio/stems/speech.wav",
        "SRC02": "audio/stems/sfx.wav",
    }
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [entry])
    profile = {
        "global": {
            "min_active_ratio": 0.3,
            "max_single_event_fraction": 0.0,
            "max_low_active_fraction": 0.0,
            "long_trailing_silence_sec": 0.5,
            "max_long_trailing_silence_fraction": 0.0,
            "long_silence_sec": 0.5,
            "max_long_silence_fraction": 0.0,
        },
        "sparse_templates": {"names": [], "max_fraction": 0.0},
        "repeated_event": {"required": False},
        "overlapping_speakers": {"required": False},
        "stem_audibility": {
            "min_active_rms_dbfs_by_kind": {"speech": -28.0, "sfx": -38.0},
            "max_below_rms_floor_fraction": 0.0,
            "min_speech_competitor_margin_db": 3.0,
            "max_low_speech_margin_fraction": 0.0,
            "min_speech_overlap_measured_fraction": 1.0,
        },
    }
    report = audit_mixture_distribution(
        manifest, profile_name="audibility", profile=profile
    )

    assert report["pass"] is expected_pass
    metrics = report["metrics"]["stem_audibility"]
    assert metrics["n_stems"] == 2
    assert metrics["speech_overlap_measured_fraction"] == 1.0
    if expected_pass:
        assert metrics["minimum_speech_competitor_margin_db"] > 3.0
    else:
        assert "speech_competitor_margin_violation_fraction" in report["failed_checks"]
        assert metrics["minimum_speech_competitor_margin_db"] < 3.0


def test_temporal_evidence_gate_recomputes_spans_from_persisted_stem(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    waveform = np.zeros(sample_rate, dtype=np.float32)
    time = np.arange(int(0.6 * sample_rate), dtype=np.float32) / sample_rate
    waveform[int(0.2 * sample_rate) : int(0.8 * sample_rate)] = (
        0.2 * np.sin(2 * np.pi * 330.0 * time)
    )
    stems_dir = tmp_path / "audio" / "stems"
    stems_dir.mkdir(parents=True)
    sf.write(stems_dir / "sfx.wav", waveform, sample_rate, subtype="PCM_16")
    entry = _entry(
        "temporal",
        ["sfx.wav"],
        duration=1.0,
        event_spans=[[(0.2, 0.8)]],
        track_spans=[[(0.2, 0.8)]],
        source_kinds=["sfx"],
    )
    entry.sample_rate = sample_rate
    entry.stem_paths = {"SRC01": "audio/stems/sfx.wav"}
    entry.target_ledger["events"][0]["attributes"] = {"source_id": "SRC01"}
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [entry])
    profile = {
        "global": {
            "min_active_ratio": 0.0,
            "max_single_event_fraction": 1.0,
            "max_low_active_fraction": 1.0,
            "long_trailing_silence_sec": 99.0,
            "max_long_trailing_silence_fraction": 1.0,
            "long_silence_sec": 99.0,
            "max_long_silence_fraction": 1.0,
        },
        "sparse_templates": {"names": [], "max_fraction": 1.0},
        "repeated_event": {"required": False},
        "overlapping_speakers": {"required": False},
        "temporal_evidence": {
            "min_span_iou": 0.98,
            "max_boundary_error_sec": 0.1,
            "max_violation_fraction": 0.0,
        },
    }

    passed = audit_mixture_distribution(
        manifest, profile_name="temporal", profile=profile
    )
    checks = {item["name"]: item["pass"] for item in passed["checks"]}
    assert checks["stem_ledger_temporal_violation_fraction"] is True

    entry.target_ledger["events"][0]["spans"] = [
        {"start_sec": 0.0, "end_sec": 0.6}
    ]
    write_manifest(manifest, [entry])
    failed = audit_mixture_distribution(
        manifest, profile_name="temporal", profile=profile
    )
    assert "stem_ledger_temporal_violation_fraction" in failed["failed_checks"]
    assert failed["metrics"]["temporal_evidence"]["n_violations"] == 1


def test_validate_experiment_data_cli_builds_complete_gate(tmp_path: Path) -> None:
    manifests = {}
    review_dir = tmp_path / "recipe_reviews"
    review_dir.mkdir()
    for split in ("train", "val", "test"):
        manifest = tmp_path / f"{split}.jsonl"
        recipe_hash = f"{split}-recipe-plan-hash"
        entries = [
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
        ]
        for entry in entries:
            entry.scene["recipe_metadata"] = {
                "recipe_plan_sha256": recipe_hash
            }
        write_manifest(
            manifest,
            entries,
        )
        manifests[split] = manifest
        (review_dir / f"{split}_recipe_review.json").write_text(
            json.dumps(
                {
                    "schema_version": "sceneledger.recipe_human_review.v1",
                    "pass": True,
                    "n_expected": 3,
                    "recipe_plan_sha256": recipe_hash,
                }
            ),
            encoding="utf-8",
        )

    quality_config = tmp_path / "quality.yaml"
    quality_config.write_text(
        json.dumps({"profiles": {"test": _profile()}}), encoding="utf-8"
    )
    preflight_path = _write_preflight(
        tmp_path / "scene_plan_preflight.json",
        manifests,
        quality_config,
        profile="test",
    )
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
                "test",
                "--scene-plan-preflight",
                str(preflight_path),
                "--recipe-review-dir",
                str(review_dir),
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

    audit_csv = output_dir / "human_audit_tasks.csv"
    audit_metadata = output_dir / "human_audit_tasks.meta.json"
    assert (
        human_audit_main(
            [
                "prepare",
                "--manifest",
                str(manifests["test"]),
                "--data-gate-summary",
                str(summary_path),
                "--split-contract",
                str(output_dir / "split_contract.json"),
                "--expected-split",
                "test",
                "--per-template",
                "1",
                "--output-csv",
                str(audit_csv),
                "--output-metadata",
                str(audit_metadata),
            ]
        )
        == 0
    )
    audit_payload = json.loads(audit_metadata.read_text(encoding="utf-8"))
    assert audit_payload["dataset_id"] == summary["dataset_id"]
    assert audit_payload["n_tasks"] == 3

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
                "--inference-report",
                str(infer_report),
                "--output",
                str(metrics),
            ]
        )
        == 0
    )
    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert metrics_payload["n_samples"] == 3
    assert metrics_payload["strict_format_success_rate"] == 1.0
    assert metrics_payload["format_status_complete"] is True
    assert metrics_payload["experiment_contract"]["split"] == "test"
    assert (
        metrics_payload["experiment_contract"]["inference_report_sha256"]
        == file_sha256(infer_report)
    )

    (review_dir / "test_recipe_review.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="recipe review.*changed"):
        require_experiment_data_summary(
            summary_path, output_dir / "split_contract.json"
        )


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


def test_scene_plan_gate_rejects_defined_but_unsampled_complex_templates() -> None:
    pool = SyntheticSourcePool(index_range=(0, 20))
    sampler = SceneGraphSampler(pool, SceneSamplerConfig(stable_unique_source_ids=True))
    simple_scenes = [
        sampler.sample(f"simple-{index}", index, "speech_over_music")
        for index in range(20)
    ]
    profile = {
        "complexity": {
            "simple_max_sources": 2,
            "complex_min_sources": 5,
            "min_mean_source_count": 3.0,
            "simple_fraction_range": [0.1, 0.4],
            "medium_fraction_range": [0.2, 0.7],
            "complex_fraction_range": [0.2, 0.5],
            "required_templates": {"complex_cocktail": 0.05},
        }
    }

    report = audit_scene_plan_distribution(simple_scenes, profile)

    assert report["pass"] is False
    assert "mean_source_count" in report["failed_checks"]
    assert "complex_source_fraction" in report["failed_checks"]
    assert "required_template:complex_cocktail" in report["failed_checks"]


def test_scene_plan_hash_changes_when_scene_conditions_change() -> None:
    pool = SyntheticSourcePool(index_range=(0, 20))
    sampler = SceneGraphSampler(pool, SceneSamplerConfig(stable_unique_source_ids=True))
    scene = sampler.sample("scene", 7, "speech_over_music")
    original = scene_plan_sha256([scene])
    scene.conditions.ducking_enabled = not scene.conditions.ducking_enabled

    assert scene_plan_sha256([scene]) != original


def test_scene_plan_source_diversity_fails_on_two_reused_untraceable_sources() -> None:
    scenes = []
    for index in range(10):
        scenes.append(
            Scene(
                scene_id=f"legacy_{index}",
                seed=index,
                duration=8.0,
                template="speech_with_sfx",
                sources=[
                    PlacedSource("SP01", "speech", f"speech_{index % 2}.wav", 0.0, 0.0, "text"),
                    PlacedSource("SF01", "sfx", f"sfx_{index % 2}.wav", 1.0, -3.0, "sound"),
                ],
            )
        )
    profile = {
        "source_diversity": {
            "min_unique_sources_by_kind": {"speech": 5, "sfx": 5},
            "min_unique_groups_by_kind": {"speech": 3, "sfx": 3},
            "max_source_reuse_fraction": 0.3,
        }
    }

    report = audit_scene_plan_distribution(scenes, profile)

    assert report["pass"] is False
    assert "source_diversity:speech:unique_sources" in report["failed_checks"]
    assert "source_diversity:speech:reuse_fraction" in report["failed_checks"]
    assert "all_sources_have_provenance" in report["failed_checks"]


def test_scene_plan_rejects_truncated_transcript_source() -> None:
    scene = Scene(
        scene_id="truncated-speech",
        seed=1,
        duration=5.0,
        template="speech_with_sfx",
        sources=[
            PlacedSource(
                "SP01",
                "speech",
                "utterance",
                1.0,
                0.0,
                "full transcript",
                source_duration_sec=4.5,
            ),
            PlacedSource("SF01", "sfx", "event", 0.0, -4.0, "sound"),
        ],
    )

    report = audit_scene_plan_distribution([scene], {})

    assert report["pass"] is False
    assert "noncontinuous_sources_fit_scene" in report["failed_checks"]


def test_complex_templates_do_not_reuse_a_source_recording() -> None:
    pool = SyntheticSourcePool(index_range=(0, 20))
    sampler = SceneGraphSampler(pool, SceneSamplerConfig(stable_unique_source_ids=True))
    for template in ("complex_cocktail", "rich_band", "multi_event_dense"):
        for seed in range(25):
            scene = sampler.sample(f"{template}-{seed}", seed, template)
            paths = [source.path for source in scene.sources]
            assert len(paths) == len(set(paths))


def test_sampler_fails_when_pool_cannot_supply_unique_sources() -> None:
    pool = SyntheticSourcePool(index_range=(0, 0))
    sampler = SceneGraphSampler(pool, SceneSamplerConfig(stable_unique_source_ids=True))

    with pytest.raises(ValueError, match="enough unique recordings"):
        sampler.sample("too-small", 1, "complex_cocktail")
