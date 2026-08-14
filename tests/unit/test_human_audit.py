from __future__ import annotations

import json
from pathlib import Path

import pytest

from sceneledger.data.human_audit import (
    build_human_audit,
    read_human_audit,
    require_human_audit_summary,
    summarize_human_audit,
    write_human_audit,
)
from sceneledger.data.manifests import ManifestEntry, write_manifest


def _entry(
    sample_id: str, template: str, *, overlap: bool = False, speech: bool = False
) -> ManifestEntry:
    track_spans = [(0.0, 4.0), (1.0, 3.0)] if overlap else [(0.0, 4.0), (5.0, 7.0)]
    tracks = [
        {
            "id": f"T{index}",
            "kind": "speech" if speech and index == 1 else ("music" if index == 1 else "sfx"),
            "spans": [{"start_sec": start, "end_sec": end}],
            "confidence": 1.0,
        }
        for index, (start, end) in enumerate(track_spans, 1)
    ]
    events = [
        {
            "id": f"E{index:03d}",
            "type": "speech" if speech and index == 1 else ("music" if index == 1 else "sfx"),
            "track_id": f"T{index}",
            "spans": [{"start_sec": start, "end_sec": end}],
            "text": "spoken words" if speech and index == 1 else ("music" if index == 1 else "sound effect"),
            "confidence": 1.0,
        }
        for index, (start, end) in enumerate(track_spans, 1)
    ]
    return ManifestEntry(
        scene={
            "scene_id": sample_id,
            "seed": 1,
            "duration": 8.0,
            "template": template,
            "sources": [],
        },
        mixture_path=f"audio/{sample_id}.wav",
        stem_paths={"M1": f"audio/stems/{sample_id}_M1.wav"},
        mixture_hash="mixture",
        dry_mixture_hash="dry",
        stem_hashes={},
        activity_hashes={},
        target_ledger={
            "schema_version": "0.2.0",
            "sample_id": sample_id,
            "duration_sec": 8.0,
            "time_resolution_sec": 0.1,
            "tracks": tracks,
            "events": events,
        },
        sample_rate=24000,
    )


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            *[_entry(f"alpha-{index}", "alpha") for index in range(4)],
            *[
                _entry(f"overlap-{index}", "overlap", overlap=True)
                for index in range(4)
            ],
        ],
    )
    return path


def _prepare(tmp_path: Path, quality_report: dict | None = None):
    rows, metadata = build_human_audit(
        _manifest(tmp_path),
        quality_report or {"violation_samples": []},
        dataset_id="dataset-test",
        per_template=2,
        max_violation_samples=20,
        seed="fixed-seed",
    )
    csv_path = tmp_path / "audit.csv"
    metadata_path = tmp_path / "audit.meta.json"
    write_human_audit(rows, metadata, csv_path, metadata_path)
    return rows, metadata, csv_path, metadata_path


def _complete(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row.update(
            {
                "reviewer": "reviewer-1",
                "reviewed_at_utc": "2026-08-12T06:00:00Z",
                "event_audibility": "pass",
                "caption_accuracy": "pass",
                "speech_intelligibility": "not_required",
                "speech_transcript_accuracy": "not_required",
                "timestamp_alignment": "pass",
                "overlap_rendering": (
                    "pass" if row["overlap_review_required"] == "yes" else "not_required"
                ),
                "long_silence": "absent",
                "clipping": "absent",
                "stem_mixture_consistency": (
                    "pass" if row["stem_review_required"] == "yes" else "not_required"
                ),
                "severity": "none",
                "overall_decision": "pass",
            }
        )


def test_build_audit_is_deterministic_and_oversamples_violations(tmp_path: Path) -> None:
    initial, _, _, _ = _prepare(tmp_path / "initial")
    selected = {row["sample_id"] for row in initial}
    extra = next(f"alpha-{index}" for index in range(4) if f"alpha-{index}" not in selected)
    quality = {
        "violation_samples": [
            {"sample_id": extra, "reasons": ["long_silence", "low_active_ratio"]}
        ]
    }
    rows_a, meta_a, _, _ = _prepare(tmp_path / "a", quality)
    rows_b, meta_b, _, _ = _prepare(tmp_path / "b", quality)
    assert rows_a == rows_b
    assert meta_a["audit_id"] == meta_b["audit_id"]
    assert meta_a["n_tasks"] == 5
    extra_row = next(row for row in rows_a if row["sample_id"] == extra)
    assert extra_row["selection_reason"] == "quality_violation"
    assert extra_row["quality_violation_reasons"] == "long_silence;low_active_ratio"


def test_completed_clean_audit_passes(tmp_path: Path) -> None:
    rows, _, csv_path, metadata_path = _prepare(tmp_path)
    _complete(rows)
    write_human_audit(
        rows,
        json.loads(metadata_path.read_text(encoding="utf-8")),
        csv_path,
        metadata_path,
    )
    summary = summarize_human_audit(csv_path, metadata_path)
    assert summary["pass"] is True
    assert summary["n_completed"] == len(rows)
    assert summary["failed_checks"] == []
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    loaded = require_human_audit_summary(
        summary_path, expected_dataset_id="dataset-test"
    )
    assert loaded["audit_id"] == summary["audit_id"]

    csv_path.write_text(csv_path.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after summary creation"):
        require_human_audit_summary(summary_path)


def test_incomplete_or_tampered_audit_fails_closed(tmp_path: Path) -> None:
    _, _, csv_path, metadata_path = _prepare(tmp_path)
    summary = summarize_human_audit(csv_path, metadata_path)
    assert summary["pass"] is False
    assert "all_tasks_completed" in summary["failed_checks"]

    rows = read_human_audit(csv_path)
    rows[0]["sample_id"] = "tampered"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    write_human_audit(rows, metadata, csv_path, metadata_path)
    with pytest.raises(ValueError, match="sample IDs or ordering changed"):
        summarize_human_audit(csv_path, metadata_path)


def test_repeated_same_template_issue_is_systematic_failure(tmp_path: Path) -> None:
    rows, _, csv_path, metadata_path = _prepare(tmp_path)
    _complete(rows)
    affected = [row for row in rows if row["template"] == "alpha"]
    assert len(affected) == 2
    for row in affected:
        row["event_audibility"] = "fail"
        row["severity"] = "minor"
        row["overall_decision"] = "fail"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    write_human_audit(rows, metadata, csv_path, metadata_path)
    summary = summarize_human_audit(
        csv_path,
        metadata_path,
        max_total_failures=10,
        template_failure_threshold=2,
    )
    assert summary["pass"] is False
    assert "no_template_systematic_failure" in summary["failed_checks"]
    assert summary["template_failures"] == [
        {"template": "alpha", "criterion": "event_audibility", "count": 2}
    ]


def test_audit_requires_enough_samples_per_template(tmp_path: Path) -> None:
    manifest = tmp_path / "small.jsonl"
    write_manifest(manifest, [_entry("only-one", "alpha")])
    with pytest.raises(ValueError, match="insufficient template coverage"):
        build_human_audit(
            manifest,
            {"violation_samples": []},
            dataset_id="dataset-test",
            per_template=2,
        )


def test_speech_tasks_require_intelligibility_and_transcript_review(tmp_path: Path) -> None:
    manifest = tmp_path / "speech.jsonl"
    write_manifest(manifest, [_entry("speech-one", "speech", speech=True)])
    rows, metadata = build_human_audit(
        manifest,
        {"violation_samples": []},
        dataset_id="speech-dataset",
        per_template=1,
    )
    assert rows[0]["speech_review_required"] == "yes"
    _complete(rows)
    rows[0]["speech_intelligibility"] = "not_required"
    rows[0]["speech_transcript_accuracy"] = "not_required"
    csv_path = tmp_path / "speech.csv"
    metadata_path = tmp_path / "speech.meta.json"
    write_human_audit(rows, metadata, csv_path, metadata_path)

    summary = summarize_human_audit(csv_path, metadata_path)

    assert summary["pass"] is False
    assert "answer_values_valid" in summary["failed_checks"]


def test_all_samples_mode_reviews_every_manifest_entry(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    rows, metadata = build_human_audit(
        manifest,
        {"violation_samples": []},
        dataset_id="all-samples",
        per_template=1,
        all_samples=True,
    )
    assert len(rows) == 8
    assert metadata["sampling"]["all_samples"] is True
    assert {row["selection_reason"] for row in rows} == {"all_samples"}
