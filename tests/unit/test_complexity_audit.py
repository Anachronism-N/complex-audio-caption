from __future__ import annotations

from pathlib import Path

from sceneledger.data.complexity_audit import audit_manifest_complexity
from sceneledger.data.manifests import ManifestEntry, write_manifest


def _entry(
    scene_id: str,
    spans: list[tuple[float, float]],
    *,
    complete_provenance: bool = True,
) -> ManifestEntry:
    sources = []
    tracks = []
    events = []
    for index, (start, end) in enumerate(spans):
        source_id = f"S{index + 1}"
        kind = "speech" if index < 2 else ("ambience" if index == 2 else "sfx")
        sources.append(
            {
                "source_id": source_id,
                "kind": kind,
                "path": f"audio/{scene_id}_{source_id}.wav",
                "source_group": f"group:{scene_id}:{source_id}",
                "source_dataset": "unit-test",
                "source_file_sha256": "a" * 64,
                "source_labels": [f"label-{index}"],
            }
        )
        track_id = f"T{index + 1}"
        event_id = f"E{index + 1:03d}"
        span = {"start_sec": start, "end_sec": end}
        tracks.append({"id": track_id, "kind": kind, "spans": [span]})
        events.append(
            {
                "id": event_id,
                "type": kind,
                "track_id": track_id,
                "spans": [span],
                "text": f"event {index}",
            }
        )
    if sources and not complete_provenance:
        sources[0]["path"] = "real:speech"
        sources[0].pop("source_file_sha256")
    return ManifestEntry(
        scene={
            "scene_id": scene_id,
            "duration": 10.0,
            "template": "unit-test",
            "sources": sources,
        },
        mixture_path=f"audio/{scene_id}.wav",
        stem_paths={},
        mixture_hash="",
        dry_mixture_hash="",
        stem_hashes={},
        activity_hashes={},
        target_ledger={"tracks": tracks, "events": events},
        sample_rate=16000,
    )


def _profile() -> dict:
    return {
        "description": "unit-test",
        "complex_definition": {
            "min_sources": 5,
            "min_events": 5,
            "min_overlap_ratio": 0.2,
            "min_simultaneous_tracks": 3,
        },
        "simple_definition": {
            "max_sources": 2,
            "max_events": 2,
            "max_overlap_ratio": 0.05,
            "max_temporal_transitions": 4,
        },
        "gates": {
            "min_scenes": 2,
            "min_complex_scene_fraction": 1.0,
            "max_simple_scene_fraction": 0.0,
            "max_sequential_only_fraction": 0.0,
            "max_block_like_fraction": 0.0,
            "min_multi_voice_scene_fraction": 1.0,
            "min_provenance_complete_fraction": 1.0,
        },
    }


def test_complexity_audit_accepts_overlapping_provenance_complete_scenes(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "complex.jsonl"
    spans = [(0.0, 3.0), (0.5, 3.5), (0.0, 10.0), (2.0, 4.0), (6.0, 7.0)]
    write_manifest(manifest, [_entry("a", spans), _entry("b", spans)])

    report = audit_manifest_complexity(manifest, _profile())

    assert report["pass"] is True
    assert report["summary"]["complex_scene_fraction"] == 1.0
    assert report["summary"]["multi_voice_scene_fraction"] == 1.0
    assert report["summary"]["provenance_complete_fraction"] == 1.0


def test_complexity_audit_rejects_simple_and_provenance_free_scenes(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "simple.jsonl"
    write_manifest(
        manifest,
        [
            _entry("simple", [(0.0, 2.0), (6.0, 8.0)], complete_provenance=False),
            _entry("complex", [(0.0, 3.0), (0.5, 3.5), (0.0, 10.0), (2.0, 4.0), (6.0, 7.0)]),
        ],
    )

    report = audit_manifest_complexity(manifest, _profile())

    assert report["pass"] is False
    assert report["summary"]["simple_scene_fraction"] == 0.5
    assert report["summary"]["provenance_complete_fraction"] == 0.5
    failed = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "min_complex_scene_fraction" in failed
    assert "max_simple_scene_fraction" in failed
    assert "min_provenance_complete_fraction" in failed
