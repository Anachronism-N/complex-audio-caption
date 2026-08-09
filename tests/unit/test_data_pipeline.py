"""CPU-only tests for manifest auditing, splitting, and MOSS export."""

from __future__ import annotations

import json
from pathlib import Path

from sceneledger.cli.prepare_moss_sft import export_moss_sft
from sceneledger.cli.prepare_sources import main as prepare_sources_main
from sceneledger.data.datamodule import group_split, source_leakage
from sceneledger.data.manifests import (
    ManifestEntry,
    audit_manifest_structure,
    write_manifest,
)
from sceneledger.data.renderer import RESIDUAL_STEM_ID
from sceneledger.data.schema import Ledger
from sceneledger.data.source_catalog import load_source_catalog


def _entry(scene_id: str, paths: list[str]) -> ManifestEntry:
    sources = [
        {
            "source_id": f"FX{index:02d}",
            "kind": "sfx",
            "path": path,
            "onset": 0.0,
            "gain_db": 0.0,
            "text": "event",
        }
        for index, path in enumerate(paths, 1)
    ]
    component_ids = [source["source_id"] for source in sources] + [RESIDUAL_STEM_ID]
    ledger = Ledger(sample_id=scene_id, duration_sec=10.0)
    return ManifestEntry(
        scene={
            "scene_id": scene_id,
            "seed": 1,
            "duration": 10.0,
            "template": "isolated_sfx",
            "sources": sources,
        },
        mixture_path=f"audio/{scene_id}.wav",
        stem_paths={key: f"stems/{scene_id}_{key}.wav" for key in component_ids},
        mixture_hash="mix",
        dry_mixture_hash="dry",
        stem_hashes={key: key for key in component_ids},
        activity_hashes={source["source_id"]: "mask" for source in sources},
        target_ledger=ledger.model_dump(mode="json"),
        sample_rate=24000,
    )


def test_group_split_uses_transitive_source_components():
    entries = [
        _entry("ab", ["a.wav", "b.wav"]),
        _entry("ac", ["a.wav", "c.wav"]),
        _entry("d", ["d.wav"]),
        _entry("e", ["e.wav"]),
    ]
    train, val = group_split(entries, val_fraction=0.34, seed=7)
    folds = {
        entry.scene["scene_id"]: "train" for entry in train
    } | {entry.scene["scene_id"]: "val" for entry in val}
    assert folds["ab"] == folds["ac"]
    assert source_leakage(train, val) == set()


def test_group_split_uses_original_source_group_across_different_segments():
    first = _entry("song_a_1", ["segment_1.wav"])
    second = _entry("song_a_2", ["segment_2.wav"])
    first.scene["sources"][0]["source_group"] = "song_a"
    second.scene["sources"][0]["source_group"] = "song_a"
    third = _entry("song_b", ["segment_3.wav"])
    train, val = group_split([first, second, third], val_fraction=0.5, seed=2)
    folds = {entry.scene["scene_id"]: "train" for entry in train} | {
        entry.scene["scene_id"]: "val" for entry in val
    }
    assert folds["song_a_1"] == folds["song_a_2"]
    assert source_leakage(train, val) == set()


def test_manifest_audit_rejects_duplicate_source_ids():
    entry = _entry("collision", ["speech.wav", "sfx.wav"])
    entry.scene["sources"][1]["source_id"] = entry.scene["sources"][0]["source_id"]
    report = audit_manifest_structure([entry])
    assert not report.ok()
    assert any("duplicate source IDs" in error for error in report.errors)


def test_export_moss_sft_writes_official_conversations(tmp_path: Path):
    entries = [
        _entry("s1", ["a.wav"]),
        _entry("s2", ["b.wav"]),
        _entry("s3", ["c.wav"]),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, entries)
    output = tmp_path / "sft"
    metadata = export_moss_sft(
        manifest_path=manifest,
        audio_base=tmp_path,
        output_dir=output,
        val_fraction=0.34,
        seed=3,
        allow_missing_audio=True,
    )
    assert metadata["n_total"] == 3
    assert metadata["source_leakage_count"] == 0
    assert metadata["missing_audio_count"] == 3
    rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    conversation = rows[0]["conversation"]
    assert [message["message_type"] for message in conversation] == [
        "audio",
        "text",
        "text",
    ]
    assert "Atomic grammar" in conversation[1]["content"]
    assert conversation[2]["content"] == "<empty/>"
    assert (output / "val_manifest.jsonl").exists()
    assert (output / "val_references.jsonl").exists()


def test_source_catalog_requires_verbatim_real_lyrics(tmp_path: Path):
    catalog = tmp_path / "sources.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "path": "vocal.wav",
                "kind": "vocal",
                "text": "take me home",
                "source_group": "song-1",
                "verbatim": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="verbatim=true"):
        load_source_catalog(catalog, require_files=False)


def test_sft_export_rejects_synthetic_vocal_lyrics(tmp_path: Path):
    entry = _entry("fake_lyrics", ["vocal:001"])
    entry.scene["sources"][0]["kind"] = "vocal"
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [entry])
    import pytest

    with pytest.raises(ValueError, match="synthetic vocal placeholders"):
        export_moss_sft(
            manifest_path=manifest,
            audio_base=tmp_path,
            output_dir=tmp_path / "sft",
            include_lyrics=True,
            allow_missing_audio=True,
        )


def test_prepare_sources_merges_catalogs_and_rejects_cross_catalog_duplicates(
    tmp_path: Path,
):
    speech = tmp_path / "speech.jsonl"
    sfx = tmp_path / "sfx.jsonl"
    speech.write_text(
        json.dumps(
            {
                "path": "speech.wav",
                "kind": "speech",
                "text": "hello",
                "source_group": "speaker-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sfx.write_text(
        json.dumps(
            {
                "path": "sfx.wav",
                "kind": "sfx",
                "text": "a click",
                "source_group": "recording-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "combined.jsonl"
    assert (
        prepare_sources_main(
            [
                "--input",
                str(speech),
                "--input",
                str(sfx),
                "--output",
                str(output),
                "--allow-missing",
            ]
        )
        == 0
    )
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    import pytest

    with pytest.raises(ValueError, match="duplicate waveform across"):
        prepare_sources_main(
            [
                "--input",
                str(speech),
                "--input",
                str(speech),
                "--output",
                str(output),
                "--allow-missing",
            ]
        )

    with pytest.raises(ValueError, match="missing required kinds.*vocal"):
        prepare_sources_main(
            [
                "--input",
                str(speech),
                "--output",
                str(output),
                "--allow-missing",
                "--require-kind",
                "vocal",
            ]
        )
