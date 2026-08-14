from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from sceneledger.data.musdb18 import convert_musdb18_records
from scripts.materialize_musdb18_stems import materialize_track


def _write_audio(path: Path, amplitude: float, *, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(
        path,
        amplitude * np.sin(2 * np.pi * 220.0 * time),
        sample_rate,
        subtype="PCM_24",
    )


def test_materialize_and_import_musdb18_stems(tmp_path: Path) -> None:
    hq_track = tmp_path / "hq" / "train" / "Artist - Song"
    for name, amplitude in (
        ("drums.wav", 0.05),
        ("bass.wav", 0.04),
        ("other.wav", 0.03),
        ("vocals.wav", 0.02),
    ):
        _write_audio(hq_track / name, amplitude)
    materialized = tmp_path / "materialized" / "train" / hq_track.name
    materialize_track(hq_track, materialized, block_frames=1024)
    test_track = tmp_path / "materialized" / "test" / "Other - Test Song"
    _write_audio(test_track / "accompaniment.wav", 0.1)
    _write_audio(test_track / "vocals.wav", 0.03)

    tracklist = tmp_path / "tracklist.csv"
    with tracklist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Track Name", "Genre", "Source", "License"])
        writer.writerow([hq_track.name, "Rock", "MedleyDB", "CC BY-NC-SA"])
        writer.writerow([test_track.name, "Jazz", "DSD", "Restricted"])
    records = convert_musdb18_records(
        tmp_path / "materialized",
        tracklist_path=tracklist,
        allowed_licenses={"CC BY-NC-SA"},
    )

    assert len(records) == 2
    assert {record.kind for record in records} == {"music", "vocal"}
    assert {record.split for record in records} == {"train"}
    assert {record.source_group for record in records} == {
        "musdb18-track:artist - song"
    }
    music = next(record for record in records if record.kind == "music")
    assert music.audio_path.endswith("accompaniment.wav")
    assert "without lead vocals" in music.caption

    drums, _ = sf.read(hq_track / "drums.wav", dtype="float32")
    bass, _ = sf.read(hq_track / "bass.wav", dtype="float32")
    other, _ = sf.read(hq_track / "other.wav", dtype="float32")
    accompaniment, _ = sf.read(materialized / "accompaniment.wav", dtype="float32")
    assert np.allclose(accompaniment, drums + bass + other, atol=1e-6)
