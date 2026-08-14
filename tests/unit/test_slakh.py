from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sceneledger.data.slakh import SLAKH_LICENSE, convert_slakh_records


def _write_track(
    root: Path,
    split: str,
    track: str,
    uuid: str,
    *,
    voice_like: bool = False,
) -> None:
    directory = root / split / track
    directory.mkdir(parents=True)
    (directory / "mix.flac").write_bytes(b"placeholder")
    classes = ["Piano", "Guitar", "Drums", "Bass"]
    if voice_like:
        classes[-1] = "Synth Voice"
    metadata = {
        "UUID": uuid,
        "stems": {
            f"S{index:02d}": {
                "audio_rendered": True,
                "inst_class": instrument,
                "midi_program_name": instrument,
                "plugin_name": f"{instrument}.nkm",
            }
            for index, instrument in enumerate(classes)
        },
    }
    (directory / "metadata.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )


def test_slakh_redux_import_tracks_midi_uuid_and_instruments(tmp_path: Path) -> None:
    _write_track(tmp_path, "train", "Track00001", "uuid-1")
    _write_track(tmp_path, "validation", "Track00002", "uuid-2")
    _write_track(tmp_path, "test", "Track00003", "uuid-3")
    _write_track(tmp_path, "train", "Track00004", "uuid-4", voice_like=True)

    records = convert_slakh_records(tmp_path, split_variant="Slakh2100-redux")

    assert len(records) == 3
    assert {record.split for record in records} == {"train", "val", "test"}
    assert all(record.kind == "music" for record in records)
    assert all(record.license == SLAKH_LICENSE for record in records)
    assert all("instrumental music" in record.caption for record in records)
    assert all(any(group.startswith("slakh-midi:") for group in record.leakage_groups) for record in records)


def test_slakh_rejects_original_split_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="orig split is forbidden"):
        convert_slakh_records(tmp_path, split_variant="Slakh2100-orig")


def test_slakh_rejects_cross_split_midi_uuid(tmp_path: Path) -> None:
    _write_track(tmp_path, "train", "Track00001", "same")
    _write_track(tmp_path, "validation", "Track00002", "uuid-2")
    _write_track(tmp_path, "test", "Track00003", "same")

    with pytest.raises(ValueError, match="UUID leakage"):
        convert_slakh_records(tmp_path, split_variant="Slakh2100-redux")
