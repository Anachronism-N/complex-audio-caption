from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from sceneledger.data.audio import save_audio
from sceneledger.data.carc import build_exact_carc
from sceneledger.data.manifest import SourceRecord, write_source_manifest
from sceneledger.data.renderer import render_tac_dataset
from sceneledger.data.validate import validate_rendered_dataset
from sceneledger.types import read_jsonl


def _tone(path: Path, frequency: float, duration: float, sample_rate: int = 8000) -> None:
    time = np.arange(round(duration * sample_rate)) / sample_rate
    save_audio(path, (0.1 * np.sin(2 * np.pi * frequency * time)).astype(np.float32), sample_rate)


def _source_manifest(root: Path) -> Path:
    specs = [
        ("speech", 220.0, 1.0, "speaker one"),
        ("speech", 260.0, 1.2, "speaker two"),
        ("music", 440.0, 2.0, "steady music"),
        ("sfx", 880.0, 0.4, "a beep"),
        ("ambience", 90.0, 3.0, "background hum"),
    ]
    records = []
    for index, (kind, frequency, duration, text) in enumerate(specs):
        path = root / kind / f"{index}.wav"
        _tone(path, frequency, duration)
        records.append(
            SourceRecord(
                id=f"source_{index}",
                path=str(path),
                type=kind,  # type: ignore[arg-type]
                duration_sec=duration,
                sample_rate=8000,
                text=text,
                group_id=f"group_{index}",
            )
        )
    manifest = root / "sources.jsonl"
    write_source_manifest(manifest, records)
    return manifest


def test_renderer_is_replayable_and_stems_reconstruct(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path / "source")
    config = {
        "seed": 7,
        "sample_count": 2,
        "sample_rate": 8000,
        "duration_sec": [3.0, 3.0],
        "gain_db": [-3.0, -3.0],
        "save_stems": True,
        "templates": [{"name": "test", "sources": ["speech", "music", "sfx"]}],
    }
    config_path = tmp_path / "render.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = tmp_path / "render_a"
    second = tmp_path / "render_b"
    render_tac_dataset(manifest, config_path, first)
    render_tac_dataset(manifest, config_path, second)
    assert _sha(first / "audio/tac_0000000.wav") == _sha(second / "audio/tac_0000000.wav")
    ledgers = list(read_jsonl(first / "ledgers.jsonl"))
    assert len(ledgers) == 2
    assert len(ledgers[0].events) == 3

    mixture = _read_wav(first / "audio/tac_0000000.wav")
    stems = [_read_wav(path) for path in sorted((first / "stems/tac_0000000").glob("*.wav"))]
    assert np.max(np.abs(mixture - np.sum(stems, axis=0))) < 2e-4
    assert validate_rendered_dataset(first)["valid"]


def test_exact_carc_pair_is_exact(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path / "source")
    config = {
        "seed": 3,
        "pair_count": 2,
        "sample_rate": 8000,
        "duration_sec": 3.0,
        "snr_db": [0.0, 0.0],
        "audible_snr_db": -2.0,
        "hidden_snr_db": -10.0,
    }
    config_path = tmp_path / "carc.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "carc"
    build_exact_carc(manifest, manifest, config_path, output)
    row = json.loads((output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    before = _read_wav(output / row["before_path"])
    after = _read_wav(output / row["after_path"])
    target = _read_wav(output / row["target_path"])
    shifted_after = _read_wav(output / row["shifted_after_path"])
    shifted_target = _read_wav(output / row["shifted_target_path"])
    assert np.max(np.abs(after - before - target)) < 2e-4
    assert np.max(np.abs(shifted_after - before - shifted_target)) < 2e-4
    assert row["audibility_target"] == "must_add"
    assert row["shift_delta_sec"] != 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_wav(path: Path) -> np.ndarray:
    _, audio = wavfile.read(path)
    return audio.astype(np.float32) / 32768.0
