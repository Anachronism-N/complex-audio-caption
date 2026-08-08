from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scipy.io import wavfile

try:
    import soundfile as sf
except ImportError:
    sf = None

from .datasets import sha256_file
from .manifest import SourceRecord, write_source_manifest

_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"}
_SOURCE_TYPES = {"speech", "lys", "music", "sfx", "ambience"}


def build_source_manifest(
    input_root: str | Path,
    output_path: str | Path,
    *,
    default_type: str | None = None,
    metadata_path: str | Path | None = None,
    license_name: str | None = None,
) -> list[SourceRecord]:
    root = Path(input_root).resolve()
    metadata = _load_metadata(metadata_path)
    records: list[SourceRecord] = []
    for path in sorted(
        item for item in root.rglob("*") if item.suffix.lower() in _AUDIO_EXTENSIONS
    ):
        relative = path.relative_to(root).as_posix()
        info = _probe_audio(path)
        row = metadata.get(relative, metadata.get(path.name, {}))
        source_type = row.get("type") or default_type or _infer_type(path, root)
        if source_type not in _SOURCE_TYPES:
            raise ValueError(
                f"Cannot infer source type for {relative}; put it under a supported type "
                "directory, provide --type, or add metadata."
            )
        digest = sha256_file(path)
        record_id = row.get("id") or f"{source_type}_{digest[:16]}"
        text = row.get("text") or row.get("caption") or row.get("transcript") or source_type
        group_id = row.get("group_id") or row.get("speaker_id") or row.get("work_id") or digest
        records.append(
            SourceRecord(
                id=str(record_id),
                path=str(path),
                type=source_type,
                duration_sec=info["duration_sec"],
                sample_rate=info["sample_rate"],
                text=str(text),
                group_id=str(group_id),
                language=row.get("language"),
                verbatim=_to_bool(row.get("verbatim", source_type in {"speech", "lys"})),
                license=row.get("license", license_name),
                sha256=digest,
                metadata={"relative_path": relative, **_extra_metadata(row)},
            )
        )
    write_source_manifest(output_path, records)
    return records


def assign_group_splits(
    records: Iterable[SourceRecord],
    *,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
    seed: int = 20260808,
) -> list[SourceRecord]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must leave a non-empty test interval")
    group_split: dict[str, str] = {}
    output: list[SourceRecord] = []
    for record in records:
        if record.group_id not in group_split:
            digest = hashlib.sha256(f"{seed}:{record.group_id}".encode()).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            if value < train_ratio:
                split = "train"
            elif value < train_ratio + validation_ratio:
                split = "validation"
            else:
                split = "test"
            group_split[record.group_id] = split
        record.split = group_split[record.group_id]
        output.append(record)
    return output


def _probe_audio(path: Path) -> dict[str, Any]:
    if sf is not None:
        try:
            info = sf.info(str(path))
            return {"duration_sec": float(info.duration), "sample_rate": int(info.samplerate)}
        except RuntimeError:
            pass
    if path.suffix.lower() == ".wav":
        try:
            sample_rate, audio = wavfile.read(str(path), mmap=True)
            return {"duration_sec": len(audio) / sample_rate, "sample_rate": int(sample_rate)}
        except (ValueError, OSError):
            pass
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=sample_rate",
            "-select_streams",
            "a:0",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Cannot probe {path}; install ffmpeg or convert it to WAV/FLAC"
        ) from exc
    value = json.loads(result.stdout)
    return {
        "duration_sec": float(value["format"]["duration"]),
        "sample_rate": int(value["streams"][0]["sample_rate"]),
    }


def _load_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    elif metadata_path.suffix.lower() == ".json":
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("items", [])
    elif metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Metadata must be JSONL, JSON or CSV")
    result = {}
    for row in rows:
        key = row.get("relative_path") or row.get("path") or row.get("filename")
        if not key:
            raise ValueError("Every metadata row needs relative_path/path/filename")
        result[str(key).replace("\\", "/")] = row
    return result


def _infer_type(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts[:-1]:
        if part.lower() in _SOURCE_TYPES:
            return part.lower()
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _extra_metadata(row: dict[str, Any]) -> dict[str, Any]:
    known = {
        "relative_path",
        "path",
        "filename",
        "id",
        "type",
        "text",
        "caption",
        "transcript",
        "group_id",
        "speaker_id",
        "work_id",
        "language",
        "verbatim",
        "license",
    }
    return {
        key: value for key, value in row.items() if key not in known and value not in {None, ""}
    }
