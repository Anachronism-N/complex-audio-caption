"""Canonical metadata catalog for real isolated audio sources.

The renderer must never invent speech transcripts or lyrics for file-backed
audio.  A source catalog binds each waveform to its acoustically supported
text and to a leakage group (speaker, song, or original media item).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SourceKind = Literal["speech", "vocal", "music", "sfx", "ambience"]
_KINDS = {"speech", "vocal", "music", "sfx", "ambience"}


@dataclass(frozen=True)
class SourceRecord:
    path: str
    kind: SourceKind
    text: str
    source_group: str
    identity: str | None = None
    language: str | None = None
    verbatim: bool | None = None
    license: str | None = None
    dataset: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def load_source_catalog(
    path: str | Path,
    *,
    audio_root: str | Path | None = None,
    require_files: bool = True,
) -> list[SourceRecord]:
    """Load and validate CSV/JSONL source metadata.

    Relative waveform paths are resolved against ``audio_root`` when given,
    otherwise against the catalog directory.  Speech and vocal rows require
    text; vocal rows additionally require ``verbatim=true``.
    """
    catalog_path = Path(path).resolve()
    root = Path(audio_root).resolve() if audio_root else catalog_path.parent
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(_load_rows(catalog_path), 1):
        raw_path = str(row.get("path", "")).strip()
        kind = str(row.get("kind", "")).strip().lower()
        text = str(row.get("text", "")).strip()
        source_group = str(row.get("source_group", "")).strip()
        if not raw_path:
            raise ValueError(f"catalog row {index}: missing path")
        if kind not in _KINDS:
            raise ValueError(f"catalog row {index}: invalid kind {kind!r}")
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = root / resolved
        resolved = resolved.resolve()
        key = str(resolved)
        if key in seen:
            raise ValueError(f"catalog row {index}: duplicate path {key}")
        if require_files and not resolved.is_file():
            raise FileNotFoundError(f"catalog row {index}: audio file not found: {resolved}")
        if not text:
            raise ValueError(
                f"catalog row {index}: {kind} requires an acoustically supported label"
            )
        verbatim = _coerce_bool(row.get("verbatim"))
        if kind == "vocal" and verbatim is not True:
            raise ValueError(
                f"catalog row {index}: vocal lyrics require verbatim=true; "
                "use kind=music for unintelligible or wordless vocals"
            )
        if not source_group:
            raise ValueError(
                f"catalog row {index}: source_group is required for leakage-safe splitting"
            )
        seen.add(key)
        records.append(
            SourceRecord(
                path=key,
                kind=kind,  # type: ignore[arg-type]
                text=text,
                source_group=source_group,
                identity=str(row.get("identity", "")).strip() or None,
                language=str(row.get("language", "")).strip() or None,
                verbatim=verbatim,
                license=str(row.get("license", "")).strip() or None,
                dataset=str(row.get("dataset", "")).strip() or None,
            )
        )
    if not records:
        raise ValueError(f"source catalog is empty: {catalog_path}")
    return records


def write_source_catalog(path: str | Path, records: list[SourceRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["SourceRecord", "load_source_catalog", "write_source_catalog"]
