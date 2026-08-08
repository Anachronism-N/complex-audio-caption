from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceType = Literal["speech", "lys", "music", "sfx", "ambience"]


@dataclass
class SourceRecord:
    id: str
    path: str
    type: SourceType
    duration_sec: float
    sample_rate: int
    text: str
    group_id: str
    language: str | None = None
    verbatim: bool = False
    license: str | None = None
    sha256: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: Path | None = None) -> SourceRecord:
        path = Path(str(value["path"]))
        if base_dir is not None and not path.is_absolute():
            path = (base_dir / path).resolve()
        return cls(
            id=str(value["id"]),
            path=str(path),
            type=value["type"],
            duration_sec=float(value["duration_sec"]),
            sample_rate=int(value["sample_rate"]),
            text=str(value.get("text", value["type"])),
            group_id=str(value.get("group_id", value["id"])),
            language=value.get("language"),
            verbatim=bool(value.get("verbatim", False)),
            license=value.get("license"),
            sha256=value.get("sha256"),
            split=value.get("split"),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self, relative_to: Path | None = None) -> dict[str, Any]:
        value = asdict(self)
        if relative_to is not None:
            try:
                value["path"] = str(Path(self.path).resolve().relative_to(relative_to.resolve()))
            except ValueError:
                value["path"] = self.path
        return {key: item for key, item in value.items() if item is not None}


def load_source_manifest(path: str | Path) -> list[SourceRecord]:
    manifest = Path(path).resolve()
    records: list[SourceRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(SourceRecord.from_dict(json.loads(line), manifest.parent))
            except Exception as exc:
                raise ValueError(f"{manifest}:{line_number}: {exc}") from exc
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in source manifest: {manifest}")
    return records


def write_source_manifest(
    path: str | Path, records: Iterable[SourceRecord], relative_paths: bool = True
) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            base = output.parent if relative_paths else None
            handle.write(
                json.dumps(record.to_dict(base), ensure_ascii=False, sort_keys=True) + "\n"
            )
