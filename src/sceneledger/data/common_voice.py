"""Import a locally acquired Mozilla Common Voice release safely.

Common Voice is obtained by the user from Mozilla Data Collective.  This
module deliberately has no downloader: Mozilla asks that access and further
downloads happen through MDC rather than third-party mirrors.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord

COMMON_VOICE_LICENSE = "CC0-1.0"
COMMON_VOICE_URL = "https://commonvoice.mozilla.org/en/datasets"
SPLIT_FILES = {"train.tsv": "train", "dev.tsv": "val", "test.tsv": "test"}


def _speaker_token(client_id: str) -> str:
    return hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:20]


def _safe_clip_path(root: Path, relative: str) -> Path:
    clip_root = (root / "clips").resolve()
    path = (clip_root / relative).resolve()
    try:
        path.relative_to(clip_root)
    except ValueError as exc:
        raise ValueError(f"Common Voice clip path escapes clips/: {relative!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Common Voice clip is missing: {path}")
    return path


def convert_common_voice_records(
    root: str | Path,
    *,
    release: str,
    locale: str,
    min_up_votes: int = 2,
    max_down_votes: int | None = None,
    max_per_speaker: int | None = None,
    drop_missing_client_id: bool = True,
) -> list[SourceRecord]:
    """Convert official train/dev/test TSV files to exact speech records.

    Speaker identifiers are hashed before being written to the catalog.  A
    speaker occurring in more than one official split is rejected because it
    would invalidate speaker-disjoint evaluation.
    """
    if not release.strip() or not locale.strip():
        raise ValueError("Common Voice release and locale must be explicit")
    if min_up_votes < 0 or (max_down_votes is not None and max_down_votes < 0):
        raise ValueError("vote thresholds must be non-negative")
    if max_per_speaker is not None and max_per_speaker <= 0:
        raise ValueError("max_per_speaker must be positive")

    base = Path(root).expanduser().resolve()
    records: list[SourceRecord] = []
    seen_paths: set[str] = set()
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    speaker_counts: dict[tuple[str, str], int] = defaultdict(int)

    for tsv_name, split in SPLIT_FILES.items():
        tsv_path = base / tsv_name
        if not tsv_path.is_file():
            raise FileNotFoundError(
                f"official Common Voice split file is missing: {tsv_path}"
            )
        with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"client_id", "path", "sentence"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(
                    f"invalid Common Voice TSV {tsv_path}: missing "
                    f"{sorted(required - set(reader.fieldnames or []))}"
                )
            for line_no, row in enumerate(reader, 2):
                relative = str(row.get("path", "")).strip().replace("\\", "/")
                sentence = str(row.get("sentence", "")).strip()
                client_id = str(row.get("client_id", "")).strip()
                if not relative or not sentence:
                    raise ValueError(f"empty path/sentence in {tsv_path}:{line_no}")
                if not client_id:
                    if drop_missing_client_id:
                        continue
                    raise ValueError(
                        f"missing client_id in {tsv_path}:{line_no}; speaker leakage is unknown"
                    )
                try:
                    up_votes = int(str(row.get("up_votes", "0") or "0"))
                    down_votes = int(str(row.get("down_votes", "0") or "0"))
                except ValueError as exc:
                    raise ValueError(f"invalid votes in {tsv_path}:{line_no}") from exc
                if up_votes < min_up_votes:
                    continue
                if max_down_votes is not None and down_votes > max_down_votes:
                    continue
                row_locale = str(row.get("locale", "")).strip()
                if row_locale and row_locale.casefold() != locale.casefold():
                    raise ValueError(
                        f"locale mismatch in {tsv_path}:{line_no}: "
                        f"expected={locale!r} observed={row_locale!r}"
                    )
                if relative in seen_paths:
                    raise ValueError(f"Common Voice clip appears more than once: {relative}")
                seen_paths.add(relative)
                clip = _safe_clip_path(base, relative)
                speaker = _speaker_token(client_id)
                speaker_splits[speaker].add(split)
                count_key = (split, speaker)
                if (
                    max_per_speaker is not None
                    and speaker_counts[count_key] >= max_per_speaker
                ):
                    continue
                speaker_counts[count_key] += 1
                path_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
                labels = ["scripted_speech", f"locale:{locale}"]
                for column in ("accent", "accents", "variant"):
                    value = str(row.get(column, "")).strip()
                    if value:
                        labels.append(f"{column}:{value}")
                records.append(
                    SourceRecord(
                        source_id=f"commonvoice:{release}:{locale}:{path_hash}",
                        kind="speech",
                        audio_path=clip.relative_to(base).as_posix(),
                        # Do not namespace by release/locale: if Mozilla keeps a
                        # client ID stable, combining releases or languages must
                        # still collapse that contributor into one leakage group.
                        source_group=f"commonvoice-speaker:{speaker}",
                        leakage_groups=[],
                        labels=labels,
                        caption=sentence,
                        dataset=f"Mozilla Common Voice/{release}/{locale}",
                        license=COMMON_VOICE_LICENSE,
                        annotation_origin="dataset",
                        text_is_verbatim=True,
                        identity=f"speaker:{speaker}",
                        language=locale,
                        attribution=f"Mozilla Common Voice {release}, locale {locale}",
                        original_url=COMMON_VOICE_URL,
                        split=split,
                    )
                )

    overlap = {
        speaker: sorted(splits)
        for speaker, splits in speaker_splits.items()
        if len(splits) > 1
    }
    if overlap:
        examples = list(sorted(overlap.items()))[:10]
        raise ValueError(
            "Common Voice speaker overlap across train/val/test; do not use this "
            f"release split without regrouping: {examples}"
        )
    if not records:
        raise ValueError("Common Voice conversion selected zero utterances")
    return records


__all__ = [
    "COMMON_VOICE_LICENSE",
    "COMMON_VOICE_URL",
    "SPLIT_FILES",
    "convert_common_voice_records",
]
