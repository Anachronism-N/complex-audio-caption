from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..types import read_jsonl
from .audio import load_audio
from .datasets import sha256_file
from .manifest import load_source_manifest


def validate_source_manifest(path: str | Path, verify_hashes: bool = False) -> dict[str, Any]:
    records = load_source_manifest(path)
    errors: list[str] = []
    group_splits: dict[str, set[str]] = {}
    for record in records:
        audio_path = Path(record.path)
        if not audio_path.is_file():
            errors.append(f"missing: {record.id}: {audio_path}")
            continue
        if verify_hashes and record.sha256:
            actual = sha256_file(audio_path)
            if actual != record.sha256:
                errors.append(f"hash: {record.id}: {actual} != {record.sha256}")
        if record.split:
            group_splits.setdefault(record.group_id, set()).add(record.split)
    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"group leakage: {group_id}: {sorted(splits)}")
    return {"records": len(records), "errors": errors, "valid": not errors}


def validate_rendered_dataset(output_dir: str | Path, tolerance: float = 2e-4) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    ledgers = {ledger.sample_id: ledger for ledger in read_jsonl(root / "ledgers.jsonl")}
    rows = [
        json.loads(line)
        for line in (root / "render_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors: list[str] = []
    maximum_reconstruction_error = 0.0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in ledgers:
            errors.append(f"missing ledger: {sample_id}")
            continue
        sample_rate = int(row["sample_rate"])
        mixture = load_audio(root / row["mixture_path"], sample_rate)
        stem_paths = [item.get("stem_path") for item in row["sources"]]
        if all(stem_paths):
            stems = [load_audio(root / path, sample_rate) for path in stem_paths]
            if row.get("residual_stem_path"):
                stems.append(load_audio(root / row["residual_stem_path"], sample_rate))
            reconstruction = np.sum(np.stack(stems), axis=0)
            error = float(np.max(np.abs(mixture - reconstruction)))
            maximum_reconstruction_error = max(maximum_reconstruction_error, error)
            if error > tolerance:
                errors.append(f"reconstruction: {sample_id}: {error:.6g} > {tolerance}")
        expected_samples = round(float(row["duration_sec"]) * sample_rate)
        if abs(len(mixture) - expected_samples) > 1:
            errors.append(f"duration: {sample_id}: {len(mixture)} != {expected_samples}")
    missing_rows = sorted(set(ledgers) - {row["sample_id"] for row in rows})
    errors.extend(f"missing render row: {sample_id}" for sample_id in missing_rows)
    return {
        "samples": len(rows),
        "valid": not errors,
        "errors": errors,
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "tolerance": tolerance,
    }
