from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from sceneledger.data.source_captioning import (
    build_source_caption_plan,
    run_source_caption_plan,
    validate_source_caption_audit,
    write_source_caption_audit,
)
from sceneledger.data.source_catalog import SourceRecord, probe_source_record, write_source_catalog


class _FakeCaptioner:
    def infer(self, audio_path: str, prompt: str, *, sample_id: str, duration: float) -> str:
        if "atomic 0.1-second" in prompt:
            return "<sfx><|t_000|>a labeled sound<|t_004|></sfx>"
        return "A clearly audible labeled sound occurs."


def _tone(path: Path, frequency: float) -> None:
    sr = 8000
    time = np.arange(sr // 2, dtype=np.float32) / sr
    sf.write(path, 0.1 * np.sin(2 * np.pi * frequency * time), sr, subtype="PCM_16")


def test_paired_source_caption_plan_is_frozen_resumable_and_auditable(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    records = []
    for index, label in enumerate(("dog", "rain")):
        filename = f"{label}.wav"
        _tone(audio_root / filename, 200 + index * 100)
        record = SourceRecord(
            source_id=f"source:{label}",
            kind="sfx" if label == "dog" else "ambience",
            audio_path=filename,
            source_group=f"group:{label}",
            labels=[label],
            caption=f"Label: {label}",
            dataset="unit-test",
            license="CC0-1.0",
            annotation_origin="dataset",
            split="test",
        )
        records.append(probe_source_record(record, audio_root=audio_root))
    catalog = tmp_path / "all.jsonl"
    write_source_catalog(catalog, records)

    plan_path = tmp_path / "plan.json"
    plan = build_source_caption_plan(catalog, audio_root, plan_path)
    assert plan["n_labels"] == 2
    assert plan["n_generations"] == 4

    results_path = tmp_path / "results.jsonl"
    report = run_source_caption_plan(plan_path, results_path, _FakeCaptioner())
    assert report["pass"] is True
    assert report["n_structured_parseable"] == 2
    resumed = run_source_caption_plan(plan_path, results_path, _FakeCaptioner(), resume=True)
    assert resumed["n_results"] == 4

    audit_path = tmp_path / "audit.csv"
    assert write_source_caption_audit(results_path, audit_path) == 4
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for row in rows:
        row["label_correct_y_n"] = "y"
        row["all_audible_events_covered_y_n"] = "y"
        row["hallucination_free_y_n"] = "y"
        row["temporal_claims_present_y_n"] = "y" if row["prompt_mode"] == "structured" else "n"
        row["temporal_claims_supported_y_n_or_na"] = "y" if row["prompt_mode"] == "structured" else "na"
        row["structured_format_usable_y_n_or_na"] = "y" if row["prompt_mode"] == "structured" else "na"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    gate = validate_source_caption_audit(
        plan_path,
        results_path,
        audit_path,
        tmp_path / "audit_report.json",
    )
    assert gate["pass"] is True
    assert gate["rates"]["structured"]["temporal_supported"] == 1.0
    persisted = json.loads((tmp_path / "audit_report.json").read_text(encoding="utf-8"))
    assert persisted["results_sha256"] == gate["results_sha256"]
