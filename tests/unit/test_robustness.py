"""Tests for robustness stratification."""

from __future__ import annotations

import json
from pathlib import Path

from sceneledger.eval.metrics import SampleMetrics
from sceneledger.eval.robustness import (
    load_conditions_from_manifest,
    robustness_report,
    stratify,
)


def _sm(sample_id: str, f1: float, **kw) -> SampleMetrics:
    base = dict(
        sample_id=sample_id,
        n_ref=2,
        n_hyp=2,
        n_matched=2,
        event_precision=0.9,
        event_recall=0.9,
        event_f1=f1,
        caption_token_f1=0.8,
        seg_f1_100ms=0.9,
        onset_mae=0.05,
        offset_mae=0.05,
        onset_p90=0.1,
        offset_p90=0.1,
        tolerance_acc_010=0.9,
        tolerance_acc_025=0.9,
        tolerance_acc_050=0.9,
        tolerance_acc_100=0.9,
        hallucination=0,
        omission=0,
        source_count_mae=0.0,
        pointer_accuracy=1.0,
    )
    base.update(kw)
    return SampleMetrics(**base)


def test_stratify_by_template_and_source_count():
    samples = [_sm("a", 0.9), _sm("b", 0.7), _sm("c", 0.8)]
    conds = {
        "a": {"template": "speech_over_music", "sources": [{}, {}], "conditions": {"t60_sec": 0.2, "overlap_ratio": 0.05}},
        "b": {"template": "speech_music_sfx", "sources": [{}, {}, {}], "conditions": {"t60_sec": 0.8, "overlap_ratio": 0.4}},
        "c": {"template": "speech_over_music", "sources": [{}, {}], "conditions": {"t60_sec": 0.5, "overlap_ratio": 0.2}},
    }
    strat = stratify(samples, conds)
    assert "speech_over_music" in strat["template"]
    assert strat["template"]["speech_over_music"]["n"] == 2
    assert strat["source_count"]["2"]["n"] == 2
    assert strat["source_count"]["3"]["n"] == 1
    # t60 buckets: a=0.2 -> <0.3, c=0.5 -> <0.6, b=0.8 -> <0.9
    assert "<0.3" in strat["t60_sec"]
    assert strat["t60_sec"]["<0.3"]["n"] == 1


def test_robustness_report_round_trip(tmp_path: Path):
    # write a fake metrics report + manifest
    metrics = {
        "samples": [
            _sm("a", 0.9).to_dict(),
            _sm("b", 0.6).to_dict(),
        ]
    }
    mp = tmp_path / "metrics.json"
    mp.write_text(json.dumps(metrics))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"scene": {"scene_id": "a", "template": "isolated_sfx", "sources": [{}], "conditions": {"t60_sec": 0.2, "overlap_ratio": 0.0}}}) + "\n"
        + json.dumps({"scene": {"scene_id": "b", "template": "speech_music_sfx", "sources": [{}, {}, {}], "conditions": {"t60_sec": 0.9, "overlap_ratio": 0.5}}}) + "\n"
    )
    out = tmp_path / "robustness.json"
    robustness_report(mp, manifest, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert "template" in loaded
    # isolated_sfx has 1 source, speech_music_sfx has 3
    assert loaded["source_count"]["1"]["n"] == 1
    assert loaded["source_count"]["3"]["n"] == 1


def test_load_conditions_from_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"scene": {"scene_id": "x", "template": "t", "sources": []}}) + "\n"
    )
    conds = load_conditions_from_manifest(manifest)
    assert "x" in conds
    assert conds["x"]["template"] == "t"
