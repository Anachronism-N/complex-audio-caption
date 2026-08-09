import json
import zipfile
from pathlib import Path

import pytest

from sceneledger.repro.tag2021 import (
    ReproductionError,
    _make_random_query_ablation,
    _read_json_table,
    _safe_extract,
    audit_paper_data,
    collect_paper_results,
)


def _row(audio: Path, phrase: str, start_word: int) -> dict:
    return {
        "audiocap_id": 1,
        "filename": str(audio),
        "caption": "a dog barks and a bell rings",
        "tokens": ["a", "dog", "barks", "and", "a", "bell", "rings"],
        "soundtag": phrase,
        "start_word": start_word,
        "timestamps": [[0.1, 0.9]],
    }


def test_read_json_table_accepts_pandas_column_orientation(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "filename": {"0": "a.wav", "1": "b.wav"},
                "soundtag": {"0": "dog", "1": "bell"},
            }
        ),
        encoding="utf-8",
    )
    assert _read_json_table(path) == [
        {"filename": "a.wav", "soundtag": "dog"},
        {"filename": "b.wav", "soundtag": "bell"},
    ]


def test_random_query_ablation_changes_query_but_not_target(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    rows = [_row(audio, "a dog barks", 1), _row(audio, "a bell rings", 5)]
    changed, count = _make_random_query_ablation(rows, seed=1)
    assert count == 1
    assert [item["soundtag"] for item in changed] == ["a dog barks", "a dog barks"]
    assert [item["timestamps"] for item in changed] == [item["timestamps"] for item in rows]
    assert [item["start_word"] for item in changed] == [1, 5]


def test_audit_detects_valid_small_no_leak_fixture(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    for split in ("train", "val", "test"):
        audio = tmp_path / f"{split}.wav"
        audio.write_bytes(b"RIFF-test")
        label = prepared / split / "label.json"
        label.parent.mkdir(parents=True)
        label.write_text(json.dumps([_row(audio, f"{split} sound", 0)]), encoding="utf-8")
    report = audit_paper_data(prepared, strict_counts=False)
    assert report["valid"] is True
    assert report["cross_split_leakage"] == {}


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("../escape.txt", "bad")
    with pytest.raises(ReproductionError, match="Unsafe archive member"):
        _safe_extract(archive, tmp_path / "out")


def test_collect_results_parses_and_applies_acceptance(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    event = """Overall metrics
F-measure
  F-measure (F1-score) : 28.30 %
  Precision : 28.60 %
  Recall : 27.90 %
"""
    random_event = event.replace("28.30", "19.60")
    (experiment / "paper_event.txt").write_text(event, encoding="utf-8")
    (experiment / "paper_psds.txt").write_text("PSD-Score: 0.14700\n", encoding="utf-8")
    (experiment / "random_query_event.txt").write_text(random_event, encoding="utf-8")
    output = tmp_path / "metrics.json"
    collect_paper_results(experiment, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["pass"] is True
    assert result["metrics"]["event_f1"] == pytest.approx(0.283)
    assert result["metrics"]["random_query_event_f1"] == pytest.approx(0.196)
