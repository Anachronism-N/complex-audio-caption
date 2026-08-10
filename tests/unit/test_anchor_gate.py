"""Tests for the executable anchor-first experiment gate."""

import json
from pathlib import Path

from scripts.repro.require_anchor_pass import main


def test_anchor_gate_requires_existing_passing_summary(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert main([str(missing)]) == 2

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"pass": False}), encoding="utf-8")
    assert main([str(summary)]) == 3

    summary.write_text(json.dumps({"pass": True}), encoding="utf-8")
    assert main([str(summary)]) == 0
