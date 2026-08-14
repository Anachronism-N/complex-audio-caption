from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_real_speech_sfx_pilot_config import _passed_audit, _rms_ready_catalog


def _write_audit(path: Path, *, required_splits: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "pass": True,
                "required_splits": required_splits,
                "counts_by_split_kind": {
                    "test": {"speech": 10, "sfx": 3, "ambience": 3}
                },
            }
        ),
        encoding="utf-8",
    )


def test_pilot_config_requires_test_fold_source_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_audit(audit, required_splits=[])

    with pytest.raises(ValueError, match="required-split test"):
        _passed_audit(
            audit,
            "source audit",
            required_test_kinds={"speech"},
            minimum_test_per_kind=10,
        )


def test_pilot_config_accepts_sufficient_test_fold_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    _write_audit(audit, required_splits=["test"])

    assert _passed_audit(
        audit,
        "source audit",
        required_test_kinds={"sfx", "ambience"},
        minimum_test_per_kind=3,
    ) == audit.resolve()


def test_pilot_config_rejects_catalog_without_active_rms(tmp_path: Path) -> None:
    catalog = tmp_path / "test.jsonl"
    catalog.write_text(
        json.dumps({"source_id": "old", "rms_dbfs": -20.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rerun source prepare"):
        _rms_ready_catalog(catalog, "old catalog")


def test_pilot_config_accepts_active_rms_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "test.jsonl"
    catalog.write_text(
        json.dumps(
            {"source_id": "new", "rms_dbfs": -24.0, "active_rms_dbfs": -20.0}
        )
        + "\n",
        encoding="utf-8",
    )
    assert _rms_ready_catalog(catalog, "new catalog") == catalog.resolve()
