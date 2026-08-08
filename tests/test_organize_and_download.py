from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sceneledger.data.audio import save_audio
from sceneledger.data.datasets import download_datasets, safe_extract
from sceneledger.data.organize import assign_group_splits, build_source_manifest


def test_organizer_and_group_split(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    for name in ("a", "b"):
        save_audio(root / "speech" / f"{name}.wav", np.zeros(1600, dtype=np.float32), 16000)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "relative_path,text,group_id,language\n"
        "speech/a.wav,hello,same_speaker,en\n"
        "speech/b.wav,world,same_speaker,en\n",
        encoding="utf-8",
    )
    records = build_source_manifest(root, tmp_path / "sources.jsonl", metadata_path=metadata)
    split = assign_group_splits(records, seed=1)
    assert len(records) == 2
    assert split[0].split == split[1].split
    assert split[0].text == "hello"


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        payload = b"bad"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe archive"):
        safe_extract(archive, tmp_path / "out")


def test_registry_requires_explicit_license_acceptance(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
datasets:
  manual_set:
    kind: manual
    homepage: https://example.test/data
    license: test terms
    license_url: https://example.test/license
    requires_acceptance: true
    instructions: obtain it manually
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="explicit license"):
        download_datasets(registry, ["manual_set"], tmp_path / "raw", accepted_licenses=set())
    result = download_datasets(
        registry,
        ["manual_set"],
        tmp_path / "raw",
        accepted_licenses={"manual_set"},
    )
    assert result[0]["status"] == "manual"
