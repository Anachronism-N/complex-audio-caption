"""Dataset helpers and leakage-safe splits for rendered TAC manifests.

The split/audit functions deliberately do not require PyTorch. Audio loading
and ``TacMiniDataset`` become active when the optional MOSS runtime is
installed.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

try:  # split/audit commands must work in the CPU-only base environment
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - dependency-minimal environment
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.data.schema import Ledger
from sceneledger.models.target_formatter import (
    StyleConfig,
    format_atomic_caption,
    format_xml_caption,
)

MOSS_INPUT_SAMPLE_RATE = 16000


@dataclass
class DatamoduleConfig:
    manifest_path: str
    audio_base_dir: str = "."
    sample_rate: int = MOSS_INPUT_SAMPLE_RATE
    max_audio_seconds: float = 30.0
    style: str = "brief"
    target_mode: str = "atomic"
    group_key: str = "source_id"
    val_group_fraction: float = 0.1


def _load_audio(path: str, sample_rate: int, max_seconds: float):
    if torch is None:
        raise RuntimeError("audio loading requires the optional 'moss' dependencies")
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        factor = gcd(int(sr), int(sample_rate))
        wav = resample_poly(
            wav.astype(np.float64), int(sample_rate) // factor, int(sr) // factor
        ).astype(np.float32)
    max_n = int(max_seconds * sample_rate)
    if wav.shape[0] > max_n:
        wav = wav[:max_n]
    elif wav.shape[0] < max_n:
        wav = np.pad(wav, (0, max_n - wav.shape[0]))
    return torch.from_numpy(wav.astype(np.float32))


def _group_key(entry: ManifestEntry, key: str) -> str:
    """Stable per-scene key; source leakage is handled by ``group_split``."""
    if key == "source_id":
        paths = sorted(_source_paths(entry))
        if not paths:
            return f"empty:{entry.scene.get('scene_id', 'unknown')}"
        return hashlib.sha1("|".join(paths).encode()).hexdigest()[:12]
    if key == "template":
        return entry.scene.get("template", "unknown")
    return key


def _source_paths(entry: ManifestEntry) -> set[str]:
    return {
        str(source.get("source_group") or source["path"])
        for source in entry.scene.get("sources", [])
        if source.get("path")
    }


def _source_components(entries: list[ManifestEntry]) -> list[list[ManifestEntry]]:
    """Connected scene components under shared raw-source paths.

    Scenes ``{a,b}`` and ``{a,c}`` belong to one component even though their
    per-scene source sets differ. Union-find also captures longer chains.
    """
    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_scene_by_source: dict[str, int] = {}
    for index, entry in enumerate(entries):
        for source_path in sorted(_source_paths(entry)):
            previous = first_scene_by_source.setdefault(source_path, index)
            union(index, previous)

    components: dict[int, list[ManifestEntry]] = {}
    for index, entry in enumerate(entries):
        components.setdefault(find(index), []).append(entry)
    return [components[key] for key in sorted(components)]


def source_leakage(
    train: list[ManifestEntry], val: list[ManifestEntry]
) -> set[str]:
    """Return raw source paths present in both folds."""
    train_sources = {path for entry in train for path in _source_paths(entry)}
    val_sources = {path for entry in val for path in _source_paths(entry)}
    return train_sources & val_sources


class TacMiniDataset(Dataset):
    """One item per rendered manifest entry."""

    def __init__(self, config: DatamoduleConfig, entries: list[ManifestEntry]):
        if torch is None:
            raise RuntimeError("TacMiniDataset requires the optional 'moss' dependencies")
        self.config = config
        self.entries = entries
        self._style_cfg = StyleConfig()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        audio_path = Path(self.config.audio_base_dir) / entry.mixture_path
        wav = _load_audio(
            str(audio_path), self.config.sample_rate, self.config.max_audio_seconds
        )
        ledger = Ledger.model_validate(entry.target_ledger)
        atomic = format_atomic_caption(ledger, self.config.style, self._style_cfg)
        xml = format_xml_caption(ledger, self.config.style, self._style_cfg)
        return {
            "sample_id": entry.scene["scene_id"],
            "audio": wav,
            "audio_path": str(audio_path),
            "duration": float(entry.scene["duration"]),
            "target_atomic": atomic,
            "target_xml": xml,
            "ledger": ledger,
            "group": _group_key(entry, self.config.group_key),
        }


def build_dataset(config: DatamoduleConfig) -> TacMiniDataset:
    return TacMiniDataset(config, read_manifest(config.manifest_path))


def group_split(
    entries: list[ManifestEntry],
    val_fraction: float,
    group_key: str = "source_id",
    seed: int = 20260808,
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    """Leak-free split with transitive grouping for shared raw sources."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if not entries:
        return [], []

    if group_key == "source_id":
        grouped_entries = _source_components(entries)
    else:
        groups: dict[str, list[ManifestEntry]] = {}
        for entry in entries:
            groups.setdefault(_group_key(entry, group_key), []).append(entry)
        grouped_entries = [groups[key] for key in sorted(groups)]

    rng = random.Random(seed)
    rng.shuffle(grouped_entries)
    n_val = max(1, int(round(len(grouped_entries) * val_fraction)))
    if len(grouped_entries) > 1:
        n_val = min(n_val, len(grouped_entries) - 1)
    val = [entry for group in grouped_entries[:n_val] for entry in group]
    train = [entry for group in grouped_entries[n_val:] for entry in group]
    leaked = source_leakage(train, val) if group_key == "source_id" else set()
    if leaked:  # defensive invariant; impossible by construction
        raise AssertionError(f"source leakage after group split: {sorted(leaked)[:5]}")
    return train, val


def collate_for_inference(batch: list[dict]) -> dict:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "audio": [item["audio"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
        "duration": [item["duration"] for item in batch],
        "group": [item["group"] for item in batch],
    }


__all__ = [
    "DatamoduleConfig",
    "MOSS_INPUT_SAMPLE_RATE",
    "TacMiniDataset",
    "build_dataset",
    "collate_for_inference",
    "group_split",
    "source_leakage",
]
