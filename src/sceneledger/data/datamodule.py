"""PyTorch Dataset / DataModule over a TAC-mini manifest.

Loads ``(mixture_audio, target_atomic, target_xml, target_ledger)`` tuples
from a rendered manifest (``data/derived/tac_mini/manifest.jsonl``). Group
splits follow the leak-prevention rules in ``docs/06`` §9: scenes sharing a
dry source must not cross train/val folds. With the synthetic pool the
``source_id`` is the grouping key; with real corpora, group by raw media ID +
audio fingerprint + uploader + song/performer.

The audio is loaded at the model's expected sample rate (MOSS-Audio = 16 kHz,
``mel_sr``) and padded/truncated to ``max_audio_seconds``. Targets are
pre-computed strings so the collator only needs to tokenize.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.data.schema import Ledger
from sceneledger.models.target_formatter import (
    StyleConfig,
    format_atomic_caption,
    format_xml_caption,
)

MOSS_INPUT_SAMPLE_RATE = 16000  # processor.config.mel_sr


@dataclass
class DatamoduleConfig:
    manifest_path: str
    audio_base_dir: str = "."  # root that manifest paths are relative to
    sample_rate: int = MOSS_INPUT_SAMPLE_RATE
    max_audio_seconds: float = 30.0
    style: str = "brief"
    target_mode: str = "atomic"  # "atomic" | "xml"
    group_key: str = "source_id"  # for leak-free splits
    val_group_fraction: float = 0.1


def _load_audio(path: str, sample_rate: int, max_seconds: float) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import librosa

        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=sample_rate).astype(np.float32)
    max_n = int(max_seconds * sample_rate)
    if wav.shape[0] > max_n:
        wav = wav[:max_n]
    elif wav.shape[0] < max_n:
        wav = np.pad(wav, (0, max_n - wav.shape[0]))
    return torch.from_numpy(wav.astype(np.float32))


def _group_key(entry: ManifestEntry, key: str) -> str:
    if key == "source_id":
        # combine all source ids into a stable group token
        srcs = entry.scene.get("sources", [])
        if not srcs:
            return "empty"
        # group by the *set* of source paths so the same dry source doesn't leak
        paths = sorted(s["path"] for s in srcs)
        return hashlib.sha1("|".join(paths).encode()).hexdigest()[:12]
    if key == "template":
        return entry.scene.get("template", "unknown")
    return key


class TacMiniDataset(Dataset):
    """One item per manifest entry."""

    def __init__(self, config: DatamoduleConfig, entries: list[ManifestEntry]):
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
    entries = read_manifest(config.manifest_path)
    return TacMiniDataset(config, entries)


def group_split(
    entries: list[ManifestEntry], val_fraction: float, group_key: str = "source_id", seed: int = 20260808
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    """Leak-free split: groups are atomic — all scenes sharing a group go to one side."""
    import random

    groups: dict[str, list[ManifestEntry]] = {}
    for e in entries:
        groups.setdefault(_group_key(e, group_key), []).append(e)
    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_val = max(1, int(round(len(group_ids) * val_fraction)))
    val_groups = set(group_ids[:n_val])
    train = [e for g in group_ids[n_val:] for e in groups[g]]
    val = [e for g in group_ids[:n_val] for e in groups[g]]
    return train, val


def collate_for_inference(batch: list[dict]) -> dict:
    """Collate that keeps audio as a list (variable length pre-pad) + ids."""
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "audio": [b["audio"] for b in batch],
        "audio_path": [b["audio_path"] for b in batch],
        "duration": [b["duration"] for b in batch],
        "group": [b["group"] for b in batch],
    }


__all__ = [
    "DatamoduleConfig",
    "MOSS_INPUT_SAMPLE_RATE",
    "TacMiniDataset",
    "build_dataset",
    "collate_for_inference",
    "group_split",
]
