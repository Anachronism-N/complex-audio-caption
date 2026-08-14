"""Scene-graph sampler for TAC-style synthetic mixtures.

Builds a :class:`Scene` describing a set of placed sources (type, onset, gain,
repeat, RIR, echo, identity, caption text) plus acoustic conditions and
supervision parameters. Sampling ranges follow ``docs/06`` §3.3.

Two source-pool backends are supported:

* :class:`FileSourcePool`  -- reads single-source audio from disk (licensed
  corpora; the user points it at LibriSpeech / Freesound / MUSDB18 etc.).
* :class:`SyntheticSourcePool` -- generates deterministic placeholder audio
  (sine tones, noise bursts, chord progressions) so the renderer and
  supervision targets can be developed and tested before any licensed audio
  is wired in. Synthetic sources carry plausible caption text per type.

The first TAC-mini version (``docs/11`` §5) targets 3 core templates
(speech+music, music+sfx, speech+music+sfx) plus ``isolated_sfx`` and
``repeated_event`` from the paper-spec template list.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from sceneledger.data.schema import TIME_RESOLUTION_SEC

SourceKind = Literal["speech", "vocal", "music", "sfx", "ambience"]
TemplateID = Literal[
    "isolated_sfx",
    "speech_over_music",
    "speech_with_sfx",
    "speech_ambience_sfx",
    "music_with_sfx",
    "speech_music_sfx",
    "repeated_event",
    "ambient_with_intermittent_sfx",
    "lyrics_over_music",
    "speech_music_lyrics_sfx",
    "overlapping_speakers",
    "random_mix",
    "complex_cocktail",
    "rich_band",
    "multi_event_dense",
]

# Sampling ranges (docs/06 §3.3). These are project choices, not TAC values.
DURATION_RANGE = (10.0, 30.0)
GAIN_DB_RANGE = (-12.0, 3.0)
FG_BG_SNR_RANGE = (-10.0, 20.0)  # foreground/background relative level
T60_RANGE = (0.1, 1.2)
ECHO_DELAY_MS_RANGE = (80, 500)
ECHO_ATTEN_DB_RANGE = (-18.0, -3.0)
REPEAT_RANGE = (1, 5)
MERGE_THRESHOLD_RANGE = (0.1, 1.0)
RESOLUTIONS = (0.1, 0.5, 1.0)
STYLES = ("keyword", "brief", "detailed")
ACTIVITY_THRESHOLD_RANGE = (0.03, 0.12)


@dataclass
class PlacedSource:
    source_id: str
    kind: SourceKind
    # audio provider key (file path for FileSourcePool, synthetic key otherwise)
    path: str
    onset: float  # seconds
    gain_db: float
    text: str  # caption text for the events derived from this source
    identity: str | None = None  # speaker_1 / singer_1 / None
    # Stable identity of the original recording/work/speaker group.  It is
    # intentionally distinct from the per-scene source_id and is used for
    # leakage-safe split checks.
    source_group: str | None = None
    leakage_groups: list[str] = field(default_factory=list)
    # Primary semantic class first, followed by optional secondary dataset
    # labels.  Keeping this separate from free-form text makes source-bank
    # class coverage auditable without parsing captions.
    source_labels: list[str] = field(default_factory=list)
    source_dataset: str | None = None
    source_license: str | None = None
    annotation_origin: str | None = None
    text_is_verbatim: bool = False
    # Cryptographic identity of the dry source waveform.  The stable catalog
    # source ID in ``path`` is convenient for sampling, while this digest proves
    # that the rendered scene used the exact file that passed source review.
    source_file_sha256: str | None = None
    source_duration_sec: float | None = None
    source_rms_dbfs: float | None = None
    source_active_rms_dbfs: float | None = None
    repeat: int = 1
    repeat_gap_s: float = 0.0
    rir_id: str | None = None
    t60_sec: float | None = None
    is_foreground: bool = True
    # New manifests can request seamless background extension.  The default
    # remains False so old frozen manifests replay bit-for-bit.
    loop_to_scene: bool = False

    def event_type(self) -> str:
        """Map source kind to event type tag (speech/lys/music/sfx)."""
        if self.kind == "speech":
            return "speech"
        if self.kind == "vocal":
            return "lys"
        if self.kind == "ambience":
            return "sfx"
        return self.kind  # music, sfx


@dataclass
class Conditions:
    noise_snr_db: float | None = None
    echo_delay_ms: int | None = None
    echo_atten_db: float | None = None
    t60_sec: float | None = None
    codec: str | None = None
    overlap_ratio: float | None = None
    ducking_enabled: bool = False
    ducking_depth_db: float | None = None


@dataclass
class Supervision:
    style: str = "brief"
    activity_threshold: float = 0.05
    merge_threshold_s: float = 0.25
    resolution_s: float = TIME_RESOLUTION_SEC


@dataclass
class Scene:
    scene_id: str
    seed: int
    duration: float
    template: TemplateID
    sources: list[PlacedSource]
    conditions: Conditions = field(default_factory=Conditions)
    supervision: Supervision = field(default_factory=Supervision)
    sample_rate: int = 24000

    def to_manifest_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "duration": self.duration,
            "template": self.template,
            "sample_rate": self.sample_rate,
            "sources": [_source_dict(s) for s in self.sources],
            "conditions": _conditions_dict(self.conditions),
            "supervision": _supervision_dict(self.supervision),
        }


def _source_dict(s: PlacedSource) -> dict:
    payload = {
        "source_id": s.source_id,
        "kind": s.kind,
        "path": s.path,
        "onset": s.onset,
        "gain_db": s.gain_db,
        "text": s.text,
        "identity": s.identity,
        "source_group": s.source_group,
        "leakage_groups": s.leakage_groups,
        "source_labels": s.source_labels,
        "source_dataset": s.source_dataset,
        "source_license": s.source_license,
        "annotation_origin": s.annotation_origin,
        "text_is_verbatim": s.text_is_verbatim,
        "source_file_sha256": s.source_file_sha256,
        "source_duration_sec": s.source_duration_sec,
        "source_rms_dbfs": s.source_rms_dbfs,
        "source_active_rms_dbfs": s.source_active_rms_dbfs,
        "repeat": s.repeat,
        "repeat_gap_s": s.repeat_gap_s,
        "rir_id": s.rir_id,
        "t60_sec": s.t60_sec,
        "is_foreground": s.is_foreground,
    }
    if s.loop_to_scene:
        payload["loop_to_scene"] = True
    return payload


def _conditions_dict(c: Conditions) -> dict:
    return {
        "noise_snr_db": c.noise_snr_db,
        "echo_delay_ms": c.echo_delay_ms,
        "echo_atten_db": c.echo_atten_db,
        "t60_sec": c.t60_sec,
        "codec": c.codec,
        "overlap_ratio": c.overlap_ratio,
        "ducking_enabled": c.ducking_enabled,
        "ducking_depth_db": c.ducking_depth_db,
    }


def _supervision_dict(s: Supervision) -> dict:
    return {
        "style": s.style,
        "activity_threshold": s.activity_threshold,
        "merge_threshold_s": s.merge_threshold_s,
        "resolution_s": s.resolution_s,
    }


# --------------------------------------------------------------------------- #
# source pools
# --------------------------------------------------------------------------- #
class SourcePool(Protocol):
    """Provides a mono waveform + duration for a source key."""

    def load(self, key: str, sample_rate: int) -> tuple[np.ndarray, float]:  # pragma: no cover
        ...

    def pick(self, kind: SourceKind, rng: random.Random) -> str:  # pragma: no cover
        ...

    def metadata(self, key: str) -> dict[str, object]:  # pragma: no cover
        ...


@dataclass
class FileSourcePool:
    """Reads single-source audio from a ``{kind: [paths]}`` mapping."""

    by_kind: dict[str, list[str]]

    def pick(self, kind: SourceKind, rng: random.Random) -> str:
        paths = self.by_kind.get(kind, [])
        if not paths:
            raise ValueError(f"no sources registered for kind {kind!r}")
        return rng.choice(paths)

    def load(self, key: str, sample_rate: int) -> tuple[np.ndarray, float]:
        import math

        import soundfile as sf
        from scipy.signal import resample_poly

        wav, sr = sf.read(key, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            divisor = math.gcd(int(sr), int(sample_rate))
            wav = resample_poly(
                wav.astype(np.float64),
                sample_rate // divisor,
                sr // divisor,
            )
        return wav.astype(np.float32), float(len(wav) / sample_rate)

    def metadata(self, key: str) -> dict[str, object]:
        # Legacy path-only pools have no trustworthy semantic annotation.
        return {}


@dataclass
class CatalogSourcePool:
    """Real sources backed by a validated :mod:`source_catalog` JSONL file."""

    catalog_path: str
    audio_root: str | None = None
    audit_report_path: str | None = None
    expected_split: str | None = None

    def __post_init__(self) -> None:
        from pathlib import Path

        from sceneledger.data.source_catalog import file_sha256, read_source_catalog

        catalog = Path(self.catalog_path).expanduser().resolve()
        if self.audit_report_path:
            import json

            audit_path = Path(self.audit_report_path).expanduser().resolve()
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("pass") is not True:
                raise ValueError(f"source audit did not pass: {audit_path}")
            preparation_path = Path(str(audit.get("preparation_report_path", "")))
            if not preparation_path.is_absolute():
                preparation_path = (audit_path.parent / preparation_path).resolve()
            expected_preparation_hash = audit.get("preparation_report_sha256")
            observed_preparation_hash = (
                file_sha256(preparation_path) if preparation_path.is_file() else None
            )
            if (
                not expected_preparation_hash
                or observed_preparation_hash != expected_preparation_hash
            ):
                raise ValueError(
                    "source preparation report changed after human audit: "
                    f"{preparation_path}"
                )
            artifact = (audit.get("catalog_artifacts") or {}).get(catalog.name)
            expected_hash = artifact.get("sha256") if artifact else None
            observed_hash = file_sha256(catalog)
            if not expected_hash or expected_hash != observed_hash:
                raise ValueError(
                    f"catalog is not bound to passed source audit: {catalog} "
                    f"expected={expected_hash!r} observed={observed_hash!r}"
                )
        root = Path(self.audio_root).expanduser().resolve() if self.audio_root else catalog.parent
        records = read_source_catalog(catalog)
        observed_splits = {record.split for record in records}
        if None in observed_splits or len(observed_splits) != 1:
            raise ValueError(
                f"catalog must contain exactly one frozen split, observed={sorted(str(x) for x in observed_splits)}"
            )
        observed_split = next(iter(observed_splits))
        if self.expected_split and observed_split != self.expected_split:
            raise ValueError(
                f"catalog split mismatch: expected={self.expected_split!r} observed={observed_split!r}"
            )
        self._records = {record.source_id: record for record in records}
        self._by_kind: dict[str, list[str]] = {}
        self._resolved_paths: dict[str, Path] = {}
        self._verified_audio_hashes: set[str] = set()
        for record in self._records.values():
            self._by_kind.setdefault(record.kind, []).append(record.source_id)
            resolved = (root / record.audio_path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"catalog audio path escapes audio_root: {record.audio_path}"
                ) from exc
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"catalog audio file is missing: source={record.source_id} path={resolved}"
                )
            if self.audit_report_path:
                if not record.file_sha256:
                    raise ValueError(
                        f"audited catalog source lacks file SHA-256: {record.source_id}"
                    )
            self._resolved_paths[record.source_id] = resolved
        self._audio_root = root

    def _verify_audio(self, key: str) -> None:
        if not self.audit_report_path or key in self._verified_audio_hashes:
            return
        from sceneledger.data.source_catalog import file_sha256

        record = self._records[key]
        observed = file_sha256(self._resolved_paths[key])
        if observed != record.file_sha256:
            raise ValueError(
                "source audio changed after catalog preparation: "
                f"source={record.source_id} expected={record.file_sha256} "
                f"observed={observed}"
            )
        self._verified_audio_hashes.add(key)

    def pick(self, kind: SourceKind, rng: random.Random) -> str:
        keys = self._by_kind.get(kind, [])
        if not keys:
            raise ValueError(f"no catalog sources registered for kind {kind!r}")
        by_class: dict[str, list[str]] = {}
        for key in keys:
            labels = self._records[key].labels
            by_class.setdefault(labels[0] if labels else "<unlabeled>", []).append(key)
        return rng.choice(by_class[rng.choice(sorted(by_class))])

    def keys(self, kind: SourceKind) -> list[str]:
        """Return a copy so the sampler can exhaustively filter by duration."""
        return list(self._by_kind.get(kind, []))

    def candidates(
        self,
        kind: SourceKind,
        rng: random.Random,
        *,
        max_duration: float | None = None,
        limit: int = 256,
    ) -> list[str]:
        keys = [
            key
            for key in self._by_kind.get(kind, [])
            if max_duration is None
            or self._records[key].duration_sec is None
            or float(self._records[key].duration_sec) <= max_duration
        ]
        # Uniformly cycle over primary semantic classes.  This prevents a
        # long-tailed bank from returning 256 examples of the few dominant
        # classes while still sampling recordings randomly inside each class.
        queues: dict[str, list[str]] = {}
        for key in keys:
            labels = self._records[key].labels
            queues.setdefault(labels[0] if labels else "<unlabeled>", []).append(key)
        for queue in queues.values():
            rng.shuffle(queue)
        output: list[str] = []
        while queues and len(output) < limit:
            class_cycle = sorted(queues)
            rng.shuffle(class_cycle)
            for label in class_cycle:
                output.append(queues[label].pop())
                if not queues[label]:
                    del queues[label]
                if len(output) >= limit:
                    break
        return output

    def load(self, key: str, sample_rate: int) -> tuple[np.ndarray, float]:
        self._verify_audio(key)
        return FileSourcePool(by_kind={}).load(str(self._resolved_paths[key]), sample_rate)

    def metadata(self, key: str) -> dict[str, object]:
        self._verify_audio(key)
        record = self._records[key]
        return {
            "text": record.caption,
            "identity": record.identity,
            "source_group": record.source_group,
            "leakage_groups": record.leakage_groups,
            "source_labels": record.labels,
            "source_dataset": record.dataset,
            "source_license": record.license,
            "annotation_origin": record.annotation_origin,
            "text_is_verbatim": record.text_is_verbatim,
            "source_file_sha256": record.file_sha256,
            "source_duration_sec": record.duration_sec,
            "source_rms_dbfs": record.rms_dbfs,
            "source_active_rms_dbfs": record.active_rms_dbfs,
        }


@dataclass
class CatalogSetSourcePool:
    """Compose independently prepared/audited catalogs without rebasing paths."""

    catalogs: list[CatalogSourcePool]
    sampling_weights: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.catalogs:
            raise ValueError("catalog set must contain at least one catalog")
        if self.sampling_weights is None:
            self.sampling_weights = [1.0] * len(self.catalogs)
        if len(self.sampling_weights) != len(self.catalogs) or any(
            weight <= 0 for weight in self.sampling_weights
        ):
            raise ValueError("catalog sampling weights must be positive and match catalogs")
        self._owner: dict[str, CatalogSourcePool] = {}
        self._by_kind: dict[str, list[str]] = {}
        collisions: list[str] = []
        content_owner: dict[str, int] = {}
        group_owner: dict[str, int] = {}
        cross_catalog_duplicates: list[dict[str, object]] = []
        for catalog_index, catalog in enumerate(self.catalogs):
            for source_id, record in catalog._records.items():
                if source_id in self._owner:
                    collisions.append(source_id)
                    continue
                self._owner[source_id] = catalog
                self._by_kind.setdefault(record.kind, []).append(source_id)
                if record.content_sha256:
                    owner = content_owner.setdefault(record.content_sha256, catalog_index)
                    if owner != catalog_index:
                        cross_catalog_duplicates.append(
                            {
                                "source_id": source_id,
                                "reason": "content_sha256",
                                "value": record.content_sha256,
                                "catalogs": [owner, catalog_index],
                            }
                        )
                for group in {record.source_group, *record.leakage_groups}:
                    owner = group_owner.setdefault(group, catalog_index)
                    if owner != catalog_index:
                        cross_catalog_duplicates.append(
                            {
                                "source_id": source_id,
                                "reason": "source/leakage group",
                                "value": group,
                                "catalogs": [owner, catalog_index],
                            }
                        )
        if collisions:
            raise ValueError(f"catalog set contains duplicate source IDs: {sorted(collisions)[:20]}")
        if cross_catalog_duplicates:
            raise ValueError(
                "catalog set contains cross-catalog source leakage: "
                f"{cross_catalog_duplicates[:20]}"
            )

    def pick(self, kind: SourceKind, rng: random.Random) -> str:
        available = [
            index
            for index, catalog in enumerate(self.catalogs)
            if catalog._by_kind.get(kind)
        ]
        if not available:
            raise ValueError(f"no catalog-set sources registered for kind {kind!r}")
        weights = [self.sampling_weights[index] for index in available]
        selected = rng.choices(available, weights=weights, k=1)[0]
        return self.catalogs[selected].pick(kind, rng)

    def keys(self, kind: SourceKind) -> list[str]:
        return list(self._by_kind.get(kind, []))

    def candidates(
        self,
        kind: SourceKind,
        rng: random.Random,
        *,
        max_duration: float | None = None,
        limit: int = 256,
    ) -> list[str]:
        """Weighted catalog permutation without letting a large bank dominate."""
        queues: list[list[str]] = []
        for catalog in self.catalogs:
            queues.append(
                catalog.candidates(
                    kind,
                    rng,
                    max_duration=max_duration,
                    limit=limit,
                )
            )
        output: list[str] = []
        while any(queues) and len(output) < limit:
            available = [index for index, queue in enumerate(queues) if queue]
            weights = [self.sampling_weights[index] for index in available]
            selected = rng.choices(available, weights=weights, k=1)[0]
            output.append(queues[selected].pop(0))
        return output

    def load(self, key: str, sample_rate: int) -> tuple[np.ndarray, float]:
        return self._owner[key].load(key, sample_rate)

    def metadata(self, key: str) -> dict[str, object]:
        return self._owner[key].metadata(key)


@dataclass
class SyntheticSourcePool:
    """Generates deterministic placeholder audio so the renderer is testable.

    Each source key encodes its type and an index (e.g. ``"speech:03"``); the
    pool synthesizes a short clip whose envelope and spectral content are
    plausible for the type. Captions are deterministic per key.
    """

    sample_rate: int = 24000
    seed: int = 20260808
    index_range: tuple[int, int] = (0, 999)

    def __post_init__(self) -> None:
        low, high = self.index_range
        if low < 0 or high < low:
            raise ValueError(f"invalid synthetic source index_range: {self.index_range}")

    def pick(self, kind: SourceKind, rng: random.Random) -> str:
        idx = rng.randint(*self.index_range)
        return f"{kind}:{idx:03d}"

    def metadata(self, key: str) -> dict[str, object]:
        return {}

    def load(self, key: str, sample_rate: int) -> tuple[np.ndarray, float]:
        import hashlib

        kind, _, idx_s = key.partition(":")
        idx = int(idx_s) if idx_s else 0
        sr = sample_rate
        # Deterministic seed: use hashlib (NOT Python hash(), which is
        # randomized per-process via PYTHONHASHSEED and breaks cross-process
        # reproducibility).
        kind_hash = int(hashlib.sha256(kind.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(self.seed + idx + 1000 * (kind_hash % 100000))
        if kind == "speech":
            dur = rng.uniform(1.5, 5.0)
            wav = self._synth_speech(sr, dur, rng, idx)
        elif kind == "vocal":
            dur = rng.uniform(2.0, 6.0)
            wav = self._synth_vocal(sr, dur, rng, idx)
        elif kind == "music":
            dur = rng.uniform(8.0, 15.0)
            wav = self._synth_music(sr, dur, rng, idx)
        elif kind == "sfx":
            dur = rng.uniform(0.3, 1.5)
            wav = self._synth_sfx(sr, dur, rng, idx)
        elif kind == "ambience":
            dur = rng.uniform(10.0, 20.0)
            wav = self._synth_ambience(sr, dur, rng)
        else:
            dur = 1.0
            wav = rng.standard_normal(int(sr * dur)).astype(np.float32) * 0.1
        # normalize peak to 0.9
        peak = float(np.max(np.abs(wav)) + 1e-9)
        wav = (wav / peak) * 0.9
        return wav.astype(np.float32), float(len(wav) / sr)

    @staticmethod
    def _synth_speech(sr: int, dur: float, rng: np.random.Generator, idx: int) -> np.ndarray:
        """Formant-ish tones with syllable envelopes (not real speech)."""
        n = int(sr * dur)
        t = np.arange(n) / sr
        # syllable rate ~3-5 Hz
        syl_rate = rng.uniform(3.0, 5.0)
        env = 0.5 * (1 + np.sin(2 * np.pi * syl_rate * t))
        env *= (t > 0.05).astype(float)  # short onset
        f0 = rng.uniform(90, 220)
        formants = [rng.uniform(500, 900), rng.uniform(1100, 1700), rng.uniform(2400, 3200)]
        sig = np.zeros(n, dtype=np.float64)
        for f in formants:
            sig += np.sin(2 * np.pi * f * t) * np.exp(-((t * f * 0.001) % 1.0) * 3)
        sig *= env
        # add pitch
        sig += 0.3 * np.sin(2 * np.pi * f0 * t) * env
        return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)

    @staticmethod
    def _synth_vocal(sr: int, dur: float, rng: np.random.Generator, idx: int) -> np.ndarray:
        """Singing-like sustained tones with vibrato and breath envelope."""
        n = int(sr * dur)
        t = np.arange(n) / sr
        # higher pitch than speech, with vibrato
        f0 = rng.uniform(220, 440)
        vibrato = 0.02 * f0 * np.sin(2 * np.pi * 5.0 * t)
        carrier = np.sin(2 * np.pi * (f0 + vibrato) * t)
        # harmonics for vocal timbre
        sig = carrier + 0.4 * np.sin(2 * np.pi * 2 * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        # breath envelope: sustained notes with gaps
        note_dur = rng.uniform(0.8, 2.0)
        env = np.zeros(n)
        pos = 0
        while pos < n:
            note_len = int(min(note_dur, (n - pos) / sr) * sr)
            fade = int(0.05 * sr)
            if note_len > 2 * fade:
                env[pos:pos + fade] = np.linspace(0, 1, fade)
                env[pos + fade:pos + note_len - fade] = 1.0
                env[pos + note_len - fade:pos + note_len] = np.linspace(1, 0, fade)
            else:
                env[pos:pos + note_len] = 1.0
            pos += note_len + int(rng.uniform(0.1, 0.3) * sr)
        sig *= env[:n]
        return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)

    @staticmethod
    def _synth_music(sr: int, dur: float, rng: np.random.Generator, idx: int) -> np.ndarray:
        """A slow chord progression (root + fifth + octave)."""
        n = int(sr * dur)
        roots = [220.0, 246.94, 196.0, 174.61]
        chord_period = dur / max(1, len(roots))
        sig = np.zeros(n, dtype=np.float64)
        for i, root in enumerate(roots):
            lo = int(i * chord_period * sr)
            hi = min(n, int((i + 1) * chord_period * sr))
            tt = np.arange(hi - lo) / sr
            chord = (
                np.sin(2 * np.pi * root * tt)
                + 0.7 * np.sin(2 * np.pi * root * 1.5 * tt)
                + 0.5 * np.sin(2 * np.pi * root * 2 * tt)
            )
            # gentle tremolo
            chord *= 0.9 + 0.1 * np.sin(2 * np.pi * 2.0 * tt)
            # fade between chords
            fade = int(0.05 * sr)
            if fade > 0 and hi - lo > 2 * fade:
                env = np.ones(hi - lo)
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] = np.linspace(1, 0, fade)
                chord *= env
            sig[lo:hi] += chord
        # add a soft kick every 0.5s
        for beat in range(int(dur / 0.5)):
            pos = int(beat * 0.5 * sr)
            if pos < n:
                length = min(int(0.08 * sr), n - pos)
                decay = np.exp(-np.arange(length) / (0.02 * sr))
                sig[pos : pos + length] += 0.4 * decay * np.sin(
                    2 * np.pi * 60 * np.arange(length) / sr
                )
        return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)

    @staticmethod
    def _synth_sfx(sr: int, dur: float, rng: np.random.Generator, idx: int) -> np.ndarray:
        """A transient noise burst with fast decay."""
        n = int(sr * dur)
        noise = rng.standard_normal(n).astype(np.float64)
        decay = np.exp(-np.arange(n) / (0.05 * sr))
        sig = noise * decay
        # bandpass-ish: combine with a tonal component
        t = np.arange(n) / sr
        f = rng.uniform(800, 3000)
        sig += 0.5 * np.sin(2 * np.pi * f * t) * decay
        return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)

    @staticmethod
    def _synth_ambience(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
        """Ambience bed with rain/wind/room-tone character.

        Improvements over v1:
        - Pink noise (not white) for more natural spectral balance
        - Slow amplitude modulation for natural起伏
        - Periodic droplet clicks for rain-like character
        - Low-pass filtered for warmth
        """
        from scipy.signal import firwin, lfilter

        n = int(sr * dur)
        # Pink-ish noise via a vectorized one-pole filter. This is numerically
        # identical to the previous Python recurrence, but avoids iterating
        # over hundreds of thousands of samples per rendered source.
        white = rng.standard_normal(n).astype(np.float64)
        pink = lfilter([0.01], [1.0, -0.99], white)
        # Low-pass for warmth (cutoff ~2kHz)
        try:
            b = firwin(65, 2000 / (sr / 2))
            pink = lfilter(b, 1.0, pink)
        except Exception:
            pass
        # Slow amplitude modulation (0.1-0.3 Hz natural起伏)
        t = np.arange(n) / sr
        mod_freq = rng.uniform(0.1, 0.3)
        modulation = 0.7 + 0.3 * np.sin(2 * np.pi * mod_freq * t)
        sig = pink * modulation
        # Add periodic droplet clicks for rain character (~2-5 Hz)
        click_rate = rng.uniform(2, 5)
        n_clicks = int(dur * click_rate)
        for _ in range(n_clicks):
            pos = int(rng.integers(0, max(1, n - int(0.01 * sr))))
            click_len = int(0.005 * sr)
            if pos + click_len < n:
                click = rng.standard_normal(click_len) * np.exp(-np.arange(click_len) / 3)
                sig[pos:pos + click_len] += 0.05 * click
        return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------- #
# captions
# --------------------------------------------------------------------------- #
_SYNTH_CAPTIONS = {
    "speech": [
        "一名说话者正在讲话。",
        "A speaker is talking.",
        "说话者描述当前场景。",
    ],
    "vocal": [
        '"take me home tonight"',
        '"la la la, dancing in the rain"',
        '"don\'t stop believing, hold on to that feeling"',
        '"月亮代表我的心"',
    ],
    "music": [
        "背景音乐持续播放，节奏平稳。",
        "Background music plays steadily.",
        "电子伴奏循环播放。",
    ],
    "sfx": [
        "一次玻璃破碎声。",
        "A glass breaking sound.",
        "短促的撞击声。",
        "远处的犬吠。",
    ],
    "ambience": [
        "持续的环境噪声。",
        "Ambient room tone.",
        "雨声背景。",
    ],
}


def _caption_for(kind: SourceKind, rng: random.Random) -> str:
    return rng.choice(_SYNTH_CAPTIONS.get(kind, ["An audible event."]))


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #
@dataclass
class SceneSamplerConfig:
    sample_rate: int = 24000
    duration_range: tuple[float, float] = DURATION_RANGE
    template_duration_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    gain_db_range: tuple[float, float] = GAIN_DB_RANGE
    kind_gain_db_offsets: dict[str, float] = field(default_factory=dict)
    # Optional post-gain active-RMS targets for audited real sources.  When
    # present, gain is derived from measured active RMS rather than raw peaks
    # or a whole-file RMS diluted by long silence.
    target_active_rms_dbfs_by_kind: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    # Fail instead of applying an extreme correction to an anomalously quiet
    # or loud source.  ``None`` preserves legacy manifests/configurations.
    max_abs_source_gain_db: float | None = None
    fg_bg_snr_range: tuple[float, float] = FG_BG_SNR_RANGE
    t60_range: tuple[float, float] = T60_RANGE
    echo_delay_ms_range: tuple[int, int] = ECHO_DELAY_MS_RANGE
    echo_atten_db_range: tuple[float, float] = ECHO_ATTEN_DB_RANGE
    repeat_range: tuple[int, int] = REPEAT_RANGE
    merge_threshold_range: tuple[float, float] = MERGE_THRESHOLD_RANGE
    resolutions: tuple[float, ...] = RESOLUTIONS
    styles: tuple[str, ...] = STYLES
    activity_threshold_range: tuple[float, float] = ACTIVITY_THRESHOLD_RANGE
    foreground_onset_fraction_range: tuple[float, float] | None = None
    loop_background_to_scene: bool = False
    enforce_speaker_overlap: bool = False
    dense_repeated_event: bool = False
    spread_repeated_event: bool = False
    stable_unique_source_ids: bool = False
    ducking_probability: float = 0.7
    ducking_depth_db_range: tuple[float, float] = (2.0, 5.0)
    # probability of applying RIR / echo to a scene
    p_rir: float = 0.5
    p_echo: float = 0.3


def _snr_to_gain_db(snr_db: float, ref_gain_db: float) -> float:
    """Foreground at ref_gain; background at ref_gain - snr."""
    return ref_gain_db - snr_db


class SceneGraphSampler:
    """Samples :class:`Scene` objects from templates and a source pool."""

    def __init__(self, pool: SourcePool, config: SceneSamplerConfig | None = None):
        self.pool = pool
        self.config = config or SceneSamplerConfig()

    def sample(
        self,
        scene_id: str,
        seed: int,
        template: TemplateID,
    ) -> Scene:
        rng = random.Random(seed)
        cfg = self.config
        duration_range = cfg.template_duration_ranges.get(template, cfg.duration_range)
        duration = round(rng.uniform(*duration_range) / TIME_RESOLUTION_SEC) * TIME_RESOLUTION_SEC
        duration = round(duration, 6)

        sources = self._place_sources(template, duration, rng)
        conditions = self._sample_conditions(rng, sources)
        supervision = self._sample_supervision(rng)
        return Scene(
            scene_id=scene_id,
            seed=seed,
            duration=duration,
            template=template,
            sources=sources,
            conditions=conditions,
            supervision=supervision,
            sample_rate=cfg.sample_rate,
        )

    def _place_sources(
        self, template: TemplateID, duration: float, rng: random.Random
    ) -> list[PlacedSource]:
        cfg = self.config
        fg_gain = rng.uniform(*cfg.gain_db_range)
        bg_gain = _snr_to_gain_db(rng.uniform(*cfg.fg_bg_snr_range), fg_gain)
        source_serial = 0
        selected_paths: set[str] = set()
        selected_leakage_groups: set[str] = set()
        selected_voice_identities: set[str] = set()

        def _src(kind: SourceKind, *, fg: bool, identity: str | None = None) -> PlacedSource:
            nonlocal source_serial
            source_serial += 1
            key = ""
            metadata: dict[str, object] = {}
            candidates_fn = getattr(self.pool, "candidates", None)
            keys_fn = getattr(self.pool, "keys", None)
            if callable(candidates_fn):
                candidates = list(
                    candidates_fn(
                        kind,
                        rng,
                        max_duration=(
                            duration - 0.05
                            if kind not in ("music", "ambience")
                            else None
                        ),
                    )
                )
            elif callable(keys_fn):
                candidates = list(keys_fn(kind))
                rng.shuffle(candidates)
            else:
                candidates = [self.pool.pick(kind, rng) for _ in range(128)]
            for candidate in candidates:
                if candidate in selected_paths:
                    continue
                metadata_fn = getattr(self.pool, "metadata", None)
                candidate_metadata = metadata_fn(candidate) if callable(metadata_fn) else {}
                candidate_duration = candidate_metadata.get("source_duration_sec")
                candidate_groups = {
                    str(candidate_metadata.get("source_group") or ""),
                    *(
                        str(item)
                        for item in candidate_metadata.get("leakage_groups", [])
                    ),
                } - {""}
                if candidate_groups & selected_leakage_groups:
                    continue
                # Speech, vocals and discrete SFX carry utterance/event-level
                # text.  Cropping their tail while keeping the complete text
                # would manufacture unsupported supervision.
                if (
                    kind not in ("music", "ambience")
                    and candidate_duration is not None
                    and float(candidate_duration) > duration - 0.05
                ):
                    continue
                if kind in ("speech", "vocal"):
                    voice_identity = str(
                        candidate_metadata.get("identity")
                        or candidate_metadata.get("source_group")
                        or candidate
                    )
                    if voice_identity in selected_voice_identities:
                        continue
                    selected_voice_identities.add(voice_identity)
                key = candidate
                metadata = candidate_metadata
                selected_paths.add(candidate)
                selected_leakage_groups.update(candidate_groups)
                break
            if not key:
                raise ValueError(
                    f"source pool cannot provide enough unique recordings for "
                    f"template={template!r}, kind={kind!r}"
                )
            target_range = cfg.target_active_rms_dbfs_by_kind.get(kind)
            measured_rms = candidate_metadata.get("source_active_rms_dbfs")
            if target_range is not None:
                if measured_rms is None:
                    raise ValueError(
                        "active-RMS-normalized sampling requires "
                        f"source_active_rms_dbfs: {key}"
                    )
                if len(target_range) != 2 or target_range[0] > target_range[1]:
                    raise ValueError(
                        f"invalid target RMS range for {kind}: {target_range}"
                    )
                target_rms = rng.uniform(*target_range)
                gain = target_rms - float(measured_rms)
                if (
                    cfg.max_abs_source_gain_db is not None
                    and abs(gain) > cfg.max_abs_source_gain_db
                ):
                    raise ValueError(
                        f"RMS normalization for {key} requires {gain:.2f} dB, "
                        f"above max_abs_source_gain_db={cfg.max_abs_source_gain_db}"
                    )
            else:
                gain = (fg_gain if fg else bg_gain) + float(
                    cfg.kind_gain_db_offsets.get(kind, 0.0)
                )
            # Background starts at zero and may be extended by the renderer.
            # Foreground is distributed across the clip instead of being
            # restricted to the first 40%, which previously created long tails.
            if kind in ("music", "ambience"):
                onset = 0.0
            else:
                source_duration = (
                    float(metadata["source_duration_sec"])
                    if metadata.get("source_duration_sec") is not None
                    else None
                )
                latest_onset = (
                    max(0.0, duration - source_duration - 0.05)
                    if source_duration is not None
                    else duration
                )
                if cfg.foreground_onset_fraction_range is None:
                    desired_low, desired_high = 0.2, max(0.3, duration * 0.4)
                else:
                    low_fraction, high_fraction = cfg.foreground_onset_fraction_range
                    desired_low = max(0.1, duration * low_fraction)
                    desired_high = max(0.2, duration * high_fraction)
                latest_grid_onset = (
                    int((latest_onset + 1e-9) / TIME_RESOLUTION_SEC)
                    * TIME_RESOLUTION_SEC
                )
                onset_low = min(desired_low, latest_grid_onset)
                onset_high = min(max(desired_high, onset_low), latest_grid_onset)
                onset = round(
                    rng.uniform(onset_low, onset_high)
                    / TIME_RESOLUTION_SEC
                ) * TIME_RESOLUTION_SEC
                onset = min(onset, latest_grid_onset)
            repeat = 1
            repeat_gap = 0.0
            if template == "repeated_event" and kind == "sfx":
                repeat = rng.randint(*cfg.repeat_range)
                if cfg.spread_repeated_event:
                    # Spread instances over a meaningful part of the clip instead
                    # of packing them into the opening seconds.
                    repeat_gap = round(
                        rng.uniform(1.2, max(1.3, duration * 0.2)), 3
                    )
                else:
                    repeat_gap = round(rng.uniform(0.2, 1.0), 3)
            rir_id = None
            t60 = None
            if rng.random() < cfg.p_rir:
                t60 = round(rng.uniform(*cfg.t60_range), 3)
                rir_id = f"room_{rng.randint(1, 8):02d}"
            resolved_identity = str(metadata.get("identity") or identity) if metadata.get("identity") or identity else None
            text = str(metadata.get("text") or _caption_for(kind, rng))
            return PlacedSource(
                source_id=(
                    f"{kind[:2].upper()}{source_serial:02d}"
                    if cfg.stable_unique_source_ids
                    else f"{kind[0].upper()}{rng.randint(1, 99):02d}"
                ),
                kind=kind,
                path=key,
                onset=round(onset, 6),
                gain_db=round(gain, 3),
                text=text,
                identity=resolved_identity,
                source_group=(str(metadata["source_group"]) if metadata.get("source_group") else None),
                leakage_groups=[str(item) for item in metadata.get("leakage_groups", [])],
                source_labels=[str(item) for item in metadata.get("source_labels", [])],
                source_dataset=(str(metadata["source_dataset"]) if metadata.get("source_dataset") else None),
                source_license=(str(metadata["source_license"]) if metadata.get("source_license") else None),
                annotation_origin=(str(metadata["annotation_origin"]) if metadata.get("annotation_origin") else None),
                text_is_verbatim=bool(metadata.get("text_is_verbatim", False)),
                source_file_sha256=(str(metadata["source_file_sha256"]) if metadata.get("source_file_sha256") else None),
                source_duration_sec=(float(metadata["source_duration_sec"]) if metadata.get("source_duration_sec") is not None else None),
                source_rms_dbfs=(float(metadata["source_rms_dbfs"]) if metadata.get("source_rms_dbfs") is not None else None),
                source_active_rms_dbfs=(float(metadata["source_active_rms_dbfs"]) if metadata.get("source_active_rms_dbfs") is not None else None),
                repeat=repeat,
                repeat_gap_s=repeat_gap,
                rir_id=rir_id,
                t60_sec=t60,
                is_foreground=fg,
                loop_to_scene=cfg.loop_background_to_scene
                and kind in ("music", "ambience"),
            )

        if template == "isolated_sfx":
            return [_src("sfx", fg=True)]
        if template == "speech_over_music":
            return [_src("music", fg=False), _src("speech", fg=True, identity="S1")]
        if template == "speech_with_sfx":
            return [_src("speech", fg=True, identity="S1"), _src("sfx", fg=True)]
        if template == "speech_ambience_sfx":
            return [
                _src("ambience", fg=False),
                _src("speech", fg=True, identity="S1"),
                _src("sfx", fg=True),
            ]
        if template == "music_with_sfx":
            return [_src("music", fg=False), _src("sfx", fg=True)]
        if template == "speech_music_sfx":
            return [
                _src("music", fg=False),
                _src("speech", fg=True, identity="S1"),
                _src("sfx", fg=True),
            ]
        if template == "repeated_event":
            if cfg.dense_repeated_event:
                # Retain the multi-span repeated SFX target while ensuring the
                # main distribution is not a long, otherwise-empty clip.
                sources = [_src("ambience", fg=False), _src("sfx", fg=True)]
                sources[1].onset = round(
                    rng.uniform(0.2, max(0.3, duration * 0.15))
                    / TIME_RESOLUTION_SEC
                ) * TIME_RESOLUTION_SEC
                return sources
            return [_src("sfx", fg=True)]
        if template == "ambient_with_intermittent_sfx":
            return [_src("ambience", fg=False), _src("sfx", fg=True)]
        # B3 templates: lyrics + overlapping speakers
        if template == "lyrics_over_music":
            return [_src("music", fg=False), _src("vocal", fg=True, identity="V1")]
        if template == "speech_music_lyrics_sfx":
            return [
                _src("music", fg=False),
                _src("speech", fg=True, identity="S1"),
                _src("vocal", fg=True, identity="V1"),
                _src("sfx", fg=True),
            ]
        if template == "overlapping_speakers":
            speakers = [
                _src("speech", fg=True, identity="S1"),
                _src("speech", fg=True, identity="S2"),
            ]
            if cfg.enforce_speaker_overlap:
                first_onset = round(
                    rng.uniform(0.1, max(0.2, min(1.0, duration * 0.15)))
                    / TIME_RESOLUTION_SEC
                ) * TIME_RESOLUTION_SEC
                second_onset = max(
                    0.0,
                    first_onset
                    + round(rng.uniform(-0.3, 0.5) / TIME_RESOLUTION_SEC)
                    * TIME_RESOLUTION_SEC,
                )
                speakers[0].onset = round(first_onset, 6)
                speakers[1].onset = round(second_onset, 6)
            return speakers
        if template == "random_mix":
            # B2-no-template ablation: fully random source selection + placement
            n_sources = rng.randint(2, 4)
            kinds_pool = ["speech", "music", "sfx", "vocal"]
            sources = []
            for _ in range(n_sources):
                kind = rng.choice(kinds_pool)
                identity = None
                if kind == "speech":
                    identity = f"S{len([s for s in sources if s.kind=='speech'])+1}"
                elif kind == "vocal":
                    identity = f"V{len([s for s in sources if s.kind=='vocal'])+1}"
                # override onset to be fully random (not template-structured)
                s = _src(kind, fg=True, identity=identity)
                s.onset = round(rng.uniform(0.0, max(0.1, duration * 0.5)) / TIME_RESOLUTION_SEC) * TIME_RESOLUTION_SEC
                sources.append(s)
            return sources
        # New complex templates for higher data complexity (docs/15 problem 7)
        if template == "complex_cocktail":
            # 3-4 speakers + background music + occasional sfx
            n_speakers = rng.randint(3, 4)
            sources = [_src("music", fg=False)]
            for si in range(n_speakers):
                sources.append(_src("speech", fg=True, identity=f"S{si+1}"))
            if rng.random() < 0.5:
                sources.append(_src("sfx", fg=True))
            return sources
        if template == "rich_band":
            # music + vocals + 2 sfx + ambience = 5 sources
            return [
                _src("ambience", fg=False),
                _src("music", fg=False),
                _src("vocal", fg=True, identity="V1"),
                _src("sfx", fg=True),
                _src("sfx", fg=True),
            ]
        if template == "multi_event_dense":
            # 2-3 sfx + speech + music = 4-5 sources, dense events
            sources = [_src("music", fg=False), _src("speech", fg=True, identity="S1")]
            for _ in range(rng.randint(2, 3)):
                sources.append(_src("sfx", fg=True))
            return sources
        raise ValueError(f"unknown template {template!r}")

    def _sample_conditions(
        self, rng: random.Random, sources: list[PlacedSource]
    ) -> Conditions:
        cfg = self.config
        echo_delay = None
        echo_atten = None
        if rng.random() < cfg.p_echo:
            echo_delay = rng.randint(*cfg.echo_delay_ms_range)
            echo_atten = round(rng.uniform(*cfg.echo_atten_db_range), 2)
        t60 = None
        # scene-level T60 if any source has RIR
        for s in sources:
            if s.t60_sec is not None:
                t60 = s.t60_sec
                break
        has_voice = any(source.kind in ("speech", "vocal") for source in sources)
        has_background = any(source.kind in ("music", "ambience") for source in sources)
        ducking_enabled = (
            has_voice
            and has_background
            and rng.random() < cfg.ducking_probability
        )
        ducking_depth_db = (
            round(rng.uniform(*cfg.ducking_depth_db_range), 3)
            if ducking_enabled
            else None
        )
        return Conditions(
            noise_snr_db=None,
            echo_delay_ms=echo_delay,
            echo_atten_db=echo_atten,
            t60_sec=t60,
            codec=None,
            overlap_ratio=None,
            ducking_enabled=ducking_enabled,
            ducking_depth_db=ducking_depth_db,
        )

    def _sample_supervision(self, rng: random.Random) -> Supervision:
        cfg = self.config
        return Supervision(
            style=rng.choice(cfg.styles),
            activity_threshold=round(rng.uniform(*cfg.activity_threshold_range), 4),
            merge_threshold_s=round(rng.uniform(*cfg.merge_threshold_range), 3),
            resolution_s=rng.choice(cfg.resolutions),
        )


__all__ = [
    "CatalogSetSourcePool",
    "CatalogSourcePool",
    "Conditions",
    "FileSourcePool",
    "PlacedSource",
    "Scene",
    "SceneGraphSampler",
    "SceneSamplerConfig",
    "SourcePool",
    "Supervision",
    "SyntheticSourcePool",
]
