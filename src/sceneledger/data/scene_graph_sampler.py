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

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np

from sceneledger.data.schema import TIME_RESOLUTION_SEC

SourceKind = Literal["speech", "music", "sfx", "ambience"]
TemplateID = Literal[
    "isolated_sfx",
    "speech_over_music",
    "music_with_sfx",
    "speech_music_sfx",
    "repeated_event",
    "ambient_with_intermittent_sfx",
    "lyrics_over_music",
    "speech_music_lyrics_sfx",
    "overlapping_speakers",
    "random_mix",
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
    repeat: int = 1
    repeat_gap_s: float = 0.0
    rir_id: str | None = None
    t60_sec: float | None = None
    is_foreground: bool = True

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
    return {
        "source_id": s.source_id,
        "kind": s.kind,
        "path": s.path,
        "onset": s.onset,
        "gain_db": s.gain_db,
        "text": s.text,
        "identity": s.identity,
        "repeat": s.repeat,
        "repeat_gap_s": s.repeat_gap_s,
        "rir_id": s.rir_id,
        "t60_sec": s.t60_sec,
        "is_foreground": s.is_foreground,
    }


def _conditions_dict(c: Conditions) -> dict:
    return {
        "noise_snr_db": c.noise_snr_db,
        "echo_delay_ms": c.echo_delay_ms,
        "echo_atten_db": c.echo_atten_db,
        "t60_sec": c.t60_sec,
        "codec": c.codec,
        "overlap_ratio": c.overlap_ratio,
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
        import soundfile as sf

        wav, sr = sf.read(key, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            import librosa

            wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=sample_rate)
        return wav.astype(np.float32), float(len(wav) / sample_rate)


@dataclass
class SyntheticSourcePool:
    """Generates deterministic placeholder audio so the renderer is testable.

    Each source key encodes its type and an index (e.g. ``"speech:03"``); the
    pool synthesizes a short clip whose envelope and spectral content are
    plausible for the type. Captions are deterministic per key.
    """

    sample_rate: int = 24000
    seed: int = 20260808

    def pick(self, kind: SourceKind, rng: random.Random) -> str:
        idx = rng.randint(0, 999)
        return f"{kind}:{idx:03d}"

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
        t = np.arange(n) / sr
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
        from scipy.signal import lfilter, firwin

        n = int(sr * dur)
        # Pink noise via filtered white noise (Voss-McCartney approximation)
        white = rng.standard_normal(n).astype(np.float64)
        # accumulate every 2^k sample for pink spectrum
        pink = np.zeros(n)
        prev = 0.0
        for i in range(n):
            prev = 0.99 * prev + 0.01 * white[i]
            pink[i] = prev
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
    gain_db_range: tuple[float, float] = GAIN_DB_RANGE
    fg_bg_snr_range: tuple[float, float] = FG_BG_SNR_RANGE
    t60_range: tuple[float, float] = T60_RANGE
    echo_delay_ms_range: tuple[int, int] = ECHO_DELAY_MS_RANGE
    echo_atten_db_range: tuple[float, float] = ECHO_ATTEN_DB_RANGE
    repeat_range: tuple[int, int] = REPEAT_RANGE
    merge_threshold_range: tuple[float, float] = MERGE_THRESHOLD_RANGE
    resolutions: tuple[float, ...] = RESOLUTIONS
    styles: tuple[str, ...] = STYLES
    activity_threshold_range: tuple[float, float] = ACTIVITY_THRESHOLD_RANGE
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
        # Per-template duration ranges to avoid long clips with few events
        template_durations = {
            "isolated_sfx": (3, 8),
            "repeated_event": (5, 10),
            "ambient_with_intermittent_sfx": (10, 20),
            "overlapping_speakers": (5, 15),
            "random_mix": (8, 20),
        }
        dur_range = template_durations.get(template, cfg.duration_range)
        duration = round(rng.uniform(*dur_range) / TIME_RESOLUTION_SEC) * TIME_RESOLUTION_SEC
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

        def _src(kind: SourceKind, *, fg: bool, identity: str | None = None) -> PlacedSource:
            key = self.pool.pick(kind, rng)
            gain = fg_gain if fg else bg_gain
            # onset: foreground sources placed with margin; background at 0
            if kind in ("music", "ambience"):
                onset = 0.0
            else:
                onset = round(
                    rng.uniform(0.2, max(0.3, duration * 0.4)) / TIME_RESOLUTION_SEC
                ) * TIME_RESOLUTION_SEC
            repeat = 1
            repeat_gap = 0.0
            if template == "repeated_event":
                repeat = rng.randint(*cfg.repeat_range)
                repeat_gap = round(rng.uniform(0.2, 1.0), 3)
            rir_id = None
            t60 = None
            if rng.random() < cfg.p_rir:
                t60 = round(rng.uniform(*cfg.t60_range), 3)
                rir_id = f"room_{rng.randint(1, 8):02d}"
            return PlacedSource(
                source_id=f"{kind[0].upper()}{rng.randint(1, 99):02d}",
                kind=kind,
                path=key,
                onset=round(onset, 6),
                gain_db=round(gain, 3),
                text=_caption_for(kind, rng),
                identity=identity,
                repeat=repeat,
                repeat_gap_s=repeat_gap,
                rir_id=rir_id,
                t60_sec=t60,
                is_foreground=fg,
            )

        if template == "isolated_sfx":
            return [_src("sfx", fg=True)]
        if template == "speech_over_music":
            return [_src("music", fg=False), _src("speech", fg=True, identity="S1")]
        if template == "music_with_sfx":
            return [_src("music", fg=False), _src("sfx", fg=True)]
        if template == "speech_music_sfx":
            return [
                _src("music", fg=False),
                _src("speech", fg=True, identity="S1"),
                _src("sfx", fg=True),
            ]
        if template == "repeated_event":
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
            return [
                _src("speech", fg=True, identity="S1"),
                _src("speech", fg=True, identity="S2"),
            ]
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
        return Conditions(
            noise_snr_db=None,
            echo_delay_ms=echo_delay,
            echo_atten_db=echo_atten,
            t60_sec=t60,
            codec=None,
            overlap_ratio=None,
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
