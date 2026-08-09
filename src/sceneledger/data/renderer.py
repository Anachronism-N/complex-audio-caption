"""Deterministic renderer: turn a :class:`Scene` into audio + supervision target.

Pipeline per source (``docs/06`` §3):

    load -> gain -> fade -> repeat -> RIR -> place at onset
    sum stems -> mixture -> scene-level echo

Each source's activity is computed from its placed stem (RMS, ``activity.py``)
and converted to 0.1 s spans. The supervision target is a :class:`Ledger`
built from those spans + source metadata, so P3 (B1/B2) can train directly on
``(mixture_audio, target_ledger)`` pairs.

Determinism: given the same ``Scene`` (seed) and source pool, the mixture
waveform hash is identical. Validation/test mixes are pre-generated and frozen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from sceneledger.data.activity import ActivityResult, compute_activity
from sceneledger.data.scene_graph_sampler import (
    PlacedSource,
    Scene,
    SourcePool,
)
from sceneledger.data.schema import (
    SCHEMA_VERSION,
    TIME_RESOLUTION_SEC,
    Event,
    Ledger,
    Provenance,
    Span,
    Track,
)
from sceneledger.data.schema import (
    Conditions as LedgerConditions,
)

# type ordering for stable event serialization (docs/06 §5.2)
_TYPE_ORDER = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}
RENDERER_VERSION = "0.3.0"
RESIDUAL_STEM_ID = "__residual__"


@dataclass
class RenderedSource:
    placed: PlacedSource
    stem: np.ndarray  # [N] placed within clip (post gain/fade/repeat/RIR)
    activity: ActivityResult


@dataclass
class RenderOutput:
    scene: Scene
    mixture: np.ndarray  # [N] float32, post-echo + clipping guard
    dry_mixture: np.ndarray  # [N] float32, sum of scaled semantic stems
    residual_stem: np.ndarray  # [N] float32, echo/mastering residual
    stems: list[RenderedSource]
    target_ledger: Ledger
    sample_rate: int

    def waveform_hash(self, wav: np.ndarray) -> str:
        return hashlib.sha256(wav.tobytes()).hexdigest()[:16]

    def mixture_hash(self) -> str:
        return self.waveform_hash(self.mixture)

    def stem_hashes(self) -> dict[str, str]:
        hashes = {
            rs.placed.source_id: self.waveform_hash(rs.stem) for rs in self.stems
        }
        hashes[RESIDUAL_STEM_ID] = self.waveform_hash(self.residual_stem)
        return hashes


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #
def _db_to_gain(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def _apply_gain(wav: np.ndarray, gain_db: float) -> np.ndarray:
    return wav * _db_to_gain(gain_db)


def _apply_fade(wav: np.ndarray, sample_rate: int, fade_s: float = 0.01) -> np.ndarray:
    n = wav.shape[-1]
    fade = int(fade_s * sample_rate)
    if fade <= 0 or n < 2 * fade:
        return wav
    out = wav.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
    out[..., :fade] *= ramp
    out[..., -fade:] *= ramp[::-1]
    return out


def _repeat_source(wav: np.ndarray, repeat: int, gap_s: float, sample_rate: int) -> np.ndarray:
    if repeat <= 1:
        return wav
    gap = int(gap_s * sample_rate)
    pad = np.zeros(gap, dtype=wav.dtype)
    parts = []
    for i in range(repeat):
        if i > 0:
            parts.append(pad)
        parts.append(wav)
    return np.concatenate(parts)


def _synth_rir(sample_rate: int, t60: float, rng: np.random.Generator) -> np.ndarray:
    """A simple decaying-noise impulse response of length ~min(3*T60, 1.0)s."""
    length = max(1, int(sample_rate * max(0.05, min(t60 * 3.0, 1.0))))
    noise = rng.standard_normal(length).astype(np.float64)
    t = np.arange(length) / sample_rate
    # early reflection spike + exponential decay toward T60
    decay = np.exp(-6.9 * t / max(t60, 0.05))
    rir = noise * decay
    rir[0] = 1.0
    # normalize energy
    rir /= np.sqrt(np.sum(rir**2) + 1e-12)
    return rir.astype(np.float32)


def _apply_rir(wav: np.ndarray, sample_rate: int, t60: float, seed: int) -> np.ndarray:
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    rir = _synth_rir(sample_rate, t60, rng)
    # FFT convolution is O(N log N); np.convolve is O(N*M) and far too slow
    # for long clips with multi-second RIRs.
    conv = fftconvolve(wav.astype(np.float64), rir, mode="full")[: wav.shape[-1]]
    # match peak to avoid level blow-up
    peak_in = float(np.max(np.abs(wav)) + 1e-9)
    peak_out = float(np.max(np.abs(conv)) + 1e-9)
    conv *= peak_in / peak_out
    return conv.astype(np.float32)


def _apply_echo(
    wav: np.ndarray, sample_rate: int, delay_ms: int, atten_db: float
) -> np.ndarray:
    delay = int(sample_rate * delay_ms / 1000.0)
    if delay <= 0 or delay >= wav.shape[-1]:
        return wav
    out = wav.copy().astype(np.float64)
    gain = _db_to_gain(atten_db)
    out[delay:] += gain * wav[:-delay]
    peak = float(np.max(np.abs(out)) + 1e-9)
    out *= (float(np.max(np.abs(wav)) + 1e-9)) / peak
    return out.astype(np.float32)


def _place_at_onset(wav: np.ndarray, onset: float, duration: float, sample_rate: int) -> np.ndarray:
    n_clip = int(round(duration * sample_rate))
    out = np.zeros(n_clip, dtype=np.float32)
    start = int(round(onset * sample_rate))
    end = min(n_clip, start + wav.shape[-1])
    if start >= n_clip:
        return out
    out[start:end] = wav[: end - start]
    return out


# --------------------------------------------------------------------------- #
# activity -> spans -> ledger
# --------------------------------------------------------------------------- #
def _continuous(kind: str) -> bool:
    return kind in ("music", "ambience")


def _build_events_for_source(
    src: PlacedSource, activity: ActivityResult, track_id: str, eid_start: int
) -> list[Event]:
    """Convert a source's activity spans into one or more events.

    * speech: one event per span (each utterance).
    * music / ambience: one event over the union of all spans.
    * sfx: one event with multiple spans (repeated instances of same source).
    """
    spans = activity.spans
    if not spans:
        return []
    etype = src.event_type()
    span_objs = [Span(start_sec=s, end_sec=e) for s, e in spans]
    events: list[Event] = []

    if etype == "speech":
        for i, sp in enumerate(span_objs):
            events.append(
                Event(
                    id=f"E{eid_start + i:03d}",
                    type="speech",
                    track_id=track_id,
                    spans=[sp],
                    text=src.text,
                    verbatim=False,
                    confidence=0.95,
                )
            )
        return events

    if etype == "sfx":
        # one event with all spans (repeated same source) — exercises multi-span
        events.append(
            Event(
                id=f"E{eid_start:03d}",
                type="sfx",
                track_id=track_id,
                spans=span_objs,
                text=src.text,
                confidence=0.93,
            )
        )
        return events

    # music / ambience -> one event over the union
    events.append(
        Event(
            id=f"E{eid_start:03d}",
            type=etype,  # type: ignore[arg-type]
            track_id=track_id,
            spans=span_objs,
            text=src.text,
            confidence=0.94,
        )
    )
    return events


def _build_ledger(scene: Scene, rendered: list[RenderedSource]) -> Ledger:
    """Assemble the canonical supervision target from rendered sources."""
    tracks: list[Track] = []
    events: list[Event] = []

    # tracks first (T1.. in source order)
    track_id_by_src: dict[str, str] = {}
    for i, rs in enumerate(rendered):
        tid = f"T{i + 1}"
        track_id_by_src[rs.placed.source_id] = tid
        spans = [
            Span(start_sec=s, end_sec=e) for s, e in rs.activity.spans
        ] or [Span(start_sec=0.0, end_sec=min(0.1, scene.duration))]
        tracks.append(
            Track(
                id=tid,
                kind=rs.placed.kind,  # type: ignore[arg-type]
                identity=rs.placed.identity,
                spans=spans,
                confidence=0.95,
            )
        )

    # events, with stable ID assignment
    eid = 1
    raw_events: list[Event] = []
    for rs in rendered:
        evs = _build_events_for_source(
            rs.placed, rs.activity, track_id_by_src[rs.placed.source_id], eid
        )
        eid += len(evs)
        raw_events.extend(evs)

    # stable order: onset asc, then type order, then source id
    def _sort_key(e: Event) -> tuple:
        return (
            round(e.start_sec(), 6),
            _TYPE_ORDER.get(e.type, 9),
            e.id,
        )

    events = sorted(raw_events, key=_sort_key)

    # conditions
    overlap_ratio = _overlap_ratio(rendered, scene.duration)
    cond = scene.conditions
    ledger_cond = LedgerConditions(
        domain=scene.template,
        snr_db=cond.noise_snr_db,
        t60_sec=cond.t60_sec,
        echo=cond.echo_delay_ms is not None,
        codec=cond.codec,
        overlap_ratio=overlap_ratio,
    )

    return Ledger(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        sample_id=scene.scene_id,
        duration_sec=scene.duration,
        time_resolution_sec=TIME_RESOLUTION_SEC,  # type: ignore[arg-type]
        conditions=ledger_cond,
        tracks=tracks,
        events=events,
        provenance=Provenance(
            label_level="B",
            source_dataset="tac_mini_synthetic",
            renderer_manifest_uri=None,
            license_status="synthetic",
        ),
    )


def _overlap_ratio(rendered: list[RenderedSource], duration: float) -> float:
    """Fraction of clip time with >= 2 simultaneously active sources."""
    if not rendered:
        return 0.0
    n_frames = int(round(duration / TIME_RESOLUTION_SEC))
    if n_frames <= 0:
        return 0.0
    active = np.zeros(n_frames, dtype=np.int32)
    for rs in rendered:
        mask = rs.activity.activity_mask
        resolution = rs.activity.resolution_sec
        for index in np.flatnonzero(mask):
            start = int(round(index * resolution / TIME_RESOLUTION_SEC))
            end = int(round((index + 1) * resolution / TIME_RESOLUTION_SEC))
            start = max(0, min(start, n_frames))
            end = max(start, min(end, n_frames))
            active[start:end] += 1
    overlap = int(np.sum(active >= 2))
    return round(overlap / n_frames, 6)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def render_scene(scene: Scene, pool: SourcePool) -> RenderOutput:
    """Render a :class:`Scene` to mixture + stems + supervision ledger."""
    sr = scene.sample_rate
    n_clip = int(round(scene.duration * sr))
    rendered: list[RenderedSource] = []

    source_ids = [source.source_id for source in scene.sources]
    if len(source_ids) != len(set(source_ids)):
        duplicates = sorted({sid for sid in source_ids if source_ids.count(sid) > 1})
        raise ValueError(f"scene {scene.scene_id} has duplicate source IDs: {duplicates}")

    for idx, src in enumerate(scene.sources):
        wav, _dur = pool.load(src.path, sr)
        wav = wav.astype(np.float32)
        wav = _apply_gain(wav, src.gain_db)
        wav = _apply_fade(wav, sr, fade_s=0.01)
        wav = _repeat_source(wav, src.repeat, src.repeat_gap_s, sr)
        if src.t60_sec is not None:
            wav = _apply_rir(wav, sr, src.t60_sec, seed=scene.seed + idx * 7919)
        placed = _place_at_onset(wav, src.onset, scene.duration, sr)
        activity = compute_activity(
            placed,
            sr,
            activity_threshold=scene.supervision.activity_threshold,
            resolution_sec=scene.supervision.resolution_s,
            merge_threshold_s=scene.supervision.merge_threshold_s,
            duration_sec=scene.duration,
            is_continuous=_continuous(src.kind),
        )
        rendered.append(RenderedSource(placed=src, stem=placed, activity=activity))

    mixture = np.zeros(n_clip, dtype=np.float32)
    for rs in rendered:
        mixture += rs.stem
    dry_mixture = mixture.copy()

    # scene-level echo applied to the mixture
    if scene.conditions.echo_delay_ms is not None and scene.conditions.echo_atten_db is not None:
        mixture = _apply_echo(
            mixture, sr, scene.conditions.echo_delay_ms, scene.conditions.echo_atten_db
        )

    # Represent scene-level echo/mastering as an explicit residual.  Apply one
    # common scale to the mixture, semantic stems, and residual so that saved
    # components remain physically auditable:
    #     mixture ~= sum(semantic stems) + residual.
    residual = np.asarray(mixture - dry_mixture, dtype=np.float32)
    component_peak = max(
        [float(np.max(np.abs(mixture))), float(np.max(np.abs(residual)))]
        + [float(np.max(np.abs(rs.stem))) for rs in rendered]
    )
    master_scale = min(1.0, 0.98 / component_peak) if component_peak > 0 else 1.0
    if master_scale != 1.0:
        mixture = np.asarray(mixture * master_scale, dtype=np.float32)
        for rs in rendered:
            rs.stem = np.asarray(rs.stem * master_scale, dtype=np.float32)
    dry_mixture = np.zeros(n_clip, dtype=np.float32)
    for rs in rendered:
        dry_mixture += rs.stem
    residual = np.asarray(mixture - dry_mixture, dtype=np.float32)

    target_ledger = _build_ledger(scene, rendered)
    return RenderOutput(
        scene=scene,
        mixture=np.asarray(mixture, dtype=np.float32),
        dry_mixture=dry_mixture,
        residual_stem=residual,
        stems=rendered,
        target_ledger=target_ledger,
        sample_rate=sr,
    )


__all__ = [
    "RESIDUAL_STEM_ID",
    "RENDERER_VERSION",
    "RenderOutput",
    "RenderedSource",
    "render_scene",
]
