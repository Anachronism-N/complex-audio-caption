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
from sceneledger.data.scene_graph_sampler import PlacedSource, Scene, SourcePool
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


def _semantic_confidence(source: PlacedSource) -> float:
    """Conservative semantic confidence based on actual annotation origin."""
    return {
        "human": 1.0,
        "dataset": 0.98,
        "asr": 0.80,
        "audio_model": 0.60,
        "llm_rewrite": 0.50,
    }.get(source.annotation_origin, 0.50)


@dataclass
class RenderedSource:
    placed: PlacedSource
    stem: np.ndarray  # [N] placed within clip (post gain/fade/repeat/RIR)
    activity: ActivityResult


@dataclass
class RenderOutput:
    scene: Scene
    mixture: np.ndarray  # [N] float32, post-echo + clipping guard
    dry_mixture: np.ndarray  # [N] float32, exact sum of persisted stems, before echo
    stems: list[RenderedSource]
    target_ledger: Ledger
    sample_rate: int

    def waveform_hash(self, wav: np.ndarray) -> str:
        return hashlib.sha256(wav.tobytes()).hexdigest()[:16]

    def mixture_hash(self) -> str:
        return self.waveform_hash(self.mixture)

    def stem_hashes(self) -> dict[str, str]:
        return {rs.placed.source_id: self.waveform_hash(rs.stem) for rs in self.stems}


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


def _apply_fade_by_kind(wav: np.ndarray, kind: str, sample_rate: int) -> np.ndarray:
    """Source-type-dependent fade for natural boundaries (docs/15 fix)."""
    fade_map = {
        "ambience": 1.5,   # gradual 1.5s fade for environmental sounds
        "music": 0.5,      # 0.5s for music
        "vocal": 0.05,     # 50ms for vocals
        "speech": 0.05,    # 50ms for speech
        "sfx": 0.05,       # 50ms for sfx (transient, short fade)
    }
    fade_s = fade_map.get(kind, 0.05)
    return _apply_fade(wav, sample_rate, fade_s)


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


def _loop_to_duration(
    wav: np.ndarray, duration_s: float, sample_rate: int, crossfade_s: float = 0.05
) -> np.ndarray:
    """Loop a background source with short crossfades and crop to duration."""
    target = max(0, int(round(duration_s * sample_rate)))
    if target == 0:
        return np.zeros(0, dtype=np.float32)
    if wav.size == 0:
        return np.zeros(target, dtype=np.float32)
    if wav.shape[-1] >= target:
        return wav[:target].astype(np.float32, copy=True)

    overlap = min(
        int(round(crossfade_s * sample_rate)),
        max(0, wav.shape[-1] // 4),
    )
    out = wav.astype(np.float32, copy=True)
    while out.shape[-1] < target:
        if overlap <= 0:
            out = np.concatenate([out, wav])
            continue
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        blended = out[-overlap:] * (1.0 - ramp) + wav[:overlap] * ramp
        out = np.concatenate([out[:-overlap], blended, wav[overlap:]])
    return out[:target].astype(np.float32, copy=False)


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

    * speech: one event per source utterance; pauses become multiple spans.
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
        # A catalog source is one transcript-aligned utterance.  Energy VAD
        # may split it at pauses, but copying the full transcript into every
        # active region creates false repeated speech supervision.  Preserve
        # one utterance event and attach every supported activity span.
        events.append(
            Event(
                id=f"E{eid_start:03d}",
                type="speech",
                track_id=track_id,
                spans=span_objs,
                text=src.text,
                verbatim=src.text_is_verbatim,
                confidence=_semantic_confidence(src),
            )
        )
        return events

    if etype == "lys":
        # A source event is only verbatim when its words were supplied by a
        # human or a transcript-bearing dataset.  Model-generated lyrics are
        # semantic hypotheses and must not be promoted to exact quotations.
        for i, sp in enumerate(span_objs):
            events.append(
                Event(
                    id=f"E{eid_start + i:03d}",
                    type="lys",
                    track_id=track_id,
                    spans=[sp],
                    text=src.text,
                    verbatim=src.text_is_verbatim,
                    confidence=_semantic_confidence(src),
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
                confidence=_semantic_confidence(src),
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
            confidence=_semantic_confidence(src),
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

    datasets = sorted({rs.placed.source_dataset for rs in rendered if rs.placed.source_dataset})
    licenses = sorted({rs.placed.source_license for rs in rendered if rs.placed.source_license})
    is_catalog_scene = bool(datasets)
    # Programmatic mixing of known single sources is Level B: source identity,
    # placement and transforms are exact, while real-scene naturalness and any
    # model-derived semantic attributes are not promoted to Level A truth.
    label_level = "B" if is_catalog_scene else "model_prediction"
    return Ledger(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        sample_id=scene.scene_id,
        duration_sec=scene.duration,
        time_resolution_sec=TIME_RESOLUTION_SEC,  # type: ignore[arg-type]
        conditions=ledger_cond,
        tracks=tracks,
        events=events,
        provenance=Provenance(
            label_level=label_level,
            source_dataset="+".join(datasets) if datasets else "tac_mini_synthetic",
            renderer_manifest_uri=None,
            license_status="+".join(licenses) if licenses else "synthetic",
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
        L = min(len(mask), n_frames)
        active[:L] += mask[:L]
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

    for idx, src in enumerate(scene.sources):
        wav, _dur = pool.load(src.path, sr)
        wav = wav.astype(np.float32)
        if src.loop_to_scene:
            wav = _loop_to_duration(wav, scene.duration - src.onset, sr)
        wav = _apply_gain(wav, src.gain_db)
        wav = _apply_fade_by_kind(wav, src.kind, sr)  # source-type-dependent fade
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

    # Ducking is a deterministic, explicit scene condition. Apply it to the
    # stored background stems (not only to the final mixture), then recompute
    # activity so Ledger evidence matches the waveform the model hears.
    speech_active = np.zeros(n_clip, dtype=np.float32)
    if scene.conditions.ducking_enabled:
        for rs in rendered:
            if rs.placed.kind in ("speech", "vocal"):
                frame_size = int(0.05 * sr)
                for i in range(0, len(rs.stem), frame_size):
                    rms = np.sqrt(np.mean(rs.stem[i : i + frame_size] ** 2))
                    if rms > 0.01:
                        speech_active[i : i + frame_size] = 1.0
    if speech_active.any():
        from scipy.signal import lfilter

        smooth = lfilter(np.ones(10) / 10, [1.0], speech_active)
        speech_active = np.clip(smooth, 0, 1)
        duck_depth_db = float(scene.conditions.ducking_depth_db or 0.0)
        duck_factor = 10.0 ** (-duck_depth_db / 20.0)
        duck_gain = 1.0 - (1.0 - duck_factor) * speech_active
        for rs in rendered:
            if rs.placed.kind in ("music", "ambience"):
                rs.stem = (rs.stem * duck_gain[: len(rs.stem)]).astype(np.float32)
                rs.activity = compute_activity(
                    rs.stem,
                    sr,
                    activity_threshold=scene.supervision.activity_threshold,
                    resolution_sec=scene.supervision.resolution_s,
                    merge_threshold_s=scene.supervision.merge_threshold_s,
                    duration_sec=scene.duration,
                    is_continuous=_continuous(rs.placed.kind),
                )

    mixture = np.zeros(n_clip, dtype=np.float32)
    for rs in rendered:
        mixture += rs.stem
    dry_mixture = mixture.copy()

    # scene-level echo applied to the mixture
    if scene.conditions.echo_delay_ms is not None and scene.conditions.echo_atten_db is not None:
        mixture = _apply_echo(
            mixture, sr, scene.conditions.echo_delay_ms, scene.conditions.echo_atten_db
        )

    # Apply one master gain to mixture, dry mixture and every persisted stem.
    # Scaling only the final mixture would make saved stems louder than their
    # actual contribution and invalidate stem-level audibility/SNR audits.
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.99:
        master_gain = 0.99 / peak
        mixture = (mixture * master_gain).astype(np.float32)
        for rs in rendered:
            rs.stem = (rs.stem * master_gain).astype(np.float32)
            rs.activity = compute_activity(
                rs.stem,
                sr,
                activity_threshold=scene.supervision.activity_threshold,
                resolution_sec=scene.supervision.resolution_s,
                merge_threshold_s=scene.supervision.merge_threshold_s,
                duration_sec=scene.duration,
                is_continuous=_continuous(rs.placed.kind),
            )
        # Re-sum the scaled float32 stems so the persisted-stem invariant is
        # exact even after per-array rounding.
        dry_mixture = np.zeros(n_clip, dtype=np.float32)
        for rs in rendered:
            dry_mixture += rs.stem

    target_ledger = _build_ledger(scene, rendered)
    return RenderOutput(
        scene=scene,
        mixture=mixture,
        dry_mixture=dry_mixture,
        stems=rendered,
        target_ledger=target_ledger,
        sample_rate=sr,
    )


__all__ = ["RenderOutput", "RenderedSource", "render_scene"]
