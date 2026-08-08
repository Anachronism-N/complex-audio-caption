"""Short-time RMS activity detection for source stems.

Implements the TAC-style activity mask (``docs/06_tac_reproduction_protocol.md``
§3.2):

    r_i(t) = sqrt( mean_{n in [t, t+W)} a_i[n]^2 )
    m_i(t) = 1[ r_i(t) > delta_act * r_i^max ]

Internal hop is 10 ms; the binary mask is then aggregated to the target
output resolution (default 0.1 s). Gaps shorter than ``merge_threshold`` are
merged so that brief silences inside a continuous source do not split one
event into many. The raw RMS curve is retained for debugging and for the
``B2-complex`` per-type smoothing that comes later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sceneledger.data.schema import TIME_RESOLUTION_SEC


@dataclass
class ActivityResult:
    """Per-stem activity analysis."""

    rms_curve: np.ndarray  # [T_hop] short-time RMS at hop rate
    hop_sec: float
    activity_mask: np.ndarray  # [T_res] binary at target resolution
    resolution_sec: float
    spans: list[tuple[float, float]]  # contiguous (start, end) in seconds

    def active_duration(self) -> float:
        return round(sum(e - s for s, e in self.spans), 6)


def compute_rms(
    waveform: np.ndarray, sample_rate: int, hop_ms: float = 10.0, window_ms: float | None = None
) -> np.ndarray:
    """Short-time RMS with a frame length equal to hop (no overlap) by default.

    ``waveform`` may be [N] or [C, N]; mono-mixed internally.
    """
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    win = hop if window_ms is None else max(1, int(round(sample_rate * window_ms / 1000.0)))
    n = waveform.shape[0]
    n_frames = max(0, n // hop)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    trimmed = waveform[: n_frames * hop]
    frames = trimmed.reshape(n_frames, hop)
    # use the full window (may equal hop) -> reshape accordingly
    if win == hop:
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    else:
        # windowed RMS centered on each hop
        rms = np.empty(n_frames, dtype=np.float64)
        half = win // 2
        for i in range(n_frames):
            lo = max(0, i * hop - half)
            hi = min(n, i * hop - half + win)
            seg = waveform[lo:hi].astype(np.float64)
            rms[i] = np.sqrt(np.mean(seg ** 2)) if seg.size else 0.0
    return rms


def _aggregate_to_resolution(
    mask_hop: np.ndarray, hop_sec: float, resolution_sec: float
) -> np.ndarray:
    """Aggregate a hop-rate binary mask to the target resolution (OR within frame)."""
    if mask_hop.size == 0:
        return np.zeros(0, dtype=np.int8)
    ratio = max(1, int(round(resolution_sec / hop_sec)))
    n_res = mask_hop.size // ratio
    trimmed = mask_hop[: n_res * ratio].reshape(n_res, ratio)
    return (trimmed.any(axis=1)).astype(np.int8)


def _mask_to_spans(
    mask: np.ndarray, resolution_sec: float
) -> list[tuple[float, float]]:
    """Convert a binary mask at ``resolution_sec`` into contiguous (start, end) spans."""
    spans: list[tuple[float, float]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            spans.append((round(start * resolution_sec, 6), round(i * resolution_sec, 6)))
    if in_run:
        spans.append((round(start * resolution_sec, 6), round(len(mask) * resolution_sec, 6)))
    return spans


def _merge_close_spans(
    spans: list[tuple[float, float]], merge_threshold: float
) -> list[tuple[float, float]]:
    """Merge spans whose gap is strictly less than ``merge_threshold`` seconds."""
    if not spans:
        return []
    spans = sorted(spans)
    merged: list[tuple[float, float]] = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s - le < merge_threshold - 1e-9:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def compute_activity(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    activity_threshold: float = 0.05,
    hop_ms: float = 10.0,
    resolution_sec: float = TIME_RESOLUTION_SEC,
    merge_threshold_s: float = 0.25,
    duration_sec: float | None = None,
    is_continuous: bool = False,
) -> ActivityResult:
    """Compute the activity mask and contiguous spans for one stem.

    Parameters
    ----------
    activity_threshold:
        ``delta_act`` in the TAC formula; the mask is active where
        ``rms > threshold * rms.max()``.
    merge_threshold_s:
        gaps shorter than this are merged (TAC ``merge_threshold``).
    is_continuous:
        if True (music / ambience), the whole stem is considered one span and
        relative-peak thresholding is skipped (continuous sources would
        otherwise be fragmented by their internal dynamics). A very low floor
        is still applied to drop digital silence.
    duration_sec:
        explicit clip duration; if None, inferred from waveform length.
    """
    if duration_sec is None:
        duration_sec = waveform.shape[-1] / sample_rate
    hop_sec = hop_ms / 1000.0
    rms = compute_rms(waveform, sample_rate, hop_ms=hop_ms)

    if is_continuous:
        peak = float(rms.max()) if rms.size else 0.0
        floor = max(1e-6, 1e-4 * peak) if peak > 0 else 1e-6
        mask_hop = (rms > floor).astype(np.int8)
    else:
        peak = float(rms.max()) if rms.size else 0.0
        thr = activity_threshold * peak
        mask_hop = (rms > thr).astype(np.int8) if peak > 0 else np.zeros_like(rms, dtype=np.int8)

    mask_res = _aggregate_to_resolution(mask_hop, hop_sec, resolution_sec)
    # pad to full duration so spans reach clip end when appropriate
    n_res_target = int(round(duration_sec / resolution_sec))
    if mask_res.size < n_res_target:
        mask_res = np.concatenate(
            [mask_res, np.zeros(n_res_target - mask_res.size, dtype=np.int8)]
        )
    elif mask_res.size > n_res_target:
        mask_res = mask_res[:n_res_target]

    spans = _mask_to_spans(mask_res, resolution_sec)
    spans = _merge_close_spans(spans, merge_threshold_s)
    # drop zero-length spans (can arise when a single active frame sits at a boundary)
    spans = [(s, e) for s, e in spans if e > s + 1e-9]

    return ActivityResult(
        rms_curve=rms,
        hop_sec=hop_sec,
        activity_mask=mask_res,
        resolution_sec=resolution_sec,
        spans=spans,
    )


__all__ = ["ActivityResult", "compute_activity", "compute_rms"]
