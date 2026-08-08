"""Unit tests for RMS activity detection."""

from __future__ import annotations

import numpy as np

from sceneledger.data.activity import compute_activity, compute_rms


def _sr() -> int:
    return 24000


def test_rms_of_silence_is_zero():
    wav = np.zeros(_sr(), dtype=np.float32)
    rms = compute_rms(wav, _sr(), hop_ms=10.0)
    assert np.all(rms == 0.0)


def test_rms_of_full_scale_tone_matches_hand_calc():
    sr = _sr()
    t = np.arange(sr) / sr
    # 1 Hz sine, amplitude 1 -> RMS over a full period = 1/sqrt(2) ~ 0.7071
    wav = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
    rms = compute_rms(wav, sr, hop_ms=10.0)
    # every 10ms frame contains many cycles at 1Hz? No: 1Hz = 1 cycle/sec, 10ms = 0.01 cycle.
    # Use a higher frequency so each frame is representative.
    wav = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
    rms = compute_rms(wav, sr, hop_ms=10.0)
    assert np.allclose(rms, 1.0 / np.sqrt(2), atol=1e-3)


def test_activity_mask_isolated_burst():
    sr = _sr()
    dur = 2.0
    wav = np.zeros(int(sr * dur), dtype=np.float32)
    # active burst [0.5, 1.0]
    lo = int(0.5 * sr)
    hi = int(1.0 * sr)
    wav[lo:hi] = 0.5 * np.sin(2 * np.pi * 220 * np.arange(hi - lo) / sr)
    act = compute_activity(wav, sr, activity_threshold=0.1, resolution_sec=0.1, merge_threshold_s=0.2)
    # spans should cover roughly [0.5, 1.0]
    assert len(act.spans) == 1
    s, e = act.spans[0]
    assert abs(s - 0.5) <= 0.1
    assert abs(e - 1.0) <= 0.1


def test_merge_threshold_merges_close_spans():
    sr = _sr()
    dur = 3.0
    wav = np.zeros(int(sr * dur), dtype=np.float32)
    for lo, hi in [(0.2, 0.4), (0.5, 0.7)]:  # 0.1s gap between 0.4 and 0.5
        a = int(lo * sr)
        b = int(hi * sr)
        wav[a:b] = 0.5 * np.sin(2 * np.pi * 220 * np.arange(b - a) / sr)
    # gap = 0.1; merge_threshold 0.2 -> merged into one span
    act = compute_activity(wav, sr, activity_threshold=0.1, resolution_sec=0.1, merge_threshold_s=0.2)
    assert len(act.spans) == 1
    # merge_threshold 0.05 -> stays two spans
    act2 = compute_activity(wav, sr, activity_threshold=0.1, resolution_sec=0.1, merge_threshold_s=0.05)
    assert len(act2.spans) == 2


def test_continuous_source_not_fragmented():
    sr = _sr()
    dur = 4.0
    t = np.arange(int(sr * dur)) / sr
    # music-like with amplitude dips to ~0 but never silent
    wav = (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)) * np.sin(2 * np.pi * 220 * t)
    wav = wav.astype(np.float32)
    act = compute_activity(wav, sr, is_continuous=True, resolution_sec=0.1, merge_threshold_s=0.2)
    # continuous -> single span covering the whole clip
    assert len(act.spans) == 1
    assert act.spans[0][0] <= 0.1
    assert act.spans[0][1] >= dur - 0.2
