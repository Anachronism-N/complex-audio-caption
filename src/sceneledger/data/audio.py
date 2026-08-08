from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

try:
    import soundfile as sf
except ImportError:  # WAV remains supported through scipy.
    sf = None


def load_audio(path: str | Path, sample_rate: int, mono: bool = True) -> np.ndarray:
    audio_path = Path(path)
    if sf is not None:
        audio, input_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    elif audio_path.suffix.lower() == ".wav":
        input_rate, audio = wavfile.read(str(audio_path))
        audio = _pcm_to_float(audio)
        if audio.ndim == 1:
            audio = audio[:, None]
    else:
        raise RuntimeError("Install sceneledger[audio] to read non-WAV audio")
    if mono:
        audio = audio.mean(axis=1)
    if input_rate != sample_rate:
        gcd = math.gcd(input_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, input_rate // gcd, axis=0).astype(
            np.float32
        )
    return np.asarray(audio, dtype=np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(audio, dtype=np.float32)
    pcm = np.round(np.clip(values, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(output), sample_rate, pcm)


def db_to_amplitude(gain_db: float) -> float:
    return float(10 ** (gain_db / 20.0))


def rms(audio: np.ndarray, epsilon: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64))) + epsilon))


def scale_to_snr(
    source: np.ndarray, background: np.ndarray, snr_db: float
) -> tuple[np.ndarray, float]:
    source_rms = rms(source)
    background_rms = rms(background)
    desired_source_rms = background_rms * db_to_amplitude(snr_db)
    gain = desired_source_rms / max(source_rms, 1e-8)
    return np.asarray(source * gain, dtype=np.float32), float(gain)


def frame_activity(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_sec: float = 0.1,
    threshold_db_below_peak: float = 35.0,
    absolute_floor_dbfs: float = -60.0,
) -> np.ndarray:
    frame_size = max(1, round(frame_sec * sample_rate))
    sample_count = len(audio)
    frame_count = math.ceil(sample_count / frame_size)
    padded = np.pad(
        np.asarray(audio, dtype=np.float32), (0, frame_count * frame_size - sample_count)
    )
    frames = padded.reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    dbfs = 20 * np.log10(frame_rms + 1e-12)
    peak_dbfs = float(np.max(dbfs)) if len(dbfs) else -120.0
    threshold = max(absolute_floor_dbfs, peak_dbfs - threshold_db_below_peak)
    return dbfs >= threshold


def activity_to_spans(
    activity: np.ndarray,
    *,
    frame_sec: float = 0.1,
    merge_gap_sec: float = 0.2,
    minimum_duration_sec: float = 0.1,
) -> list[tuple[float, float]]:
    active = np.asarray(activity, dtype=bool)
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    merge_frames = round(merge_gap_sec / frame_sec)
    raw: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous - 1 <= merge_frames:
            previous = index
            continue
        raw.append((start, previous + 1))
        start = previous = index
    raw.append((start, previous + 1))
    return [
        (round(start * frame_sec, 6), round(end * frame_sec, 6))
        for start, end in raw
        if (end - start) * frame_sec + 1e-9 >= minimum_duration_sec
    ]


def apply_rir(audio: np.ndarray, rir: np.ndarray, output_length: int) -> np.ndarray:
    if rir.ndim > 1:
        rir = rir.mean(axis=-1)
    normalized_rir = rir / max(float(np.sqrt(np.sum(np.square(rir)))), 1e-8)
    convolved = fftconvolve(audio, normalized_rir, mode="full")[:output_length]
    return np.asarray(convolved, dtype=np.float32)


def peak_normalize_group(
    arrays: list[np.ndarray], peak: float = 0.98
) -> tuple[list[np.ndarray], float]:
    if not arrays:
        return [], 1.0
    mixture = np.sum(np.stack(arrays), axis=0)
    maximum = float(np.max(np.abs(mixture))) if len(mixture) else 0.0
    scale = min(1.0, peak / maximum) if maximum > 0 else 1.0
    return [np.asarray(array * scale, dtype=np.float32) for array in arrays], scale


def _pcm_to_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return np.asarray(audio, dtype=np.float32)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    maximum = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
    return (audio.astype(np.float32) / maximum).astype(np.float32)
