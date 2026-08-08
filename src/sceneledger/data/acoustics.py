from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from .audio import db_to_amplitude, rms


def apply_echo(
    audio: np.ndarray,
    sample_rate: int,
    *,
    delay_sec: float,
    decay: float,
    repeats: int = 2,
) -> np.ndarray:
    """Apply a causal finite echo while preserving the input length."""
    values = np.asarray(audio, dtype=np.float32)
    delay_samples = max(1, round(delay_sec * sample_rate))
    output = values.copy()
    for repeat in range(1, repeats + 1):
        offset = repeat * delay_samples
        if offset >= len(values):
            break
        output[offset:] += values[:-offset] * float(decay**repeat)
    return output


def apply_scene_degradation(
    audio: np.ndarray,
    sample_rate: int,
    config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic TAC++ recording-channel corruptions.

    The caller can retain ``output - input`` as a residual stem. This keeps the
    renderer exactly reconstructable even for nonlinear compression and clipping.
    """
    values = np.asarray(audio, dtype=np.float32).copy()
    settings = config or {}
    metadata: dict[str, Any] = {"operations": []}

    if _sample_enabled(settings, "noise_probability", rng):
        color = str(_choice(settings.get("noise_color", ["white", "pink", "brown"]), rng))
        snr_db = _sample_range(settings.get("snr_db", [0.0, 20.0]), rng)
        noise = _colored_noise(len(values), color, rng)
        signal_rms = rms(values)
        noise_gain = signal_rms / max(rms(noise) * db_to_amplitude(snr_db), 1e-8)
        values += noise * noise_gain
        metadata.update(noise_color=color, snr_db=snr_db, noise_gain=float(noise_gain))
        metadata["operations"].append("noise")

    if _sample_enabled(settings, "device_filter_probability", rng):
        nyquist = sample_rate / 2.0
        low_hz = min(_sample_range(settings.get("low_cut_hz", [40.0, 300.0]), rng), nyquist * 0.8)
        high_hz = min(
            _sample_range(settings.get("high_cut_hz", [3000.0, 11000.0]), rng),
            nyquist * 0.98,
        )
        if high_hz > low_hz * 1.1:
            sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
            values = sosfilt(sos, values).astype(np.float32)
            metadata.update(low_cut_hz=low_hz, high_cut_hz=high_hz)
            metadata["operations"].append("device_filter")

    if _sample_enabled(settings, "compression_probability", rng):
        threshold_dbfs = _sample_range(
            settings.get("compression_threshold_dbfs", [-24.0, -8.0]), rng
        )
        ratio = max(1.0, _sample_range(settings.get("compression_ratio", [2.0, 8.0]), rng))
        threshold = db_to_amplitude(threshold_dbfs)
        magnitude = np.abs(values)
        compressed = np.where(
            magnitude <= threshold,
            magnitude,
            threshold * np.power(magnitude / max(threshold, 1e-8), 1.0 / ratio),
        )
        values = np.sign(values) * compressed
        metadata.update(compression_threshold_dbfs=threshold_dbfs, compression_ratio=ratio)
        metadata["operations"].append("compression")

    if _sample_enabled(settings, "clipping_probability", rng):
        threshold = _sample_range(settings.get("clipping_threshold", [0.35, 0.9]), rng)
        threshold = float(np.clip(threshold, 0.05, 1.0))
        values = np.clip(values, -threshold, threshold)
        metadata["clipping_threshold"] = threshold
        metadata["operations"].append("clipping")

    metadata["operations"] = list(metadata["operations"])
    return np.asarray(values, dtype=np.float32), metadata


def _colored_noise(length: int, color: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(length).astype(np.float64)
    if color == "white":
        return white.astype(np.float32)
    if color not in {"pink", "brown"}:
        raise ValueError(f"Unsupported noise color: {color}")
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(length)
    exponent = 0.5 if color == "pink" else 1.0
    scale = np.ones_like(frequencies)
    nonzero = frequencies > 0
    scale[nonzero] = 1.0 / np.power(frequencies[nonzero], exponent)
    scale[~nonzero] = 0.0
    colored = np.fft.irfft(spectrum * scale, n=length)
    colored -= np.mean(colored)
    colored /= max(float(np.std(colored)), 1e-8)
    return colored.astype(np.float32)


def _sample_enabled(config: dict[str, Any], key: str, rng: np.random.Generator) -> bool:
    return rng.random() < float(config.get(key, 0.0))


def _sample_range(value: float | list[float], rng: np.random.Generator) -> float:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected [minimum, maximum], got {value}")
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def _choice(value: str | list[str], rng: np.random.Generator) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("Choice list must not be empty")
        return value[int(rng.integers(len(value)))]
    return value
