from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..types import Event, Span, quantize_time
from .audio import load_audio, rms, save_audio, scale_to_snr
from .manifest import load_source_manifest


def build_exact_carc(
    background_manifest: str | Path,
    source_manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    backgrounds = load_source_manifest(background_manifest)
    sources = load_source_manifest(source_manifest)
    output = Path(output_dir).resolve()
    (output / "audio").mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 20260808))
    pair_count = int(config.get("pair_count", 5000))
    sample_rate = int(config.get("sample_rate", 24000))
    duration_sec = float(config.get("duration_sec", 20.0))
    total_samples = round(duration_sec * sample_rate)
    snr_range = config.get("snr_db", [-5.0, 5.0])
    audible_snr = float(config.get("audible_snr_db", -2.0))
    hidden_snr = float(config.get("hidden_snr_db", -10.0))
    rows: list[dict[str, Any]] = []

    for index in range(pair_count):
        rng = np.random.default_rng(seed + index)
        background_record = backgrounds[int(rng.integers(len(backgrounds)))]
        candidate_sources = [
            record
            for record in sources
            if record.duration_sec <= duration_sec - 0.1
            and Path(record.path).resolve() != Path(background_record.path).resolve()
            and not (
                record.sha256
                and background_record.sha256
                and record.sha256 == background_record.sha256
            )
        ]
        if not candidate_sources:
            raise ValueError("No intervention source shorter than configured duration")
        source_record = candidate_sources[int(rng.integers(len(candidate_sources)))]
        background = _fit_background(
            load_audio(background_record.path, sample_rate), total_samples, rng
        )
        source = load_audio(source_record.path, sample_rate)
        if rms(source) < 1e-5:
            raise ValueError(f"Intervention source is effectively silent: {source_record.path}")
        max_frame = (total_samples - len(source)) // round(0.1 * sample_rate)
        if max_frame < 1:
            raise ValueError("Intervention source leaves no room for a non-zero time shift")
        placement_frame = int(rng.integers(max_frame + 1))
        placement = placement_frame * round(0.1 * sample_rate)
        region = background[placement : placement + len(source)]
        snr_db = float(rng.uniform(snr_range[0], snr_range[1]))
        scaled_source, source_gain = scale_to_snr(source, region, snr_db)
        target_stem = np.zeros(total_samples, dtype=np.float32)
        target_stem[placement : placement + len(source)] = scaled_source
        shifted_frame = _different_frame(placement_frame, max_frame, rng)
        shifted_placement = shifted_frame * round(0.1 * sample_rate)
        shifted_target_stem = np.zeros(total_samples, dtype=np.float32)
        shifted_target_stem[shifted_placement : shifted_placement + len(source)] = scaled_source
        master_scale = _common_master_scale(background, target_stem, shifted_target_stem)
        background_scaled = background * master_scale
        target_scaled = target_stem * master_scale
        shifted_target_scaled = shifted_target_stem * master_scale
        mixed = background_scaled + target_scaled
        shifted_mixed = background_scaled + shifted_target_scaled

        pair_id = f"carc_{index:07d}"
        before_path = output / "audio" / f"{pair_id}_before.wav"
        after_path = output / "audio" / f"{pair_id}_after.wav"
        target_path = output / "audio" / f"{pair_id}_target.wav"
        shifted_after_path = output / "audio" / f"{pair_id}_shifted_after.wav"
        shifted_target_path = output / "audio" / f"{pair_id}_shifted_target.wav"
        save_audio(before_path, background_scaled, sample_rate)
        save_audio(after_path, mixed, sample_rate)
        save_audio(target_path, target_scaled, sample_rate)
        save_audio(shifted_after_path, shifted_mixed, sample_rate)
        save_audio(shifted_target_path, shifted_target_scaled, sample_rate)
        start = quantize_time(placement / sample_rate)
        end = quantize_time((placement + len(source)) / sample_rate)
        shifted_start = quantize_time(shifted_placement / sample_rate)
        shifted_end = quantize_time((shifted_placement + len(source)) / sample_rate)
        event_type = "sfx" if source_record.type == "ambience" else source_record.type
        delta_event = Event(
            id="E_delta",
            type=event_type,  # type: ignore[arg-type]
            track_id="T_delta",
            spans=[Span(start, min(end, duration_sec))],
            text=source_record.text,
            confidence=1.0,
            language=source_record.language,
            verbatim=source_record.verbatim if event_type in {"speech", "lys"} else None,
        )
        if snr_db >= audible_snr:
            audibility = "must_add"
        elif snr_db <= hidden_snr:
            audibility = "must_not_add"
        else:
            audibility = "ignore_positive_loss"
        rows.append(
            {
                "pair_id": pair_id,
                "before_path": str(before_path.relative_to(output)),
                "after_path": str(after_path.relative_to(output)),
                "target_path": str(target_path.relative_to(output)),
                "shifted_after_path": str(shifted_after_path.relative_to(output)),
                "shifted_target_path": str(shifted_target_path.relative_to(output)),
                "operations": ["add", "remove", "shift"],
                "background_id": background_record.id,
                "source_id": source_record.id,
                "placement_sample": placement,
                "shifted_placement_sample": shifted_placement,
                "shift_delta_sec": quantize_time((shifted_placement - placement) / sample_rate),
                "snr_db": snr_db,
                "source_gain": source_gain,
                "master_scale": master_scale,
                "measured_target_rms": rms(target_scaled),
                "audibility_target": audibility,
                "delta_event": delta_event_to_dict(delta_event),
                "shifted_delta_event": delta_event_to_dict(
                    Event(
                        id="E_delta",
                        type=event_type,  # type: ignore[arg-type]
                        track_id="T_delta",
                        spans=[Span(shifted_start, min(shifted_end, duration_sec))],
                        text=source_record.text,
                        confidence=1.0,
                        language=source_record.language,
                        verbatim=(
                            source_record.verbatim if event_type in {"speech", "lys"} else None
                        ),
                    )
                ),
                "seed": seed + index,
            }
        )

    manifest = output / "pairs.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"pairs": len(rows), "seed": seed, "manifest": str(manifest)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fit_background(audio: np.ndarray, total_samples: int, rng: np.random.Generator) -> np.ndarray:
    if not len(audio):
        raise ValueError("Background audio is empty")
    if len(audio) >= total_samples:
        start = int(rng.integers(len(audio) - total_samples + 1))
        return np.asarray(audio[start : start + total_samples], dtype=np.float32)
    repeats = int(np.ceil(total_samples / max(len(audio), 1)))
    return np.tile(audio, repeats)[:total_samples].astype(np.float32)


def _different_frame(placement_frame: int, max_frame: int, rng: np.random.Generator) -> int:
    if max_frame == 0:
        return placement_frame
    shifted = int(rng.integers(max_frame + 1))
    if shifted == placement_frame:
        shifted = (shifted + 1) % (max_frame + 1)
    return shifted


def _common_master_scale(
    background: np.ndarray,
    target: np.ndarray,
    shifted_target: np.ndarray,
    peak: float = 0.98,
) -> float:
    maximum = max(
        float(np.max(np.abs(background + target))),
        float(np.max(np.abs(background + shifted_target))),
    )
    return min(1.0, peak / maximum) if maximum > 0 else 1.0


def delta_event_to_dict(event: Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "type": event.type,
        "track_id": event.track_id,
        "spans": [{"start_sec": span.start_sec, "end_sec": span.end_sec} for span in event.spans],
        "text": event.text,
        "confidence": event.confidence,
        "language": event.language,
        "verbatim": event.verbatim,
    }
