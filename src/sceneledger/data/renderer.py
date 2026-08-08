from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..serialization import serialize_tagged_caption
from ..types import Event, Evidence, Ledger, Span, Track, quantize_time, write_jsonl
from .acoustics import apply_echo, apply_scene_degradation
from .audio import (
    activity_to_spans,
    apply_rir,
    db_to_amplitude,
    frame_activity,
    load_audio,
    peak_normalize_group,
    save_audio,
)
from .manifest import SourceRecord, load_source_manifest


@dataclass
class RenderedSource:
    source_id: str
    source_path: str
    type: str
    track_id: str
    event_id: str
    placement_sample: int
    gain_db: float
    rir_path: str | None
    echo_delay_sec: float | None
    echo_decay: float | None
    stem_path: str | None


def render_tac_dataset(
    source_manifest: str | Path, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    records = load_source_manifest(source_manifest)
    output = Path(output_dir).resolve()
    (output / "audio").mkdir(parents=True, exist_ok=True)
    if bool(config.get("save_stems", True)):
        (output / "stems").mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 20260808))
    sample_count = int(config.get("sample_count", 500))
    ledgers: list[Ledger] = []
    render_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index in range(sample_count):
        rng = np.random.default_rng(seed + index)
        try:
            ledger, row = _render_one(index, rng, records, config, output)
            ledgers.append(ledger)
            render_rows.append(row)
        except Exception as exc:
            failures.append(f"sample {index}: {type(exc).__name__}: {exc}")
            if len(failures) > max(10, sample_count // 10):
                raise RuntimeError(
                    f"Too many renderer failures. Last error: {failures[-1]}"
                ) from exc

    write_jsonl(output / "ledgers.jsonl", ledgers)
    with (output / "render_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in render_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "requested": sample_count,
        "rendered": len(ledgers),
        "failed": len(failures),
        "failures": failures[:50],
        "seed": seed,
        "config": str(Path(config_path).resolve()),
        "source_manifest": str(Path(source_manifest).resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _render_one(
    index: int,
    rng: np.random.Generator,
    records: list[SourceRecord],
    config: dict[str, Any],
    output: Path,
) -> tuple[Ledger, dict[str, Any]]:
    sample_rate = int(config.get("sample_rate", 24000))
    duration_range = config.get("duration_sec", [10.0, 20.0])
    duration_sec = quantize_time(float(rng.uniform(duration_range[0], duration_range[1])))
    total_samples = round(duration_sec * sample_rate)
    templates = config["templates"]
    weights = np.asarray([float(item.get("weight", 1.0)) for item in templates], dtype=float)
    template = templates[int(rng.choice(len(templates), p=weights / weights.sum()))]
    source_types = list(template["sources"])

    selected: list[SourceRecord] = []
    for source_type in source_types:
        candidates = [
            record
            for record in records
            if record.type == source_type
            and record not in selected
            and record.duration_sec <= duration_sec + 1e-6
        ]
        if not candidates:
            raise ValueError(f"No source for type={source_type} shorter than {duration_sec}s")
        selected.append(candidates[int(rng.integers(len(candidates)))])

    rir_records: list[SourceRecord] = []
    rir_manifest = config.get("rir_manifest")
    if rir_manifest:
        rir_records = load_source_manifest(rir_manifest)

    stems: list[np.ndarray] = []
    ledger_tracks: list[Track] = []
    ledger_events: list[Event] = []
    rendered_sources: list[RenderedSource] = []
    sample_id = f"tac_{index:07d}"
    gain_range = config.get("gain_db", [-8.0, 2.0])
    frame_sec = float(config.get("time_resolution_sec", 0.1))
    merge_gap = float(config.get("merge_gap_sec", 0.2))
    echo_probability = float(config.get("echo_probability", 0.0))

    for source_index, record in enumerate(selected, 1):
        audio = load_audio(record.path, sample_rate)
        if not len(audio):
            raise ValueError(f"Empty audio: {record.path}")
        if len(audio) > total_samples:
            raise ValueError(f"Source longer than scene: {record.path}")
        max_placement_frames = (total_samples - len(audio)) // round(frame_sec * sample_rate)
        placement_frame = int(rng.integers(max_placement_frames + 1))
        placement_sample = placement_frame * round(frame_sec * sample_rate)
        gain_db = float(rng.uniform(gain_range[0], gain_range[1]))
        transformed = audio * db_to_amplitude(gain_db)

        rir_path: str | None = None
        if rir_records and rng.random() < float(config.get("rir_probability", 0.0)):
            rir_record = rir_records[int(rng.integers(len(rir_records)))]
            rir = load_audio(rir_record.path, sample_rate)
            transformed = apply_rir(transformed, rir, len(transformed))
            rir_path = rir_record.path

        echo_delay_sec: float | None = None
        echo_decay: float | None = None
        if rng.random() < echo_probability:
            echo_delay_sec = _sample_range(config.get("echo_delay_sec", [0.08, 0.35]), rng)
            echo_decay = _sample_range(config.get("echo_decay", [0.2, 0.6]), rng)
            transformed = apply_echo(
                transformed,
                sample_rate,
                delay_sec=echo_delay_sec,
                decay=echo_decay,
                repeats=int(config.get("echo_repeats", 2)),
            )

        stem = np.zeros(total_samples, dtype=np.float32)
        end_sample = min(total_samples, placement_sample + len(transformed))
        stem[placement_sample:end_sample] = transformed[: end_sample - placement_sample]
        stems.append(stem)

        span_pairs = _placed_activity_spans(
            audio,
            placement_frame,
            duration_sec,
            sample_rate,
            frame_sec,
            merge_gap,
            float(config.get("activity_threshold_db", 35.0)),
        )
        if not span_pairs:
            start = quantize_time(placement_sample / sample_rate)
            end = quantize_time(min(duration_sec, (placement_sample + len(audio)) / sample_rate))
            if end <= start:
                end = quantize_time(start + frame_sec)
            span_pairs = [(start, end)]
        spans = [Span(start, min(end, duration_sec)) for start, end in span_pairs]
        evidence_pairs = _placed_activity_spans(
            transformed,
            placement_frame,
            duration_sec,
            sample_rate,
            frame_sec,
            merge_gap,
            float(config.get("evidence_threshold_db", 45.0)),
        )
        evidence_spans = [Span(start, min(end, duration_sec)) for start, end in evidence_pairs]
        track_id = f"T{source_index}"
        event_id = f"E{source_index}"
        track_kind = "vocal" if record.type == "lys" else record.type
        event_type = "sfx" if record.type == "ambience" else record.type
        ledger_tracks.append(
            Track(
                id=track_id,
                kind=track_kind,  # type: ignore[arg-type]
                spans=spans,
                confidence=1.0,
                identity=record.group_id if record.type in {"speech", "lys"} else None,
                evidence=Evidence(
                    method="exact_renderer_stem", spans=evidence_spans, audio_support=1.0
                ),
                attributes={"source_record_id": record.id},
            )
        )
        ledger_events.append(
            Event(
                id=event_id,
                type=event_type,  # type: ignore[arg-type]
                track_id=track_id,
                spans=spans,
                text=record.text,
                confidence=1.0,
                language=record.language,
                verbatim=record.verbatim if event_type in {"speech", "lys"} else None,
                evidence=Evidence(
                    method="exact_renderer_stem", spans=evidence_spans, audio_support=1.0
                ),
            )
        )
        rendered_sources.append(
            RenderedSource(
                source_id=record.id,
                source_path=record.path,
                type=record.type,
                track_id=track_id,
                event_id=event_id,
                placement_sample=placement_sample,
                gain_db=gain_db,
                rir_path=rir_path,
                echo_delay_sec=echo_delay_sec,
                echo_decay=echo_decay,
                stem_path=None,
            )
        )

    stems, master_scale = peak_normalize_group(stems)
    clean_mixture = np.sum(np.stack(stems), axis=0)
    degraded_mixture, degradation = apply_scene_degradation(
        clean_mixture, sample_rate, config.get("scene_degradation"), rng
    )
    residual = degraded_mixture - clean_mixture
    component_peak = max(
        [float(np.max(np.abs(degraded_mixture))), float(np.max(np.abs(residual)))]
        + [float(np.max(np.abs(stem))) for stem in stems]
    )
    degradation_scale = min(1.0, 0.98 / component_peak) if component_peak > 0 else 1.0
    stems = [np.asarray(stem * degradation_scale, dtype=np.float32) for stem in stems]
    residual = np.asarray(residual * degradation_scale, dtype=np.float32)
    mixture = np.asarray(degraded_mixture * degradation_scale, dtype=np.float32)
    mixture_path = output / "audio" / f"{sample_id}.wav"
    save_audio(mixture_path, mixture, sample_rate)
    if bool(config.get("save_stems", True)):
        for stem_index, stem in enumerate(stems, 1):
            stem_path = output / "stems" / sample_id / f"T{stem_index}.wav"
            save_audio(stem_path, stem, sample_rate)
            rendered_sources[stem_index - 1].stem_path = str(stem_path.relative_to(output))
            waveform_uri = str(stem_path.relative_to(output))
            ledger_tracks[stem_index - 1].evidence.waveform_uri = waveform_uri  # type: ignore[union-attr]
            ledger_events[stem_index - 1].evidence.waveform_uri = waveform_uri  # type: ignore[union-attr]
        if degradation["operations"]:
            residual_path = output / "stems" / sample_id / "T_residual.wav"
            save_audio(residual_path, residual, sample_rate)
            residual_uri: str | None = str(residual_path.relative_to(output))
        else:
            residual_uri = None
    else:
        residual_uri = None

    conditions = {
        "domain": "tac_pp" if degradation["operations"] or echo_probability else "tac_mini",
        "overlap_ratio": _overlap_ratio(stems),
        "snr_db": degradation.get("snr_db"),
        "noise_color": degradation.get("noise_color"),
        "device_filter": "device_filter" in degradation["operations"],
        "compression": "compression" in degradation["operations"],
        "clipping": "clipping" in degradation["operations"],
        "echo": any(item.echo_delay_sec is not None for item in rendered_sources),
    }

    ledger = Ledger(
        sample_id=sample_id,
        duration_sec=duration_sec,
        tracks=ledger_tracks,
        events=ledger_events,
        conditions=conditions,
        provenance={
            "label_level": "B",
            "source_dataset": "source_manifest",
            "renderer_manifest_uri": "render_manifest.jsonl",
        },
    )
    ledger.validate()
    row = {
        "sample_id": sample_id,
        "mixture_path": str(mixture_path.relative_to(output)),
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "seed": int(config.get("seed", 20260808)) + index,
        "template": template.get("name", "+".join(source_types)),
        "master_scale": master_scale,
        "degradation_scale": degradation_scale,
        "acoustic_degradation": degradation,
        "residual_stem_path": residual_uri,
        "sources": [asdict(item) for item in rendered_sources],
        "caption": serialize_tagged_caption(ledger),
    }
    return ledger, row


def _overlap_ratio(stems: list[np.ndarray]) -> float:
    if not stems:
        return 0.0
    active = np.stack([np.abs(stem) > 1e-5 for stem in stems])
    return float(np.mean(np.sum(active, axis=0) >= 2))


def _placed_activity_spans(
    audio: np.ndarray,
    placement_frame: int,
    duration_sec: float,
    sample_rate: int,
    frame_sec: float,
    merge_gap: float,
    threshold_db: float,
) -> list[tuple[float, float]]:
    activity = frame_activity(
        audio,
        sample_rate,
        frame_sec=frame_sec,
        threshold_db_below_peak=threshold_db,
    )
    full_activity = np.concatenate([np.zeros(placement_frame, dtype=bool), activity])
    frame_count = round(duration_sec / frame_sec)
    full_activity = np.pad(full_activity, (0, max(0, frame_count - len(full_activity))))[
        :frame_count
    ]
    return activity_to_spans(
        full_activity,
        frame_sec=frame_sec,
        merge_gap_sec=merge_gap,
        minimum_duration_sec=frame_sec,
    )


def _sample_range(value: float | list[float], rng: np.random.Generator) -> float:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected [minimum, maximum], got {value}")
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)
