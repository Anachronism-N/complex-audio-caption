#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sceneledger.integrations.moss import MossInferenceAdapter
from sceneledger.types import Ledger, read_jsonl

TRACK_TYPES = {"speech": 0, "vocal": 1, "music": 2, "sfx": 3, "ambience": 4, "residual": 5}
EVENT_TYPES = {"speech": 0, "lys": 1, "music": 2, "sfx": 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ledgers", required=True)
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    ledgers = {ledger.sample_id: ledger for ledger in read_jsonl(args.ledgers)}
    render_root = (
        Path(args.data_root).resolve()
        if args.data_root
        else Path(args.render_manifest).resolve().parent
    )
    rows = [
        json.loads(line)
        for line in Path(args.render_manifest).read_text(encoding="utf-8").splitlines()
        if line
    ]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = MossInferenceAdapter(
        args.upstream_root, args.model_path, device=args.device, enable_time_marker=True
    )
    index = []
    for row in rows:
        ledger = ledgers[row["sample_id"]]
        audio_path = Path(row["mixture_path"])
        if not audio_path.is_absolute():
            audio_path = render_root / audio_path
        raw_features = adapter.extract_encoder_features(audio_path)
        frame_count = round(ledger.duration_sec / ledger.time_resolution_sec)
        features = interpolate_features(raw_features, frame_count)
        targets = ledger_targets(ledger, frame_count)
        target_path = output / f"{ledger.sample_id}.npz"
        np.savez_compressed(target_path, features=features, **targets)
        index.append({"sample_id": ledger.sample_id, "path": target_path.name})
    (output / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index), encoding="utf-8"
    )


def interpolate_features(features: np.ndarray, frame_count: int) -> np.ndarray:
    if len(features) == frame_count:
        return features.astype(np.float32)
    source = np.linspace(0.0, 1.0, len(features))
    target = np.linspace(0.0, 1.0, frame_count)
    columns = [np.interp(target, source, features[:, index]) for index in range(features.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def ledger_targets(ledger: Ledger, frame_count: int) -> dict[str, np.ndarray]:
    track_by_id = {track.id: index for index, track in enumerate(ledger.tracks)}
    track_activity = _stack_masks(
        [spans_to_mask(track.spans, frame_count) for track in ledger.tracks], frame_count
    )
    event_activity = _stack_masks(
        [spans_to_mask(event.spans, frame_count) for event in ledger.events], frame_count
    )
    if any(event.track_id is None for event in ledger.events):
        raise ValueError(f"All training events require track_id: {ledger.sample_id}")
    return {
        "track_type": np.asarray(
            [TRACK_TYPES[track.kind] for track in ledger.tracks], dtype=np.int64
        ),
        "track_activity": track_activity,
        "event_type": np.asarray(
            [EVENT_TYPES[event.type] for event in ledger.events], dtype=np.int64
        ),
        "event_activity": event_activity,
        "event_track": np.asarray(
            [track_by_id[event.track_id] for event in ledger.events], dtype=np.int64
        ),
    }


def spans_to_mask(spans, frame_count: int) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=np.float32)
    for span in spans:
        start = max(0, round(span.start_sec * 10))
        end = min(frame_count, round(span.end_sec * 10))
        mask[start:end] = 1.0
    return mask


def _stack_masks(masks: list[np.ndarray], frame_count: int) -> np.ndarray:
    return np.stack(masks) if masks else np.zeros((0, frame_count), dtype=np.float32)


if __name__ == "__main__":
    main()
