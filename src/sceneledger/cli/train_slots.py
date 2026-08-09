"""Cache MOSS features, train the S1a event-slot probe, and evaluate it.

S1a predicts an unordered set of event types and 100 ms activity masks. It is
an event-localization probe, not yet the final text/track SceneLedger model.
The command is intentionally fail-closed around feature-cache identity and
source leakage so server results remain auditable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sceneledger.eval.selection import (
    coverage_aware_metrics,
    select_eventness_threshold,
)

CACHE_VERSION = "s1-moss-features-v2"


def _load_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S1 config must contain a YAML mapping")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_file_name(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()  # noqa: S324
    return f"{digest}.pt"


def _manifest_paths(cfg: dict) -> list[Path]:
    data_cfg = cfg["data"]
    train_path = data_cfg.get("train_manifest_path")
    val_path = data_cfg.get("val_manifest_path")
    if bool(train_path) != bool(val_path):
        raise ValueError(
            "data.train_manifest_path and data.val_manifest_path must be set together"
        )
    paths = [Path(train_path), Path(val_path)] if train_path else []
    if not paths:
        manifest_path = data_cfg.get("manifest_path")
        if not manifest_path:
            raise ValueError("set either an unsplit manifest or explicit train/val manifests")
        paths = [Path(manifest_path)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"S1 manifest file(s) missing: {missing}")
    return [path.resolve() for path in paths]


def _model_identity(model_path: str | Path) -> dict:
    root = Path(model_path).expanduser().resolve()
    identity: dict[str, object] = {"path": str(root)}
    hashes: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        candidate = root / name
        if candidate.is_file():
            hashes[name] = _sha256_file(candidate)
    identity["metadata_sha256"] = hashes
    return identity


def _cache_signature(cfg: dict, sample_ids: list[str]) -> dict:
    paths = _manifest_paths(cfg)
    payload = {
        "cache_version": CACHE_VERSION,
        "manifests": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in paths
        ],
        "model": _model_identity(cfg["model"]["path"]),
        "model_dtype": cfg["model"].get("dtype", "bfloat16"),
        "max_audio_seconds": float(cfg["data"].get("max_audio_seconds", 30.0)),
        "feature_storage_dtype": "float16",
        "sample_count": len(sample_ids),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(sample_ids)).encode("utf-8")
        ).hexdigest(),
    }
    return {**payload, "signature_sha256": _json_hash(payload)}


def _load_audio(path: str | Path, sample_rate: int, max_seconds: float) -> np.ndarray:
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    wav, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if source_rate != sample_rate:
        factor = gcd(int(source_rate), int(sample_rate))
        wav = resample_poly(
            wav.astype(np.float64),
            int(sample_rate) // factor,
            int(source_rate) // factor,
        ).astype(np.float32)
    return wav[: int(max_seconds * sample_rate)]


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {name}") from exc


def _all_entries(cfg: dict):
    from sceneledger.data.manifests import read_manifest

    entries = [entry for path in _manifest_paths(cfg) for entry in read_manifest(path)]
    sample_ids = [str(entry.scene["scene_id"]) for entry in entries]
    duplicates = sorted(
        {sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate sample IDs across S1 manifests: {duplicates[:10]}")
    return entries


def cache_features(cfg: dict, force: bool = False) -> Path:
    """Extract MOSS audio embeddings with a content-addressed cache contract."""
    cache_dir = Path(cfg["data"].get("feature_cache", "/tmp/s1_features")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "cache_manifest.json"
    entries = _all_entries(cfg)
    sample_ids = [str(entry.scene["scene_id"]) for entry in entries]
    signature = _cache_signature(cfg, sample_ids)
    expected_files = [_cache_file_name(sample_id) for sample_id in sample_ids]

    existing = None
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = all((cache_dir / name).is_file() for name in expected_files)
    if (
        not force
        and existing is not None
        and existing.get("signature_sha256") == signature["signature_sha256"]
        and complete
    ):
        print(f"[s1] verified feature cache at {cache_dir}", file=sys.stderr)
        return cache_dir
    if not force and existing is not None:
        raise RuntimeError(
            "feature cache identity/completeness differs from this run; "
            "inspect cache_manifest.json and rerun with --force-cache"
        )

    from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig

    print("[s1] loading MOSS for feature extraction", file=sys.stderr, flush=True)
    adapter = MossAdapter(
        MossAdapterConfig(
            model_path=cfg["model"]["path"],
            device=cfg["model"]["device"],
            dtype=cfg["model"].get("dtype", "bfloat16"),
        )
    )
    adapter._load()
    model = adapter._model
    processor = adapter._processor
    model.eval()

    audio_root = Path(cfg["data"]["audio_base_dir"]).resolve()
    sample_rate = int(processor.config.mel_sr)
    max_seconds = float(cfg["data"].get("max_audio_seconds", 30.0))
    input_dtype = _torch_dtype(cfg["model"].get("dtype", "bfloat16"))
    started = time.time()
    for index, entry in enumerate(entries, start=1):
        sample_id = str(entry.scene["scene_id"])
        cache_path = cache_dir / _cache_file_name(sample_id)
        audio_path = audio_root / entry.mixture_path
        if not audio_path.is_file():
            raise FileNotFoundError(f"mixture audio missing: {audio_path}")
        wav = _load_audio(audio_path, sample_rate, max_seconds)
        inputs = processor(text="x", audios=[wav], return_tensors="pt")
        audio_data = inputs["audio_data"].to(cfg["model"]["device"]).to(input_dtype)
        audio_lengths = inputs["audio_data_seqlens"].to(cfg["model"]["device"])
        with torch.inference_mode():
            audio_embeddings, _ = model.get_audio_features(audio_data, audio_lengths)
            audio_embeddings = model.audio_adapter(audio_embeddings)
        torch.save(
            {
                "features": audio_embeddings[0].to(dtype=torch.float16).cpu(),
                "sample_id": sample_id,
                "duration_sec": float(entry.scene["duration"]),
            },
            cache_path,
        )
        if index % 100 == 0 or index == len(entries):
            elapsed = time.time() - started
            print(
                f"[s1] cached {index}/{len(entries)} ({elapsed:.0f}s)",
                file=sys.stderr,
                flush=True,
            )

    metadata = {
        **signature,
        "created_unix": time.time(),
        "files": dict(zip(sample_ids, expected_files, strict=True)),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return cache_dir


def _set_reproducible(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _split_entries(cfg: dict):
    from sceneledger.data.datamodule import group_split, source_leakage
    from sceneledger.data.manifests import audit_manifest_structure, read_manifest

    data_cfg = cfg["data"]
    train_path = data_cfg.get("train_manifest_path")
    val_path = data_cfg.get("val_manifest_path")
    if train_path and val_path:
        base_train_entries = read_manifest(train_path)
        val_entries = read_manifest(val_path)
    else:
        entries = read_manifest(data_cfg["manifest_path"])
        base_train_entries, val_entries = group_split(
            entries,
            val_fraction=float(cfg["train"].get("val_fraction", 0.1)),
            group_key=data_cfg.get("group_key", "source_id"),
            seed=int(cfg["train"]["seed"]),
        )

    calibration_fraction = float(cfg["train"].get("calibration_fraction", 0.0))
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "train.calibration_fraction must be between 0 and 1; "
            "the reported validation fold cannot select checkpoints or thresholds"
        )
    train_entries, calibration_entries = group_split(
        base_train_entries,
        val_fraction=calibration_fraction,
        group_key=data_cfg.get("group_key", "source_id"),
        seed=int(cfg["train"]["seed"]) + 1,
    )

    for name, entries in (
        ("train", train_entries),
        ("calibration", calibration_entries),
        ("val", val_entries),
    ):
        audit = audit_manifest_structure(entries)
        if not audit.ok():
            raise ValueError(f"{name} manifest failed audit: {audit.errors[:5]}")
    if not train_entries or not calibration_entries or not val_entries:
        raise ValueError("S1 requires non-empty train, calibration, and validation folds")
    fold_pairs = (
        ("train/calibration", train_entries, calibration_entries),
        ("train/val", train_entries, val_entries),
        ("calibration/val", calibration_entries, val_entries),
    )
    for label, left, right in fold_pairs:
        leakage = source_leakage(left, right)
        if leakage:
            raise ValueError(
                f"source leakage between {label} folds: {sorted(leakage)[:10]}"
            )
    return train_entries, calibration_entries, val_entries


def _event_targets(entry) -> list[dict]:
    from sceneledger.data.schema import Ledger

    ledger = Ledger.model_validate(entry.target_ledger)
    return [
        {
            "type": event.type,
            "spans": [span.model_dump(mode="json") for span in event.spans],
        }
        for event in ledger.events
    ]


def _load_fold(entries, cache_dir: Path) -> list[dict]:
    dataset = []
    for entry in entries:
        sample_id = str(entry.scene["scene_id"])
        cache_path = cache_dir / _cache_file_name(sample_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"cached feature missing for {sample_id}: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("sample_id") != sample_id:
            raise ValueError(f"cache sample mismatch for {sample_id}")
        dataset.append(
            {
                "features": payload["features"].float(),
                "events": _event_targets(entry),
                "sample_id": sample_id,
                "duration": float(entry.scene["duration"]),
                "target_ledger": entry.target_ledger,
            }
        )
    return dataset


def _loss_kwargs(cfg: dict) -> dict[str, float]:
    loss_cfg = cfg.get("loss", {})
    return {
        "eventness_weight": float(loss_cfg.get("eventness_weight", 1.0)),
        "type_weight": float(loss_cfg.get("type_weight", 1.0)),
        "activity_weight": float(loss_cfg.get("activity_weight", 2.0)),
        "boundary_weight": float(loss_cfg.get("boundary_weight", 1.0)),
        "activity_cost_weight": float(loss_cfg.get("activity_cost_weight", 2.0)),
        "boundary_cost_weight": float(loss_cfg.get("boundary_cost_weight", 1.0)),
        "positive_weight_scale": float(loss_cfg.get("positive_weight_scale", 1.0)),
        "max_positive_weight": float(loss_cfg.get("max_positive_weight", 20.0)),
    }


def _sample_loss(model, sample: dict, device: str, loss_kwargs: dict):
    from sceneledger.losses.set_prediction import _events_to_targets, set_prediction_loss

    features = sample["features"].unsqueeze(0).to(device)
    outputs = model(features)
    targets = [
        _events_to_targets(sample["events"], int(outputs["n_frames"]), model.n_slots)
    ]
    return set_prediction_loss(outputs, targets, **loss_kwargs)


def _validation_loss(model, dataset: list[dict], device: str, loss_kwargs: dict) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for sample in dataset:
            losses.append(float(_sample_loss(model, sample, device, loss_kwargs)["loss"]))
    model.train()
    return sum(losses) / len(losses)


def _warmup_cosine_multiplier(
    step: int, *, warmup_steps: int, total_steps: int
) -> float:
    """Linear warmup followed by cosine decay, expressed as an LR multiplier."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(0, min(warmup_steps, total_steps))
    if warmup_steps and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    step: int,
    best_calibration_loss: float,
    rng: random.Random,
    config_hash: str,
) -> None:
    torch.save(
        {
            "checkpoint_version": 3,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_calibration_loss": best_calibration_loss,
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "config_sha256": config_hash,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    rng: random.Random | None = None,
    expected_config_hash: str | None = None,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_version = payload.get("checkpoint_version")
    if checkpoint_version != 3:
        raise ValueError(
            f"checkpoint version {checkpoint_version!r} is incompatible with the "
            "dual-head S1 runner (expected 3); retrain with the current protocol"
        )
    checkpoint_hash = payload.get("config_sha256")
    if expected_config_hash and checkpoint_hash != expected_config_hash:
        raise ValueError(
            f"checkpoint config hash {checkpoint_hash} != current {expected_config_hash}"
        )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        parameter_device = next(model.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(parameter_device)
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if rng is not None and payload.get("python_rng_state") is not None:
        rng.setstate(payload["python_rng_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return payload


def _build_model(cfg: dict, feature_dim: int):
    from sceneledger.models.event_slots import EventSlotDecoder

    model_cfg = cfg["model"]
    return EventSlotDecoder(
        feature_dim=feature_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        n_slots=int(model_cfg.get("n_slots", 24)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        max_duration_sec=float(cfg["data"].get("max_audio_seconds", 30.0)),
        use_temporal_embedding=bool(model_cfg.get("use_temporal_embedding", True)),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clip_event_spans(events: list[dict], duration: float) -> list[dict]:
    clipped_events = []
    for event in events:
        spans = []
        for span in event["spans"]:
            start = max(0.0, float(span["start_sec"]))
            end = min(duration, float(span["end_sec"]))
            if end > start:
                spans.append({"start_sec": start, "end_sec": end})
        if spans:
            event["spans"] = spans
            clipped_events.append(event)
    return clipped_events


def _prediction_rows(
    model,
    dataset: list[dict],
    *,
    device: str,
    decode_mode: str,
    eventness_threshold: float,
    activity_threshold: float,
) -> list[dict]:
    predictions = []
    with torch.inference_mode():
        for sample in dataset:
            features = sample["features"].unsqueeze(0).to(device)
            events = model.predict(
                features,
                eventness_threshold=eventness_threshold,
                activity_threshold=activity_threshold,
                decode_mode=decode_mode,
            )[0]
            events = _clip_event_spans(events, sample["duration"])
            events.sort(key=lambda event: event["spans"][0]["start_sec"])
            prediction_events = [
                {
                    "id": f"E{index:03d}",
                    "type": event["type"],
                    "track_id": None,
                    "spans": event["spans"],
                    "text": event["type"],
                    "confidence": event["confidence"],
                }
                for index, event in enumerate(events, start=1)
            ]
            predictions.append(
                {
                    "schema_version": "0.2.0",
                    "sample_id": sample["sample_id"],
                    "duration_sec": sample["duration"],
                    "time_resolution_sec": 0.1,
                    "tracks": [],
                    "events": prediction_events,
                }
            )
    return predictions


def evaluate_slots(
    cfg: dict,
    *,
    model,
    val_data: list[dict],
    out_dir: Path,
    checkpoint_path: Path,
    split_name: str = "val",
    eventness_threshold_override: float | None = None,
) -> dict:
    """Evaluate event-only predictions and persist machine-readable evidence."""
    from sceneledger.eval.metrics import evaluate_corpus

    device = cfg["model"]["device"]
    eval_cfg = cfg.get("evaluation", {})
    eventness_threshold = (
        float(eventness_threshold_override)
        if eventness_threshold_override is not None
        else float(eval_cfg.get("eventness_threshold", 0.5))
    )
    activity_threshold = float(eval_cfg.get("activity_threshold", 0.5))
    tiou_threshold = float(eval_cfg.get("tiou_threshold", 0.3))
    primary_decode_mode = str(eval_cfg.get("primary_decode_mode", "hybrid"))
    decode_modes = list(eval_cfg.get("decode_modes", [primary_decode_mode]))
    if primary_decode_mode not in decode_modes:
        decode_modes.insert(0, primary_decode_mode)
    decode_modes = list(dict.fromkeys(decode_modes))
    invalid_modes = set(decode_modes) - {"activity", "boundary", "hybrid"}
    if invalid_modes:
        raise ValueError(f"unsupported evaluation decode modes: {sorted(invalid_modes)}")

    model.eval()
    references = [sample["target_ledger"] for sample in val_data]
    reference_path = out_dir / f"{split_name}_references.jsonl"
    _write_jsonl(reference_path, references)
    metrics_by_decode = {}
    full_metrics_by_decode = {}
    for decode_mode in decode_modes:
        predictions = _prediction_rows(
            model,
            val_data,
            device=device,
            decode_mode=decode_mode,
            eventness_threshold=eventness_threshold,
            activity_threshold=activity_threshold,
        )
        prediction_path = out_dir / f"{split_name}_predictions_{decode_mode}.jsonl"
        metric_path = out_dir / f"{split_name}_metrics_{decode_mode}.json"
        _write_jsonl(prediction_path, predictions)
        corpus = evaluate_corpus(
            prediction_path,
            reference_path,
            tiou_threshold=tiou_threshold,
            min_text_similarity=0.0,
        )
        metrics = corpus.to_dict()
        metric_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        compact = {
            "macro_event_f1": metrics["macro_event_f1"],
            "macro_event_precision": metrics["macro_event_precision"],
            "macro_event_recall": metrics["macro_event_recall"],
            "macro_seg_f1_100ms": metrics["macro_seg_f1_100ms"],
            "mean_onset_mae": metrics["mean_onset_mae"],
            "mean_offset_mae": metrics["mean_offset_mae"],
            "total_hallucination": metrics["total_hallucination"],
            "total_omission": metrics["total_omission"],
            **coverage_aware_metrics(metrics),
        }
        metrics_by_decode[decode_mode] = compact
        full_metrics_by_decode[decode_mode] = metrics

    primary_metrics = full_metrics_by_decode[primary_decode_mode]
    primary_predictions = (
        out_dir / f"{split_name}_predictions_{primary_decode_mode}.jsonl"
    )
    (out_dir / f"{split_name}_predictions.jsonl").write_text(
        primary_predictions.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(primary_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment": "S1a-dual-temporal-head",
        "status": "completed",
        "scope": (
            "event type, 100 ms multi-span activity, and boundary envelope; "
            "no caption text or tracks"
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "git_commit": _git_commit(),
        "config_sha256": _json_hash(cfg),
        "n_validation": len(val_data),
        "split_name": split_name,
        "n_samples": len(val_data),
        "eventness_threshold": eventness_threshold,
        "activity_threshold": activity_threshold,
        "tiou_threshold": tiou_threshold,
        "primary_decode_mode": primary_decode_mode,
        "decode_modes": decode_modes,
        "metrics_path": str(
            (out_dir / f"{split_name}_metrics.json").resolve()
        ),
        "metrics": metrics_by_decode[primary_decode_mode],
        "metrics_by_decode": metrics_by_decode,
    }
    summary_name = "run_summary.json" if split_name == "val" else f"{split_name}_summary.json"
    (out_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def calibrate_eventness_threshold(
    cfg: dict,
    *,
    model,
    calibration_data: list[dict],
    out_dir: Path,
    checkpoint_path: Path,
) -> float:
    """Tune eventness only on a source-disjoint calibration fold."""
    eval_cfg = cfg.get("evaluation", {})
    thresholds = sorted(
        {float(value) for value in eval_cfg.get("calibration_thresholds", [0.5])}
    )
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("evaluation.calibration_thresholds must lie within (0, 1)")
    primary_mode = str(eval_cfg.get("primary_decode_mode", "hybrid"))
    sweep_cfg = copy.deepcopy(cfg)
    sweep_cfg.setdefault("evaluation", {})["decode_modes"] = [primary_mode]
    sweep_root = out_dir / "calibration_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for threshold in thresholds:
        threshold_dir = sweep_root / f"eventness_{threshold:.3f}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        summary = evaluate_slots(
            sweep_cfg,
            model=model,
            val_data=calibration_data,
            out_dir=threshold_dir,
            checkpoint_path=checkpoint_path,
            split_name="calibration",
            eventness_threshold_override=threshold,
        )
        row = {
            "threshold": threshold,
            **summary["metrics_by_decode"][primary_mode],
        }
        rows.append(row)
    selected = select_eventness_threshold(rows)
    artifact = {
        "selection_split": "calibration",
        "selection_metric": "micro_event_f1",
        "tie_breakers": [
            "micro_event_recall",
            "negative_total_hallucination",
            "higher_threshold",
        ],
        "primary_decode_mode": primary_mode,
        "selected_threshold": selected["threshold"],
        "candidates": rows,
        "warning": (
            "boundary MAE is conditional on matched events and is never the "
            "threshold-selection objective"
        ),
    }
    (out_dir / "threshold_selection.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return float(selected["threshold"])


def _evaluate_with_calibrated_threshold(
    cfg: dict,
    *,
    model,
    calibration_data: list[dict],
    val_data: list[dict],
    out_dir: Path,
    checkpoint_path: Path,
) -> dict:
    selected_threshold = calibrate_eventness_threshold(
        cfg,
        model=model,
        calibration_data=calibration_data,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
    )
    summary = evaluate_slots(
        cfg,
        model=model,
        val_data=val_data,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        split_name="val",
        eventness_threshold_override=selected_threshold,
    )
    summary["threshold_selected_on"] = "calibration"
    summary["threshold_selection_path"] = str(
        (out_dir / "threshold_selection.json").resolve()
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def train_slots(
    cfg: dict,
    cache_dir: Path,
    *,
    resume_from: str | Path | None = None,
    evaluate_checkpoint: str | Path | None = None,
) -> dict:
    """Train S1a with leakage-safe folds, validation, and resumable checkpoints."""
    train_entries, calibration_entries, val_entries = _split_entries(cfg)
    train_data = _load_fold(train_entries, cache_dir)
    calibration_data = _load_fold(calibration_entries, cache_dir)
    val_data = _load_fold(val_entries, cache_dir)
    train_cfg = cfg["train"]
    seed = int(train_cfg["seed"])
    deterministic = bool(train_cfg.get("deterministic", True))
    _set_reproducible(seed, deterministic)

    device = cfg["model"]["device"]
    model = _build_model(cfg, train_data[0]["features"].shape[-1]).to(device)
    out_dir = Path(train_cfg.get("output_dir", "outputs/s1_event_slots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _json_hash(cfg)
    split_payload = {
        "seed": seed,
        "group_key": cfg["data"].get("group_key", "source_id"),
        "source_leakage_count": 0,
        "train": [sample["sample_id"] for sample in train_data],
        "calibration": [sample["sample_id"] for sample in calibration_data],
        "val": [sample["sample_id"] for sample in val_data],
    }
    (out_dir / "split.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "experiment": "S1a-dual-temporal-head",
        "git_commit": _git_commit(),
        "config": cfg,
        "config_sha256": config_hash,
        "cache_manifest_sha256": _sha256_file(cache_dir / "cache_manifest.json"),
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "n_train": len(train_data),
        "n_calibration": len(calibration_data),
        "n_val": len(val_data),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if evaluate_checkpoint:
        checkpoint_path = Path(evaluate_checkpoint).resolve()
        _load_checkpoint(
            checkpoint_path,
            model=model,
            expected_config_hash=config_hash,
        )
        return _evaluate_with_calibrated_threshold(
            cfg,
            model=model,
            calibration_data=calibration_data,
            val_data=val_data,
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    total_steps = int(train_cfg["steps"])
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda current_step: _warmup_cosine_multiplier(
            current_step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    rng = random.Random(seed)
    start_step = 0
    best_calibration_loss = float("inf")
    if resume_from:
        resumed = _load_checkpoint(
            Path(resume_from).resolve(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            expected_config_hash=config_hash,
        )
        start_step = int(resumed["step"])
        best_calibration_loss = float(resumed["best_calibration_loss"])
        print(f"[s1] resumed at step {start_step}", file=sys.stderr)

    loss_kwargs = _loss_kwargs(cfg)
    eval_every = int(train_cfg.get("eval_every", 500))
    log_every = int(train_cfg.get("log_every", 100))
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    step = start_step
    started = time.time()
    model.train()
    while step < total_steps:
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for sample_index in order:
            if step >= total_steps:
                break
            loss_values = _sample_loss(
                model, train_data[sample_index], device, loss_kwargs
            )
            loss = loss_values["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("max_grad_norm", 1.0))
            )
            optimizer.step()
            scheduler.step()
            step += 1

            if step % log_every == 0 or step == total_steps:
                print(
                    f"[s1] step={step}/{total_steps} loss={float(loss):.4f} "
                    f"eventness={float(loss_values['eventness_loss']):.4f} "
                    f"type={float(loss_values['type_loss']):.4f} "
                    f"activity={float(loss_values['activity_loss']):.4f} "
                    f"boundary={float(loss_values['boundary_loss']):.4f} "
                    f"elapsed={time.time() - started:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            if step % eval_every == 0 or step == total_steps:
                calibration_loss = _validation_loss(
                    model, calibration_data, device, loss_kwargs
                )
                print(
                    f"[s1] step={step} calibration_loss={calibration_loss:.6f}",
                    file=sys.stderr,
                )
                if calibration_loss < best_calibration_loss:
                    best_calibration_loss = calibration_loss
                    _save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_calibration_loss=best_calibration_loss,
                        rng=rng,
                        config_hash=config_hash,
                    )
                _save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_calibration_loss=best_calibration_loss,
                    rng=rng,
                    config_hash=config_hash,
                )

    if not best_path.is_file():
        raise RuntimeError("training completed without producing a best checkpoint")
    _load_checkpoint(best_path, model=model, expected_config_hash=config_hash)
    return _evaluate_with_calibrated_threshold(
        cfg,
        model=model,
        calibration_data=calibration_data,
        val_data=val_data,
        out_dir=out_dir,
        checkpoint_path=best_path,
    )


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model_path:
        cfg["model"]["path"] = args.model_path
    if args.device:
        cfg["model"]["device"] = args.device
    if args.train_manifest:
        cfg["data"]["train_manifest_path"] = args.train_manifest
    if args.val_manifest:
        cfg["data"]["val_manifest_path"] = args.val_manifest
    if args.audio_base:
        cfg["data"]["audio_base_dir"] = args.audio_base
    if args.feature_cache:
        cfg["data"]["feature_cache"] = args.feature_cache
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.n_slots is not None:
        cfg["model"]["n_slots"] = args.n_slots
    if args.disable_temporal_embedding:
        cfg["model"]["use_temporal_embedding"] = False
    if args.positive_weight_scale is not None:
        cfg.setdefault("loss", {})["positive_weight_scale"] = args.positive_weight_scale
    if args.activity_weight is not None:
        cfg.setdefault("loss", {})["activity_weight"] = args.activity_weight
    if args.boundary_weight is not None:
        cfg.setdefault("loss", {})["boundary_weight"] = args.boundary_weight
    if args.activity_cost_weight is not None:
        cfg.setdefault("loss", {})[
            "activity_cost_weight"
        ] = args.activity_cost_weight
    if args.boundary_cost_weight is not None:
        cfg.setdefault("loss", {})[
            "boundary_cost_weight"
        ] = args.boundary_cost_weight
    if args.primary_decode_mode:
        cfg.setdefault("evaluation", {})[
            "primary_decode_mode"
        ] = args.primary_decode_mode
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-train-slots")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--audio-base")
    parser.add_argument("--feature-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-slots", type=int)
    parser.add_argument("--disable-temporal-embedding", action="store_true")
    parser.add_argument("--positive-weight-scale", type=float)
    parser.add_argument("--activity-weight", type=float)
    parser.add_argument("--boundary-weight", type=float)
    parser.add_argument("--activity-cost-weight", type=float)
    parser.add_argument("--boundary-cost-weight", type=float)
    parser.add_argument(
        "--primary-decode-mode",
        choices=("activity", "boundary", "hybrid"),
    )
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-checkpoint")
    args = parser.parse_args(argv)
    if bool(args.train_manifest) != bool(args.val_manifest):
        parser.error("--train-manifest and --val-manifest must be supplied together")
    if args.resume and args.evaluate_checkpoint:
        parser.error("--resume and --evaluate-checkpoint are mutually exclusive")

    cfg = _apply_overrides(_load_config(args.config), args)
    cache_dir = cache_features(cfg, force=args.force_cache)
    if args.cache_only:
        print(json.dumps({"cache_dir": str(cache_dir), "status": "complete"}))
        return 0
    summary = train_slots(
        cfg,
        cache_dir,
        resume_from=args.resume,
        evaluate_checkpoint=args.evaluate_checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
