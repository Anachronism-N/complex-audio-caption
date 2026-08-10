"""Exploratory S1a-v3 joint training on an accepted B3-valid release.

Key change from v2: features are computed WITH gradient through the last 4
encoder layers + audio_adapter, so the features adapt to event detection.
The LLM and first 28 encoder layers stay frozen.

This is not the primary S1 protocol: checkpoint/threshold selection still need
to be integrated with ``sceneledger.cli.train_slots`` before paper reporting.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")

from sceneledger.data.manifests import read_manifest
from sceneledger.data.reproduction import require_b3_data_summary
from sceneledger.data.schema import Ledger
from sceneledger.losses.set_prediction_v2 import (
    _events_to_targets_v2,
    set_prediction_loss_v2,
)
from sceneledger.models.event_slots_v2 import EventSlotDecoderV2
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig


def _load_audio(path, sr_target, max_sec):
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sr_target:
        g = gcd(int(sr), int(sr_target))
        wav = resample_poly(
            wav.astype(np.float64), int(sr_target) // g, int(sr) // g
        ).astype(np.float32)
    return wav[: int(max_sec * sr_target)]


def _load_fold(entries, audio_base: Path) -> list[dict]:
    dataset = []
    for entry in entries:
        ledger = Ledger.model_validate(entry.target_ledger)
        events = [
            {"type": event.type, "onset": event.start_sec(), "offset": event.end_sec()}
            for event in ledger.events
        ]
        dataset.append(
            {
                "audio_path": str(audio_base / entry.mixture_path),
                "events": events,
                "sample_id": entry.scene["scene_id"],
                "duration": float(entry.scene["duration"]),
                "ref": entry.target_ledger,
            }
        )
    return dataset


def _lr_multiplier(update: int, *, warmup_updates: int, total_updates: int) -> float:
    if update < warmup_updates:
        return max(1e-8, (update + 1) / max(1, warmup_updates))
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    device = os.environ.get("DEVICE", "cuda:0")
    model_path = os.environ.get("MODEL_DIR", "/tmp/moss_weights")
    b3_root = Path(os.environ.get("B3_WORK_DIR", "runs/b3_valid")).resolve()
    output_dir = Path(
        os.environ.get("S1V3_OUTPUT_DIR", "outputs/s1_event_slots_v3")
    ).resolve()
    data_summary = require_b3_data_summary(
        b3_root / "data_reproduction_summary.json",
        expected_dataset_id=os.environ.get("B3_DATASET_ID"),
    )
    seed = int(os.environ.get("SEED", "20260808"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("[s1v3] loading MOSS model ...", file=sys.stderr, flush=True)
    adapter = MossAdapter(
        MossAdapterConfig(model_path=model_path, device=device, dtype="bfloat16")
    )
    adapter._load()
    model = adapter._model
    processor = adapter._processor

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last 4 encoder layers + audio_adapter
    n_enc_layers = len(model.audio_encoder.layers)
    for i in range(n_enc_layers - 4, n_enc_layers):
        for p in model.audio_encoder.layers[i].parameters():
            p.requires_grad = True
    for p in model.audio_adapter.parameters():
        p.requires_grad = True
    print(
        f"[s1v3] unfrozen encoder layers "
        f"{n_enc_layers - 4}..{n_enc_layers - 1} + adapter",
        file=sys.stderr,
        flush=True,
    )

    # build slot decoder
    slot_decoder = EventSlotDecoderV2(
        feature_dim=2560, hidden_dim=768, n_slots=8, n_heads=8, n_layers=6,
    ).to(device)

    # collect trainable params
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable += list(slot_decoder.parameters())
    n_trainable = sum(parameter.numel() for parameter in trainable)
    print(
        f"[s1v3] trainable params: {n_trainable / 1e6:.1f}M",
        file=sys.stderr,
        flush=True,
    )

    # differential LR: encoder layers get 10x lower LR for stability
    enc_params = [p for p in model.parameters() if p.requires_grad]
    dec_params = list(slot_decoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": 1e-5},
            {"params": dec_params, "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    total_steps = int(os.environ.get("S1V3_STEPS", "3000"))
    warmup_steps = int(os.environ.get("S1V3_WARMUP_STEPS", "500"))
    grad_accum = int(os.environ.get("S1V3_GRAD_ACCUM", "4"))
    if total_steps <= 0 or grad_accum <= 0:
        raise ValueError("S1V3_STEPS and S1V3_GRAD_ACCUM must be positive")
    if total_steps % grad_accum:
        raise ValueError("S1V3_STEPS must be divisible by S1V3_GRAD_ACCUM")
    total_updates = math.ceil(total_steps / grad_accum)
    warmup_updates = math.ceil(warmup_steps / grad_accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: _lr_multiplier(
            update,
            warmup_updates=warmup_updates,
            total_updates=total_updates,
        ),
    )

    # load data
    train_entries = read_manifest(b3_root / "sft/train_manifest.jsonl")
    val_entries = read_manifest(b3_root / "sft/val_manifest.jsonl")
    audio_base = b3_root / "data"
    sr = processor.config.mel_sr
    max_sec = 30.0
    train = _load_fold(train_entries, audio_base)
    val = _load_fold(val_entries, audio_base)
    if not train or not val:
        raise ValueError("accepted B3 train and validation folds must be non-empty")
    print(f"[s1v3] {len(train)} train, {len(val)} val", file=sys.stderr, flush=True)

    rng = random.Random(seed)
    step = 0
    update_step = 0
    t0 = time.time()
    model.eval()  # eval mode for encoder (no dropout in frozen layers)
    slot_decoder.train()
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        rng.shuffle(train)
        for sample in train:
            if step >= total_steps:
                break
            # load audio + process
            wav = _load_audio(sample["audio_path"], sr, max_sec)
            inputs = processor(text="x", audios=[wav], return_tensors="pt")
            audio_data = inputs["audio_data"].to(device).to(torch.bfloat16)
            audio_seqlens = inputs["audio_data_seqlens"].to(device)

            # forward through encoder WITH gradient on unfrozen layers
            audio_embeds, _ = model.get_audio_features(audio_data, audio_seqlens)
            features = model.audio_adapter(audio_embeds).float()  # [1, T, 2560] cast to fp32 for slot decoder

            # slot decoder forward
            outputs = slot_decoder(features)
            targets = [_events_to_targets_v2(sample["events"], 8)]
            loss_dict = set_prediction_loss_v2(outputs, targets, boundary_weight=5.0)
            raw_loss = loss_dict["loss"]
            (raw_loss / grad_accum).backward()
            should_update = (step + 1) % grad_accum == 0 or step + 1 == total_steps
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if (step + 1) % 100 == 0:
                print(
                    f"[s1v3] step {step + 1}/{total_steps} "
                    f"update={update_step}/{total_updates} loss={raw_loss.item():.4f} "
                    f"(ev={loss_dict['eventness_loss'].item():.3f} "
                    f"ty={loss_dict['type_loss'].item():.3f} "
                    f"bnd={loss_dict['boundary_loss'].item():.3f}) "
                    f"({time.time() - t0:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )
            step += 1

    # save
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(slot_decoder.state_dict(), out_dir / "slot_decoder_v3.pt")
    # save unfrozen encoder weights
    enc_state = {
        key: value.cpu()
        for key, value in model.audio_encoder.layers[n_enc_layers - 4 :]
        .state_dict()
        .items()
    }
    torch.save(enc_state, out_dir / "encoder_last4.pt")
    torch.save(model.audio_adapter.state_dict(), out_dir / "adapter.pt")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "exploratory_not_primary_s1_protocol",
                "dataset_id": data_summary["dataset_id"],
                "b3_work_dir": str(b3_root),
                "seed": seed,
                "total_micro_steps": total_steps,
                "total_optimizer_updates": update_step,
                "warmup_updates": warmup_updates,
                "gradient_accumulation": grad_accum,
                "encoder_learning_rate": 1e-5,
                "decoder_learning_rate": 1e-4,
                "n_train": len(train),
                "n_val": len(val),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # evaluate
    slot_decoder.eval()
    model.eval()
    preds, refs = [], []
    for sample in val:
        wav = _load_audio(sample["audio_path"], sr, max_sec)
        inputs = processor(text="x", audios=[wav], return_tensors="pt")
        audio_data = inputs["audio_data"].to(device).to(torch.bfloat16)
        audio_seqlens = inputs["audio_data_seqlens"].to(device)
        with torch.no_grad():
            audio_embeds, _ = model.get_audio_features(audio_data, audio_seqlens)
            features = model.audio_adapter(audio_embeds).float()
            predicted = slot_decoder.predict(features, threshold=0.2)
        pe = predicted[0]
        pred_events = []
        for i, e in enumerate(pe):
            onset = max(0.0, min(e["onset"], sample["duration"] - 0.1))
            offset = max(onset + 0.1, min(e["offset"], sample["duration"]))
            pred_events.append(
                {
                    "id": f"E{i + 1:03d}",
                    "type": e["type"],
                    "track_id": None,
                    "spans": [{"start_sec": onset, "end_sec": offset}],
                    "text": e["type"],
                    "confidence": e["confidence"],
                }
            )
        preds.append(
            {
                "schema_version": "0.2.0",
                "sample_id": sample["sample_id"],
                "duration_sec": sample["duration"],
                "time_resolution_sec": 0.1,
                "tracks": [],
                "events": pred_events,
            }
        )
        refs.append(sample["ref"])

    pp, rp = out_dir / "predictions.jsonl", out_dir / "references.jsonl"
    with pp.open("w", encoding="utf-8") as handle:
        for prediction in preds:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with rp.open("w", encoding="utf-8") as handle:
        for reference in refs:
            handle.write(json.dumps(reference, ensure_ascii=False) + "\n")

    from sceneledger.eval.metrics import evaluate_corpus

    c = evaluate_corpus(pp, rp)
    print(f"\n=== S1a-v3 joint val ({len(val)} clips) ===", file=sys.stderr)
    print(f"event-F1:   {c.macro_event_f1:.3f}", file=sys.stderr)
    print(f"precision:  {c.macro_event_precision:.3f}", file=sys.stderr)
    print(f"recall:     {c.macro_event_recall:.3f}", file=sys.stderr)
    print(f"onset-MAE:  {c.mean_onset_mae:.3f}s", file=sys.stderr)
    print(f"offset-MAE: {c.mean_offset_mae:.3f}s", file=sys.stderr)
    print(f"halluc:     {c.total_hallucination}", file=sys.stderr)
    print(f"omission:   {c.total_omission}", file=sys.stderr)
    print(f"per-type:   {json.dumps(c.per_type, ensure_ascii=False)}", file=sys.stderr)
    (out_dir / "val_metrics.json").write_text(
        json.dumps(c.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
