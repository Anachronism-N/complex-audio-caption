"""S1a-v3: Joint training of MOSS encoder last-4 layers + slot decoder.

Key change from v2: features are computed WITH gradient through the last 4
encoder layers + audio_adapter, so the features adapt to event detection.
The LLM and first 28 encoder layers stay frozen.
"""

from __future__ import annotations
import json, sys, time, random, gc
from pathlib import Path
import torch, torch.nn as nn
import yaml
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
from sceneledger.models.event_slots_v2 import EventSlotDecoderV2
from sceneledger.losses.set_prediction_v2 import set_prediction_loss_v2, _events_to_targets_v2
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger


def _load_audio(path, sr_target, max_sec):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2: wav = wav.mean(axis=1)
    if sr != sr_target:
        g = gcd(int(sr), int(sr_target))
        wav = resample_poly(wav.astype(np.float64), int(sr_target)//g, int(sr)//g).astype(np.float32)
    return wav[:int(max_sec * sr_target)]


def main():
    device = "cuda:0"
    print("[s1v3] loading MOSS model ...", file=sys.stderr, flush=True)
    adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device=device, dtype="bfloat16"))
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
    print(f"[s1v3] unfrozen encoder layers {n_enc_layers-4}..{n_enc_layers-1} + adapter", file=sys.stderr, flush=True)

    # build slot decoder
    slot_decoder = EventSlotDecoderV2(
        feature_dim=2560, hidden_dim=768, n_slots=8, n_heads=8, n_layers=6,
    ).to(device)

    # collect trainable params
    trainable = [p for p in model.parameters() if p.requires_grad] + list(slot_decoder.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"[s1v3] trainable params: {n_train/1e6:.1f}M", file=sys.stderr, flush=True)

    # differential LR: encoder layers get 10x lower LR for stability
    enc_params = [p for p in model.parameters() if p.requires_grad]
    dec_params = list(slot_decoder.parameters())
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": 1e-5},   # encoder: gentle
        {"params": dec_params, "lr": 1e-4},   # decoder: aggressive
    ], weight_decay=0.01)
    total_steps = 3000
    warmup = 500  # longer warmup for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    grad_accum = 4  # gradient accumulation for stability

    # load data
    entries = read_manifest("data/derived/b3_unified/manifest.jsonl")
    audio_base = "/tmp/b3_unified"
    sr = processor.config.mel_sr
    max_sec = 30.0
    dataset = []
    for entry in entries:
        ledger = Ledger.model_validate(entry.target_ledger)
        events = [{"type": ev.type, "onset": ev.start_sec(), "offset": ev.end_sec()} for ev in ledger.events]
        dataset.append({"audio_path": str(Path(audio_base) / entry.mixture_path),
                        "events": events, "sample_id": entry.scene["scene_id"],
                        "duration": float(entry.scene["duration"]), "ref": entry.target_ledger})

    n_train = int(len(dataset) * 0.9)
    train, val = dataset[:n_train], dataset[n_train:]
    print(f"[s1v3] {len(train)} train, {len(val)} val", file=sys.stderr, flush=True)

    rng = random.Random(42)
    step = 0
    t0 = time.time()
    model.eval()  # eval mode for encoder (no dropout in frozen layers)

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
            loss = loss_dict["loss"] / grad_accum
            optimizer.zero_grad()
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                if step < warmup:
                    lr_scale = (step + 1) / warmup
                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["lr"] * lr_scale if step == 0 else pg["lr"]
                optimizer.step()
                scheduler.step()

            if (step + 1) % 100 == 0:
                print(f"[s1v3] step {step+1}/{total_steps} loss={loss.item():.4f} "
                      f"(ev={loss_dict['eventness_loss'].item():.3f} "
                      f"ty={loss_dict['type_loss'].item():.3f} "
                      f"bnd={loss_dict['boundary_loss'].item():.3f}) "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
            step += 1

    # save
    out_dir = Path("outputs/s1_event_slots_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(slot_decoder.state_dict(), out_dir / "slot_decoder_v3.pt")
    # save unfrozen encoder weights
    enc_state = {k: v.cpu() for k, v in model.audio_encoder.layers[n_enc_layers-4:].state_dict().items()}
    torch.save(enc_state, out_dir / "encoder_last4.pt")
    torch.save(model.audio_adapter.state_dict(), out_dir / "adapter.pt")

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
            pred_events.append({"id": f"E{i+1:03d}", "type": e["type"], "track_id": None,
                                "spans": [{"start_sec": onset, "end_sec": offset}],
                                "text": e["type"], "confidence": e["confidence"]})
        preds.append({"schema_version": "0.2.0", "sample_id": sample["sample_id"],
                      "duration_sec": sample["duration"], "time_resolution_sec": 0.1,
                      "tracks": [], "events": pred_events})
        refs.append(sample["ref"])

    pp, rp = out_dir / "predictions.jsonl", out_dir / "references.jsonl"
    with open(pp, "w") as f:
        for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(rp, "w") as f:
        for r in refs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
    (out_dir / "val_metrics.json").write_text(json.dumps(c.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import numpy as np
    main()
