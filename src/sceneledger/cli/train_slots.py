"""S1 event-slot training: cache MOSS features, train slot decoder, evaluate.

This is S1a (event-only, no text). The slot decoder predicts event sets
(type + activity mask) via permutation-invariant Hungarian matching.

::

    conda run -n moss-audio python -m sceneledger.cli.train_slots \
      --config configs/model/s1_event_slots.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_audio(path, sr_target, max_sec):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sr_target:
        g = gcd(int(sr), int(sr_target))
        wav = resample_poly(wav.astype(np.float64), int(sr_target) // g, int(sr) // g).astype(np.float32)
    return wav[: int(max_sec * sr_target)]


def cache_features(cfg, force=False):
    """Extract and cache MOSS audio encoder features for all clips."""
    cache_dir = Path(cfg["data"].get("feature_cache", "/tmp/s1_features"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "done.flag").exists() and not force:
        print(f"[s1] features already cached at {cache_dir}", file=sys.stderr)
        return cache_dir

    import sys as _sys
    repo = Path(__file__).resolve().parents[2] / "third_party" / "MOSS-Audio"
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
    from sceneledger.data.manifests import read_manifest

    print("[s1] loading MOSS model for feature extraction ...", file=sys.stderr, flush=True)
    adapter = MossAdapter(MossAdapterConfig(
        model_path=cfg["model"]["path"], device=cfg["model"]["device"], dtype=cfg["model"]["dtype"]
    ))
    adapter._load()
    model = adapter._model
    processor = adapter._processor
    model.eval()

    entries = read_manifest(cfg["data"]["manifest_path"])
    audio_base = cfg["data"]["audio_base_dir"]
    sr = processor.config.mel_sr
    max_sec = cfg["data"].get("max_audio_seconds", 30.0)

    print(f"[s1] caching features for {len(entries)} clips ...", file=sys.stderr, flush=True)
    t0 = time.time()
    for i, entry in enumerate(entries):
        cache_path = cache_dir / f"{entry.scene['scene_id']}.pt"
        if cache_path.exists() and not force:
            continue
        audio_path = str(Path(audio_base) / entry.mixture_path)
        wav = _load_audio(audio_path, sr, max_sec)
        inputs = processor(text="x", audios=[wav], return_tensors="pt")
        audio_data = inputs["audio_data"].to(cfg["model"]["device"]).to(torch.bfloat16)
        audio_seqlens = inputs["audio_data_seqlens"].to(cfg["model"]["device"])
        with torch.no_grad():
            audio_embeds, deepstack = model.get_audio_features(audio_data, audio_seqlens)
            audio_embeds = model.audio_adapter(audio_embeds)
        # save as float16 to save space
        torch.save({
            "features": audio_embeds[0].float().cpu(),
            "sample_id": entry.scene["scene_id"],
            "duration": float(entry.scene["duration"]),
            "target_ledger": entry.target_ledger,
        }, cache_path)
        if (i + 1) % 100 == 0:
            print(f"[s1] cached {i+1}/{len(entries)} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    (cache_dir / "done.flag").touch()
    print(f"[s1] feature caching done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    return cache_dir


def train_slots(cfg, cache_dir):
    """Train the event slot decoder on cached features."""
    from sceneledger.models.event_slots import EventSlotDecoder
    from sceneledger.losses.set_prediction import set_prediction_loss, _events_to_targets
    from sceneledger.data.manifests import read_manifest
    from sceneledger.data.schema import Ledger, Span, Event

    device = cfg["model"]["device"]
    tcfg = cfg["train"]

    # load all cached features + targets
    entries = read_manifest(cfg["data"]["manifest_path"])
    dataset = []
    for entry in entries:
        cache_path = cache_dir / f"{entry.scene['scene_id']}.pt"
        if not cache_path.exists():
            continue
        data = torch.load(cache_path, weights_only=False)
        ledger = Ledger.model_validate(entry.target_ledger)
        events = []
        for ev in ledger.events:
            events.append({
                "type": ev.type,
                "onset": ev.start_sec(),
                "offset": ev.end_sec(),
            })
        dataset.append({
            "features": data["features"],
            "events": events,
            "sample_id": entry.scene["scene_id"],
            "duration": float(entry.scene["duration"]),
        })

    n_train = int(len(dataset) * (1 - tcfg.get("val_fraction", 0.1)))
    train_data = dataset[:n_train]
    val_data = dataset[n_train:]
    print(f"[s1] {len(train_data)} train, {len(val_data)} val", file=sys.stderr, flush=True)

    # build model
    feature_dim = dataset[0]["features"].shape[-1]
    model = EventSlotDecoder(
        feature_dim=feature_dim,
        hidden_dim=cfg["model"].get("hidden_dim", 768),
        n_slots=cfg["model"].get("n_slots", 24),
        n_heads=cfg["model"].get("n_heads", 8),
        n_layers=cfg["model"].get("n_layers", 4),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["steps"])

    import random
    rng = random.Random(tcfg["seed"])
    step = 0
    t0 = time.time()
    while step < tcfg["steps"]:
        rng.shuffle(train_data)
        for sample in train_data:
            if step >= tcfg["steps"]:
                break
            features = sample["features"].unsqueeze(0).to(device)
            outputs = model(features)
            T_100 = outputs["n_frames"]
            targets = [_events_to_targets(sample["events"], T_100, model.n_slots)]
            loss_dict = set_prediction_loss(outputs, targets)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0:
                print(
                    f"[s1] step {step+1}/{tcfg['steps']} loss={loss.item():.4f} "
                    f"(ev={loss_dict['eventness_loss'].item():.3f} "
                    f"ty={loss_dict['type_loss'].item():.3f} "
                    f"act={loss_dict['activity_loss'].item():.3f}) "
                    f"({time.time()-t0:.0f}s)",
                    file=sys.stderr, flush=True,
                )
            step += 1

    # save model
    out_dir = Path(tcfg.get("output_dir", "outputs/s1_event_slots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "slot_decoder.pt")
    print(f"[s1] saved to {out_dir / 'slot_decoder.pt'}", file=sys.stderr, flush=True)

    # evaluate on val set
    model.eval()
    predictions = []
    references = []
    for sample in val_data:
        features = sample["features"].unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model.predict(features)
        pred_events = preds[0]
        # build prediction ledger
        pred_ledger_events = []
        for i, pe in enumerate(pred_events):
            pred_ledger_events.append({
                "id": f"E{i+1:03d}",
                "type": pe["type"],
                "track_id": None,
                "spans": [{"start_sec": pe["onset"], "end_sec": pe["offset"]}],
                "text": pe["type"],
                "confidence": pe["confidence"],
            })
        pred_ledger = {
            "schema_version": "0.2.0",
            "sample_id": sample["sample_id"],
            "duration_sec": sample["duration"],
            "time_resolution_sec": 0.1,
            "tracks": [],
            "events": pred_ledger_events,
        }
        predictions.append(pred_ledger)
        references.append(sample["target_ledger"] if isinstance(sample.get("target_ledger"), dict) else
                         next(e.target_ledger for e in entries if e.scene["scene_id"] == sample["sample_id"]))

    # write predictions + references for evaluation
    pred_path = out_dir / "predictions.jsonl"
    ref_path = out_dir / "references.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(ref_path, "w") as f:
        for r in references:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[s1] predictions -> {pred_path}", file=sys.stderr, flush=True)
    print(f"[s1] references -> {ref_path}", file=sys.stderr, flush=True)

    # quick eval
    from sceneledger.eval.metrics import evaluate_corpus
    corpus = evaluate_corpus(pred_path, ref_path)
    print(f"\n=== S1a val results ({len(val_data)} clips) ===", file=sys.stderr)
    print(f"event-F1:     {corpus.macro_event_f1:.3f}", file=sys.stderr)
    print(f"precision:    {corpus.macro_event_precision:.3f}", file=sys.stderr)
    print(f"recall:       {corpus.macro_event_recall:.3f}", file=sys.stderr)
    print(f"onset-MAE:    {corpus.mean_onset_mae:.3f}s", file=sys.stderr)
    print(f"offset-MAE:   {corpus.mean_offset_mae:.3f}s", file=sys.stderr)
    print(f"halluc:       {corpus.total_hallucination}", file=sys.stderr)
    print(f"omission:     {corpus.total_omission}", file=sys.stderr)

    # save metrics
    import json as _json
    (out_dir / "val_metrics.json").write_text(
        _json.dumps(corpus.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sceneledger-train-slots")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    cache_dir = cache_features(cfg, force=args.force_cache)
    train_slots(cfg, cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
