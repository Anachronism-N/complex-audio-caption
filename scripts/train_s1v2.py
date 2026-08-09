"""Train S1a-v2 (boundary regression) on cached features + evaluate."""

from __future__ import annotations
import json, sys, time, random
from pathlib import Path
import torch, yaml
from sceneledger.models.event_slots_v2 import EventSlotDecoderV2
from sceneledger.losses.set_prediction_v2 import set_prediction_loss_v2, _events_to_targets_v2
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger


def main():
    cfg = yaml.safe_load(Path("configs/model/s1_event_slots.yaml").read_text())
    cache_dir = Path(cfg["data"]["feature_cache"])
    device = cfg["model"]["device"]
    tcfg = cfg["train"]

    entries = read_manifest(cfg["data"]["manifest_path"])
    dataset = []
    for entry in entries:
        cp = cache_dir / f"{entry.scene['scene_id']}.pt"
        if not cp.exists():
            continue
        data = torch.load(cp, weights_only=False)
        ledger = Ledger.model_validate(entry.target_ledger)
        events = [{"type": ev.type, "onset": ev.start_sec(), "offset": ev.end_sec()} for ev in ledger.events]
        dataset.append({"features": data["features"], "events": events,
                        "sample_id": entry.scene["scene_id"], "duration": float(entry.scene["duration"])})

    n_train = int(len(dataset) * 0.9)
    train, val = dataset[:n_train], dataset[n_train:]
    print(f"[s1v2] {len(train)} train, {len(val)} val", file=sys.stderr, flush=True)

    feat_dim = dataset[0]["features"].shape[-1]
    model = EventSlotDecoderV2(
        feature_dim=feat_dim, hidden_dim=768, n_slots=8, n_heads=8, n_layers=6,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # warmup + cosine
    warmup = 200
    total_steps = 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    rng = random.Random(42)
    step = 0
    t0 = time.time()
    while step < total_steps:
        rng.shuffle(train)
        for sample in train:
            if step >= total_steps:
                break
            features = sample["features"].unsqueeze(0).to(device)
            outputs = model(features)
            targets = [_events_to_targets_v2(sample["events"], 8)]
            loss_dict = set_prediction_loss_v2(outputs, targets, boundary_weight=5.0)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # warmup
            if step < warmup:
                lr = 1e-4 * (step + 1) / warmup
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            optimizer.step()
            scheduler.step()
            if (step + 1) % 500 == 0:
                print(f"[s1v2] step {step+1}/{total_steps} loss={loss.item():.4f} "
                      f"(ev={loss_dict['eventness_loss'].item():.3f} "
                      f"ty={loss_dict['type_loss'].item():.3f} "
                      f"bnd={loss_dict['boundary_loss'].item():.3f}) "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
            step += 1

    # save
    out_dir = Path("outputs/s1_event_slots_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "slot_decoder_v2.pt")

    # evaluate on val
    model.eval()
    preds, refs = [], []
    for sample in val:
        features = sample["features"].unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = model.predict(features, threshold=0.35)
        pe = predicted[0]
        pred_events = []
        for i, e in enumerate(pe):
            pred_events.append({"id": f"E{i+1:03d}", "type": e["type"], "track_id": None,
                                "spans": [{"start_sec": e["onset"], "end_sec": e["offset"]}],
                                "text": e["type"], "confidence": e["confidence"]})
        preds.append({"schema_version": "0.2.0", "sample_id": sample["sample_id"],
                      "duration_sec": sample["duration"], "time_resolution_sec": 0.1,
                      "tracks": [], "events": pred_events})
        # find reference
        ref_entry = next(e for e in entries if e.scene["scene_id"] == sample["sample_id"])
        refs.append(ref_entry.target_ledger)

    pp = out_dir / "predictions.jsonl"
    rp = out_dir / "references.jsonl"
    with open(pp, "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(rp, "w") as f:
        for r in refs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from sceneledger.eval.metrics import evaluate_corpus
    c = evaluate_corpus(pp, rp)
    print(f"\n=== S1a-v2 val ({len(val)} clips) ===", file=sys.stderr)
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
    main()
