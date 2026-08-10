"""S1a 32-sample overfit test (docs/11 §10: '先 overfit 32 条样本').

If the slot decoder can't overfit 32 samples, there's an architecture bug.
If it can, the issue is generalization (capacity/data), not architecture.
"""
import sys, json, time, random
from pathlib import Path
import torch
sys.path.insert(0, "src")
from sceneledger.models.event_slots_v2 import EventSlotDecoderV2
from sceneledger.losses.set_prediction_v2 import set_prediction_loss_v2, _events_to_targets_v2
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger

device = "cuda:0"

# load cached features
entries = read_manifest("data/derived/b3_unified/manifest.jsonl")
cache_dir = Path("/tmp/s1_features")
dataset = []
for entry in entries:
    cp = cache_dir / f"{entry.scene['scene_id']}.pt"
    if not cp.exists(): continue
    data = torch.load(cp, weights_only=False)
    ledger = Ledger.model_validate(entry.target_ledger)
    events = [{"type": ev.type, "onset": ev.start_sec(), "offset": ev.end_sec()} for ev in ledger.events]
    dataset.append({"features": data["features"], "events": events,
                    "sample_id": entry.scene["scene_id"], "duration": float(entry.scene["duration"]),
                    "ref": entry.target_ledger})

# take only 32 samples
train_32 = dataset[:32]
print(f"[overfit] 32 samples, max events per clip: {max(len(d['events']) for d in train_32)}", file=sys.stderr, flush=True)

# build model — larger capacity for overfit test
feat_dim = train_32[0]["features"].shape[-1]
model = EventSlotDecoderV2(
    feature_dim=feat_dim, hidden_dim=1024, n_slots=8, n_heads=8, n_layers=8,  # larger
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # lower LR
rng = random.Random(42)
total_steps = 5000
t0 = time.time()

for step in range(total_steps):
    sample = train_32[step % 32]
    features = sample["features"].unsqueeze(0).to(device)
    outputs = model(features)
    targets = [_events_to_targets_v2(sample["events"], 8)]
    loss_dict = set_prediction_loss_v2(outputs, targets, boundary_weight=1.0)  # lower boundary weight
    loss = loss_dict["loss"]
    # skip NaN steps
    if torch.isnan(loss) or torch.isinf(loss):
        optimizer.zero_grad()
        continue
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # tighter clipping
    optimizer.step()

    if (step + 1) % 500 == 0:
        # evaluate on the 32 training samples
        model.eval()
        n_correct = 0
        n_total = 0
        for s in train_32:
            feats = s["features"].unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model.predict(feats, threshold=0.3)
            n_pred = len(preds[0])
            n_gt = len(s["events"])
            n_total += n_gt
            if n_pred == n_gt:
                n_correct += n_gt
        model.train()
        print(f"[overfit] step {step+1}/{total_steps} loss={loss.item():.4f} "
              f"(ev={loss_dict['eventness_loss'].item():.3f} "
              f"ty={loss_dict['type_loss'].item():.3f} "
              f"bnd={loss_dict['boundary_loss'].item():.3f}) "
              f"count_match={n_correct}/{n_total} ({time.time()-t0:.0f}s)",
              file=sys.stderr, flush=True)

# final eval
model.eval()
preds, refs = [], []
for s in train_32:
    feats = s["features"].unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model.predict(feats, threshold=0.3)
    pe = predicted[0]
    pred_events = []
    for i, e in enumerate(pe):
        onset = max(0.0, min(e["onset"], s["duration"] - 0.1))
        offset = max(onset + 0.1, min(e["offset"], s["duration"]))
        pred_events.append({"id": f"E{i+1:03d}", "type": e["type"], "track_id": None,
                            "spans": [{"start_sec": onset, "end_sec": offset}],
                            "text": e["type"], "confidence": e["confidence"]})
    preds.append({"schema_version": "0.2.0", "sample_id": s["sample_id"],
                  "duration_sec": s["duration"], "time_resolution_sec": 0.1,
                  "tracks": [], "events": pred_events})
    refs.append(s["ref"])

out_dir = Path("reports/s1_overfit32")
out_dir.mkdir(parents=True, exist_ok=True)
pp, rp = out_dir / "predictions.jsonl", out_dir / "references.jsonl"
with open(pp, "w") as f:
    for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(rp, "w") as f:
    for r in refs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

from sceneledger.eval.metrics import evaluate_corpus
c = evaluate_corpus(pp, rp)
print(f"\n=== S1a 32-sample overfit ===", file=sys.stderr)
print(f"event-F1:   {c.macro_event_f1:.3f}", file=sys.stderr)
print(f"precision:  {c.macro_event_precision:.3f}", file=sys.stderr)
print(f"recall:     {c.macro_event_recall:.3f}", file=sys.stderr)
print(f"onset-MAE:  {c.mean_onset_mae:.3f}s", file=sys.stderr)
print(f"offset-MAE: {c.mean_offset_mae:.3f}s", file=sys.stderr)
print(f"halluc:     {c.total_hallucination}", file=sys.stderr)
print(f"omit:       {c.total_omission}", file=sys.stderr)
