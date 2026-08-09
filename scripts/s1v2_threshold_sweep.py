"""Threshold sweep for S1a-v2."""
import sys, json, torch, pathlib, tempfile
sys.path.insert(0, "src")
from sceneledger.models.event_slots_v2 import EventSlotDecoderV2
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger
from sceneledger.eval.metrics import evaluate_corpus

device = "cuda:0"
entries = read_manifest("data/derived/b3_unified/manifest.jsonl")
cache_dir = "/tmp/s1_features"

dataset = []
for entry in entries:
    cp = pathlib.Path(cache_dir) / f"{entry.scene['scene_id']}.pt"
    if not cp.exists():
        continue
    data = torch.load(cp, weights_only=False)
    ledger = Ledger.model_validate(entry.target_ledger)
    events = [{"type": ev.type, "onset": ev.start_sec(), "offset": ev.end_sec()} for ev in ledger.events]
    dataset.append({"features": data["features"], "events": events,
                    "sample_id": entry.scene["scene_id"], "duration": float(entry.scene["duration"]),
                    "ref": entry.target_ledger})

n_train = int(len(dataset) * 0.9)
val = dataset[n_train:]

feat_dim = val[0]["features"].shape[-1]
model = EventSlotDecoderV2(feature_dim=feat_dim, hidden_dim=768, n_slots=8, n_heads=8, n_layers=6).to(device)
model.load_state_dict(torch.load("outputs/s1_event_slots_v2/slot_decoder_v2.pt"))
model.eval()

for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    preds, refs = [], []
    for sample in val:
        features = sample["features"].unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = model.predict(features, threshold=threshold)
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
    d = pathlib.Path(tempfile.mkdtemp())
    pp, rp = d / "p.jsonl", d / "r.jsonl"
    pp.write_text("\n".join(json.dumps(p) for p in preds))
    rp.write_text("\n".join(json.dumps(r) for r in refs))
    c = evaluate_corpus(pp, rp)
    print(f"thr={threshold:.2f}: F1={c.macro_event_f1:.3f} P={c.macro_event_precision:.3f} R={c.macro_event_recall:.3f} "
          f"onset={c.mean_onset_mae:.3f} offset={c.mean_offset_mae:.3f} halluc={c.total_hallucination} omit={c.total_omission}")
