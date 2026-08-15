"""Generate held-out review CSV: 20 clips with GT + model predictions side by side."""
import json, csv

# Load predictions
preds = {}
with open("reports/b3_real_v6_3k_heldout_predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        preds[d["sample_id"]] = d

# Load held-out manifest
manifest = [json.loads(l) for l in open("/tmp/heldout_manifest.jsonl")]

out_path = "data/derived/real_mix_v6/heldout_review.csv"
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow([
        "clip_id", "audio_path", "duration", "scene_name",
        "gt_n_events", "gt_events",
        "pred_n_events", "pred_events", "pred_format_ok",
        "match", "errors", "notes"
    ])
    for m in manifest:
        sid = m["scene"]["scene_id"]
        gt_ledger = m["target_ledger"]
        gt_events = gt_ledger["events"]
        gt_str = " | ".join(
            f'{e["type"]} [{e["spans"][0]["start_sec"]:.1f}-{e["spans"][-1]["end_sec"]:.1f}] {e["text"][:40]}'
            for e in gt_events
        )

        pred = preds.get(sid, {})
        pred_events = pred.get("events", [])
        pred_ok = pred.get("strict_format_success", True)
        pred_str = " | ".join(
            f'{e.get("type","?")} [{e.get("spans",[{}])[0].get("start_sec",0):.1f}-{e.get("spans",[{}])[-1].get("end_sec",0):.1f}] {e.get("text","")[:40]}'
            for e in pred_events
        ) if pred_events else "(parse failed)"

        # Simple match check
        gt_set = set((e["type"], round(e["spans"][0]["start_sec"], 1)) for e in gt_events)
        pred_set = set((e.get("type","?"), round(e.get("spans",[{}])[0].get("start_sec",0), 1)) for e in pred_events) if pred_events else set()
        n_correct = len(gt_set & pred_set)
        n_halluc = len(pred_set - gt_set)
        n_omit = len(gt_set - pred_set)
        match = "correct" if n_halluc == 0 and n_omit == 0 else (
            "halluc" if n_halluc > 0 and n_omit == 0 else
            "omit" if n_omit > 0 and n_halluc == 0 else "mixed"
        )
        errors = f"correct={n_correct} halluc={n_halluc} omit={n_omit}"

        w.writerow([
            sid, f"audio/{sid}.wav", m["scene"]["duration"], m["scene"]["template"],
            len(gt_events), gt_str,
            len(pred_events), pred_str, "Y" if pred_ok else "N",
            match, errors, ""
        ])

print(f"Wrote {len(manifest)} clips to {out_path}")
print("\nClips:")
for m in manifest:
    sid = m["scene"]["scene_id"]
    pred = preds.get(sid, {})
    gt_n = len(m["target_ledger"]["events"])
    pred_n = len(pred.get("events", []))
    match = "✓" if gt_n == pred_n else "✗"
    print(f"  {sid}: gt={gt_n} pred={pred_n} {match}")
