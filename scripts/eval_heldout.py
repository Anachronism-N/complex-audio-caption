"""Evaluate B3-real-v6-3k on held-out clips (not in training set)."""
import sys, json
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
from sceneledger.eval.parser import parse_model_output
from peft import PeftModel
import soundfile as sf

adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device="cuda:0", dtype="bfloat16"))
adapter._load()
adapter._model = PeftModel.from_pretrained(adapter._model, "outputs/b3_real_v6_3k/lora")

manifest = [json.loads(l) for l in open("data/derived/real_mix_v6/manifest_compat.jsonl")]
held_out = manifest[180:]  # last 20 clips, not in training

prompt = "List every audible event with type, onset, offset, and description."

results = []
for i, m in enumerate(held_out):
    sid = m["scene"]["scene_id"]
    path = f"/tmp/real_mix_v6/audio/{sid}.wav"
    gt_ledger = m["target_ledger"]
    try:
        raw = adapter.infer(path, prompt, sample_id=sid, duration=m["scene"]["duration"])
    except Exception as e:
        raw = f"ERROR: {e}"
    ledger, ok = parse_model_output(raw, sid, m["scene"]["duration"])
    results.append({"sample_id": sid, "raw": raw[:200], "ok": ok, "n_events": len(ledger.events),
                    "gt_events": len(gt_ledger["events"])})
    print(f"{i+1}/20 {sid}: ok={ok} pred_ev={len(ledger.events)} gt_ev={len(gt_ledger['events'])}", file=sys.stderr, flush=True)

# Write predictions for evaluation
with open("/tmp/heldout_predictions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps({"sample_id": r["sample_id"], "raw_text": r["raw"],
                          "events": [], "ledger": {"events": []}}) + "\n")

ok_count = sum(1 for r in results if r["ok"])
print(f"\n=== Held-out Evaluation (20 clips, NOT in training) ===")
print(f"Format OK: {ok_count}/20 ({ok_count*5}%)")
print(f"Mean pred events: {sum(r['n_events'] for r in results)/len(results):.1f}")
print(f"Mean GT events: {sum(r['gt_events'] for r in results)/len(results):.1f}")
print(f"\nSample outputs:")
for r in results[:5]:
    print(f"  {r['sample_id']}: ok={r['ok']} pred={r['n_events']} gt={r['gt_events']}")
    print(f"    {r['raw'][:120]}")
