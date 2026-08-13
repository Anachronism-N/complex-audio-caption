"""Run B3-slot-aware-5k on 50 real-audio mixtures to test real-audio performance."""
import sys, json, time
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
from sceneledger.eval.parser import parse_model_output

adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device="cuda:0", dtype="bfloat16"))
adapter._load()
from peft import PeftModel
adapter._model = PeftModel.from_pretrained(adapter._model, "outputs/b3_slot_aware_5k/lora")

manifest = [json.loads(l) for l in open("data/derived/real_mix/manifest.jsonl")]
prompt = "List every audible event with type, onset, offset, and description."

results = []
t0 = time.time()
for i, m in enumerate(manifest[:50]):
    sid = m["scene_id"]
    path = f"/tmp/real_mix/audio/{sid}.wav"
    try:
        raw = adapter.infer(path, prompt, sample_id=sid, duration=8.0)
    except Exception as e:
        raw = f"ERROR: {e}"
    ledger, ok = parse_model_output(raw, sid, 8.0)
    results.append({"sample_id": sid, "raw": raw[:200], "parsed": ok, "n_events": len(ledger.events)})
    if (i+1) % 10 == 0:
        print(f"{i+1}/50 ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

ok_count = sum(1 for r in results if r["parsed"])
print(f"\n=== B3-slot-aware-5k on REAL audio (50 clips) ===")
print(f"Format OK: {ok_count}/50 ({ok_count*2}%)")
print(f"Mean events: {sum(r['n_events'] for r in results)/len(results):.1f}")
print(f"\nSample outputs:")
for r in results[:5]:
    print(f"  {r['sample_id']}: parsed={r['parsed']} n_ev={r['n_events']}")
    print(f"    raw: {r['raw'][:120]}")
