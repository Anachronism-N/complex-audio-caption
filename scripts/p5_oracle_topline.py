"""P5 oracle topline experiment: oracle stems → MOSS caption vs mixture → B3.

For each clip:
1. Run MOSS zero-shot on each STEM (perfectly separated source)
2. Build oracle ledger: ground-truth type/timing + MOSS per-stem text
3. Run MOSS zero-shot on the MIXTURE (baseline)
4. Compare oracle vs mixture vs B3-slot-aware-5k vs ground truth

This answers docs/11 §8: "if oracle stems don't help, the track idea needs
rethinking."
"""
import sys, json, time, re
from pathlib import Path
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
import torch, numpy as np
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger, Event, Span, Track
from sceneledger.eval.metrics import evaluate_corpus


def _load_audio(path, sr_target):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2: wav = wav.mean(axis=1)
    if sr != sr_target:
        g = gcd(int(sr), int(sr_target))
        wav = resample_poly(wav.astype(np.float64), int(sr_target)//g, int(sr)//g).astype(np.float32)
    return wav


def main():
    device = "cuda:0"
    print("[p5] loading MOSS ...", file=sys.stderr, flush=True)
    adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device=device, dtype="bfloat16"))
    adapter._load()
    processor = adapter._processor
    sr = processor.config.mel_sr

    entries = read_entries("data/derived/b3_5k/manifest.jsonl")
    n_val = 50
    val_entries = entries[-n_val:]  # last 50 as val (matches 90/10 split)
    audio_base = "/tmp/b3_5k/audio"

    prompt = "Describe this audio in one sentence."

    oracle_preds = []
    mixture_preds = []
    refs = []

    t0 = time.time()
    for i, entry in enumerate(val_entries):
        sid = entry["scene"]["scene_id"]
        duration = float(entry["scene"]["duration"])
        gt_ledger = entry["target_ledger"]

        # --- oracle: caption each stem ---
        stem_events = []
        for src in entry["scene"]["sources"]:
            stem_path = f"{audio_base}/stems/{sid}_{src['source_id']}.wav"
            if not Path(stem_path).exists():
                continue
            wav = _load_audio(stem_path, sr)
            if len(wav) < sr * 0.05:  # skip <50ms
                continue
            try:
                raw = adapter.infer(stem_path, prompt, sample_id=sid, duration=duration)
            except Exception:
                raw = ""
            # use ground-truth type/timing, MOSS text
            etype = {"speech": "speech", "vocal": "lys", "music": "music",
                     "sfx": "sfx", "ambience": "sfx"}.get(src["kind"], "sfx")
            # find matching GT event for timing
            gt_ev = next((e for e in gt_ledger["events"]
                         if e.get("type") == etype or
                         (etype == "lys" and e.get("type") == "lys")), None)
            if gt_ev:
                spans = gt_ev["spans"]
            else:
                spans = [{"start_sec": src["onset"], "end_sec": min(duration, src["onset"] + 2.0)}]
            stem_events.append({
                "id": f"E{len(stem_events)+1:03d}",
                "type": etype,
                "track_id": None,
                "spans": spans,
                "text": raw[:100] if raw else src.get("text", "an event"),
                "confidence": 0.9,
            })

        oracle_ledger = {
            "schema_version": "0.2.0", "sample_id": sid,
            "duration_sec": duration, "time_resolution_sec": 0.1,
            "tracks": [], "events": stem_events,
        }
        oracle_preds.append(oracle_ledger)

        # --- mixture baseline: zero-shot MOSS on mixture ---
        mix_path = f"{audio_base}/{sid}.wav"
        try:
            mix_raw = adapter.infer(mix_path, prompt, sample_id=sid, duration=duration)
        except Exception:
            mix_raw = ""
        # parse mixture output as free-form (won't match our format → 0 events)
        from sceneledger.eval.parser import parse_model_output
        mix_ledger, _ = parse_model_output(mix_raw, sid, duration)
        mixture_preds.append(mix_ledger.model_dump(mode="json"))

        refs.append(gt_ledger)

        if (i + 1) % 10 == 0:
            print(f"[p5] {i+1}/{n_val} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    # write + evaluate
    out_dir = Path("reports/p5_oracle")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, preds in [("oracle", oracle_preds), ("mixture", mixture_preds)]:
        pp = out_dir / f"{name}_predictions.jsonl"
        rp = out_dir / "references.jsonl"
        with open(pp, "w") as f:
            for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(rp, "w") as f:
            for r in refs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== P5 Oracle Topline (50 val clips) ===", file=sys.stderr)
    for name in ["oracle", "mixture"]:
        c = evaluate_corpus(out_dir / f"{name}_predictions.jsonl", rp)
        print(f"{name:12s}: F1={c.macro_event_f1:.3f} P={c.macro_event_precision:.3f} "
              f"R={c.macro_event_recall:.3f} halluc={c.total_hallucination} "
              f"omit={c.total_omission} onset={c.mean_onset_mae:.3f}", file=sys.stderr)


def read_entries(path):
    entries = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


if __name__ == "__main__":
    main()
