"""P5 predicted-stems topline: Demucs separation → MOSS caption per stem.

Completes docs/11 §8 condition 3: "predicted stems → expert caption".
Compares with oracle stems (condition 2) and mixture (condition 1).
"""
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
import torch, numpy as np
import soundfile as sf
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
from sceneledger.data.manifests import read_manifest
from sceneledger.data.schema import Ledger
from sceneledger.data.activity import compute_activity
from sceneledger.eval.metrics import evaluate_corpus

device = "cuda:0"

def load_mono(path, sr_target=24000):
    from scipy.signal import resample_poly
    from math import gcd
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2: wav = wav.mean(axis=1)
    if sr != sr_target:
        g = gcd(int(sr), int(sr_target))
        wav = resample_poly(wav.astype(np.float64), int(sr_target)//g, int(sr)//g).astype(np.float32)
    return wav

def main():
    # load Demucs
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    print("[p5-pred] loading Demucs ...", file=sys.stderr, flush=True)
    demucs = get_model("htdemucs").to(device)
    demucs.eval()
    demucs_sources = demucs.sources  # ['drums', 'bass', 'other', 'vocals']
    print(f"[p5-pred] Demucs sources: {demucs_sources}", file=sys.stderr, flush=True)

    # load MOSS
    print("[p5-pred] loading MOSS ...", file=sys.stderr, flush=True)
    adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device=device, dtype="bfloat16"))
    adapter._load()
    processor = adapter._processor
    sr_moss = processor.config.mel_sr

    entries = read_manifest("data/derived/b3_5k/manifest.jsonl")
    n_val = 20
    val_entries = entries[-n_val:]
    audio_base = "/tmp/b3_5k/audio"
    prompt = "Describe this audio in one sentence."

    # stem → event type mapping
    stem_type_map = {"vocals": "speech", "other": "music", "drums": "sfx", "bass": "music"}

    preds = []
    refs = []
    t0 = time.time()

    for i, entry in enumerate(val_entries):
        sid = entry.scene["scene_id"]
        duration = float(entry.scene["duration"])
        gt_ledger = entry.target_ledger

        # 1. Demucs separation
        mix_path = f"{audio_base}/{sid}.wav"
        wav = load_mono(mix_path)
        wav_stereo = np.stack([wav, wav])
        wav_t = torch.from_numpy(wav_stereo).unsqueeze(0).to(device)
        with torch.no_grad():
            stems = apply_model(demucs, wav_t, split=True, overlap=0.25)

        # 2. For each stem: compute activity + MOSS caption
        events = []
        for j, sname in enumerate(demucs_sources):
            stem_audio = stems[0, j].cpu().numpy().mean(axis=0)  # mono
            rms = float(np.sqrt(np.mean(stem_audio**2)))
            if rms < 0.005:  # skip near-silent stems
                continue

            # compute activity (onset/offset)
            act = compute_activity(stem_audio.astype(np.float32), 24000,
                                   activity_threshold=0.1, resolution_sec=0.1,
                                   merge_threshold_s=0.3, duration_sec=duration,
                                   is_continuous=(sname in ("other", "bass")))
            if not act.spans:
                continue

            # MOSS caption on this stem
            stem_path = f"/tmp/p5pred_{sid}_{sname}.wav"
            sf.write(stem_path, stem_audio, 24000)
            try:
                raw = adapter.infer(stem_path, prompt, sample_id=f"{sid}_{sname}", duration=duration)
            except Exception:
                raw = ""
            Path(stem_path).unlink(missing_ok=True)

            etype = stem_type_map.get(sname, "sfx")
            for k, (s, e) in enumerate(act.spans[:3]):  # max 3 spans per stem
                events.append({
                    "id": f"E{len(events)+1:03d}",
                    "type": etype,
                    "track_id": None,
                    "spans": [{"start_sec": round(s, 1), "end_sec": round(e, 1)}],
                    "text": raw[:80] if raw else f"demucs {sname}",
                    "confidence": 0.7,
                })

        pred_ledger = {
            "schema_version": "0.2.0", "sample_id": sid,
            "duration_sec": duration, "time_resolution_sec": 0.1,
            "tracks": [], "events": events,
        }
        preds.append(pred_ledger)
        refs.append(gt_ledger)

        if (i + 1) % 5 == 0:
            print(f"[p5-pred] {i+1}/{n_val} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    # evaluate
    out_dir = Path("reports/p5_predicted")
    out_dir.mkdir(parents=True, exist_ok=True)
    pp, rp = out_dir / "predictions.jsonl", out_dir / "references.jsonl"
    with open(pp, "w") as f:
        for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(rp, "w") as f:
        for r in refs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = evaluate_corpus(pp, rp)
    print(f"\n=== P5 Predicted-Stems Topline ({n_val} clips) ===", file=sys.stderr)
    print(f"event-F1:   {c.macro_event_f1:.3f}", file=sys.stderr)
    print(f"precision:  {c.macro_event_precision:.3f}", file=sys.stderr)
    print(f"recall:     {c.macro_event_recall:.3f}", file=sys.stderr)
    print(f"onset-MAE:  {c.mean_onset_mae:.3f}s", file=sys.stderr)
    print(f"offset-MAE: {c.mean_offset_mae:.3f}s", file=sys.stderr)
    print(f"halluc:     {c.total_hallucination}", file=sys.stderr)
    print(f"omit:       {c.total_omission}", file=sys.stderr)
    print(f"per-type:   {json.dumps(c.per_type, ensure_ascii=False)}", file=sys.stderr)
    (out_dir / "metrics.json").write_text(json.dumps(c.to_dict(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
