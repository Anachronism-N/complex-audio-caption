"""Run MOSS zero-shot on 10 real ESC-50 clips to validate real audio capability."""
import sys, json, random, os
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
import soundfile as sf

adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device="cuda:0", dtype="bfloat16"))
adapter._load()

# Load ESC-50 category map
with open("/tmp/real_audio/esc50_category_map.json") as f:
    cat_map = json.load(f)

# Select 10 diverse samples
all_files = list(cat_map.keys())
rng = random.Random(42)
rng.shuffle(all_files)
selected = all_files[:10]

prompt = "Describe every audible event in this audio clip."

print("=== MOSS zero-shot on REAL ESC-50 audio ===\n")
for fname in selected:
    cat = cat_map[fname]
    path = f"/tmp/real_audio/esc50_wav/{fname}"
    wav, sr = sf.read(path)
    dur = len(wav) / sr
    try:
        out = adapter.infer(path, prompt, sample_id=fname, duration=dur)
    except Exception as e:
        out = f"ERROR: {e}"
    print(f"--- {fname} (category={cat}, {dur:.1f}s) ---")
    print(out[:300])
    print()
