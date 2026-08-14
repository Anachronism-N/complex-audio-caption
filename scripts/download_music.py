"""Download MUSDB18-HQ accompaniment stems (pure instrumental, no vocals)."""
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'
from huggingface_hub import snapshot_download

# MUSDB18: 150 tracks, each with drums/bass/other/vocals stems
# 'other' = accompaniment without vocals = pure instrumental
print("=== Downloading MUSDB18 ===")
try:
    path = snapshot_download('salu133445/musdb18', repo_type='dataset',
        local_dir='/tmp/real_audio/musdb18', max_workers=4)
    print(f"Downloaded to: {path}")
    # Count files
    wavs = []
    for p,_,fs in os.walk(path):
        for f in fs:
            if f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.stem'):
                wavs.append(os.path.join(p, f))
    print(f"Audio files: {len(wavs)}")
    if wavs:
        for w in wavs[:5]:
            print(f"  {w}")
except Exception as e:
    print(f"MUSDB18 failed: {e}")

# Try alternative: GTZAN (music genre classification, instrumental+voice)
print("\n=== Trying GTZAN ===")
try:
    path = snapshot_download('marsyas/gtzan', repo_type='dataset',
        local_dir='/tmp/real_audio/gtzan', max_workers=4)
    wavs = []
    for p,_,fs in os.walk(path):
        for f in fs:
            if f.endswith('.wav') or f.endswith('.au'):
                wavs.append(os.path.join(p, f))
    print(f"GTZAN files: {len(wavs)}")
except Exception as e:
    print(f"GTZAN failed: {e}")
