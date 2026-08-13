"""Download LibriSpeech dev-clean + FMA small subset for expanded source library."""
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'
from huggingface_hub import snapshot_download

# LibriSpeech dev-clean: ~270 speakers, 5.4 hours, with transcriptions
print("=== Downloading LibriSpeech dev-clean ===")
path = snapshot_download('openslr/librispeech_asr', repo_type='dataset',
    local_dir='/tmp/real_audio/librispeech', allow_patterns=['dev-clean/*'], max_workers=4)
print(f"Downloaded to: {path}")

flacs = []
txts = []
for p,_,fs in os.walk(path):
    for f in fs:
        full = os.path.join(p, f)
        if f.endswith('.flac'):
            flacs.append(full)
        elif f.endswith('.trans.txt'):
            txts.append(full)
print(f"FLAC files: {len(flacs)}")
print(f"Trans files: {len(txts)}")
if txts:
    with open(txts[0]) as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(f"  {line.strip()[:80]}")

# FMA small: 8000 music tracks, diverse genres
print("\n=== Downloading FMA small ===")
try:
    fma_path = snapshot_download('calixtemayoraz/FMA-music-dataset', repo_type='dataset',
        local_dir='/tmp/real_audio/fma', max_workers=4)
    mp3s = []
    for p,_,fs in os.walk(fma_path):
        for f in fs:
            if f.endswith('.mp3'):
                mp3s.append(os.path.join(p, f))
    print(f"MP3 files: {len(mp3s)}")
except Exception as e:
    print(f"FMA failed: {e}")

# UrbanSound8K via HF
print("\n=== Downloading UrbanSound8K ===")
try:
    us8k_path = snapshot_download('danavery/urbansound8K', repo_type='dataset',
        local_dir='/tmp/real_audio/urbansound8k', max_workers=4)
    wavs = []
    for p,_,fs in os.walk(us8k_path):
        for f in fs:
            if f.endswith('.wav'):
                wavs.append(os.path.join(p, f))
    print(f"WAV files: {len(wavs)}")
except Exception as e:
    print(f"UrbanSound8K failed: {e}")
