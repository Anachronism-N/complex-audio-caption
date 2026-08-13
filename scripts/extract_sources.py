"""Extract audio from FMA + UrbanSound8K parquet files."""
import pandas as pd, os, soundfile as sf

# Extract UrbanSound8K (10 classes: dog_bark, engine_idling, etc.)
print("=== Extracting UrbanSound8K ===")
us8k_out = '/tmp/real_audio/urbansound8k_wav'
os.makedirs(us8k_out, exist_ok=True)
us8k_classes = {}
for shard in sorted(os.listdir('/tmp/real_audio/urbansound8k/data/'))[:4]:  # first 4 shards
    path = f'/tmp/real_audio/urbansound8k/data/{shard}'
    if not os.path.exists(path): continue
    try:
        df = pd.read_parquet(path)
        print(f"  {shard}: {len(df)} rows, cols: {df.columns.tolist()}")
        if 'audio' in df.columns and 'class' in df.columns:
            for _, row in df.iterrows():
                cls = row['class']
                us8k_classes.setdefault(cls, 0)
                if us8k_classes[cls] < 20:  # 20 per class
                    audio_bytes = row['audio']['bytes'] if isinstance(row['audio'], dict) else row['audio']
                    fname = f"{cls}_{us8k_classes[cls]:04d}.wav"
                    with open(f"{us8k_out}/{fname}", 'wb') as f:
                        f.write(audio_bytes)
                    us8k_classes[cls] += 1
    except Exception as e:
        print(f"  {shard} failed: {e}")
        continue
    break  # just first shard for now
print(f"UrbanSound8K extracted: {sum(us8k_classes.values())} files, {len(us8k_classes)} classes")
print(f"Classes: {dict(list(us8k_classes.items())[:5])}")

# Extract FMA
print("\n=== Extracting FMA ===")
fma_out = '/tmp/real_audio/fma_mp3'
os.makedirs(fma_out, exist_ok=True)
for shard in sorted(os.listdir('/tmp/real_audio/fma/data/'))[:2]:
    path = f'/tmp/real_audio/fma/data/{shard}'
    if not os.path.exists(path): continue
    try:
        df = pd.read_parquet(path)
        print(f"  {shard}: {len(df)} rows, cols: {df.columns.tolist()}")
        # Check for audio column
        for col in df.columns:
            if 'audio' in col.lower():
                print(f"    found audio column: {col}")
        break
    except Exception as e:
        print(f"  {shard} failed: {e}")
        continue

# Check LibriSpeech structure
print("\n=== LibriSpeech ===")
for root, dirs, files in os.walk('/tmp/real_audio/librispeech'):
    if files:
        print(f"  {root}: {files[:3]}")
        break
