"""Extract MUSDB18-HQ stems: vocals + accompaniment (drums+bass+other).
Saves to /tmp/real_audio/musdb18_stems/"""
import pandas as pd, os, numpy as np, soundfile as sf
from pathlib import Path

SRC = '/apdcephfs_fsgm3/share_303700817/yikaihuang/dataset/caption/musdb18hq_parquet/data'
OUT = Path('/tmp/real_audio/musdb18_stems')
OUT.mkdir(parents=True, exist_ok=True)

vocals_out = OUT / 'vocals'
accomp_out = OUT / 'accompaniment'
vocals_out.mkdir(exist_ok=True)
accomp_out.mkdir(exist_ok=True)

n_extracted = 0
for shard in sorted(os.listdir(SRC))[:10]:  # first 10 shards = ~130 tracks
    path = f'{SRC}/{shard}'
    try:
        df = pd.read_parquet(path)
    except:
        continue
    # Group by track (path without /stem.wav)
    tracks = {}
    for _, row in df.iterrows():
        track_name = row['path'].rsplit('/', 1)[0].replace('musdb18hq/train/', '').replace(' ', '_')
        instrument = row['instrument']
        if track_name not in tracks:
            tracks[track_name] = {}
        tracks[track_name][instrument] = row['audio']['bytes']

    for track_name, stems in tracks.items():
        if 'vocals' in stems and 'drums' in stems and 'bass' in stems and 'other' in stems:
            # Save vocals
            vocals_path = vocals_out / f'{track_name}_vocals.wav'
            if not vocals_path.exists():
                with open(vocals_path, 'wb') as f:
                    f.write(stems['vocals'])

            # Save accompaniment (drums + bass + other)
            # Read all stems as numpy arrays
            import io
            drums_wav, sr = sf.read(io.BytesIO(stems['drums']), dtype='float32')
            bass_wav, _ = sf.read(io.BytesIO(stems['bass']), dtype='float32')
            other_wav, _ = sf.read(io.BytesIO(stems['other']), dtype='float32')

            # Sum to accompaniment
            min_len = min(len(drums_wav), len(bass_wav), len(other_wav))
            accomp = drums_wav[:min_len] + bass_wav[:min_len] + other_wav[:min_len]
            if accomp.ndim == 2:
                accomp = accomp.mean(axis=1)
            accomp_path = accomp_out / f'{track_name}_accomp.wav'
            if not accomp_path.exists():
                sf.write(accomp_path, accomp, sr)

            n_extracted += 1
            if n_extracted <= 3:
                print(f'  {track_name}: vocals + accomp ({len(accomp)/sr:.1f}s)')

    if n_extracted >= 50:  # enough for now
        break

print(f'\nExtracted {n_extracted} tracks with vocals + accompaniment')
print(f'Vocals: {len(os.listdir(vocals_out))} files')
print(f'Accompaniment: {len(os.listdir(accomp_out))} files')
