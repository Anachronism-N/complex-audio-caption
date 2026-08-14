"""Select instrumental GTZAN clips (classical, jazz) as music sources + build v6 mix."""
import sys, json, random, time, os
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve, lfilter, firwin
from pathlib import Path
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig

# GTZAN instrumental genres (mostly no vocals)
MUSIC_SOURCES = []
for genre in ['classical', 'jazz', 'blues']:
    gdir = f'/tmp/real_audio/gtzan/genres/{genre}'
    if os.path.exists(gdir):
        for f in os.listdir(gdir):
            if f.endswith('.wav') and not f.startswith('._'):
                MUSIC_SOURCES.append((f'{gdir}/{f}', f'{genre} music'))
print(f"Music sources (GTZAN instrumental): {len(MUSIC_SOURCES)}")

SPEECH_SOURCES = [
    ("third_party/MOSS-Audio/demo/assets/audio/test_en.mp3", "English speech"),
    ("third_party/MOSS-Audio/demo/assets/audio/faker_and_chovy.mp3", "Korean speech"),
]

# Scene templates with per-type gain
SCENE_TEMPLATES = [
    {"name": "coffee_shop", "desc": "A busy coffee shop",
     "sources": [
         {"role": "music", "type": "music"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["can_opening", "washing_machine"]},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clapping", "laughing"]},
     ]},
    {"name": "street_traffic", "desc": "A busy street",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["engine", "car_horn"]},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["siren"]},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["footsteps"]},
     ]},
    {"name": "park_afternoon", "desc": "A park afternoon",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["chirping_birds", "crickets"]},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["dog", "cat"]},
         {"role": "ambience", "type": "sfx", "esc50_cats": ["wind"]},
     ]},
    {"name": "concert_outdoor", "desc": "Outdoor concert",
     "sources": [
         {"role": "music", "type": "music"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clapping", "laughing"]},
         {"role": "ambience", "type": "sfx", "esc50_cats": ["wind"]},
     ]},
    {"name": "speech_with_music", "desc": "Someone talking with background music",
     "sources": [
         {"role": "speech", "type": "speech"},
         {"role": "music", "type": "music"},
     ]},
    {"name": "speech_with_sfx", "desc": "Someone talking with sound effects",
     "sources": [
         {"role": "speech", "type": "speech"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["door_wood_knock", "clapping", "glass_breaking"]},
     ]},
    {"name": "restaurant_busy", "desc": "A busy restaurant",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["washing_machine", "vacuum_cleaner"]},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clapping", "laughing", "coughing"]},
         {"role": "music", "type": "music"},
     ]},
]

def load_audio(path, sr):
    wav, sr = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)

def apply_fade(wav, sr, fade_s=0.1):
    fade = int(fade_s * sr)
    if len(wav) > 2 * fade:
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
    return wav

def apply_compression(wav, sr, threshold=-20, ratio=3.0):
    attack = max(1, int(0.005 * sr))
    env = np.abs(wav)
    env = lfilter(np.ones(attack)/attack, [1.0], env)
    gain = np.ones_like(wav)
    above = env > 10**(threshold/20)
    gain[above] = 10**(threshold/20) / (env[above] + 1e-9)
    gain = np.clip(gain, 0, 1)
    release = max(1, int(0.05 * sr))
    gain = lfilter(np.ones(release)/release, [1.0], gain)
    return wav * gain

def apply_vocal_eq(wav, sr, boost_db=2.0):
    try:
        low = firwin(65, 1000 / (sr / 2))
        high = firwin(65, 3000 / (sr / 2))
        mid = lfilter(high, 1.0, wav) - lfilter(low, 1.0, wav)
        return wav + mid * (10**(boost_db/20) - 1)
    except: return wav

def synth_rir(sr, t60, seed):
    rng = np.random.default_rng(seed)
    length = max(1, int(sr * min(t60 * 3, 1.0)))
    noise = rng.standard_normal(length)
    t = np.arange(length) / sr
    decay = np.exp(-6.9 * t / max(t60, 0.05))
    rir = noise * decay
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir**2) + 1e-12)
    return rir.astype(np.float32)

def main():
    device = "cuda:0"
    adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device=device, dtype="bfloat16"))
    adapter._load()
    sr = 16000

    with open("/tmp/real_audio/esc50_category_map.json") as f:
        cat_map = json.load(f)
    cat_files = {}
    for fname, cat in cat_map.items():
        cat_files.setdefault(cat, []).append(fname)
    # Add UrbanSound8K
    US8K_MAP = {'dog_bark':'dog','children_playing':'laughing','car_horn':'car_horn',
                'air_conditioner':'engine','street_music':'clapping','drilling':'chainsaw',
                'siren':'siren','jackhammer':'chainsaw','engine_idling':'engine','gun_shot':'fireworks'}
    if os.path.exists('/tmp/real_audio/urbansound8k_wav'):
        for f in os.listdir('/tmp/real_audio/urbansound8k_wav'):
            if f.endswith('.wav'):
                cls = f.rsplit('_', 1)[0]
                mapped = US8K_MAP.get(cls, cls)
                cat_files.setdefault(mapped, []).append(f'/tmp/real_audio/urbansound8k_wav/{f}')

    print(f"Music: {len(MUSIC_SOURCES)}, Speech: {len(SPEECH_SOURCES)}, SFX cats: {len(cat_files)}", file=sys.stderr, flush=True)

    n_mixtures = 200
    out_dir = Path("/tmp/real_mix_v6")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260814)
    prompt = "Describe this audio in one sentence."
    t0 = time.time()
    manifest = []

    for i in range(n_mixtures):
        scene = rng.choice(SCENE_TEMPLATES)
        sid = f"rv6_{i+1:04d}"
        duration = rng.uniform(8.0, 12.0)
        n_clip = int(duration * sr)
        has_speech = any(s["type"] == "speech" for s in scene["sources"])
        duck_others = has_speech and rng.random() < 0.8
        mixture = np.zeros(n_clip, dtype=np.float32)
        events = []
        source_wavs = []
        speech_mask = np.zeros(n_clip, dtype=np.float32)
        sources_info = []

        for sc_idx, src_spec in enumerate(scene["sources"]):
            src_type = src_spec["type"]
            onset = rng.uniform(0, max(0.5, duration - 5))
            if src_type == "speech":
                gain_db = rng.uniform(0, 6)
            elif src_type == "music":
                gain_db = rng.uniform(-6, 0)  # music louder than before
            else:
                gain_db = rng.uniform(-9, -3)

            if src_type == "music":
                music_path, music_desc = rng.choice(MUSIC_SOURCES)
                wav = load_audio(music_path, sr)
                if len(wav) > n_clip:
                    start = rng.randint(0, len(wav) - n_clip)
                    wav = wav[start:start+n_clip]
                wav = apply_fade(wav, sr, 0.5)
            elif src_type == "speech":
                speech_path, speech_desc = rng.choice(SPEECH_SOURCES)
                wav = load_audio(speech_path, sr)
                if len(wav) > n_clip // 2:
                    start = rng.randint(0, len(wav) - n_clip // 2)
                    wav = wav[start:start + n_clip // 2]
                wav = apply_fade(wav, sr, 0.05)
                wav = apply_compression(wav, sr)
                wav = apply_vocal_eq(wav, sr)
            else:
                cats = src_spec.get("esc50_cats", ["dog"])
                cat = rng.choice(cats)
                fname = rng.choice(cat_files.get(cat, cat_files.get("dog")))
                path = f"/tmp/real_audio/esc50_wav/{fname}"
                if not os.path.exists(path):
                    path = fname
                wav = load_audio(path, sr)
                wav = apply_fade(wav, sr, 0.05)

            # MOSS caption this actual source
            try:
                raw_path = f"/tmp/rv6_src_{sid}_{sc_idx}.wav"
                sf.write(raw_path, wav, sr)
                caption = adapter.infer(raw_path, prompt, sample_id=f"{sid}_s{sc_idx}", duration=len(wav)/sr)
                os.unlink(raw_path)
            except:
                caption = src_spec.get("desc", src_type)

            wav = wav * (10 ** (gain_db / 20))
            if rng.random() < 0.4:
                t60 = rng.uniform(0.2, 0.8)
                rir = synth_rir(sr, t60, seed=i*100+sc_idx)
                wav = fftconvolve(wav, rir, mode="full")[:len(wav)].astype(np.float32)

            src_rms = float(np.sqrt(np.mean(wav**2)) + 1e-12)
            source_wavs.append({"wav": wav, "onset": onset, "type": src_type, "rms": src_rms})
            if src_type == "speech":
                start_s = int(onset * sr)
                end_s = min(n_clip, start_s + len(wav))
                frame_size = int(0.05 * sr)
                for fi in range(start_s, end_s - frame_size, frame_size):
                    rms = np.sqrt(np.mean(wav[fi-start_s:fi-start_s+frame_size]**2))
                    if rms > 0.01:
                        speech_mask[fi:fi+frame_size] = 1.0
            onset_sec = round(onset, 1)
            offset_sec = round(min(duration, onset + len(wav)/sr), 1)
            events.append({"id": f"E{sc_idx+1:03d}", "type": src_type, "track_id": f"T{sc_idx+1}",
                            "spans": [{"start_sec": onset_sec, "end_sec": offset_sec}],
                            "text": caption[:200], "confidence": 0.85})
            sources_info.append({"role": src_spec["role"], "type": src_type,
                                 "onset": onset_sec, "offset": offset_sec,
                                 "caption": caption[:200], "src_rms": src_rms, "gain_db": gain_db})

        # Ducking
        if duck_others and speech_mask.any():
            smooth = lfilter(np.ones(10)/10, [1.0], speech_mask)
            speech_mask_s = np.clip(smooth, 0, 1)
            duck_depth = rng.uniform(6, 8)
            duck_factor = 10 ** (-duck_depth / 20)
            duck_gain = 1.0 - (1.0 - duck_factor) * speech_mask_s
        else:
            duck_gain = np.ones(n_clip, dtype=np.float32)

        # Mix
        mixture = np.zeros(n_clip, dtype=np.float32)
        for sw in source_wavs:
            start = int(sw["onset"] * sr)
            wav = sw["wav"]
            end = min(n_clip, start + len(wav))
            if start < n_clip:
                seg = wav[:end-start]
                if sw["type"] != "speech" and duck_others:
                    seg = seg * duck_gain[start:start+len(seg)]
                mixture[start:end] += seg

        # RMS verify
        for si, sw in zip(sources_info, source_wavs):
            si["present_in_mix"] = sw["rms"] > 0.001

        peak = np.max(np.abs(mixture)) + 1e-9
        if peak > 0.99:
            mixture *= 0.99 / peak
        sf.write(str(audio_dir / f"{sid}.wav"), mixture, sr)
        ledger = {"schema_version":"0.2.0","sample_id":sid,"duration_sec":duration,
                  "time_resolution_sec":0.1,
                  "tracks":[{"id":f"T{j+1}","kind":e["type"],"spans":e["spans"],"confidence":0.85} for j,e in enumerate(events)],
                  "events":events,
                  "provenance":{"label_level":"model_prediction","source_dataset":"esc50+gtzan","license_status":"CC"}}
        manifest.append({"scene_id":sid,"scene_name":scene["name"],"scene_desc":scene["desc"],
                         "audio_path":f"audio/{sid}.wav","duration":duration,"ledger":ledger,"sources":sources_info})
        if (i+1) % 50 == 0:
            print(f"[rv6] {i+1}/{n_mixtures} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Review CSV: 10 music/speech clips
    import csv
    ms = [m for m in manifest if any(s["type"] in ("music","speech") for s in m["sources"])]
    review_10 = rng.sample(ms, min(10, len(ms)))
    with open(out_dir / "review_10.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip_id","scene_name","audio_path","duration","n_sources",
                     "source_1_role","source_1_type","source_1_onset","source_1_offset","source_1_caption","source_1_present",
                     "source_2_role","source_2_type","source_2_onset","source_2_offset","source_2_caption","source_2_present",
                     "source_3_role","source_3_type","source_3_onset","source_3_offset","source_3_caption","source_3_present",
                     "audio_natural","caption_accurate","notes"])
        for m in review_10:
            row = [m["scene_id"],m["scene_name"],m["audio_path"],f'{m["duration"]:.1f}',len(m["sources"])]
            for j in range(3):
                if j < len(m["sources"]):
                    s = m["sources"][j]
                    row.extend([s["role"],s["type"],f'{s["onset"]:.1f}s',f'{s["offset"]:.1f}s',s["caption"][:100],s.get("present_in_mix","?")])
                else:
                    row.extend([""]*6)
            row.extend(["","",""])
            w.writerow(row)
    print(f"\nWrote {n_mixtures} to {out_dir}, {len(ms)} with music/speech")
    print(f"Review: {out_dir / 'review_10.csv'}")

if __name__ == "__main__":
    main()
