"""Build real-audio mixtures v10: LLM-assisted + new datasets.

Uses:
- LibriSpeech (251 speakers, with transcriptions) for speech
- MUSDB18 accompaniment (pure instrumental) for music
- MUSDB18 vocals for vocal/lyrics
- ESC-50 + UrbanSound8K for sfx/ambience
- MOSS-Audio as LLM for scene generation + per-source captioning
"""
import sys, json, random, time, os, io
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter, firwin, resample_poly
from math import gcd
from pathlib import Path
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig

# Data sources
LIBRISPEECH_DIR = "data/sources/librispeech/train-clean-100"
MUSDB18_ACCOMP_DIR = "data/sources/musdb18_stems/accompaniment"
MUSDB18_VOCALS_DIR = "data/sources/musdb18_stems/vocals"
ESC50_DIR = "data/sources/esc50"
GTZAN_DIR = "data/sources/gtzan/genres"

def load_audio(path, sr):
    if path.endswith('.flac'):
        wav, sr_orig = sf.read(path, dtype="float32", always_2d=False)
    elif path.endswith('.mp3') or path.endswith('.m4a'):
        import subprocess, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        subprocess.run(['ffmpeg', '-i', path, '-f', 'wav', '-ar', str(sr),
                       '-ac', '1', '-y', tmp.name], capture_output=True, check=True)
        wav, sr_orig = sf.read(tmp.name, dtype="float32", always_2d=False)
        os.unlink(tmp.name)
    else:
        try:
            wav, sr_orig = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            import subprocess, tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            r = subprocess.run(['ffmpeg', '-i', path, '-f', 'wav', '-ar', str(sr),
                               '-ac', '1', '-y', tmp.name], capture_output=True)
            if r.returncode != 0: raise RuntimeError(f"Cannot load {path}")
            wav, sr_orig = sf.read(tmp.name, dtype="float32", always_2d=False)
            os.unlink(tmp.name)
    if wav.ndim == 2: wav = wav.mean(axis=1)
    if sr_orig != sr:
        g = gcd(int(sr_orig), int(sr))
        wav = resample_poly(wav.astype(np.float64), int(sr)//g, int(sr_orig)//g).astype(np.float32)
    return wav


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


def compute_rms_activity(wav, sr, threshold=0.005, frame_ms=50):
    frame_size = int(frame_ms / 1000 * sr)
    n_frames = len(wav) // frame_size
    activity = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        frame = wav[i*frame_size:(i+1)*frame_size]
        rms = np.sqrt(np.mean(frame**2))
        activity[i] = rms > threshold
    spans = []
    in_span = False
    start = 0
    for i in range(n_frames):
        if activity[i] and not in_span:
            start = i; in_span = True
        elif not activity[i] and in_span:
            spans.append((start * frame_size / sr, i * frame_size / sr)); in_span = False
    if in_span:
        spans.append((start * frame_size / sr, n_frames * frame_size / sr))
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] < 0.1:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged if merged else [(0, len(wav)/sr)]


def load_librispeech_sources():
    """Load LibriSpeech FLAC files + transcriptions."""
    sources = []
    if not os.path.exists(LIBRISPEECH_DIR):
        return sources
    for speaker in os.listdir(LIBRISPEECH_DIR)[:50]:  # 50 speakers
        spk_dir = f"{LIBRISPEECH_DIR}/{speaker}"
        if not os.path.isdir(spk_dir): continue
        for chapter in os.listdir(spk_dir):
            ch_dir = f"{spk_dir}/{chapter}"
            if not os.path.isdir(ch_dir): continue
            # Load transcriptions
            trans_file = f"{ch_dir}/{speaker}-{chapter}.trans.txt"
            trans = {}
            if os.path.exists(trans_file):
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            trans[parts[0]] = parts[1]
            # Find FLAC files
            for flac in os.listdir(ch_dir):
                if flac.endswith('.flac'):
                    uid = flac.replace('.flac', '')
                    text = trans.get(uid, "")
                    sources.append((f"{ch_dir}/{flac}", f"English speech: {text[:60]}"))
                    if len(sources) >= 200:  # limit
                        return sources
    return sources


def load_musdb18_sources():
    """Load MUSDB18 accompaniment (pure instrumental) + vocals."""
    accomp = []
    vocals = []
    if os.path.exists(MUSDB18_ACCOMP_DIR):
        for f in os.listdir(MUSDB18_ACCOMP_DIR):
            if f.endswith('.wav'):
                accomp.append((f"{MUSDB18_ACCOMP_DIR}/{f}", f"instrumental music from {f.split('_')[0]}"))
    if os.path.exists(MUSDB18_VOCALS_DIR):
        for f in os.listdir(MUSDB18_VOCALS_DIR):
            if f.endswith('.wav'):
                vocals.append((f"{MUSDB18_VOCALS_DIR}/{f}", f"vocals from {f.split('_')[0]}"))
    return accomp, vocals


def load_esc50_sources():
    """Load ESC-50 + UrbanSound8K."""
    with open("/tmp/real_audio/esc50_category_map.json") as f:
        cat_map = json.load(f)
    cat_files = {}
    for fname, cat in cat_map.items():
        cat_files.setdefault(cat, []).append(f"{ESC50_DIR}/{fname}")
    # UrbanSound8K
    US8K_MAP = {'dog_bark':'dog','children_playing':'laughing','car_horn':'car_horn',
                'air_conditioner':'engine','street_music':'clapping','drilling':'chainsaw',
                'siren':'siren','jackhammer':'chainsaw','engine_idling':'engine','gun_shot':'fireworks'}
    if os.path.exists('data/sources/urbansound8k'):
        for f in os.listdir('data/sources/urbansound8k'):
            if f.endswith('.wav'):
                cls = f.rsplit('_', 1)[0]
                mapped = US8K_MAP.get(cls, cls)
                cat_files.setdefault(mapped, []).append(f"data/sources/urbansound8k/{f}")
    return cat_files


def main():
    device = "cuda:0"
    adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device=device, dtype="bfloat16"))
    adapter._load()
    sr = 16000
    processor = adapter._processor

    # Load all sources
    speech_sources = load_librispeech_sources()
    music_sources, vocal_sources = load_musdb18_sources()
    # Add GTZAN as backup music
    for genre in ['classical', 'jazz', 'blues']:
        gdir = f"{GTZAN_DIR}/{genre}"
        if os.path.exists(gdir):
            for f in os.listdir(gdir):
                if f.endswith('.wav') and not f.startswith('._'):
                    try:
                        sf.read(f'{gdir}/{f}', frames=10)
                        music_sources.append((f'{gdir}/{f}', f'{genre} music'))
                    except: pass
    cat_files = load_esc50_sources()

    print(f"Speech (LibriSpeech): {len(speech_sources)}", file=sys.stderr, flush=True)
    print(f"Music (MUSDB18+GTZAN): {len(music_sources)}", file=sys.stderr, flush=True)
    print(f"Vocals (MUSDB18): {len(vocal_sources)}", file=sys.stderr, flush=True)
    print(f"SFX cats: {len(cat_files)} ({sum(len(v) for v in cat_files.values())} files)", file=sys.stderr, flush=True)

    available_cats = sorted(cat_files.keys())

    # LLM prompt for scene generation
    scene_prompt = f"""You are an audio scene designer. Generate a realistic audio scene.

Available: speech(English), music(instrumental), vocals(singing), sfx({', '.join(available_cats[:20])}).

Output JSON:
{{"scene_name":"short","description":"1 sentence","duration_s":8-12,"sources":[{{"role":"speech|music|sfx|ambience|vocal","category":"ESC-50 cat or genre","gain_db":-12 to 6,"onset_s":0 to duration-3,"duck_others":true if speech}}]}}

Rules: 2-4 sources, logically related, speech gain>sfx>music. Output ONLY JSON."""

    caption_prompt = "Describe this audio in one sentence."
    n_mixtures = 1000
    out_dir = Path("/tmp/real_mix_v10")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260817)
    t0 = time.time()
    manifest = []

    for i in range(n_mixtures):
        sid = f"rv10_{i+1:04d}"

        # LLM generates scene
        try:
            scene_response = adapter._model.generate(
                **processor(text=scene_prompt, return_tensors="pt").to(device),
                max_new_tokens=300, do_sample=True, temperature=0.9,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            scene_text = processor.decode(scene_response[0], skip_special_tokens=True)
            import re
            json_match = re.search(r'\{.*\}', scene_text, re.DOTALL)
            scene_config = json.loads(json_match.group()) if json_match else {}
        except:
            scene_config = {}

        if not scene_config.get("sources"):
            # Fallback: random scene WITH speech/music/vocal diversity
            roles = ["speech", "music", "sfx", "sfx", "vocal", "sfx", "speech", "music"]
            n_src = rng.randint(2, 4)
            fb_sources = []
            for r in range(n_src):
                role = rng.choice(roles)
                if role == "speech" and speech_sources:
                    fb_sources.append({"role":"speech","category":"english","gain_db":rng.uniform(0,6),
                                       "onset_s":rng.uniform(0,duration-3),"duck_others":True})
                elif role == "music" and music_sources:
                    fb_sources.append({"role":"music","category":"instrumental","gain_db":rng.uniform(-6,0),
                                       "onset_s":rng.uniform(0,duration-3),"duck_others":False})
                elif role == "vocal" and vocal_sources:
                    fb_sources.append({"role":"vocal","category":"singing","gain_db":rng.uniform(-3,3),
                                       "onset_s":rng.uniform(0,duration-3),"duck_others":False})
                else:
                    fb_sources.append({"role":"sfx","category":rng.choice(available_cats),
                                       "gain_db":rng.uniform(-9,-3),"onset_s":rng.uniform(0,duration-3),"duck_others":False})
            scene_config = {"scene_name":"random","description":"A mixed scene","duration_s":duration,"sources":fb_sources}

        duration = float(np.clip(float(scene_config.get("duration_s", 10.0)), 8.0, 12.0))
        n_clip = int(duration * sr)
        mixture = np.zeros(n_clip, dtype=np.float32)
        events = []
        source_wavs = []
        speech_mask = np.zeros(n_clip, dtype=np.float32)
        sources_info = []
        has_speech = False

        for sc_idx, src_spec in enumerate(scene_config.get("sources", [])):
            role = src_spec.get("role", "sfx")
            category = src_spec.get("category", "dog")
            gain_db = float(src_spec.get("gain_db", -6))
            onset = float(src_spec.get("onset_s", rng.uniform(0, duration-3)))

            # Load source based on role
            if role == "music":
                if music_sources:
                    music_path, music_desc = rng.choice(music_sources)
                    wav = load_audio(music_path, sr)
                    if len(wav) > n_clip:
                        start = rng.randint(0, len(wav) - n_clip)
                        wav = wav[start:start+n_clip]
                    wav = apply_fade(wav, sr, 0.5)
                else: continue
                src_type = "music"
            elif role == "speech":
                if speech_sources:
                    speech_path, speech_desc = rng.choice(speech_sources)
                    wav = load_audio(speech_path, sr)
                    if len(wav) > n_clip // 2:
                        start = rng.randint(0, len(wav) - n_clip // 2)
                        wav = wav[start:start + n_clip // 2]
                    wav = apply_fade(wav, sr, 0.05)
                    wav = apply_compression(wav, sr)
                    wav = apply_vocal_eq(wav, sr)
                    has_speech = True
                else: continue
                src_type = "speech"
            elif role == "vocal":
                if vocal_sources:
                    vocal_path, vocal_desc = rng.choice(vocal_sources)
                    wav = load_audio(vocal_path, sr)
                    if len(wav) > n_clip // 2:
                        start = rng.randint(0, len(wav) - n_clip // 2)
                        wav = wav[start:start + n_clip // 2]
                    wav = apply_fade(wav, sr, 0.05)
                else: continue
                src_type = "vocal"
            else:
                cat = category if category in cat_files else rng.choice(available_cats)
                fname = rng.choice(cat_files.get(cat, cat_files.get("dog")))
                wav = load_audio(fname, sr)
                wav = apply_fade(wav, sr, 0.05)
                src_type = "sfx"

            # MOSS caption
            try:
                raw_path = f"/tmp/rv10_src_{sid}_{sc_idx}.wav"
                sf.write(raw_path, wav, sr)
                caption = adapter.infer(raw_path, caption_prompt, sample_id=f"{sid}_s{sc_idx}", duration=len(wav)/sr)
                os.unlink(raw_path)
            except:
                caption = f"An audible {role} sound."

            wav = wav * (10 ** (gain_db / 20))
            if rng.random() < 0.4:
                t60 = rng.uniform(0.2, 0.8)
                rir = synth_rir(sr, t60, seed=i*100+sc_idx)
                wav = fftconvolve(wav, rir, mode="full")[:len(wav)].astype(np.float32)

            activity_spans = compute_rms_activity(wav, sr, threshold=0.005)
            actual_onset = onset + activity_spans[0][0]
            actual_offset = min(duration, onset + activity_spans[-1][1])
            if actual_offset <= actual_onset:
                actual_offset = min(duration, actual_onset + 0.1)

            source_wavs.append({"wav": wav, "onset": onset, "type": src_type})
            if src_type == "speech":
                start_s = int(onset * sr)
                end_s = min(n_clip, start_s + len(wav))
                frame_size = int(0.05 * sr)
                for fi in range(start_s, end_s - frame_size, frame_size):
                    rms = np.sqrt(np.mean(wav[fi-start_s:fi-start_s+frame_size]**2))
                    if rms > 0.01:
                        speech_mask[fi:fi+frame_size] = 1.0

            events.append({"id": f"E{sc_idx+1:03d}", "type": src_type, "track_id": f"T{sc_idx+1}",
                            "spans": [{"start_sec": round(actual_onset, 1), "end_sec": round(actual_offset, 1)}],
                            "text": caption[:200], "confidence": 0.85})
            sources_info.append({"role": role, "type": src_type, "onset": round(actual_onset, 1),
                                 "offset": round(actual_offset, 1), "caption": caption[:200],
                                 "gain_db": gain_db, "category": category, "placement_onset": round(onset, 1)})

        # Ducking
        duck_others = has_speech and rng.random() < 0.8
        if duck_others and speech_mask.any():
            smooth = lfilter(np.ones(10)/10, [1.0], speech_mask)
            speech_mask_s = np.clip(smooth, 0, 1)
            duck_depth = rng.uniform(6, 8)
            duck_factor = 10 ** (-duck_depth / 20)
            duck_gain = 1.0 - (1.0 - duck_factor) * speech_mask_s
        else:
            duck_gain = np.ones(n_clip, dtype=np.float32)

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

        peak = np.max(np.abs(mixture)) + 1e-9
        if peak > 0.99: mixture *= 0.99 / peak
        sf.write(str(audio_dir / f"{sid}.wav"), mixture, sr)
        ledger = {"schema_version":"0.2.0","sample_id":sid,"duration_sec":duration,
                  "time_resolution_sec":0.1,
                  "tracks":[{"id":f"T{j+1}","kind":e["type"],"spans":e["spans"],"confidence":0.85} for j,e in enumerate(events)],
                  "events":events,
                  "provenance":{"label_level":"model_prediction","source_dataset":"librispeech+musdb18+esc50","license_status":"CC"}}
        manifest.append({"scene_id":sid,"scene_name":scene_config.get("scene_name","unknown"),
                         "scene_desc":scene_config.get("description",""),"audio_path":f"audio/{sid}.wav",
                         "duration":duration,"ledger":ledger,"sources":sources_info,"llm_scene":scene_config})
        if (i+1) % 50 == 0:
            print(f"[rv10] {i+1}/{n_mixtures} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Review CSV
    import csv
    ms = [m for m in manifest if any(s["type"] in ("music","speech","vocal") for s in m["sources"])]
    review_10 = rng.sample(ms, min(10, len(ms)))
    with open(out_dir / "review_10.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip_id","scene_name","scene_desc","audio_path","duration","n_sources",
                     "source_1_role","source_1_type","source_1_category","source_1_onset","source_1_offset","source_1_caption",
                     "source_2_role","source_2_type","source_2_category","source_2_onset","source_2_offset","source_2_caption",
                     "source_3_role","source_3_type","source_3_category","source_3_onset","source_3_offset","source_3_caption",
                     "audio_natural","caption_accurate","notes"])
        for m in review_10:
            row = [m["scene_id"],m["scene_name"],m["scene_desc"],m["audio_path"],f'{m["duration"]:.1f}',len(m["sources"])]
            for j in range(3):
                if j < len(m["sources"]):
                    s = m["sources"][j]
                    row.extend([s["role"],s["type"],s.get("category",""),f'{s["onset"]:.1f}s',f'{s["offset"]:.1f}s',s["caption"][:80]])
                else:
                    row.extend([""]*6)
            row.extend(["","",""])
            w.writerow(row)
    print(f"\nWrote {n_mixtures} to {out_dir}, {len(ms)} with music/speech/vocal")
    print(f"Review: {out_dir / 'review_10.csv'}")

if __name__ == "__main__":
    main()
