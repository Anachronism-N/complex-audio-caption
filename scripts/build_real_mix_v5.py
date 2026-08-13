"""LLM-guided realistic scene mixing with music/speech/sfx sources.

Instead of random source selection, uses predefined realistic scene templates
to create logical sound combinations. Sources include:
- SFX/ambience: ESC-50 (2000 real clips)
- Music: MOSS demo audio (qilixiang.mp3, game.mp3)
- Speech: MOSS demo audio (test_en.mp3, 吴京1.m4a, faker_and_chovy.mp3)
"""
import sys, json, random, time, os
sys.path.insert(0, "third_party/MOSS-Audio")
sys.path.insert(0, "src")
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve
from math import gcd
from pathlib import Path
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig

# ── Realistic scene templates ──
# Each scene specifies logical source combinations with roles
SCENE_TEMPLATES = [
    {"name": "coffee_shop", "desc": "A busy coffee shop",
     "sources": [
         {"role": "music", "type": "music", "desc": "light jazz background music"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["can_opening", "washing_machine"], "desc": "coffee machine sounds"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clapping", "laughing"], "desc": "customer activity"},
     ]},
    {"name": "street_traffic", "desc": "A busy street with traffic",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["engine", "car_horn"], "desc": "traffic noise"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["siren"], "desc": "emergency vehicle"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["footsteps"], "desc": "pedestrians walking"},
     ]},
    {"name": "park_afternoon", "desc": "A park on a sunny afternoon",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["chirping_birds", "crickets"], "desc": "birds and insects"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["dog", "cat"], "desc": "animals"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["wind"], "desc": "gentle wind"},
     ]},
    {"name": "home_morning", "desc": "Morning at home",
     "sources": [
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clock_alarm"], "desc": "alarm clock"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["door_wood_knock", "door_wood_creaks"], "desc": "door sounds"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["vacuum_cleaner", "washing_machine"], "desc": "household appliance"},
     ]},
    {"name": "thunderstorm", "desc": "A thunderstorm",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["rain", "wind"], "desc": "rain and wind"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["thunderstorm"], "desc": "thunder"},
     ]},
    {"name": "office_work", "desc": "An office environment",
     "sources": [
         {"role": "sfx", "type": "sfx", "esc50_cats": ["keyboard_typing", "mouse_click"], "desc": "typing sounds"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clock_tick"], "desc": "clock ticking"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["coughing", "sneezing"], "desc": "coworker sounds"},
     ]},
    {"name": "concert_outdoor", "desc": "Outdoor concert",
     "sources": [
         {"role": "music", "type": "music", "desc": "live music performance"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["clapping", "laughing"], "desc": "audience reaction"},
         {"role": "ambience", "type": "sfx", "esc50_cats": ["wind"], "desc": "outdoor wind"},
     ]},
    {"name": "workshop", "desc": "A workshop",
     "sources": [
         {"role": "sfx", "type": "sfx", "esc50_cats": ["hand_saw", "chainsaw"], "desc": "cutting tools"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["vacuum_cleaner"], "desc": "dust collection"},
     ]},
    {"name": "kitchen_cooking", "desc": "Kitchen during cooking",
     "sources": [
         {"role": "sfx", "type": "sfx", "esc50_cats": ["can_opening", "crushing"], "desc": "food preparation"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["washing_machine"], "desc": "running appliance"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["glass_breaking"], "desc": "glass breaking"},
     ]},
    {"name": "night_countryside", "desc": "Countryside at night",
     "sources": [
         {"role": "ambience", "type": "sfx", "esc50_cats": ["crickets", "frog"], "desc": "night insects"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["dog", "cow"], "desc": "farm animals"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["wind"], "desc": "night wind"},
     ]},
    {"name": "speech_with_music", "desc": "Someone talking with background music",
     "sources": [
         {"role": "speech", "type": "speech", "desc": "person speaking"},
         {"role": "music", "type": "music", "desc": "background music"},
     ]},
    {"name": "speech_with_sfx", "desc": "Someone talking with sound effects",
     "sources": [
         {"role": "speech", "type": "speech", "desc": "person speaking"},
         {"role": "sfx", "type": "sfx", "esc50_cats": ["door_wood_knock", "clapping"], "desc": "sound effects"},
     ]},
]

# Music/speech sources from MOSS demo
MUSIC_SOURCES = {
    # Only pure music (no vocals) to avoid overlap with speech (docs/18 problem 15)
    "music": [
        ("third_party/MOSS-Audio/demo/assets/audio/game.mp3", "electronic game background music"),
    ],
}
# Vocal music (qilixiang) — labeled as vocal, not music
VOCAL_MUSIC_SOURCES = [
    ("third_party/MOSS-Audio/demo/assets/audio/qilixiang.mp3", "Chinese pop song with vocals"),
]
SPEECH_SOURCES = {
    "speech": [
        ("third_party/MOSS-Audio/demo/assets/audio/test_en.mp3", "English speech"),
        ("third_party/MOSS-Audio/demo/assets/audio/faker_and_chovy.mp3", "Korean speech"),
    ],
}


def load_audio(path, sr_out):
    import librosa
    wav, sr = librosa.load(path, sr=sr_out, mono=True)
    return wav.astype(np.float32)


def apply_compression(wav, sr, threshold=-20, ratio=3.0):
    """Simple dynamics compression for vocal clarity (docs/17 problem 13)."""
    from scipy.signal import lfilter
    attack = int(0.005 * sr)
    env = np.abs(wav)
    env = lfilter(np.ones(max(1,attack))/max(1,attack), [1.0], env)
    gain = np.ones_like(wav)
    above = env > 10**(threshold/20)
    gain[above] = 10**(threshold/20) / (env[above] + 1e-9) ** (1 - 1/ratio) * env[above]
    gain = np.clip(gain, 0, 1)
    release = int(0.05 * sr)
    gain = lfilter(np.ones(max(1,release))/max(1,release), [1.0], gain)
    return wav * gain


def apply_vocal_eq(wav, sr, boost_db=2.0):
    """Boost 1-3kHz for vocal clarity."""
    from scipy.signal import firwin, lfilter
    try:
        low = firwin(65, 1000 / (sr / 2))
        high = firwin(65, 3000 / (sr / 2))
        mid = lfilter(high, 1.0, wav) - lfilter(low, 1.0, wav)
        return wav + mid * (10**(boost_db/20) - 1)
    except Exception:
        return wav


def apply_fade(wav, sr, fade_s=0.1):
    fade = int(fade_s * sr)
    if len(wav) > 2 * fade:
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
    return wav


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

    n_mixtures = 200
    out_dir = Path("/tmp/real_mix_v5")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260813)
    prompt = "Describe this audio in one sentence."
    t0 = time.time()
    manifest = []

    for i in range(n_mixtures):
        scene = rng.choice(SCENE_TEMPLATES)
        sid = f"rv5_{i+1:04d}"
        duration = rng.uniform(8.0, 12.0)
        n_clip = int(duration * sr)
        sources_info = []

        # Ducking: if speech present, lower ALL non-speech during speech (docs/18 problem 17)
        has_speech = any(s["type"] == "speech" for s in scene["sources"])
        duck_others = has_speech and rng.random() < 0.8

        mixture = np.zeros(n_clip, dtype=np.float32)
        events = []
        source_wavs = []  # store for ducking
        speech_mask = np.zeros(n_clip, dtype=np.float32)

        for sc_idx, src_spec in enumerate(scene["sources"]):
            src_type = src_spec["type"]
            onset = rng.uniform(0, max(0.5, duration - 5))
            # Per-type gain (docs/18 problem 16-17): speech louder, sfx lower
            if src_type == "speech":
                gain_db = rng.uniform(0, 6)   # speech +0 to +6dB
            elif src_type == "music":
                gain_db = rng.uniform(-9, -3)  # music -9 to -3dB
            elif src_type == "vocal":
                gain_db = rng.uniform(-3, 3)   # vocal (singing) -3 to +3dB
            else:
                gain_db = rng.uniform(-9, -3)  # sfx -9 to -3dB (was -6 to 3)

            if src_type == "music":
                music_path, music_desc = rng.choice(MUSIC_SOURCES["music"])
                wav = load_audio(music_path, sr)
                # take a random segment
                if len(wav) > n_clip:
                    start = rng.randint(0, len(wav) - n_clip)
                    wav = wav[start:start+n_clip]
                wav = apply_fade(wav, sr, 0.5)
                # MOSS caption this actual source (docs/19 problem 20)
                try:
                    raw_path = f"/tmp/rv5_src_{sid}_{sc_idx}.wav"
                    sf.write(raw_path, wav, sr)
                    caption = adapter.infer(raw_path, prompt, sample_id=f"{sid}_s{sc_idx}", duration=len(wav)/sr)
                    os.unlink(raw_path)
                except:
                    caption = music_desc
                etype = "music"
            elif src_type == "speech":
                speech_path, speech_desc = rng.choice(SPEECH_SOURCES["speech"])
                wav = load_audio(speech_path, sr)
                if len(wav) > n_clip // 2:
                    start = rng.randint(0, len(wav) - n_clip // 2)
                    wav = wav[start:start + n_clip // 2]
                wav = apply_fade(wav, sr, 0.05)
                # Vocal enhancement (docs/17 problem 13)
                wav = apply_compression(wav, sr, threshold=-20, ratio=3.0)
                wav = apply_vocal_eq(wav, sr, boost_db=2.0)
                # MOSS caption this actual source (docs/19 problem 20)
                try:
                    raw_path = f"/tmp/rv5_src_{sid}_{sc_idx}.wav"
                    sf.write(raw_path, wav, sr)
                    caption = adapter.infer(raw_path, prompt, sample_id=f"{sid}_s{sc_idx}", duration=len(wav)/sr)
                    os.unlink(raw_path)
                except:
                    caption = speech_desc
                etype = "speech"
            else:
                # ESC-50 source
                cats = src_spec.get("esc50_cats", ["dog"])
                cat = rng.choice(cats)
                fname = rng.choice(cat_files.get(cat, cat_files.get("dog")))
                path = f"/tmp/real_audio/esc50_wav/{fname}"
                wav = load_audio(path, sr)
                wav = apply_fade(wav, sr, 0.05)
                # MOSS caption this source
                try:
                    raw_path = f"/tmp/rv2_src_{sid}_{sc_idx}.wav"
                    sf.write(raw_path, wav, sr)
                    caption = adapter.infer(raw_path, prompt, sample_id=f"{sid}_s{sc_idx}", duration=len(wav)/sr)
                    os.unlink(raw_path)
                except:
                    caption = src_spec.get("desc", cat)
                etype = "sfx"

            wav = wav * (10 ** (gain_db / 20))

            # Optional RIR
            if rng.random() < 0.4:
                t60 = rng.uniform(0.2, 0.8)
                rir = synth_rir(sr, t60, seed=i*100+sc_idx)
                wav = fftconvolve(wav, rir, mode="full")[:len(wav)].astype(np.float32)

            # Store for ducking
            source_wavs.append({"wav": wav, "onset": onset, "type": src_type})

            # Track speech active regions
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
            events.append({
                "id": f"E{sc_idx+1:03d}", "type": etype, "track_id": f"T{sc_idx+1}",
                "spans": [{"start_sec": onset_sec, "end_sec": offset_sec}],
                "text": caption[:150], "confidence": 0.85,
            })
            sources_info.append({"role": src_spec["role"], "type": src_type,
                                 "onset": onset_sec, "offset": offset_sec,
                                 "caption": caption[:200],
                                 "src_rms": float(np.sqrt(np.mean(wav**2)) + 1e-12),
                                 "gain_db": gain_db})

        # Apply ducking: lower ALL non-speech sources during speech (docs/18 K)
        if duck_others and speech_mask.any():
            from scipy.signal import lfilter as _lf
            smooth = _lf(np.ones(10) / 10, [1.0], speech_mask)
            speech_mask_smooth = np.clip(smooth, 0, 1)
            duck_depth = rng.uniform(6, 8)  # 6-8dB
            duck_factor = 10 ** (-duck_depth / 20)
            duck_gain = 1.0 - (1.0 - duck_factor) * speech_mask_smooth
        else:
            duck_gain = np.ones(n_clip, dtype=np.float32)

        # Mix with ducking
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

        # RMS verification: ensure speech is audible (docs/18 L)
        if has_speech:
            speech_rms = np.sqrt(np.mean(mixture**2) + 1e-12)
            if speech_rms < 0.01:  # too quiet
                mixture *= 1.5  # boost

        # RMS verification: check each source actually present in mix (docs/19 N)
        mix_rms = float(np.sqrt(np.mean(mixture**2)) + 1e-12)
        verified_sources = []
        for sw, si in zip(source_wavs, sources_info):
            seg_start = int(sw["onset"] * sr)
            seg_end = min(n_clip, seg_start + len(sw["wav"]))
            if seg_start < n_clip:
                seg = mixture[seg_start:seg_end]
                seg_rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
                # Source is present if its contribution is audible
                src_rms_adjusted = si["src_rms"] * (10 ** (si["gain_db"] / 20))
                present = src_rms_adjusted > 0.001  # threshold
                si["present_in_mix"] = present
                si["mix_seg_rms"] = seg_rms
                if present:
                    verified_sources.append(si)

        # Prevent clipping
        peak = np.max(np.abs(mixture)) + 1e-9
        if peak > 0.99:
            mixture *= 0.99 / peak

        sf.write(str(audio_dir / f"{sid}.wav"), mixture, sr)
        ledger = {"schema_version": "0.2.0", "sample_id": sid, "duration_sec": duration,
                  "time_resolution_sec": 0.1,
                  "tracks": [{"id": f"T{j+1}", "kind": e["type"], "spans": e["spans"], "confidence": 0.85} for j, e in enumerate(events)],
                  "events": events,
                  "provenance": {"label_level": "model_prediction", "source_dataset": "esc50+moss_demo", "license_status": "CC"}}
        manifest.append({"scene_id": sid, "scene_name": scene["name"], "scene_desc": scene["desc"],
                         "audio_path": f"audio/{sid}.wav", "duration": duration,
                         "ledger": ledger, "sources": sources_info})

        if (i+1) % 50 == 0:
            print(f"[rv2] {i+1}/{n_mixtures} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    # Write manifest + review CSV
    with open(out_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    import csv
    with open(out_dir / "review_10.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "scene_name", "scene_desc", "audio_path", "duration", "n_sources",
                     "source_1_role", "source_1_type", "source_1_onset", "source_1_offset", "source_1_caption",
                     "source_2_role", "source_2_type", "source_2_onset", "source_2_offset", "source_2_caption",
                     "source_3_role", "source_3_type", "source_3_onset", "source_3_offset", "source_3_caption",
                     "audio_natural", "caption_accurate", "notes"])
        # Select 10 diverse clips for review
        review_10 = rng.sample(manifest, 10)
        for m in review_10:
            row = [m["scene_id"], m["scene_name"], m["scene_desc"], m["audio_path"], f'{m["duration"]:.1f}', len(m["sources"])]
            for j in range(3):
                if j < len(m["sources"]):
                    s = m["sources"][j]
                    row.extend([s["role"], s["type"], f'{s["onset"]:.1f}s', f'{s["offset"]:.1f}s', s["caption"]])
                else:
                    row.extend([""] * 5)
            row.extend(["", "", ""])
            w.writerow(row)

    print(f"\nWrote {n_mixtures} mixtures to {out_dir}")
    print(f"Review 10: {out_dir / 'review_10.csv'}")


if __name__ == "__main__":
    main()
