# TAC-mini data card

- generated: synthetic pool, 5000 clips
- sample_rate: 24000 Hz
- duration range: 10.0–30.0 s
- event count: min=1 max=6 mean=2.27

## template distribution

- ambient_with_intermittent_sfx: 329
- isolated_sfx: 352
- lyrics_over_music: 685
- music_with_sfx: 626
- overlapping_speakers: 681
- repeated_event: 337
- speech_music_lyrics_sfx: 672
- speech_music_sfx: 649
- speech_over_music: 669

## config

```yaml
pool:
  kind: synthetic
  sample_rate: 24000
  seed: 20260808
sampler:
  sample_rate: 24000
  duration_range:
  - 10.0
  - 30.0
  gain_db_range:
  - -12.0
  - 3.0
  fg_bg_snr_range:
  - -10.0
  - 20.0
  t60_range:
  - 0.1
  - 1.2
  echo_delay_ms_range:
  - 80
  - 500
  echo_atten_db_range:
  - -18.0
  - -3.0
  repeat_range:
  - 1
  - 5
  merge_threshold_range:
  - 0.1
  - 1.0
  resolutions:
  - 0.1
  - 0.5
  - 1.0
  styles:
  - keyword
  - brief
  - detailed
  activity_threshold_range:
  - 0.03
  - 0.12
  p_rir: 0.5
  p_echo: 0.3
render:
  sample_count: 5000
  template_weights:
    speech_over_music: 2
    music_with_sfx: 2
    speech_music_sfx: 2
    isolated_sfx: 1
    repeated_event: 1
    ambient_with_intermittent_sfx: 1
    lyrics_over_music: 2
    speech_music_lyrics_sfx: 2
    overlapping_speakers: 2
  seed_base: 1947
  subgroup_count: 5
  audio_format:
    subtype: PCM_16

```