# TAC-mini data card

- generated: synthetic pool, 500 clips
- sample_rate: 24000 Hz
- duration range: 10.0–29.9 s
- event count: min=2 max=5 mean=3.04

## template distribution

- random_mix: 500

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
  sample_count: 500
  template_weights:
    random_mix: 1
  seed_base: 1947
  subgroup_count: 5
  audio_format:
    subtype: PCM_16

```