# B3-valid source data preparation

## LibriSpeech speech sources

The downloader uses the official OpenSLR SLR12 archives and published MD5
checksums. It keeps the archive, extracts one subset, checks every transcript
against a waveform, and emits a canonical SceneLedger JSONL catalog. The
default is `train-clean-100` (about 6.3 GB compressed):

```bash
bash scripts/data/download_librispeech.sh
```

For a small pipeline check, select `dev-clean`:

```bash
LIBRISPEECH_SUBSET=dev-clean \
LIBRISPEECH_ROOT=/data/librispeech \
bash scripts/data/download_librispeech.sh
```

Each speaker is one `source_group`, so no speaker can cross the later
train/validation split. The output rows carry the official `CC BY 4.0`
license metadata and verbatim transcripts.

If the archive is already extracted, skip the download and register one or
more subsets directly:

```bash
python -m sceneledger.cli.catalog_librispeech \
  --root /data/librispeech \
  --subset train-clean-100 \
  --subset train-clean-360 \
  --output /data/librispeech/source_catalog.jsonl \
  --report /data/librispeech/source_catalog.report.json
```

Official source: <https://www.openslr.org/12/>

## Singing sources such as Opencpop

Opencpop is not downloaded automatically: the official distribution requires
an application, and its `CC BY-NC-ND 4.0` terms are non-commercial. Download
it only through the official instructions and confirm that the planned use is
compatible with those terms:

- Download instructions: <https://wenet-e2e.github.io/opencpop/download/>
- License: <https://wenet-e2e.github.io/opencpop/liscense/>

After obtaining an authorized local copy, create CSV rows with the schema in
`configs/data/source_catalog.example.csv`, one row per isolated vocal clip.
Use the original song ID as `source_group`, the acoustically present lyric as
`text`, and set `kind=vocal,verbatim=true`. Canonicalize and audit it with:

```bash
python -m sceneledger.cli.prepare_sources \
  --input /data/opencpop/source_catalog.csv \
  --audio-root /data/opencpop \
  --output /data/opencpop/source_catalog.jsonl \
  --report /data/opencpop/source_catalog.report.json
```

The command fails closed on missing files, empty lyrics, missing leakage
groups, or vocal rows without `verbatim=true`. It deliberately does not infer
lyrics from filenames or generate placeholder lyric supervision.

## Merge the complete B3 catalog

The full `b3_real.yaml` template set requires all five source kinds: `speech`,
`vocal`, `music`, `sfx`, and `ambience`. Merge independently audited corpora
with repeated `--input` arguments:

```bash
python -m sceneledger.cli.prepare_sources \
  --input /data/librispeech/source_catalog_train-clean-100.jsonl \
  --input /data/opencpop/source_catalog.jsonl \
  --input /data/music_sfx/source_catalog.jsonl \
  --output /data/b3/source_catalog.jsonl \
  --report /data/b3/source_catalog.report.json \
  --require-kind speech --require-kind vocal --require-kind music \
  --require-kind sfx --require-kind ambience
```

Duplicate waveform paths across inputs are rejected. If vocal and instrumental
stems originate from the same song, give both rows the same `source_group` so
the connected-component splitter keeps the song in one fold.

## Reproduce and freeze B3-valid

First run only the source-pool gate; do not render yet:

```bash
SOURCE_CATALOG=/data/b3/source_catalog.jsonl \
SOURCE_AUDIO_ROOT=/data/b3/audio \
SOURCE_PROFILE=smoke \
WORK_DIR=/data/runs/b3_smoke \
N_SAMPLES=100 \
STAGE=sources \
bash scripts/run_b3_data.sh
```

This decodes every waveform and writes `source_inventory.jsonl` plus
`source_readiness_report.json`. The report must contain `pass=true`, no failed
checks, and a non-empty `source_pool_id`. It enforces the versioned `smoke`
quotas in `configs/data/source_readiness.yaml`, known licenses, real labels,
verbatim lyrics, exact decoded-audio deduplication, duration, RMS, and clipping
limits. Re-run only this audit with `STAGE=source-audit`.

After the source report passes and its stratified manual listening audit is
recorded, continue one stage at a time with `STAGE=render`, `export`, and
`audit`. The final stage writes `data_reproduction_summary.json`, which must
contain `pass=true` and a non-empty `dataset_id`.

After automated checks pass, listen to the stratified rows in
`data/listen_list.csv`. Render the 10k release in a new directory; do not overwrite
the accepted smoke directory with a different catalog or sample count. The full
artifact and acceptance contract is documented in
`docs/16_b3_data_reproduction.md`; the immediate source-only procedure is
`docs/17_source_pool_readiness.md`.
