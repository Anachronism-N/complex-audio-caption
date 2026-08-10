# TAG 2021 anchor reproduction

This directory pins the paper-era implementation of **Text-to-Audio Grounding:
Building Correspondence Between Captions and Sound Events**. It is the scientific
anchor for later SceneLedger work. Do not substitute the later AudioGrounding v2
labels when comparing against the ICASSP 2021 numbers.

The complete Chinese protocol and acceptance rules are in
[`docs/13_anchor_first_tag_reproduction.md`](../../docs/13_anchor_first_tag_reproduction.md).

## One-command server workflow

```bash
conda env create -f repro/tag2021/environment-legacy.yml
conda activate tag2021-legacy
PYTHONPATH=src python -m sceneledger.repro.tag2021 doctor

bash scripts/repro/tag2021/00_bootstrap.sh
bash scripts/repro/tag2021/01_download.sh
bash scripts/repro/tag2021/02_prepare.sh
bash scripts/repro/tag2021/03_features.sh
bash scripts/repro/tag2021/04_train.sh 1
bash scripts/repro/tag2021/05_evaluate.sh 1
```

Repeat training and evaluation for seeds 2 and 3, then run:

```bash
bash scripts/repro/tag2021/06_summarize.sh 1 2 3
```

The final machine-readable result is `runs/tag2021/reproduction_summary.json`.
It is a pass only when the paper metrics and random-query diagnostic are within
the tolerances frozen in `upstream.lock.yaml`.

## Archive fallback

The paper's official audio archive is hosted on Google Drive and has no published
checksum. The downloader records a local SHA-256 digest. If automated download is
blocked, download file ID `1znGt8OEBdX3uCrnIUXqLz6Pn3NabBxLs` manually:

```bash
bash scripts/repro/tag2021/01_download.sh /path/to/AudioTextGrounding.zip
```

Compare `external/tag2021/paper2021/downloads/audio_archive_provenance.json`
across machines before combining results.
