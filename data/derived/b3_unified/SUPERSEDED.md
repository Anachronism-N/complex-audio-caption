# B3 synthetic artifact is not valid lyric supervision

The committed B3 manifest and reports are retained only for engineering
traceability. They must not be used for training or paper claims.

Reasons:

- synthetic ``vocal`` waveforms contain no lexical lyrics, while the historical
  targets contain verbatim lyric strings;
- the historical split grouped whole source combinations and leaked individual
  dry sources across train/validation;
- the reported metrics cover all 500 scenes, including training scenes;
- historical atomic targets omitted source-track attributes, collapsing two
  speakers into one parsed track;
- historical event F1 admitted type/time matches with zero text overlap.

Use `configs/data/b3_real.yaml`, `configs/model/b3_valid.yaml`, and
`scripts/run_b3_valid.sh`. The SFT exporter and trainer now reject synthetic
vocal placeholders unless an explicit smoke-test override is supplied.
