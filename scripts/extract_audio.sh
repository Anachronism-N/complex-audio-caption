#!/bin/bash
# Extract audio archives for all datasets.
# Usage: bash scripts/extract_audio.sh [dataset_dir]
#   dataset_dir defaults to data/derived (extracts all)

set -e

BASE="${1:-data/derived}"

for tar_file in "$BASE"/*/audio.tar; do
    if [ -f "$tar_file" ]; then
        dir=$(dirname "$tar_file")
        echo "[extract] $tar_file -> $dir/audio/"
        (cd "$dir" && tar xf audio.tar)
        echo "[extract] done: $(find "$dir/audio" -name '*.wav' | wc -l) wav files"
    fi
done

echo "[extract] all datasets extracted."
