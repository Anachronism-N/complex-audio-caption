#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sceneledger.data.audio import save_audio
from sceneledger.data.manifest import SourceRecord, write_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    specifications = [
        ("speech", 220.0, 2.0, "a synthetic speech-like tone", False),
        ("speech", 260.0, 2.5, "another synthetic speaker tone", False),
        ("lys", 330.0, 3.0, "a synthetic sung tone", False),
        ("music", 440.0, 5.0, "steady synthetic music", False),
        ("music", 523.25, 4.0, "higher synthetic music", False),
        ("sfx", 880.0, 0.6, "a short electronic beep", False),
        ("sfx", 1200.0, 0.4, "a sharp synthetic chirp", False),
        ("ambience", 90.0, 8.0, "a low synthetic background hum", False),
    ]
    for index, (source_type, frequency, duration, text, verbatim) in enumerate(specifications):
        time = np.arange(round(duration * args.sample_rate)) / args.sample_rate
        envelope = np.sin(np.pi * np.minimum(time / 0.05, 1.0)) ** 2
        envelope *= np.sin(np.pi * np.minimum((duration - time) / 0.05, 1.0)) ** 2
        audio = (0.15 * np.sin(2 * np.pi * frequency * time) * envelope).astype(np.float32)
        path = root / source_type / f"toy_{index:02d}.wav"
        save_audio(path, audio, args.sample_rate)
        records.append(
            SourceRecord(
                id=f"toy_{source_type}_{index}",
                path=str(path),
                type=source_type,  # type: ignore[arg-type]
                duration_sec=duration,
                sample_rate=args.sample_rate,
                text=text,
                group_id=f"toy_group_{index}",
                verbatim=verbatim,
                license="project-generated test fixture; not a training corpus",
            )
        )
    write_source_manifest(root / "sources.jsonl", records)
    print(root / "sources.jsonl")


if __name__ == "__main__":
    main()
