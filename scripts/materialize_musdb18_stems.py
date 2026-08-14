"""Materialize accompaniment/vocal WAVs from a user-acquired MUSDB18-HQ tree.

The project cannot automatically accept the MUSDB18 educational-use license
for the user.  After access is obtained and MUSDB18-HQ is extracted, this
script performs only local, block-wise organization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _track_directories(root: Path) -> list[tuple[str, Path]]:
    tracks: list[tuple[str, Path]] = []
    for subset in ("train", "test"):
        directory = root / subset
        if not directory.is_dir():
            raise FileNotFoundError(f"MUSDB18-HQ subset is missing: {directory}")
        tracks.extend(
            (subset, path) for path in sorted(directory.iterdir()) if path.is_dir()
        )
    if not tracks:
        raise ValueError(f"no MUSDB18-HQ tracks found under {root}")
    return tracks


def _audio_signature(path: Path) -> tuple[int, int, int]:
    info = sf.info(path)
    return int(info.samplerate), int(info.channels), int(info.frames)


def materialize_track(track: Path, destination: Path, *, block_frames: int) -> None:
    required = [track / name for name in ("drums.wav", "bass.wav", "other.wav", "vocals.wav")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"track {track.name} is missing stems: {missing}")
    signatures = {_audio_signature(path) for path in required}
    if len(signatures) != 1:
        raise ValueError(f"track {track.name} stems disagree on sample rate/channels/frames")
    sample_rate, channels, _frames = next(iter(signatures))
    destination.mkdir(parents=True, exist_ok=True)
    accompaniment_path = destination / "accompaniment.wav"
    vocals_path = destination / "vocals.wav"
    if accompaniment_path.exists() or vocals_path.exists():
        raise FileExistsError(f"refusing to overwrite materialized track: {destination}")

    with (
        sf.SoundFile(required[0]) as drums,
        sf.SoundFile(required[1]) as bass,
        sf.SoundFile(required[2]) as other,
        # Keep stems in floating-point WAV.  A sample-accurate stem sum can
        # legitimately exceed [-1, 1]; clipping here would silently distort
        # the source before SceneLedger applies its measured-RMS gain.
        sf.SoundFile(accompaniment_path, "w", sample_rate, channels, subtype="FLOAT") as output,
    ):
        while True:
            blocks = [handle.read(block_frames, dtype="float32", always_2d=True) for handle in (drums, bass, other)]
            if not blocks[0].size:
                break
            if len({len(block) for block in blocks}) != 1:
                raise ValueError(f"track {track.name} stem reads became misaligned")
            mixture = np.sum(np.stack(blocks, axis=0), axis=0, dtype=np.float64)
            output.write(mixture.astype(np.float32))

    with sf.SoundFile(required[3]) as vocals, sf.SoundFile(
        vocals_path, "w", sample_rate, channels, subtype="FLOAT"
    ) as output:
        while True:
            block = vocals.read(block_frames, dtype="float32", always_2d=True)
            if not block.size:
                break
            output.write(block)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--block-frames", type=int, default=262144)
    args = parser.parse_args(argv)
    if args.block_frames <= 0:
        parser.error("--block-frames must be positive")
    source = Path(args.input_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output root: {output}")
    output.mkdir(parents=True)
    tracks = _track_directories(source)
    for index, (subset, track) in enumerate(tracks, 1):
        materialize_track(
            track,
            output / subset / track.name,
            block_frames=args.block_frames,
        )
        if index % 10 == 0:
            print(f"materialized={index}/{len(tracks)}", flush=True)
    print(f"output_root={output}")
    print(f"tracks={len(tracks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
