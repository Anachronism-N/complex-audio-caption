"""Download the official ESC-50 repository at a pinned commit and extract safely."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC50_URL = f"https://github.com/karolpiczak/ESC-50/archive/{ESC50_COMMIT}.zip"


def _safe_extract(archive: Path, output: Path) -> Path:
    expected_root = output / f"ESC-50-{ESC50_COMMIT}"
    if (expected_root / "meta" / "esc50.csv").is_file() and (expected_root / "audio").is_dir():
        return expected_root
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        root = output.resolve()
        for member in handle.infolist():
            destination = (root / member.filename).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe archive path: {member.filename}") from exc
        handle.extractall(output)
    if not (expected_root / "meta" / "esc50.csv").is_file():
        raise RuntimeError(f"downloaded archive does not contain official ESC-50 metadata: {archive}")
    return expected_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--archive",
        default=None,
        help="use an existing pinned ZIP instead of downloading it",
    )
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    supplied = Path(args.archive).expanduser().resolve() if args.archive else None
    archive = supplied or output / f"ESC-50-{ESC50_COMMIT}.zip"
    if not archive.is_file():
        output.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading pinned ESC-50 commit {ESC50_COMMIT} ...", flush=True)
        with urllib.request.urlopen(ESC50_URL) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(archive)
    root = _safe_extract(archive, output)
    if supplied is None and not args.keep_archive:
        archive.unlink()
    print(f"root={root}")
    print(f"metadata={root / 'meta' / 'esc50.csv'}")
    print(f"audio_root={root / 'audio'}")
    print("license=CC BY-NC 3.0 for ESC-50; CC BY 3.0 for the ESC-10 subset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
