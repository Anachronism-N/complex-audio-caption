"""Download checksum-pinned UrbanSound8K v1 from its official Zenodo record."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

ZENODO_RECORD = "1203745"
ARCHIVE_NAME = "UrbanSound8K.tar.gz"
ARCHIVE_MD5 = "9aa69802bbf37fb986f71ec1483a196e"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}/content"
EXPECTED_AUDIO_FILES = 8732


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher-provided integrity checksum
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"Range": f"bytes={offset}-"} if offset else {}
    )
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    partial.replace(destination)


def _safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            destination = (root / member.name).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe archive path: {member.name}") from exc
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported link/device in archive: {member.name}")
        handle.extractall(output, members=members)  # noqa: S202 - validated above


def _ready(root: Path) -> bool:
    metadata = root / "metadata" / "UrbanSound8K.csv"
    audio_root = root / "audio"
    return (
        metadata.is_file()
        and audio_root.is_dir()
        and sum(1 for _ in audio_root.rglob("*.wav")) == EXPECTED_AUDIO_FILES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--delete-archive-after-extract", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    supplied = Path(args.archive).expanduser().resolve() if args.archive else None
    archive = supplied or output / ARCHIVE_NAME
    if not archive.is_file():
        if supplied is not None:
            raise FileNotFoundError(f"supplied archive does not exist: {archive}")
        partial = archive.with_suffix(archive.suffix + ".part")
        if partial.is_file() and _md5(partial) == ARCHIVE_MD5:
            partial.replace(archive)
        else:
            print("downloading official UrbanSound8K (approximately 6 GB)", flush=True)
            _download(archive)
    observed = _md5(archive)
    if observed != ARCHIVE_MD5:
        raise ValueError(
            f"checksum mismatch for {archive}: expected={ARCHIVE_MD5} observed={observed}"
        )
    print(f"archive={archive}")
    print(f"md5={observed}")
    if args.download_only:
        return 0
    root = output / "UrbanSound8K"
    if not _ready(root):
        _safe_extract(archive, output)
    if not _ready(root):
        observed_files = sum(1 for _ in (root / "audio").rglob("*.wav"))
        raise RuntimeError(
            "UrbanSound8K extraction is incomplete: "
            f"expected={EXPECTED_AUDIO_FILES} observed={observed_files}"
        )
    if args.delete_archive_after_extract and supplied is None:
        archive.unlink()
    print(f"root={root}")
    print("license=CC BY-NC 3.0; non-commercial research only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
