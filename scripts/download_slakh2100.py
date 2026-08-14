"""Download checksum-pinned Slakh2100-redux from the official Zenodo record.

The archive is about 105 GB.  Downloads resume from ``.part`` when the server
honours HTTP Range, and the publisher MD5 is always checked before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

ZENODO_RECORD = "4599666"
ARCHIVE_NAME = "slakh2100_flac_redux.tar.gz"
ARCHIVE_MD5 = "f4b71b6c45ac9b506f59788456b3f0c4"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}/content"


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
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
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


def _find_root(output: Path) -> Path:
    candidates = []
    for candidate in [output, *sorted(path for path in output.rglob("*") if path.is_dir())]:
        if all((candidate / split).is_dir() for split in ("train", "test")) and (
            (candidate / "validation").is_dir() or (candidate / "val").is_dir()
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one extracted Slakh root, found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", default=None, help="use a previously downloaded archive")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--delete-archive-after-extract",
        action="store_true",
        help="explicitly remove the verified 105 GB archive after successful extraction",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    supplied = Path(args.archive).expanduser().resolve() if args.archive else None
    archive = supplied or output / ARCHIVE_NAME
    if not archive.is_file():
        if supplied is not None:
            raise FileNotFoundError(f"supplied Slakh archive does not exist: {archive}")
        partial = archive.with_suffix(archive.suffix + ".part")
        if partial.is_file() and _md5(partial) == ARCHIVE_MD5:
            partial.replace(archive)
        else:
            print(
                "downloading Slakh2100-redux (approximately 105 GB; resume enabled)",
                flush=True,
            )
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
    try:
        root = _find_root(output)
    except RuntimeError:
        _safe_extract(archive, output)
        root = _find_root(output)
    if args.delete_archive_after_extract and supplied is None:
        archive.unlink()
    print(f"root={root}")
    print("split_variant=Slakh2100-redux")
    print("license=CC BY 4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
