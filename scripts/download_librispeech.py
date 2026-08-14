"""Download checksum-pinned LibriSpeech subsets and extract them safely."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

ARCHIVES = {
    "train-clean-5": {
        "url": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
        "md5": "5df7d4e78065366204ca6845bb08f490",
    },
    "dev-clean-2": {
        "url": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
        "md5": "6d7ab67ac6a1d2c993d050e16d61080d",
    },
    "train-clean-100": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "md5": "2a93770f6d5c6c964bc36631d331a522",
    },
    "dev-clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "md5": "42e2234ba48799c1f50f24a7926300a1",
    },
    "test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
    },
}
PROFILES = {
    "pilot": ("train-clean-5", "dev-clean-2", "test-clean"),
    "full-clean": ("train-clean-100", "dev-clean", "test-clean"),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies publisher-provided data checksum
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)


def _safe_extract(archive: Path, output: Path) -> None:
    root = output.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"links are forbidden in dataset archive: {member.name}")
            destination = (root / member.name).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe archive path: {member.name}") from exc
        handle.extractall(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="pilot")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for subset in PROFILES[args.profile]:
        expected_dir = output / "LibriSpeech" / subset
        if expected_dir.is_dir() and any(expected_dir.rglob("*.flac")):
            print(f"skip extracted subset={subset}")
            continue
        metadata = ARCHIVES[subset]
        archive = output / f"{subset}.tar.gz"
        if not archive.is_file():
            print(f"downloading subset={subset} url={metadata['url']}", flush=True)
            _download(str(metadata["url"]), archive)
        observed = _md5(archive)
        if observed != metadata["md5"]:
            raise ValueError(
                f"checksum mismatch for {archive}: expected={metadata['md5']} observed={observed}"
            )
        _safe_extract(archive, output)
        if not expected_dir.is_dir() or not any(expected_dir.rglob("*.flac")):
            raise RuntimeError(f"archive did not produce expected subset: {expected_dir}")
        if not args.keep_archives:
            archive.unlink()
        print(f"ready subset={subset} root={expected_dir}")
    print(f"audio_root={output}")
    print("license=CC BY 4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
