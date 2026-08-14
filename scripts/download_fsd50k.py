"""Download checksum-pinned FSD50K release files and extract them safely.

FSD50K audio is distributed as split ZIP archives.  This downloader verifies
every Zenodo-published MD5 before asking the local ``zip`` executable to merge
the parts.  Metadata-only mode is useful for planning and unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "4060432"
BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
FILES = {
    "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485",
    "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237",
    "FSD50K.doc.zip": "3516162b82dc2945d3e7feba0904e800",
    "FSD50K.dev_audio.z01": "faa7cf4cc076fc34a44a479a5ed862a3",
    "FSD50K.dev_audio.z02": "8f9b66153e68571164fb1315d00bc7bc",
    "FSD50K.dev_audio.z03": "1196ef47d267a993d30fa98af54b7159",
    "FSD50K.dev_audio.z04": "d088ac4e11ba53daf9f7574c11cccac9",
    "FSD50K.dev_audio.z05": "81356521aa159accd3c35de22da28c7f",
    "FSD50K.dev_audio.zip": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
    "FSD50K.eval_audio.z01": "3090670eaeecc013ca1ff84fe4442aeb",
    "FSD50K.eval_audio.zip": "6fa47636c3a3ad5c7dfeba99f2637982",
}
PROFILE_FILES = {
    "metadata": (
        "FSD50K.ground_truth.zip",
        "FSD50K.metadata.zip",
        "FSD50K.doc.zip",
    ),
    "dev": (
        "FSD50K.ground_truth.zip",
        "FSD50K.metadata.zip",
        "FSD50K.doc.zip",
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
        "FSD50K.dev_audio.zip",
    ),
    "full": tuple(FILES),
}
METADATA_ARCHIVES = {
    "FSD50K.ground_truth.zip",
    "FSD50K.metadata.zip",
    "FSD50K.doc.zip",
}
EXPECTED_AUDIO_FILES = {"dev": 40966, "eval": 10231}


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher-provided integrity checksum
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(name: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE_URL}/{name}/content"
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={offset}-"} if offset else {}
    )
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)


def _safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            destination = (root / member.filename).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe archive path: {member.filename}") from exc
        handle.extractall(output)


def _metadata_ready(extracted: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            extracted / "FSD50K.ground_truth" / "dev.csv",
            extracted / "FSD50K.ground_truth" / "eval.csv",
            extracted / "FSD50K.metadata" / "dev_clips_info_FSD50K.json",
            extracted / "FSD50K.metadata" / "eval_clips_info_FSD50K.json",
            extracted / "FSD50K.metadata" / "pp_pnp_ratings_FSD50K.json",
        )
    )


def _audio_ready(extracted: Path, split: str) -> bool:
    target = extracted / f"FSD50K.{split}_audio"
    return target.is_dir() and sum(1 for _ in target.glob("*.wav")) == EXPECTED_AUDIO_FILES[split]


def _merge_split_archive(last_part: Path, destination: Path) -> None:
    zip_binary = shutil.which("zip")
    if zip_binary is None:
        raise RuntimeError(
            "FSD50K split archives require the `zip` executable; install it "
            "and rerun (for example: apt-get install zip unzip)"
        )
    subprocess.run(
        [zip_binary, "-s", "0", str(last_part), "--out", str(destination)],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_FILES), default="dev")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_dir).expanduser().resolve()
    archives = output / "archives"
    extracted = output / "FSD50K"
    archives.mkdir(parents=True, exist_ok=True)
    audio_sets = (
        ["dev"]
        if args.profile == "dev"
        else (["dev", "eval"] if args.profile == "full" else [])
    )
    needed_files = set(PROFILE_FILES[args.profile])
    if _metadata_ready(extracted):
        needed_files -= METADATA_ARCHIVES
    for split in audio_sets:
        if _audio_ready(extracted, split):
            needed_files = {
                name
                for name in needed_files
                if not name.startswith(f"FSD50K.{split}_audio.")
            }

    for name in PROFILE_FILES[args.profile]:
        if name not in needed_files:
            continue
        path = archives / name
        if not path.is_file():
            partial = path.with_suffix(path.suffix + ".part")
            if partial.is_file() and _md5(partial) == FILES[name]:
                partial.replace(path)
            else:
                print(f"downloading name={name}", flush=True)
                _download(name, path)
        observed = _md5(path)
        if observed != FILES[name]:
            raise ValueError(
                f"checksum mismatch for {path}: expected={FILES[name]} observed={observed}"
            )

    for name in METADATA_ARCHIVES:
        path = archives / name
        if path.is_file():
            _safe_extract(path, extracted)

    for split in audio_sets:
        target = extracted / f"FSD50K.{split}_audio"
        if _audio_ready(extracted, split):
            continue
        last_part = archives / f"FSD50K.{split}_audio.zip"
        merged = archives / f"FSD50K.{split}_audio.unsplit.zip"
        if not merged.is_file():
            _merge_split_archive(last_part, merged)
        _safe_extract(merged, extracted)
        if not _audio_ready(extracted, split):
            observed_count = sum(1 for _ in target.glob("*.wav")) if target.is_dir() else 0
            raise RuntimeError(
                f"split archive did not produce the complete audio set under {target}: "
                f"expected={EXPECTED_AUDIO_FILES[split]} observed={observed_count}"
            )

    ground_truth = extracted / "FSD50K.ground_truth"
    metadata = extracted / "FSD50K.metadata"
    if not _metadata_ready(extracted):
        raise RuntimeError("FSD50K release structure is incomplete after extraction")
    if not args.keep_archives:
        for name in PROFILE_FILES[args.profile]:
            (archives / name).unlink(missing_ok=True)
        for split in audio_sets:
            (archives / f"FSD50K.{split}_audio.unsplit.zip").unlink(missing_ok=True)
    print(f"root={extracted}")
    print(f"ground_truth={ground_truth}")
    print(f"metadata={metadata}")
    print("dataset_license=CC BY 4.0; each clip retains its own CC license")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
