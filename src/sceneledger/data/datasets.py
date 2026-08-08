from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetSpec:
    name: str
    kind: str
    homepage: str
    license: str
    license_url: str
    requires_acceptance: bool = True
    files: list[dict[str, Any]] = field(default_factory=list)
    repo_id: str | None = None
    revision: str | None = None
    instructions: str | None = None


def load_registry(path: str | Path) -> dict[str, DatasetSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {
        name: DatasetSpec(name=name, **value) for name, value in raw.get("datasets", {}).items()
    }


def download_datasets(
    registry_path: str | Path,
    names: list[str],
    output_root: str | Path,
    *,
    accepted_licenses: set[str],
    extract: bool = False,
) -> list[dict[str, Any]]:
    registry = load_registry(registry_path)
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; available={sorted(registry)}")
    results = []
    for name in names:
        spec = registry[name]
        if spec.requires_acceptance and name not in accepted_licenses:
            raise PermissionError(
                f"{name} requires explicit license acknowledgement. Read {spec.license_url} "
                f"and rerun with --accept-license {name}."
            )
        target = Path(output_root).resolve() / name
        target.mkdir(parents=True, exist_ok=True)
        if spec.kind == "url":
            downloaded = []
            for item in spec.files:
                filename = item.get("filename", Path(item["url"]).name)
                _validate_member(target, filename)
                file_path = target / filename
                expected = item.get("sha256")
                if not file_path.exists():
                    _download_with_resume(item["url"], file_path)
                actual = sha256_file(file_path)
                if expected and actual.lower() != expected.lower():
                    raise ValueError(f"Checksum mismatch for {file_path}: {actual} != {expected}")
                downloaded.append(str(file_path))
                if extract or bool(item.get("extract", False)):
                    safe_extract(file_path, target / "extracted")
            results.append({"name": name, "status": "downloaded", "files": downloaded})
        elif spec.kind == "huggingface":
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(
                    "Install sceneledger[download] for Hugging Face datasets"
                ) from exc
            local_dir = snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=spec.revision,
                local_dir=target,
            )
            results.append({"name": name, "status": "downloaded", "path": local_dir})
        elif spec.kind == "manual":
            marker = target / "MANUAL_DOWNLOAD.txt"
            instructions = (
                f"Homepage: {spec.homepage}\nLicense: {spec.license}\n\n{spec.instructions or ''}\n"
            )
            marker.write_text(
                instructions,
                encoding="utf-8",
            )
            results.append({"name": name, "status": "manual", "instructions": str(marker)})
        else:
            raise ValueError(f"Unsupported download kind for {name}: {spec.kind}")
    return results


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: str | Path, output_dir: str | Path) -> None:
    archive_path = Path(archive).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as handle:
            for member in handle.infolist():
                _validate_member(destination, member.filename)
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as handle:
            members = handle.getmembers()
            for member in members:
                _validate_member(destination, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not allowed: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Archive special files are not allowed: {member.name}")
            handle.extractall(destination, members=members)
        return
    raise ValueError(f"Unsupported archive: {archive_path}")


def _download_with_resume(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "SceneLedger/0.1"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        mode = "ab" if existing and status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(output)


def _validate_member(destination: Path, member_name: str) -> None:
    resolved = (destination / member_name).resolve()
    try:
        resolved.relative_to(destination)
    except ValueError as exc:
        raise ValueError(f"Unsafe archive path: {member_name}") from exc
