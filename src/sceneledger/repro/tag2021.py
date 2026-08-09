"""Reproduce the ICASSP 2021 Text-to-Audio Grounding baseline.

The upstream project changed its data format and training stack after the paper.
This module deliberately targets the paper-era commit and keeps all adaptations in
this repository. It does not modify the upstream checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

UPSTREAM_URL = "https://github.com/wsntxxn/TextToAudioGrounding.git"
UPSTREAM_COMMIT = "048f7af3d5167eeee0b7fd59aa877f46f245ff36"
PAPER_AUDIO_URL = (
    "https://drive.google.com/file/d/1znGt8OEBdX3uCrnIUXqLz6Pn3NabBxLs/view"
)
PAPER_AUDIO_FILE_ID = "1znGt8OEBdX3uCrnIUXqLz6Pn3NabBxLs"
PAPER_SOURCE_SHA256 = {
    "train.json": "2dfa6f1e67648fd6d271d0fcd463b78b84b0f520c5b601c6fac9a6cc5f978f6a",
    "val.json": "01357572098531af609cf3f40f85d977bf4ce598930323ca30b4d5b38e34cb8d",
    "test.json": "44e94db7c63afbd28bc22a317fd347e85519f946c971329ca3ae8bb57ba786c8",
    "test_meta.csv": "d3d9002c4f7e9f3ca0cd1a0a629cbe7ea1cb21824e0c7267b5c564c55e3555d5",
}
PAPER_EXPECTED_COUNTS = {
    "train": {"rows": 12_373, "unique_audio": 4_489},
    "val": {"rows": 451, "unique_audio": 31},
    "test": {"rows": 1_161, "unique_audio": 70},
}
PAPER_METRICS = {
    "event_f1": {"expected": 0.283, "absolute_tolerance": 0.03},
    "precision": {"expected": 0.286, "absolute_tolerance": 0.03},
    "recall": {"expected": 0.279, "absolute_tolerance": 0.03},
    "psds": {"expected": 0.147, "absolute_tolerance": 0.03},
    "random_query_event_f1": {"expected": 0.196, "absolute_tolerance": 0.04},
}
LEGACY_PACKAGE_VERSIONS = {
    "numpy": "1.18.0",
    "pandas": "1.1.2",
    "h5py": "2.10.0",
    "torch": "1.6.0",
    "psds-eval": "0.3.0",
    "pytorch-ignite": "0.4.1",
    "scikit-learn": "0.23.2",
    "librosa": "0.8.0",
    "sed-eval": "0.2.1",
}
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


class ReproductionError(RuntimeError):
    """Raised when a reproduction invariant is violated."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2, sort_keys=True)
        writer.write("\n")
    temporary.replace(path)


def environment_snapshot() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in LEGACY_PACKAGE_VERSIONS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    exact_legacy = all(
        versions[package] == expected
        for package, expected in LEGACY_PACKAGE_VERSIONS.items()
    )
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except (ImportError, RuntimeError) as error:
        torch_info = {"error": str(error)}
    return {
        "classification": "legacy_exact" if exact_legacy else "compatibility",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "torch_runtime": torch_info,
    }


def _read_json_table(path: Path) -> list[dict[str, Any]]:
    """Read both pandas column-oriented JSON and JSON records."""
    with path.open(encoding="utf-8") as reader:
        payload = json.load(reader)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ReproductionError(f"Expected object records in {path}")
        return payload
    if not isinstance(payload, dict) or not payload:
        raise ReproductionError(f"Unsupported JSON table in {path}")
    if not all(isinstance(column, dict) for column in payload.values()):
        raise ReproductionError(f"Unsupported JSON orientation in {path}")

    row_keys = set().union(*(column.keys() for column in payload.values()))

    def sort_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    rows: list[dict[str, Any]] = []
    for row_key in sorted(row_keys, key=sort_key):
        rows.append({name: column.get(row_key) for name, column in payload.items()})
    return rows


def _audio_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise ReproductionError(f"Audio root does not exist: {root}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        key = path.name
        if key in index and index[key].resolve() != path.resolve():
            duplicates[key].extend([index[key], path])
        else:
            index[key] = path.resolve()
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise ReproductionError(
            f"Ambiguous duplicate audio basenames under {root}: {sample}. "
            "Provide a clean extraction of the official archive."
        )
    if not index:
        raise ReproductionError(f"No supported audio files found below {root}")
    return index


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ReproductionError(f"Unsafe archive member: {member.filename}")
        zip_file.extractall(destination)


def bootstrap(repo_root: Path) -> Path:
    """Clone and detach the official repository at the paper-era commit."""
    destination = repo_root / "third_party" / "TextToAudioGrounding"
    if destination.exists() and not (destination / ".git").exists():
        raise ReproductionError(
            f"Refusing to replace non-git directory at {destination}. Move it and retry."
        )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", UPSTREAM_URL, str(destination)])
    dirty = _run(
        ["git", "status", "--porcelain"], cwd=destination, capture=True
    ).stdout.strip()
    if dirty:
        raise ReproductionError(
            f"Upstream checkout has local modifications; refusing to switch commits:\n{dirty}"
        )
    _run(["git", "fetch", "origin", UPSTREAM_COMMIT, "--depth", "1"], cwd=destination)
    _run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=destination)
    actual = _run(["git", "rev-parse", "HEAD"], cwd=destination, capture=True).stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise ReproductionError(f"Upstream commit mismatch: {actual} != {UPSTREAM_COMMIT}")
    for filename, expected_sha256 in PAPER_SOURCE_SHA256.items():
        path = destination / "data" / filename
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ReproductionError(
                f"Source hash mismatch for {filename}: {actual_sha256} != {expected_sha256}"
            )
    print(f"Pinned TextToAudioGrounding at {actual}")
    return destination


def download_paper_audio(data_root: Path, archive: Path | None = None) -> Path:
    """Download (or accept) the paper-era Google Drive audio archive and extract it."""
    downloads = data_root / "paper2021" / "downloads"
    raw = data_root / "paper2021" / "raw"
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / "AudioTextGrounding.zip"

    if archive is not None:
        archive = archive.expanduser().resolve()
        if not archive.is_file():
            raise ReproductionError(f"Archive does not exist: {archive}")
        source = archive
    else:
        source = target
        if not source.exists():
            try:
                _run(
                    [
                        sys.executable,
                        "-m",
                        "gdown",
                        "--fuzzy",
                        PAPER_AUDIO_URL,
                        "-O",
                        str(target),
                    ]
                )
            except subprocess.CalledProcessError as error:
                raise ReproductionError(
                    "The original Google Drive download failed. Download file ID "
                    f"{PAPER_AUDIO_FILE_ID} manually and rerun with --archive PATH."
                ) from error
    if not zipfile.is_zipfile(source):
        raise ReproductionError(f"Downloaded file is not a valid zip archive: {source}")

    digest = _sha256(source)
    provenance = {
        "source_url": PAPER_AUDIO_URL,
        "google_drive_file_id": PAPER_AUDIO_FILE_ID,
        "archive": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": digest,
        "note": (
            "The upstream source did not publish a checksum. This digest records the "
            "exact bytes used by this run and must be compared across machines."
        ),
    }
    _json_dump(provenance, downloads / "audio_archive_provenance.json")

    marker = raw / ".extracted.json"
    previous = None
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8")).get("sha256")
    if previous != digest:
        if raw.exists() and any(raw.iterdir()):
            raise ReproductionError(
                f"{raw} contains a different extraction. Move it aside before retrying."
            )
        _safe_extract(source, raw)
        _json_dump({"sha256": digest, "archive": str(source)}, marker)
    print(f"Audio archive ready under {raw} (sha256={digest})")
    return raw


def _validate_row(row: dict[str, Any], split: str, index: int) -> list[str]:
    errors: list[str] = []
    required = {"audiocap_id", "filename", "tokens", "soundtag", "start_word", "timestamps"}
    missing = sorted(required - row.keys())
    if missing:
        return [f"{split}[{index}] missing fields: {missing}"]
    timestamps = row["timestamps"]
    if not isinstance(timestamps, list) or not timestamps:
        return [f"{split}[{index}] has no timestamp list"]
    for segment_index, segment in enumerate(timestamps):
        if not isinstance(segment, list) or len(segment) != 2:
            errors.append(f"{split}[{index}].timestamps[{segment_index}] is not [onset, offset]")
            continue
        onset, offset = segment
        if not isinstance(onset, (int, float)) or not isinstance(offset, (int, float)):
            errors.append(f"{split}[{index}].timestamps[{segment_index}] is non-numeric")
        elif onset < 0 or offset < onset or offset > 10.1:
            errors.append(
                f"{split}[{index}].timestamps[{segment_index}] invalid: {segment}"
            )
    return errors


def audit_paper_data(
    prepared_root: Path,
    *,
    require_audio: bool = True,
    strict_counts: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "profile": "paper2021",
        "upstream_commit": UPSTREAM_COMMIT,
        "splits": {},
        "errors": [],
    }
    split_sources: dict[str, set[str]] = {}
    all_audio_paths: set[Path] = set()
    for split, expected in PAPER_EXPECTED_COUNTS.items():
        label_path = prepared_root / split / "label.json"
        if not label_path.exists():
            report["errors"].append(f"Missing label file: {label_path}")
            continue
        rows = _read_json_table(label_path)
        sources = {Path(str(row.get("filename", ""))).name for row in rows}
        split_sources[split] = sources
        positive_rows = 0
        missing_audio: list[str] = []
        row_errors: list[str] = []
        for index, row in enumerate(rows):
            row_errors.extend(_validate_row(row, split, index))
            timestamps = row.get("timestamps", [])
            if timestamps and timestamps[0] != [0, 0]:
                positive_rows += 1
            filename = Path(str(row.get("filename", "")))
            if filename.exists():
                all_audio_paths.add(filename.resolve())
            elif require_audio:
                missing_audio.append(str(filename))
        stats = {
            "rows": len(rows),
            "unique_audio": len(sources),
            "positive_rows": positive_rows,
            "missing_audio": len(missing_audio),
            "label_sha256": _sha256(label_path),
            "errors": row_errors[:50],
        }
        report["splits"][split] = stats
        if strict_counts:
            for key in ("rows", "unique_audio"):
                if stats[key] != expected[key]:
                    report["errors"].append(
                        f"{split} {key}: got {stats[key]}, expected {expected[key]}"
                    )
        if missing_audio:
            report["errors"].append(
                f"{split}: {len(missing_audio)} missing audio files; first={missing_audio[0]}"
            )
        if row_errors:
            report["errors"].append(f"{split}: {len(row_errors)} invalid annotation fields")

    leakage: dict[str, list[str]] = {}
    split_names = sorted(split_sources)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = sorted(split_sources[left] & split_sources[right])
            if overlap:
                leakage[f"{left}-{right}"] = overlap[:50]
                report["errors"].append(
                    f"Cross-split audio leakage {left}-{right}: {len(overlap)} files"
                )
    inventory = "\n".join(
        f"{path.name}\t{path.stat().st_size}" for path in sorted(all_audio_paths)
    )
    report["cross_split_leakage"] = leakage
    report["audio_inventory_sha256"] = hashlib.sha256(inventory.encode()).hexdigest()
    report["valid"] = not report["errors"]
    return report


def _make_random_query_ablation(
    rows: list[dict[str, Any]], seed: int = 1
) -> tuple[list[dict[str, Any]], int]:
    """Replace each query with a random phrase from the same clip.

    Ground-truth timestamps and identifiers remain unchanged, matching the paper's
    diagnostic of whether the model is sensitive to its text query. Sampling is with
    replacement, so a query can remain unchanged; the paper did not publish its seed,
    therefore this implementation freezes seed 1.
    """
    rng = random.Random(seed)
    phrases_by_audio: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        phrase = str(row["soundtag"])
        if phrase not in phrases_by_audio[str(row["filename"])]:
            phrases_by_audio[str(row["filename"])].append(phrase)
    changed = 0
    output: list[dict[str, Any]] = []
    for row in rows:
        current = str(row["soundtag"])
        candidates = phrases_by_audio[str(row["filename"])]
        updated = dict(row)
        if candidates:
            updated["soundtag"] = rng.choice(candidates)
            if updated["soundtag"] != current:
                changed += 1
        output.append(updated)
    return output, changed


def prepare_paper_data(repo_root: Path, data_root: Path, audio_root: Path | None = None) -> Path:
    upstream = bootstrap(repo_root)
    if audio_root is None:
        audio_root = data_root / "paper2021" / "raw"
    audio = _audio_index(audio_root.expanduser().resolve())
    prepared = data_root / "paper2021" / "prepared"
    combined: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        source_label = upstream / "data" / f"{split}.json"
        rows = _read_json_table(source_label)
        missing: list[str] = []
        resolved_rows: list[dict[str, Any]] = []
        for row in rows:
            basename = Path(str(row["filename"])).name
            path = audio.get(basename)
            if path is None:
                missing.append(basename)
                continue
            resolved = dict(row)
            resolved["filename"] = str(path)
            resolved_rows.append(resolved)
        if missing:
            raise ReproductionError(
                f"{split}: {len(missing)} labeled files are absent below {audio_root}; "
                f"first={missing[0]}"
            )
        destination = prepared / split / "label.json"
        _json_dump(resolved_rows, destination)
        combined.extend(resolved_rows)

    _json_dump(combined, prepared / "all_labels.json")
    shutil.copy2(upstream / "data" / "test_meta.csv", prepared / "test" / "meta.csv")
    _json_dump(
        {
            "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": PAPER_SOURCE_SHA256,
        },
        prepared / "source_provenance.json",
    )
    test_rows = _read_json_table(prepared / "test" / "label.json")
    random_rows, changed = _make_random_query_ablation(test_rows, seed=1)
    _json_dump(random_rows, prepared / "test" / "label_random_query_seed1.json")

    report = audit_paper_data(prepared)
    report["random_query_ablation"] = {
        "seed": 1,
        "changed_rows": changed,
        "total_rows": len(test_rows),
    }
    report["source_sha256"] = PAPER_SOURCE_SHA256
    _json_dump(report, prepared / "data_audit.json")
    if not report["valid"]:
        raise ReproductionError(
            "Prepared data failed audit:\n- " + "\n- ".join(report["errors"])
        )
    print(f"Prepared and audited paper data at {prepared}")
    return prepared


def extract_paper_features(repo_root: Path, data_root: Path) -> None:
    upstream = bootstrap(repo_root)
    prepared = data_root / "paper2021" / "prepared"
    report = audit_paper_data(prepared)
    if not report["valid"]:
        raise ReproductionError("Data audit must pass before feature extraction")
    vocab = prepared / "vocab.pkl"
    feature = prepared / "logmel.hdf5"
    key_file = prepared / "logmel.keys.txt"
    _run(
        [
            sys.executable,
            str(upstream / "utils" / "build_vocab.py"),
            str(prepared / "train" / "label.json"),
            str(vocab),
        ],
        cwd=upstream,
    )
    _run(
        [
            sys.executable,
            str(upstream / "utils" / "extract_feature.py"),
            str(prepared / "all_labels.json"),
            str(feature),
            str(key_file),
            "mfcc",
            "-n_mels",
            "64",
            "-win_length",
            "640",
            "-hop_length",
            "320",
        ],
        cwd=upstream,
    )
    if not feature.exists() or feature.stat().st_size == 0:
        raise ReproductionError("Feature extraction did not produce logmel.hdf5")
    import h5py

    expected_audio = sum(item["unique_audio"] for item in PAPER_EXPECTED_COUNTS.values())
    with h5py.File(feature, "r") as store:
        keys = list(store.keys())
        if len(keys) != expected_audio:
            raise ReproductionError(
                f"Feature key count mismatch: got {len(keys)}, expected {expected_audio}"
            )
        first_shape = store[keys[0]].shape
        if len(first_shape) != 2 or first_shape[1] != 64:
            raise ReproductionError(
                f"Expected [frames, 64] log-mel features, got {first_shape} for {keys[0]}"
            )
    print(f"Features ready: {feature}")


def render_paper_config(
    repo_root: Path,
    data_root: Path,
    run_root: Path,
    seed: int,
    num_workers: int,
) -> Path:
    upstream = bootstrap(repo_root)
    prepared = data_root / "paper2021" / "prepared"
    with (upstream / "config" / "conf.yaml").open(encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    config.update(
        {
            "outputpath": str((run_root / "experiments" / f"seed_{seed}").resolve()),
            "train_label": str((prepared / "train" / "label.json").resolve()),
            "val_label": str((prepared / "val" / "label.json").resolve()),
            "audio_feature": str((prepared / "logmel.hdf5").resolve()),
            "vocab_file": str((prepared / "vocab.pkl").resolve()),
            "seed": seed,
            "reproduction_upstream_commit": UPSTREAM_COMMIT,
        }
    )
    config["dataloader_args"]["num_workers"] = num_workers
    destination = run_root / "configs" / f"paper2021_seed{seed}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as writer:
        yaml.safe_dump(config, writer, sort_keys=False)
    return destination


def _load_upstream_runner(upstream: Path) -> Any:
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("tag_paper2021_run", upstream / "run.py")
    if spec is None or spec.loader is None:
        raise ReproductionError(f"Cannot load upstream runner from {upstream}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Runner


def train_paper(
    repo_root: Path,
    data_root: Path,
    run_root: Path,
    seed: int,
    num_workers: int,
) -> Path:
    upstream = bootstrap(repo_root)
    prepared = data_root / "paper2021" / "prepared"
    required = [prepared / "logmel.hdf5", prepared / "vocab.pkl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ReproductionError(f"Run the features stage first; missing: {missing}")
    config = render_paper_config(repo_root, data_root, run_root, seed, num_workers)
    previous_cwd = Path.cwd()
    try:
        os.chdir(upstream)
        runner_class = _load_upstream_runner(upstream)
        experiment = Path(runner_class(seed=seed).train(str(config))).resolve()
    finally:
        os.chdir(previous_cwd)
    pointer = {
        "profile": "paper2021",
        "seed": seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": str(config.resolve()),
        "experiment_path": str(experiment),
        "environment": environment_snapshot(),
    }
    _json_dump(pointer, run_root / f"seed_{seed}" / "run.json")
    _json_dump(environment_snapshot(), run_root / f"seed_{seed}" / "environment.json")
    print(f"Experiment pointer: {run_root / f'seed_{seed}' / 'run.json'}")
    return experiment


def _copy_prediction_directory(experiment: Path, name: str) -> None:
    source = experiment / "predictions_for_psds"
    destination = experiment / name
    if destination.exists():
        shutil.rmtree(destination)
    if source.exists():
        shutil.copytree(source, destination)


def evaluate_paper(
    repo_root: Path,
    data_root: Path,
    run_root: Path,
    seed: int,
    experiment_path: Path | None = None,
) -> Path:
    upstream = bootstrap(repo_root)
    prepared = data_root / "paper2021" / "prepared"
    if experiment_path is None:
        pointer_path = run_root / f"seed_{seed}" / "run.json"
        if not pointer_path.exists():
            raise ReproductionError(f"Missing training pointer: {pointer_path}")
        experiment_path = Path(
            json.loads(pointer_path.read_text(encoding="utf-8"))["experiment_path"]
        )
    experiment = experiment_path.expanduser().resolve()
    runner_class = _load_upstream_runner(upstream)
    previous_cwd = Path.cwd()
    try:
        os.chdir(upstream)
        runner = runner_class(seed=seed)
        runner.evaluate(
            str(experiment),
            str(prepared / "logmel.hdf5"),
            str(prepared / "test" / "label.json"),
            str(prepared / "test" / "meta.csv"),
            pred_file="paper_predictions.tsv",
            event_file="paper_event.txt",
            segment_file="paper_segment.txt",
            psds_file="paper_psds.txt",
        )
        _copy_prediction_directory(experiment, "predictions_paper")
        runner.evaluate(
            str(experiment),
            str(prepared / "logmel.hdf5"),
            str(prepared / "test" / "label_random_query_seed1.json"),
            str(prepared / "test" / "meta.csv"),
            pred_file="random_query_predictions.tsv",
            event_file="random_query_event.txt",
            segment_file="random_query_segment.txt",
            psds_file="random_query_psds.txt",
        )
        _copy_prediction_directory(experiment, "predictions_random_query")
    finally:
        os.chdir(previous_cwd)
    return collect_paper_results(experiment, run_root / f"seed_{seed}" / "metrics.json")


def _parse_event_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "event_f1": r"F-measure \(F1-score\)\s*:\s*([0-9.]+)\s*%",
        "precision": r"Precision\s*:\s*([0-9.]+)\s*%",
        "recall": r"Recall\s*:\s*([0-9.]+)\s*%",
    }
    parsed: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ReproductionError(f"Could not parse {name} from {path}")
        parsed[name] = float(match.group(1)) / 100.0
    return parsed


def _parse_psds(path: Path) -> float:
    match = re.search(
        r"PSD-Score\s*:\s*([0-9.]+)", path.read_text(encoding="utf-8"), re.IGNORECASE
    )
    if not match:
        raise ReproductionError(f"Could not parse PSDS from {path}")
    return float(match.group(1))


def collect_paper_results(experiment: Path, output: Path) -> Path:
    metrics = _parse_event_metrics(experiment / "paper_event.txt")
    metrics["psds"] = _parse_psds(experiment / "paper_psds.txt")
    random_metrics = _parse_event_metrics(experiment / "random_query_event.txt")
    metrics["random_query_event_f1"] = random_metrics["event_f1"]
    checks = {}
    for name, target in PAPER_METRICS.items():
        value = metrics[name]
        delta = abs(value - target["expected"])
        checks[name] = {
            "value": value,
            "expected": target["expected"],
            "absolute_delta": delta,
            "absolute_tolerance": target["absolute_tolerance"],
            "pass": delta <= target["absolute_tolerance"],
        }
    payload = {
        "profile": "paper2021",
        "upstream_commit": UPSTREAM_COMMIT,
        "experiment_path": str(experiment),
        "metrics": metrics,
        "checks": checks,
        "evaluation_environment": environment_snapshot(),
        "pass": all(check["pass"] for check in checks.values()),
    }
    _json_dump(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def summarize_runs(run_root: Path, seeds: Sequence[int], output: Path) -> Path:
    records = []
    for seed in seeds:
        path = run_root / f"seed_{seed}" / "metrics.json"
        if not path.exists():
            raise ReproductionError(f"Missing seed result: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    metric_names = list(PAPER_METRICS)
    summary_metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [record["metrics"][name] for record in records]
        mean = sum(values) / len(values)
        denominator = len(values) - 1 if len(values) > 1 else 1
        variance = sum((value - mean) ** 2 for value in values) / denominator
        target = PAPER_METRICS[name]
        summary_metrics[name] = {
            "values": values,
            "mean": mean,
            "sample_std": variance**0.5,
            "expected": target["expected"],
            "absolute_delta": abs(mean - target["expected"]),
            "absolute_tolerance": target["absolute_tolerance"],
            "pass": abs(mean - target["expected"]) <= target["absolute_tolerance"],
        }
    payload = {
        "profile": "paper2021",
        "upstream_commit": UPSTREAM_COMMIT,
        "seeds": list(seeds),
        "metrics": summary_metrics,
        "pass": all(item["pass"] for item in summary_metrics.values()),
    }
    _json_dump(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=_path, default=Path.cwd())
    parser.add_argument("--data-root", type=_path, default=Path("external/tag2021"))
    parser.add_argument("--run-root", type=_path, default=Path("runs/tag2021"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Report dependencies, GPU, and environment class")
    subparsers.add_parser("bootstrap", help="Pin the official paper-era source checkout")
    download_parser = subparsers.add_parser("download", help="Download/extract paper audio")
    download_parser.add_argument("--archive", type=_path)
    prepare_parser = subparsers.add_parser("prepare", help="Resolve and audit paper labels/audio")
    prepare_parser.add_argument("--audio-root", type=_path)
    subparsers.add_parser("audit", help="Audit prepared data without training dependencies")
    subparsers.add_parser("features", help="Extract the paper's 64-bin log-mel features")

    train_parser = subparsers.add_parser("train", help="Train one paper baseline seed")
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--num-workers", type=int, default=4)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one trained seed")
    evaluate_parser.add_argument("--seed", type=int, default=1)
    evaluate_parser.add_argument("--experiment-path", type=_path)
    collect_parser = subparsers.add_parser("collect", help="Collect an existing experiment")
    collect_parser.add_argument("--experiment-path", type=_path, required=True)
    collect_parser.add_argument("--output", type=_path, required=True)
    summarize_parser = subparsers.add_parser("summarize", help="Aggregate multiple seeds")
    summarize_parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    summarize_parser.add_argument("--output", type=_path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    run_root = (repo_root / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root

    if args.command == "doctor":
        snapshot = environment_snapshot()
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        missing = [name for name, version in snapshot["packages"].items() if version is None]
        return 2 if missing else 0
    if args.command == "bootstrap":
        bootstrap(repo_root)
    elif args.command == "download":
        download_paper_audio(data_root, args.archive)
    elif args.command == "prepare":
        prepare_paper_data(repo_root, data_root, args.audio_root)
    elif args.command == "audit":
        report = audit_paper_data(data_root / "paper2021" / "prepared")
        _json_dump(report, data_root / "paper2021" / "prepared" / "data_audit.json")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 2
    elif args.command == "features":
        extract_paper_features(repo_root, data_root)
    elif args.command == "train":
        train_paper(repo_root, data_root, run_root, args.seed, args.num_workers)
    elif args.command == "evaluate":
        evaluate_paper(repo_root, data_root, run_root, args.seed, args.experiment_path)
    elif args.command == "collect":
        collect_paper_results(args.experiment_path.resolve(), args.output.resolve())
    elif args.command == "summarize":
        output = args.output or run_root / "reproduction_summary.json"
        summarize_runs(run_root, args.seeds, output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
