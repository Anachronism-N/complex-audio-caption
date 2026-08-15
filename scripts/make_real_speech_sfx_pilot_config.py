"""Materialize the server-local config for the LibriSpeech + ESC-50 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _required_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is missing: {resolved}")
    return resolved


def _passed_audit(
    path: Path,
    description: str,
    *,
    required_test_kinds: set[str],
    minimum_test_per_kind: int,
) -> Path:
    return _passed_split_audit(
        path,
        description,
        required_split="test",
        required_kinds=required_test_kinds,
        minimum_per_kind=minimum_test_per_kind,
    )


def _passed_split_audit(
    path: Path,
    description: str,
    *,
    required_split: str,
    required_kinds: set[str],
    minimum_per_kind: int,
) -> Path:
    """Bind one prepared catalog fold to an audit that explicitly covered it."""
    resolved = _required_file(path, description)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("pass") is not True:
        raise ValueError(f"{description} has not passed: {resolved}")
    if required_split not in set(payload.get("required_splits") or []):
        raise ValueError(
            f"{description} was not validated with --required-split "
            f"{required_split}: {resolved}"
        )
    split_counts = (payload.get("counts_by_split_kind") or {}).get(
        required_split
    ) or {}
    insufficient = {
        kind: int(split_counts.get(kind, 0))
        for kind in sorted(required_kinds)
        if int(split_counts.get(kind, 0)) < minimum_per_kind
    }
    if insufficient:
        raise ValueError(
            f"{description} has insufficient reviewed {required_split} sources: "
            f"minimum={minimum_per_kind} observed={insufficient}"
        )
    return resolved


def _rms_ready_catalog(path: Path, description: str) -> Path:
    resolved = _required_file(path, description)
    missing: list[str] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        absent = [
            field
            for field in ("rms_dbfs", "active_rms_dbfs")
            if row.get(field) is None
        ]
        if absent:
            missing.append(f"line {line_number}: {','.join(absent)}")
    if missing:
        raise ValueError(
            f"{description} predates the active-RMS gate; rerun source prepare "
            f"with the current code: {resolved}; examples={missing[:5]}"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument("--librispeech-prepared", required=True)
    parser.add_argument("--esc50-audio-root", required=True)
    parser.add_argument("--esc50-prepared", required=True)
    parser.add_argument("--fsd50k-root")
    parser.add_argument("--fsd50k-prepared")
    parser.add_argument("--urbansound8k-root")
    parser.add_argument("--urbansound8k-prepared")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    template_path = repo_root / "configs" / "data" / "real_speech_sfx_pilot_test.example.yaml"
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    librispeech_root = Path(args.librispeech_root).expanduser().resolve()
    esc50_audio_root = Path(args.esc50_audio_root).expanduser().resolve()
    if not librispeech_root.is_dir():
        raise FileNotFoundError(f"LibriSpeech root is missing: {librispeech_root}")
    if not esc50_audio_root.is_dir():
        raise FileNotFoundError(f"ESC-50 audio root is missing: {esc50_audio_root}")
    if bool(args.fsd50k_root) != bool(args.fsd50k_prepared):
        parser.error("--fsd50k-root and --fsd50k-prepared must be supplied together")
    if bool(args.urbansound8k_root) != bool(args.urbansound8k_prepared):
        parser.error(
            "--urbansound8k-root and --urbansound8k-prepared must be supplied together"
        )

    prepared = [
        (
            Path(args.librispeech_prepared),
            librispeech_root,
            "LibriSpeech",
            {"speech"},
            10,
        ),
        (
            Path(args.esc50_prepared),
            esc50_audio_root,
            "ESC-50",
            {"sfx", "ambience"},
            3,
        ),
    ]
    if args.fsd50k_root:
        fsd50k_root = Path(args.fsd50k_root).expanduser().resolve()
        if not fsd50k_root.is_dir():
            raise FileNotFoundError(f"FSD50K root is missing: {fsd50k_root}")
        prepared.append(
            (
                Path(args.fsd50k_prepared),
                fsd50k_root,
                "FSD50K",
                {"sfx", "ambience"},
                3,
            )
        )
    if args.urbansound8k_root:
        urbansound8k_root = Path(args.urbansound8k_root).expanduser().resolve()
        if not urbansound8k_root.is_dir():
            raise FileNotFoundError(
                f"UrbanSound8K root is missing: {urbansound8k_root}"
            )
        prepared.append(
            (
                Path(args.urbansound8k_prepared),
                urbansound8k_root,
                "UrbanSound8K",
                {"sfx"},
                3,
            )
        )
    catalogs: list[dict[str, str]] = []
    for prepared_root, audio_root, name, required_kinds, minimum_per_kind in prepared:
        root = prepared_root.expanduser().resolve()
        catalog = _rms_ready_catalog(root / "test.jsonl", f"{name} test catalog")
        audit = _passed_audit(
            root / "source_audit_report.json",
            f"{name} source audit",
            required_test_kinds=required_kinds,
            minimum_test_per_kind=minimum_per_kind,
        )
        catalogs.append(
            {
                "catalog_path": str(catalog),
                "audio_root": str(audio_root),
                "audit_report_path": str(audit),
                "expected_split": "test",
                "sampling_weight": 1.0,
            }
        )
    config["pool"]["catalogs"] = catalogs
    if args.fsd50k_root or args.urbansound8k_root:
        config["render"]["sample_count"] = 60
        config["render"]["scene_id_prefix"] = "expanded_speech_test"

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pilot config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"config={output}")
    print("source_audits=passed")
    profile = (
        "expanded_pilot"
        if args.fsd50k_root or args.urbansound8k_root
        else "pilot"
    )
    print(
        "next=run scripts/run_real_speech_sfx_pilot.sh "
        f"CONFIG OUTPUT_DIR {profile}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
