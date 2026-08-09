"""Export a rendered SceneLedger manifest to official MOSS-Audio SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneledger.data.datamodule import group_split, source_leakage
from sceneledger.data.manifests import (
    ManifestEntry,
    audit_manifest_structure,
    file_hash,
    read_manifest,
    write_manifest,
)
from sceneledger.data.schema import SCHEMA_VERSION, Ledger
from sceneledger.models.target_formatter import (
    StyleConfig,
    canonical_prompt,
    format_atomic_caption,
    format_xml_caption,
)


def _target(entry: ManifestEntry, mode: str, style: str, include_tracks: bool) -> str:
    ledger = Ledger.model_validate(entry.target_ledger)
    if mode == "atomic":
        return format_atomic_caption(
            ledger, style=style, cfg=StyleConfig(), include_tracks=include_tracks
        )
    return format_xml_caption(ledger, style=style, cfg=StyleConfig())


def _conversation_row(
    entry: ManifestEntry,
    audio_base: Path,
    prompt: str,
    mode: str,
    style: str,
    include_tracks: bool,
) -> dict:
    audio_path = (audio_base / entry.mixture_path).resolve()
    return {
        "sample_id": entry.scene["scene_id"],
        "conversation": [
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
            {"role": "user", "message_type": "text", "content": prompt},
            {
                "role": "assistant",
                "message_type": "text",
                "content": _target(entry, mode, style, include_tracks),
            },
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_moss_sft(
    *,
    manifest_path: str | Path,
    audio_base: str | Path,
    output_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 20260808,
    group_key: str = "source_id",
    target_mode: str = "atomic",
    style: str = "brief",
    include_lyrics: bool = False,
    include_tracks: bool = False,
    allow_missing_audio: bool = False,
    allow_invalid_manifest: bool = False,
    allow_placeholder_lyrics: bool = False,
) -> dict:
    if target_mode not in {"atomic", "xml"}:
        raise ValueError("target_mode must be 'atomic' or 'xml'")
    manifest = Path(manifest_path).resolve()
    audio_root = Path(audio_base).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    entries = read_manifest(manifest)
    placeholder_lyrics = [
        entry.scene["scene_id"]
        for entry in entries
        if any(
            source.get("kind") == "vocal"
            and (
                str(source.get("path", "")).startswith("vocal:")
                or "synthetic"
                in str(entry.target_ledger.get("provenance", {}).get("source_dataset", ""))
            )
            for source in entry.scene.get("sources", [])
        )
    ]
    if include_lyrics and placeholder_lyrics and not allow_placeholder_lyrics:
        raise ValueError(
            f"{len(placeholder_lyrics)} scene(s) contain synthetic vocal placeholders; "
            "they cannot supervise verbatim <lys> text. Use a real source catalog, or "
            "pass --allow-placeholder-lyrics for renderer smoke tests only."
        )
    audit = audit_manifest_structure(entries)
    if not audit.ok() and not allow_invalid_manifest:
        preview = "\n".join(audit.errors[:10])
        raise ValueError(
            "manifest failed structural audit; re-render before training or pass "
            f"--allow-invalid-manifest for diagnosis only:\n{preview}"
        )

    missing_audio = [
        str((audio_root / entry.mixture_path).resolve())
        for entry in entries
        if not (audio_root / entry.mixture_path).exists()
    ]
    if missing_audio and not allow_missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} mixture files are missing; first={missing_audio[0]}"
        )

    train_entries, val_entries = group_split(
        entries, val_fraction=val_fraction, group_key=group_key, seed=seed
    )
    leaked = source_leakage(train_entries, val_entries) if group_key == "source_id" else set()
    if leaked:
        raise AssertionError(f"source leakage detected: {sorted(leaked)[:10]}")

    prompt = canonical_prompt(
        style=style,
        include_lyrics=include_lyrics,
        include_tracks=include_tracks,
        output_mode=target_mode,
    )
    train_rows = [
        _conversation_row(entry, audio_root, prompt, target_mode, style, include_tracks)
        for entry in train_entries
    ]
    val_rows = [
        _conversation_row(entry, audio_root, prompt, target_mode, style, include_tracks)
        for entry in val_entries
    ]
    _write_jsonl(destination / "train.jsonl", train_rows)
    _write_jsonl(destination / "val.jsonl", val_rows)
    write_manifest(destination / "train_manifest.jsonl", train_entries)
    write_manifest(destination / "val_manifest.jsonl", val_entries)
    _write_jsonl(
        destination / "val_references.jsonl",
        [entry.target_ledger for entry in val_entries],
    )

    split = {
        "train": [entry.scene["scene_id"] for entry in train_entries],
        "val": [entry.scene["scene_id"] for entry in val_entries],
    }
    (destination / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest),
        "manifest_sha256": file_hash(manifest),
        "renderer_versions": sorted(
            {entry.renderer_version or "unknown" for entry in entries}
        ),
        "audio_base": str(audio_root),
        "n_total": len(entries),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "seed": seed,
        "val_fraction": val_fraction,
        "group_key": group_key,
        "source_leakage_count": len(leaked),
        "target_mode": target_mode,
        "style": style,
        "include_lyrics": include_lyrics,
        "include_tracks": include_tracks,
        "prompt": prompt,
        "missing_audio_count": len(missing_audio),
        "structural_audit_ok": audit.ok(),
        "structural_audit_errors": audit.errors[:100],
        "placeholder_lyrics_count": len(placeholder_lyrics),
        "placeholder_lyrics_allowed": allow_placeholder_lyrics,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-prepare-moss-sft")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--group-key", default="source_id")
    parser.add_argument("--target-mode", choices=["atomic", "xml"], default="atomic")
    parser.add_argument("--style", choices=["keyword", "brief", "detailed"], default="brief")
    parser.add_argument("--include-lyrics", action="store_true")
    parser.add_argument(
        "--include-tracks",
        action="store_true",
        help="preserve source track and speaker/singer identity attributes in atomic targets",
    )
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument("--allow-invalid-manifest", action="store_true")
    parser.add_argument(
        "--allow-placeholder-lyrics",
        action="store_true",
        help="smoke-test only: allow synthetic vocal waveforms to enter an SFT export",
    )
    args = parser.parse_args(argv)
    metadata = export_moss_sft(
        manifest_path=args.manifest,
        audio_base=args.audio_base,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        group_key=args.group_key,
        target_mode=args.target_mode,
        style=args.style,
        include_lyrics=args.include_lyrics,
        include_tracks=args.include_tracks,
        allow_missing_audio=args.allow_missing_audio,
        allow_invalid_manifest=args.allow_invalid_manifest,
        allow_placeholder_lyrics=args.allow_placeholder_lyrics,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
