"""Manifest (de)serialization and validation for rendered scenes.

Each manifest entry captures everything needed to:

1. rebuild the :class:`Scene` and re-render the mixture deterministically,
2. locate the frozen audio files (mixture + stems),
3. recover the supervision target :class:`Ledger`.

The manifest is JSONL (one scene per line). ``docs/06`` §3.1 specifies a YAML
scene manifest; we embed the same fields inside a JSON line so the whole
dataset is one streamable file, and additionally store hashes + the target
ledger so downstream training never needs to re-run the renderer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sceneledger.data.renderer import RenderOutput, render_scene
from sceneledger.data.scene_graph_sampler import (
    Conditions,
    PlacedSource,
    Scene,
    SourcePool,
    Supervision,
)
from sceneledger.data.schema import Ledger

ManifestDict = dict


@dataclass
class ManifestEntry:
    scene: dict
    mixture_path: str
    stem_paths: dict[str, str]  # source_id -> path
    mixture_hash: str
    dry_mixture_hash: str
    stem_hashes: dict[str, str]
    activity_hashes: dict[str, str]
    target_ledger: dict  # canonical Ledger as JSON-able dict
    sample_rate: int

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> ManifestEntry:
        return cls(
            scene=d["scene"],
            mixture_path=d["mixture_path"],
            stem_paths=d["stem_paths"],
            mixture_hash=d["mixture_hash"],
            dry_mixture_hash=d["dry_mixture_hash"],
            stem_hashes=d["stem_hashes"],
            activity_hashes=d["activity_hashes"],
            target_ledger=d["target_ledger"],
            sample_rate=d["sample_rate"],
        )


# --------------------------------------------------------------------------- #
# Scene <-> dict (round-trippable, used for deterministic replay)
# --------------------------------------------------------------------------- #
def scene_from_dict(d: dict) -> Scene:
    sources = [
        PlacedSource(
            source_id=s["source_id"],
            kind=s["kind"],  # type: ignore[arg-type]
            path=s["path"],
            onset=s["onset"],
            gain_db=s["gain_db"],
            text=s["text"],
            identity=s.get("identity"),
            track_group=s.get("track_group"),
            source_group=s.get("source_group"),
            leakage_groups=list(s.get("leakage_groups", [])),
            source_labels=list(s.get("source_labels", [])),
            source_dataset=s.get("source_dataset"),
            source_license=s.get("source_license"),
            annotation_origin=s.get("annotation_origin"),
            text_is_verbatim=s.get("text_is_verbatim", False),
            source_file_sha256=s.get("source_file_sha256"),
            source_duration_sec=s.get("source_duration_sec"),
            source_rms_dbfs=s.get("source_rms_dbfs"),
            source_active_rms_dbfs=s.get("source_active_rms_dbfs"),
            crop_start_sec=s.get("crop_start_sec", 0.0),
            crop_duration_sec=s.get("crop_duration_sec"),
            fade_in_sec=s.get("fade_in_sec"),
            fade_out_sec=s.get("fade_out_sec"),
            repeat=s.get("repeat", 1),
            repeat_gap_s=s.get("repeat_gap_s", 0.0),
            rir_id=s.get("rir_id"),
            t60_sec=s.get("t60_sec"),
            is_foreground=s.get("is_foreground", True),
            loop_to_scene=s.get("loop_to_scene", False),
        )
        for s in d["sources"]
    ]
    cond = d.get("conditions") or {}
    sup = d.get("supervision") or {}
    return Scene(
        scene_id=d["scene_id"],
        seed=d["seed"],
        duration=d["duration"],
        template=d["template"],  # type: ignore[arg-type]
        sources=sources,
        conditions=Conditions(
            noise_snr_db=cond.get("noise_snr_db"),
            echo_delay_ms=cond.get("echo_delay_ms"),
            echo_atten_db=cond.get("echo_atten_db"),
            t60_sec=cond.get("t60_sec"),
            codec=cond.get("codec"),
            overlap_ratio=cond.get("overlap_ratio"),
            ducking_enabled=cond.get("ducking_enabled", False),
            ducking_depth_db=cond.get("ducking_depth_db"),
        ),
        supervision=Supervision(
            style=sup.get("style", "brief"),
            activity_threshold=sup.get("activity_threshold", 0.05),
            merge_threshold_s=sup.get("merge_threshold_s", 0.25),
            resolution_s=sup.get("resolution_s", 0.1),
        ),
        sample_rate=d.get("sample_rate", 24000),
        recipe_metadata=dict(d.get("recipe_metadata") or {}),
    )


# --------------------------------------------------------------------------- #
# write / read
# --------------------------------------------------------------------------- #
def write_manifest(path: str | Path, entries: Iterable[ManifestEntry]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_json_line() + "\n")
            count += 1
    return count


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    p = Path(path)
    out: list[ManifestEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ManifestEntry.from_dict(json.loads(line)))
    return out


def activity_hash(mask: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(mask.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# persistence of a single render
# --------------------------------------------------------------------------- #
def persist_render(
    out: RenderOutput, output_dir: str | Path, rel_to: str | Path | None = None
) -> ManifestEntry:
    """Write mixture + stems + return a manifest entry (paths relative to ``rel_to``)."""
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sid = out.scene.scene_id
    mixture_path = odir / f"{sid}.wav"
    sf.write(mixture_path, out.mixture, out.sample_rate, subtype="PCM_16")

    stem_paths: dict[str, str] = {}
    stem_hashes: dict[str, str] = {}
    activity_hashes: dict[str, str] = {}
    stems_dir = odir / "stems"
    stems_dir.mkdir(exist_ok=True)
    for rs in out.stems:
        sp = stems_dir / f"{sid}_{rs.placed.source_id}.wav"
        sf.write(sp, rs.stem, out.sample_rate, subtype="PCM_16")
        stem_paths[rs.placed.source_id] = str(sp)
        stem_hashes[rs.placed.source_id] = out.waveform_hash(rs.stem)
        activity_hashes[rs.placed.source_id] = activity_hash(rs.activity.activity_mask)

    def _rel(p: Path) -> str:
        if rel_to is None:
            return str(p)
        try:
            return str(p.relative_to(rel_to))
        except ValueError:
            return str(p)

    return ManifestEntry(
        scene=out.scene.to_manifest_dict(),
        mixture_path=_rel(mixture_path),
        stem_paths={k: _rel(Path(v)) for k, v in stem_paths.items()},
        mixture_hash=out.mixture_hash(),
        dry_mixture_hash=out.waveform_hash(out.dry_mixture),
        stem_hashes=stem_hashes,
        activity_hashes=activity_hashes,
        target_ledger=out.target_ledger.model_dump(mode="json"),
        sample_rate=out.sample_rate,
    )


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    n_entries: int = 0
    n_replay_ok: int = 0
    n_replay_fail: int = 0
    n_stems_sum_ok: int = 0
    n_stems_sum_fail: int = 0
    n_ledger_valid: int = 0
    n_ledger_invalid: int = 0
    failures: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return (
            self.n_replay_fail == 0
            and self.n_stems_sum_fail == 0
            and self.n_ledger_invalid == 0
        )


def validate_manifest(
    path: str | Path, pool: SourcePool, check_audio: bool = True
) -> ValidationReport:
    """Validate a manifest: deterministic replay, stems-sum, ledger validity."""
    entries = read_manifest(path)
    rep = ValidationReport(n_entries=len(entries))
    base = Path(path).parent

    for e in entries:
        scene = scene_from_dict(e.scene)
        try:
            out = render_scene(scene, pool)
        except Exception as exc:  # pragma: no cover - surfaced to user
            rep.n_replay_fail += 1
            rep.failures.append(f"{scene.scene_id}: render failed: {exc}")
            continue

        # 1. deterministic replay: mixture hash must match
        if out.mixture_hash() != e.mixture_hash:
            rep.n_replay_fail += 1
            rep.failures.append(
                f"{scene.scene_id}: mixture hash mismatch "
                f"(got {out.mixture_hash()}, expected {e.mixture_hash})"
            )
        else:
            rep.n_replay_ok += 1

        # 2. stems sum to dry mixture (exact float32 add)
        dry = np.zeros_like(out.dry_mixture)
        for rs in out.stems:
            dry += rs.stem
        if np.array_equal(dry, out.dry_mixture):
            rep.n_stems_sum_ok += 1
        else:
            rep.n_stems_sum_fail += 1
            rep.failures.append(f"{scene.scene_id}: stems do not sum to dry mixture")

        # 3. target ledger is schema-valid (re-validate via Pydantic)
        try:
            Ledger.model_validate(e.target_ledger)
            rep.n_ledger_valid += 1
        except Exception as exc:
            rep.n_ledger_invalid += 1
            rep.failures.append(f"{scene.scene_id}: invalid target ledger: {exc}")

        # 4. (optional) audio file exists
        if check_audio:
            mp = base / e.mixture_path
            if not mp.exists():
                rep.failures.append(f"{scene.scene_id}: mixture audio missing at {mp}")

    return rep


__all__ = [
    "ManifestEntry",
    "ValidationReport",
    "activity_hash",
    "persist_render",
    "read_manifest",
    "scene_from_dict",
    "validate_manifest",
    "write_manifest",
]
