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

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sceneledger.data.renderer import (
    RENDERER_VERSION,
    RESIDUAL_STEM_ID,
    RenderOutput,
    render_scene,
)
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
    renderer_version: str | None = None
    mixture_file_hash: str | None = None
    stem_file_hashes: dict[str, str] = field(default_factory=dict)

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
            renderer_version=d.get("renderer_version"),
            mixture_file_hash=d.get("mixture_file_hash"),
            stem_file_hashes=d.get("stem_file_hashes", {}),
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
            language=s.get("language"),
            verbatim=s.get("verbatim"),
            source_group=s.get("source_group"),
            license=s.get("license"),
            dataset=s.get("dataset"),
            repeat=s.get("repeat", 1),
            repeat_gap_s=s.get("repeat_gap_s", 0.0),
            rir_id=s.get("rir_id"),
            t60_sec=s.get("t60_sec"),
            is_foreground=s.get("is_foreground", True),
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
        ),
        supervision=Supervision(
            style=sup.get("style", "brief"),
            activity_threshold=sup.get("activity_threshold", 0.05),
            merge_threshold_s=sup.get("merge_threshold_s", 0.25),
            resolution_s=sup.get("resolution_s", 0.1),
        ),
        sample_rate=d.get("sample_rate", 24000),
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
    return hashlib.sha256(mask.tobytes()).hexdigest()[:16]


def file_hash(path: str | Path) -> str:
    """SHA-256 of the exact bytes persisted on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    residual_path = stems_dir / f"{sid}_{RESIDUAL_STEM_ID}.wav"
    sf.write(residual_path, out.residual_stem, out.sample_rate, subtype="PCM_16")
    stem_paths[RESIDUAL_STEM_ID] = str(residual_path)
    stem_hashes[RESIDUAL_STEM_ID] = out.waveform_hash(out.residual_stem)

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
        renderer_version=RENDERER_VERSION,
        mixture_file_hash=file_hash(mixture_path),
        stem_file_hashes={key: file_hash(path) for key, path in stem_paths.items()},
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
    n_audio_files_ok: int = 0
    n_audio_files_fail: int = 0
    n_saved_reconstruction_ok: int = 0
    n_saved_reconstruction_fail: int = 0
    failures: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return (
            self.n_replay_fail == 0
            and self.n_stems_sum_fail == 0
            and self.n_ledger_invalid == 0
            and self.n_audio_files_fail == 0
            and self.n_saved_reconstruction_fail == 0
        )


@dataclass
class StructureAuditReport:
    """Cheap manifest audit that does not render or load audio."""

    n_entries: int = 0
    n_valid: int = 0
    errors: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors and self.n_valid == self.n_entries


def audit_manifest_structure(entries: list[ManifestEntry]) -> StructureAuditReport:
    report = StructureAuditReport(n_entries=len(entries))
    seen_scene_ids: set[str] = set()
    for entry in entries:
        scene_id = str(entry.scene.get("scene_id", ""))
        entry_errors: list[str] = []
        if not scene_id:
            entry_errors.append("missing scene_id")
        elif scene_id in seen_scene_ids:
            entry_errors.append(f"duplicate scene_id {scene_id}")
        seen_scene_ids.add(scene_id)

        sources = entry.scene.get("sources", [])
        source_ids = [str(source.get("source_id", "")) for source in sources]
        duplicates = sorted(
            {source_id for source_id in source_ids if source_ids.count(source_id) > 1}
        )
        if duplicates:
            entry_errors.append(f"duplicate source IDs {duplicates}")
        if any(not source.get("path") for source in sources):
            entry_errors.append("source without path")

        expected_components = set(source_ids) | {RESIDUAL_STEM_ID}
        actual_components = set(entry.stem_paths)
        if actual_components != expected_components:
            entry_errors.append(
                "component keys differ from sources+residual "
                f"(missing={sorted(expected_components - actual_components)}, "
                f"extra={sorted(actual_components - expected_components)})"
            )
        try:
            Ledger.model_validate(entry.target_ledger)
        except Exception as exc:
            entry_errors.append(f"invalid target ledger: {exc}")

        if entry_errors:
            report.errors.extend(f"{scene_id or '<unknown>'}: {error}" for error in entry_errors)
        else:
            report.n_valid += 1
    return report


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

        # 4. (optional) persisted files match their byte hashes and the saved
        # component stems reconstruct the saved mixture within PCM16 error.
        if check_audio:
            mp = base / e.mixture_path
            component_paths = {
                key: base / rel_path for key, rel_path in e.stem_paths.items()
            }
            missing = [str(mp)] if not mp.exists() else []
            missing.extend(str(path) for path in component_paths.values() if not path.exists())
            if missing:
                rep.n_audio_files_fail += len(missing)
                rep.n_saved_reconstruction_fail += 1
                rep.failures.append(
                    f"{scene.scene_id}: missing persisted audio: {missing[:5]}"
                )
                continue

            hash_failures: list[str] = []
            if e.mixture_file_hash and file_hash(mp) != e.mixture_file_hash:
                hash_failures.append("mixture")
            for key, path in component_paths.items():
                expected = e.stem_file_hashes.get(key)
                if expected and file_hash(path) != expected:
                    hash_failures.append(key)
            if hash_failures:
                rep.n_audio_files_fail += len(hash_failures)
                rep.failures.append(
                    f"{scene.scene_id}: persisted file hash mismatch: {hash_failures}"
                )
            else:
                rep.n_audio_files_ok += 1 + len(component_paths)

            import soundfile as sf

            saved_mix, mix_sr = sf.read(mp, dtype="float32", always_2d=False)
            if saved_mix.ndim == 2:
                saved_mix = saved_mix.mean(axis=1)
            reconstructed = np.zeros_like(saved_mix)
            component_error = None
            for key, path in component_paths.items():
                stem, stem_sr = sf.read(path, dtype="float32", always_2d=False)
                if stem.ndim == 2:
                    stem = stem.mean(axis=1)
                if stem_sr != mix_sr or stem.shape != saved_mix.shape:
                    component_error = (
                        f"component {key} shape/sr {stem.shape}/{stem_sr} "
                        f"!= mixture {saved_mix.shape}/{mix_sr}"
                    )
                    break
                reconstructed += stem
            pcm_tolerance = (len(component_paths) + 1) / 32768.0
            if component_error or not np.allclose(
                reconstructed, saved_mix, atol=pcm_tolerance, rtol=0.0
            ):
                rep.n_saved_reconstruction_fail += 1
                max_error = float(np.max(np.abs(reconstructed - saved_mix)))
                rep.failures.append(
                    f"{scene.scene_id}: saved stems do not reconstruct mixture "
                    f"(max_error={max_error:.8f}, detail={component_error})"
                )
            else:
                rep.n_saved_reconstruction_ok += 1

    return rep


__all__ = [
    "ManifestEntry",
    "StructureAuditReport",
    "ValidationReport",
    "activity_hash",
    "audit_manifest_structure",
    "file_hash",
    "persist_render",
    "read_manifest",
    "scene_from_dict",
    "validate_manifest",
    "write_manifest",
]
