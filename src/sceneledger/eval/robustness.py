"""Robustness curves: stratify corpus metrics by acoustic conditions.

``configs/experiment_matrix.yaml`` lists the robustness axes that the paper
must report:

    SNR, T60, echo_delay, codec, source_count, overlap_ratio, event_duration

This module joins per-sample :class:`SampleMetrics` with per-sample conditions
(from the manifest) and produces bucketed macro-aggregates so we can plot
"metric vs. axis" curves and expose failure modes that a single headline
number hides (e.g. F1 collapse under high overlap or long T60).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from sceneledger.eval.metrics import SampleMetrics, _macro


@dataclass
class StratumMetrics:
    n: int
    event_f1: float
    event_precision: float
    event_recall: float
    seg_f1_100ms: float
    onset_mae: float
    offset_mae: float
    hallucination: int
    omission: int
    pointer_accuracy: float


def _aggregate_stratum(samples: list[SampleMetrics]) -> StratumMetrics:
    if not samples:
        return StratumMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return StratumMetrics(
        n=len(samples),
        event_f1=_macro([s.event_f1 for s in samples]),
        event_precision=_macro([s.event_precision for s in samples]),
        event_recall=_macro([s.event_recall for s in samples]),
        seg_f1_100ms=_macro([s.seg_f1_100ms for s in samples]),
        onset_mae=_macro([s.onset_mae for s in samples]),
        offset_mae=_macro([s.offset_mae for s in samples]),
        hallucination=sum(s.hallucination for s in samples),
        omission=sum(s.omission for s in samples),
        pointer_accuracy=_macro([s.pointer_accuracy for s in samples]),
    )


def _bucket(value: float | None, edges: list[float]) -> str:
    """Assign ``value`` to a bucket defined by ``edges`` (lower-inclusive)."""
    if value is None:
        return "unknown"
    for e in edges:
        if value < e:
            return f"<{e}"
    return f">={edges[-1]}"


def _bucket_continuous(
    samples: list[tuple[SampleMetrics, dict]],
    key: str,
    edges: list[float],
) -> dict[str, StratumMetrics]:
    groups: dict[str, list[SampleMetrics]] = defaultdict(list)
    for sm, cond in samples:
        v = cond.get(key)
        groups[_bucket(v, edges) if v is not None else "unknown"].append(sm)
    return {k: _aggregate_stratum(v) for k, v in sorted(groups.items())}


def _bucket_discrete(
    samples: list[tuple[SampleMetrics, dict]],
    key: str,
) -> dict[str, StratumMetrics]:
    groups: dict[str, list[SampleMetrics]] = defaultdict(list)
    for sm, cond in samples:
        v = cond.get(key)
        groups[str(v) if v is not None else "unknown"].append(sm)
    return {k: _aggregate_stratum(v) for k, v in sorted(groups.items())}


def _source_count(cond: dict) -> int:
    return len(cond.get("sources", []))


def stratify(
    samples: list[SampleMetrics],
    conditions_by_sample: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """Return ``{axis: {bucket: {metric: value}}}``.

    Conditions are read from the manifest's per-scene dict (``scene`` +
    ``conditions``). Source count is derived from the source list.
    """
    joined: list[tuple[SampleMetrics, dict]] = []
    for sm in samples:
        sid = sm.sample_id
        cond = conditions_by_sample.get(sid, {})
        # flatten: scene-level fields + conditions sub-dict
        flat = {
            "template": cond.get("template"),
            "duration": cond.get("duration"),
            "t60_sec": (cond.get("conditions") or {}).get("t60_sec"),
            "echo_delay_ms": (cond.get("conditions") or {}).get("echo_delay_ms"),
            "overlap_ratio": (cond.get("conditions") or {}).get("overlap_ratio"),
            "snr_db": (cond.get("conditions") or {}).get("noise_snr_db"),
            "codec": (cond.get("conditions") or {}).get("codec"),
            "source_count": _source_count(cond),
            "n_events": len(cond.get("events", [])) if "events" in cond else None,
        }
        joined.append((sm, flat))

    out: dict[str, dict[str, dict]] = {}

    # continuous axes
    out["t60_sec"] = {
        k: asdict(v) for k, v in _bucket_continuous(joined, "t60_sec", [0.3, 0.6, 0.9]).items()
    }
    out["overlap_ratio"] = {
        k: asdict(v)
        for k, v in _bucket_continuous(joined, "overlap_ratio", [0.1, 0.3, 0.5]).items()
    }
    out["duration"] = {
        k: asdict(v)
        for k, v in _bucket_continuous(joined, "duration", [15.0, 20.0, 25.0]).items()
    }

    # discrete axes
    out["source_count"] = {
        k: asdict(v) for k, v in _bucket_discrete(joined, "source_count").items()
    }
    out["template"] = {
        k: asdict(v) for k, v in _bucket_discrete(joined, "template").items()
    }
    out["echo"] = {
        k: asdict(v)
        for k, v in _bucket_discrete(joined, "echo_delay_ms").items()
    }

    return out


def load_conditions_from_manifest(manifest_path: str | Path) -> dict[str, dict]:
    """Read a TAC-mini manifest and return ``{sample_id: scene_dict}``.

    Merges ``overlap_ratio`` from the target ledger's conditions (computed by
    the renderer) into the scene-level conditions so the overlap axis works.
    """
    p = Path(manifest_path)
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        scene = obj.get("scene", obj)
        # merge renderer-computed conditions from the target ledger
        ledger_cond = (obj.get("target_ledger") or {}).get("conditions") or {}
        scene = dict(scene)
        merged_cond = dict(scene.get("conditions") or {})
        for k in ("overlap_ratio", "t60_sec", "snr_db", "echo"):
            if k in ledger_cond and merged_cond.get(k) is None:
                merged_cond[k] = ledger_cond[k]
        scene["conditions"] = merged_cond
        out[scene["scene_id"]] = scene
    return out


def robustness_report(
    metrics_report_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Build a robustness report from a metrics JSON + manifest."""
    metrics = json.loads(Path(metrics_report_path).read_text(encoding="utf-8"))
    conds = load_conditions_from_manifest(manifest_path)
    # reconstruct SampleMetrics from the saved per-sample dicts
    from sceneledger.eval.metrics import SampleMetrics

    samples = [SampleMetrics(**s) for s in metrics.get("samples", [])]
    strat = stratify(samples, conds)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(strat, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return strat


__all__ = [
    "StratumMetrics",
    "load_conditions_from_manifest",
    "robustness_report",
    "stratify",
]
