"""``sceneledger-render`` CLI: render a TAC-mini dataset from a config.

Mirrors the contract in ``docs/11_development_plan.md`` §5::

    python -m sceneledger.cli.render \
      --config configs/data/tac_mini.yaml \
      --output-dir data/derived/tac_mini

Produces ``manifest.jsonl``, ``data_card.md``, ``listen_list.csv`` and the
``.wav`` files. Re-run ``--validate`` to check deterministic replay, stems-sum
and ledger validity.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import yaml

from sceneledger.data.manifests import (
    ManifestEntry,
    audit_manifest_structure,
    persist_render,
    validate_manifest,
    write_manifest,
)
from sceneledger.data.renderer import render_scene
from sceneledger.data.scene_graph_sampler import (
    FileSourcePool,
    SceneGraphSampler,
    SceneSamplerConfig,
    SyntheticSourcePool,
)


def _build_pool(cfg: dict) -> object:
    pcfg = cfg.get("pool", {})
    kind = pcfg.get("kind", "synthetic")
    if kind == "synthetic":
        return SyntheticSourcePool(
            sample_rate=pcfg.get("sample_rate", 24000),
            seed=pcfg.get("seed", 20260808),
        )
    if kind == "file":
        mapping = pcfg.get("file_pool", {})
        expanded: dict[str, list[str]] = {}
        for source_kind, patterns in mapping.items():
            paths = sorted(
                {
                    str(Path(path).resolve())
                    for pattern in patterns
                    for path in glob.glob(str(pattern), recursive=True)
                    if Path(path).is_file()
                }
            )
            if not paths:
                raise ValueError(
                    f"file pool pattern(s) for {source_kind!r} matched no files: {patterns}"
                )
            expanded[source_kind] = paths
        return FileSourcePool(by_kind=expanded)
    raise ValueError(f"unknown pool kind {kind!r}")


def _build_sampler(cfg: dict, pool) -> SceneGraphSampler:
    scfg = cfg.get("sampler", {})
    config = SceneSamplerConfig(
        sample_rate=scfg.get("sample_rate", 24000),
        duration_range=tuple(scfg.get("duration_range", (10.0, 30.0))),
        gain_db_range=tuple(scfg.get("gain_db_range", (-12.0, 3.0))),
        fg_bg_snr_range=tuple(scfg.get("fg_bg_snr_range", (-10.0, 20.0))),
        t60_range=tuple(scfg.get("t60_range", (0.1, 1.2))),
        echo_delay_ms_range=tuple(scfg.get("echo_delay_ms_range", (80, 500))),
        echo_atten_db_range=tuple(scfg.get("echo_atten_db_range", (-18.0, -3.0))),
        repeat_range=tuple(scfg.get("repeat_range", (1, 5))),
        merge_threshold_range=tuple(scfg.get("merge_threshold_range", (0.1, 1.0))),
        resolutions=tuple(scfg.get("resolutions", (0.1, 0.5, 1.0))),
        styles=tuple(scfg.get("styles", ("keyword", "brief", "detailed"))),
        activity_threshold_range=tuple(scfg.get("activity_threshold_range", (0.03, 0.12))),
        p_rir=scfg.get("p_rir", 0.5),
        p_echo=scfg.get("p_echo", 0.3),
    )
    return SceneGraphSampler(pool=pool, config=config)


def _weighted_templates(weights: dict[str, float], n: int) -> list[str]:
    import random

    items = list(weights.items())
    total = sum(w for _, w in items)
    norm = [(k, w / total) for k, w in items]
    rng = random.Random(0)
    chosen: list[str] = []
    for _ in range(n):
        r = rng.random()
        acc = 0.0
        for k, p in norm:
            acc += p
            if r <= acc:
                chosen.append(k)
                break
        else:
            chosen.append(norm[-1][0])
    return chosen


def render_dataset(config_path: str, output_dir: str, limit: int | None = None) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pool = _build_pool(cfg)
    sampler = _build_sampler(cfg, pool)

    rcfg = cfg.get("render", {})
    n = rcfg.get("sample_count", 500)
    if limit is not None:
        n = min(n, limit)
    templates = _weighted_templates(rcfg.get("template_weights", {"speech_over_music": 1}), n)
    seed_base = rcfg.get("seed_base", 1947)

    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    for i, tpl in enumerate(templates):
        scene_id = f"mix_{i + 1:06d}"
        seed = seed_base + i * 31
        scene = sampler.sample(scene_id=scene_id, seed=seed, template=tpl)  # type: ignore[arg-type]
        out = render_scene(scene, pool)
        entry = persist_render(out, odir / "audio", rel_to=odir)
        entries.append(entry)
        if (i + 1) % 50 == 0:
            print(f"[render] {i + 1}/{n}", file=sys.stderr)

    manifest_path = odir / "manifest.jsonl"
    write_manifest(manifest_path, entries)
    structure_report = audit_manifest_structure(entries)
    if not structure_report.ok():
        preview = "\n".join(structure_report.errors[:20])
        raise RuntimeError(f"rendered manifest failed structural audit:\n{preview}")
    _write_data_card(odir, cfg, entries)
    _write_listen_list(odir, entries)
    print(
        f"[render] wrote {len(entries)} scenes -> {manifest_path}",
        file=sys.stderr,
    )
    return len(entries)


def _write_data_card(odir: Path, cfg: dict, entries: list[ManifestEntry]) -> None:
    from collections import Counter

    templates = Counter(e.scene["template"] for e in entries)
    durations = [e.scene["duration"] for e in entries]
    n_events = [len(e.target_ledger["events"]) for e in entries]
    lines = [
        "# TAC-mini data card",
        "",
        f"- generated: synthetic pool, {len(entries)} clips",
        f"- sample_rate: {entries[0].sample_rate if entries else 'n/a'} Hz",
        f"- duration range: {min(durations):.1f}–{max(durations):.1f} s"
        if durations
        else "- duration: n/a",
        f"- event count: min={min(n_events)} max={max(n_events)} mean={sum(n_events)/len(n_events):.2f}"
        if n_events
        else "",
        "",
        "## template distribution",
        "",
    ]
    for k, v in sorted(templates.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## config", "", "```yaml", yaml.safe_dump(cfg, sort_keys=False), "```"]
    (odir / "data_card.md").write_text("\n".join(lines), encoding="utf-8")


def _write_listen_list(odir: Path, entries: list[ManifestEntry]) -> None:
    import csv

    with (odir / "listen_list.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "template", "mixture_path", "duration", "n_events", "first_event_text"])
        for e in entries:
            evs = e.target_ledger.get("events", [])
            first_text = evs[0]["text"] if evs else ""
            w.writerow(
                [
                    e.scene["scene_id"],
                    e.scene["template"],
                    e.mixture_path,
                    e.scene["duration"],
                    len(evs),
                    first_text,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-render")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap sample count (smoke test)")
    parser.add_argument("--validate", action="store_true", help="validate after rendering")
    args = parser.parse_args(argv)

    render_dataset(args.config, args.output_dir, limit=args.limit)

    if args.validate:
        pool = _build_pool(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
        manifest = Path(args.output_dir) / "manifest.jsonl"
        rep = validate_manifest(manifest, pool, check_audio=True)
        print(
            f"[validate] replay ok={rep.n_replay_ok}/{rep.n_entries} "
            f"stems_sum ok={rep.n_stems_sum_ok} ledger_valid={rep.n_ledger_valid} "
            f"saved_reconstruction={rep.n_saved_reconstruction_ok} "
            f"failures={len(rep.failures)}",
            file=sys.stderr,
        )
        if rep.failures:
            for line in rep.failures[:20]:
                print(f"  FAIL {line}", file=sys.stderr)
        return 0 if rep.ok() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
