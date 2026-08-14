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
import sys
from pathlib import Path

import yaml

from sceneledger.data.manifests import (
    ManifestEntry,
    persist_render,
    validate_manifest,
    write_manifest,
)
from sceneledger.data.renderer import render_scene
from sceneledger.data.scene_graph_sampler import (
    CatalogSetSourcePool,
    CatalogSourcePool,
    FileSourcePool,
    Scene,
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
            index_range=tuple(pcfg.get("index_range", (0, 999))),
        )
    elif kind == "file":
        mapping = pcfg.get("file_pool", {})
        return FileSourcePool(by_kind={k: list(v) for k, v in mapping.items()})
    elif kind == "catalog":
        catalog_path = pcfg.get("catalog_path")
        if not catalog_path:
            raise ValueError("pool.catalog_path is required when pool.kind=catalog")
        audit_report_path = pcfg.get("audit_report_path")
        if not audit_report_path:
            raise ValueError(
                "pool.audit_report_path is required when pool.kind=catalog; "
                "complete and validate the human source audit before rendering"
            )
        expected_split = pcfg.get("expected_split")
        if expected_split not in {"train", "val", "test"}:
            raise ValueError(
                "pool.expected_split must be one of train/val/test when pool.kind=catalog"
            )
        return CatalogSourcePool(
            catalog_path=str(catalog_path),
            audio_root=pcfg.get("audio_root"),
            audit_report_path=str(audit_report_path),
            expected_split=str(expected_split),
        )
    elif kind == "catalog_set":
        catalog_configs = pcfg.get("catalogs")
        if not isinstance(catalog_configs, list) or not catalog_configs:
            raise ValueError("pool.catalogs must be a non-empty list for pool.kind=catalog_set")
        catalogs: list[CatalogSourcePool] = []
        sampling_weights: list[float] = []
        for index, item in enumerate(catalog_configs):
            if not isinstance(item, dict):
                raise ValueError(f"pool.catalogs[{index}] must be a mapping")
            missing = [
                field
                for field in ("catalog_path", "audio_root", "audit_report_path", "expected_split")
                if not item.get(field)
            ]
            if missing:
                raise ValueError(f"pool.catalogs[{index}] missing required fields: {missing}")
            if item["expected_split"] not in {"train", "val", "test"}:
                raise ValueError(f"pool.catalogs[{index}].expected_split is invalid")
            catalogs.append(
                CatalogSourcePool(
                    catalog_path=str(item["catalog_path"]),
                    audio_root=str(item["audio_root"]),
                    audit_report_path=str(item["audit_report_path"]),
                    expected_split=str(item["expected_split"]),
                )
            )
            sampling_weights.append(float(item.get("sampling_weight", 1.0)))
        observed_splits = {catalog.expected_split for catalog in catalogs}
        if len(observed_splits) != 1:
            raise ValueError(f"catalog_set cannot mix different splits: {sorted(observed_splits)}")
        return CatalogSetSourcePool(catalogs, sampling_weights=sampling_weights)
    raise ValueError(f"unknown pool kind {kind!r}")


def _build_sampler(cfg: dict, pool) -> SceneGraphSampler:
    scfg = cfg.get("sampler", {})
    config = SceneSamplerConfig(
        sample_rate=scfg.get("sample_rate", 24000),
        duration_range=tuple(scfg.get("duration_range", (10.0, 30.0))),
        template_duration_ranges={
            name: tuple(value)
            for name, value in scfg.get("template_duration_ranges", {}).items()
        },
        gain_db_range=tuple(scfg.get("gain_db_range", (-12.0, 3.0))),
        kind_gain_db_offsets={
            str(kind): float(offset)
            for kind, offset in scfg.get("kind_gain_db_offsets", {}).items()
        },
        target_active_rms_dbfs_by_kind={
            str(kind): tuple(float(item) for item in value)
            for kind, value in scfg.get(
                "target_active_rms_dbfs_by_kind", {}
            ).items()
        },
        max_abs_source_gain_db=(
            float(scfg["max_abs_source_gain_db"])
            if scfg.get("max_abs_source_gain_db") is not None
            else None
        ),
        fg_bg_snr_range=tuple(scfg.get("fg_bg_snr_range", (-10.0, 20.0))),
        t60_range=tuple(scfg.get("t60_range", (0.1, 1.2))),
        echo_delay_ms_range=tuple(scfg.get("echo_delay_ms_range", (80, 500))),
        echo_atten_db_range=tuple(scfg.get("echo_atten_db_range", (-18.0, -3.0))),
        repeat_range=tuple(scfg.get("repeat_range", (1, 5))),
        merge_threshold_range=tuple(scfg.get("merge_threshold_range", (0.1, 1.0))),
        resolutions=tuple(scfg.get("resolutions", (0.1, 0.5, 1.0))),
        styles=tuple(scfg.get("styles", ("keyword", "brief", "detailed"))),
        activity_threshold_range=tuple(scfg.get("activity_threshold_range", (0.03, 0.12))),
        foreground_onset_fraction_range=(
            tuple(scfg["foreground_onset_fraction_range"])
            if "foreground_onset_fraction_range" in scfg
            else None
        ),
        loop_background_to_scene=scfg.get("loop_background_to_scene", False),
        enforce_speaker_overlap=scfg.get("enforce_speaker_overlap", False),
        dense_repeated_event=scfg.get("dense_repeated_event", False),
        spread_repeated_event=scfg.get("spread_repeated_event", False),
        stable_unique_source_ids=scfg.get("stable_unique_source_ids", False),
        ducking_probability=scfg.get("ducking_probability", 0.7),
        ducking_depth_db_range=tuple(
            scfg.get("ducking_depth_db_range", (2.0, 5.0))
        ),
        p_rir=scfg.get("p_rir", 0.5),
        p_echo=scfg.get("p_echo", 0.3),
    )
    return SceneGraphSampler(pool=pool, config=config)


def _weighted_templates(
    weights: dict[str, float], n: int, seed: int = 0
) -> list[str]:
    import random

    items = list(weights.items())
    total = sum(w for _, w in items)
    norm = [(k, w / total) for k, w in items]
    rng = random.Random(seed)
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


def sample_scene_plan(config_path: str, limit: int | None = None) -> list[Scene]:
    """Sample every scene deterministically without rendering waveforms."""
    resolved_config = Path(config_path).expanduser().resolve()
    cfg = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    pool = _build_pool(cfg)
    sampler = _build_sampler(cfg, pool)
    rcfg = cfg.get("render", {})
    n = int(rcfg.get("sample_count", 500))
    recipe_plan_value = rcfg.get("recipe_plan_path")
    if recipe_plan_value:
        from sceneledger.data.scene_recipes import (
            read_inventory,
            read_recipes,
            validate_recipes,
        )
        from sceneledger.data.source_catalog import file_sha256

        inventory_value = rcfg.get("recipe_inventory_path")
        if not inventory_value:
            raise ValueError(
                "render.recipe_inventory_path is required with recipe_plan_path"
            )
        recipe_path = Path(str(recipe_plan_value)).expanduser()
        inventory_path = Path(str(inventory_value)).expanduser()
        if not recipe_path.is_absolute():
            recipe_path = (resolved_config.parent / recipe_path).resolve()
        if not inventory_path.is_absolute():
            inventory_path = (resolved_config.parent / inventory_path).resolve()
        recipes = read_recipes(recipe_path)
        inventory = read_inventory(inventory_path)
        validate_recipes(recipes, inventory)
        if len(recipes) != n:
            raise ValueError(
                "render.sample_count must exactly equal recipe plan rows: "
                f"sample_count={n} recipes={len(recipes)}"
            )
        selected = recipes[: min(n, limit)] if limit is not None else recipes
        scene_id_prefix = str(rcfg.get("scene_id_prefix", "mix"))
        recipe_hash = file_sha256(recipe_path)
        inventory_hash = file_sha256(inventory_path)
        return [
            sampler.sample(
                scene_id=f"{scene_id_prefix}_{index + 1:06d}",
                seed=recipe.seed,
                template=recipe.template,  # type: ignore[arg-type]
                label_preferences_by_kind=recipe.label_preferences_by_kind,
                recipe_metadata={
                    "recipe_id": recipe.recipe_id,
                    "proposal_source": recipe.proposal_source,
                    "context": recipe.context,
                    "difficulty": recipe.difficulty,
                    "relations": recipe.relations,
                    "rationale": recipe.rationale,
                    "label_preferences_by_kind": recipe.label_preferences_by_kind,
                    "recipe_plan_sha256": recipe_hash,
                    "recipe_inventory_sha256": inventory_hash,
                },
            )
            for index, recipe in enumerate(selected)
        ]
    if limit is not None:
        n = min(n, limit)
    templates = _weighted_templates(
        rcfg.get("template_weights", {"speech_over_music": 1}),
        n,
        seed=int(rcfg.get("template_seed", 0)),
    )
    seed_base = int(rcfg.get("seed_base", 1947))
    scene_id_prefix = str(rcfg.get("scene_id_prefix", "mix"))
    return [
        sampler.sample(
            scene_id=f"{scene_id_prefix}_{i + 1:06d}",
            seed=seed_base + i * 31,
            template=template,  # type: ignore[arg-type]
        )
        for i, template in enumerate(templates)
    ]


def render_dataset(config_path: str, output_dir: str, limit: int | None = None) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pool = _build_pool(cfg)

    rcfg = cfg.get("render", {})
    n = rcfg.get("sample_count", 500)
    if limit is not None:
        n = min(n, limit)
    scenes = sample_scene_plan(config_path, limit=limit)

    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    for i, scene in enumerate(scenes):
        out = render_scene(scene, pool)
        entry = persist_render(out, odir / "audio", rel_to=odir)
        entries.append(entry)
        if (i + 1) % 50 == 0:
            print(f"[render] {i + 1}/{n}", file=sys.stderr)

    manifest_path = odir / "manifest.jsonl"
    write_manifest(manifest_path, entries)
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
        f"- generated: {cfg.get('pool', {}).get('kind', 'synthetic')} pool, {len(entries)} clips",
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
    recipe_metadata = entries[0].scene.get("recipe_metadata") if entries else None
    if recipe_metadata:
        lines[6:6] = [
            f"- recipe_plan_sha256: {recipe_metadata.get('recipe_plan_sha256')}",
            f"- recipe_inventory_sha256: {recipe_metadata.get('recipe_inventory_sha256')}",
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
