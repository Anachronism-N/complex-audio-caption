"""Bind a validated recipe plan and inventory to an audited base mix config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sceneledger.data.scene_recipes import read_inventory, read_recipes, validate_recipes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--recipes", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--scene-id-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    base_config = Path(args.base_config).expanduser().resolve()
    recipe_path = Path(args.recipes).expanduser().resolve()
    inventory_path = Path(args.inventory).expanduser().resolve()
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    recipes = read_recipes(recipe_path)
    validate_recipes(recipes, read_inventory(inventory_path))
    if config.get("pool", {}).get("kind") not in {"catalog", "catalog_set"}:
        raise ValueError("recipe mix config must inherit an audited catalog pool")
    render = config.setdefault("render", {})
    render["sample_count"] = len(recipes)
    render["scene_id_prefix"] = args.scene_id_prefix
    render["recipe_plan_path"] = str(recipe_path)
    render["recipe_inventory_path"] = str(inventory_path)
    render.pop("template_weights", None)
    render.pop("template_seed", None)
    config["recipe_experiment"] = {
        "base_config": str(base_config),
        "recipes": str(recipe_path),
        "inventory": str(inventory_path),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"config={output}")
    print(f"recipes={len(recipes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
