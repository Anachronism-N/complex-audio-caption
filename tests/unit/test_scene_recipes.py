from __future__ import annotations

import json
from pathlib import Path

import yaml

from sceneledger.cli.render import sample_scene_plan
from sceneledger.data.scene_graph_sampler import SceneGraphSampler, SceneSamplerConfig
from sceneledger.data.scene_recipes import (
    build_label_inventory,
    compare_recipe_sets,
    compile_llm_responses,
    export_llm_tasks,
    generate_rule_recipes,
    read_recipes,
    validate_recipes,
    write_inventory,
    write_recipes,
)
from sceneledger.data.source_catalog import SourceRecord, write_source_catalog


def _record(source_id: str, kind: str, label: str | None) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        kind=kind,  # type: ignore[arg-type]
        audio_path=f"{source_id}.wav",
        source_group=f"group:{source_id}",
        labels=[label] if label else [],
        caption=f"caption {source_id}",
        dataset="fixture",
        license="CC0-1.0",
        annotation_origin="dataset",
        split="test",
    )


def _inventory(tmp_path: Path):
    catalog = tmp_path / "catalog.jsonl"
    write_source_catalog(
        catalog,
        [
            _record("speech1", "speech", None),
            _record("sfx1", "sfx", "car_horn"),
            _record("sfx2", "sfx", "door_wood_knock"),
            _record("amb1", "ambience", "rain"),
            _record("amb2", "ambience", "wind"),
        ],
    )
    return build_label_inventory([catalog])


def test_rule_recipes_are_deterministic_and_inventory_constrained(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    left = generate_rule_recipes(
        inventory,
        count=20,
        seed=42,
        template_weights={"speech_with_sfx": 1, "speech_ambience_sfx": 1},
    )
    right = generate_rule_recipes(
        inventory,
        count=20,
        seed=42,
        template_weights={"speech_with_sfx": 1, "speech_ambience_sfx": 1},
    )

    assert [recipe.model_dump() for recipe in left] == [
        recipe.model_dump() for recipe in right
    ]
    assert validate_recipes(left, inventory)["pass"] is True
    assert all(recipe.label_preferences_by_kind.get("sfx") for recipe in left)


def test_llm_compile_injects_frozen_template_seed_and_rejects_hallucinated_labels(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    tasks = tmp_path / "tasks.jsonl"
    export_llm_tasks(
        inventory,
        count=2,
        seed=7,
        template_weights={"speech_with_sfx": 1},
        output_path=tasks,
    )
    task_rows = [json.loads(line) for line in tasks.read_text(encoding="utf-8").splitlines()]
    responses = tmp_path / "responses.jsonl"
    with responses.open("w", encoding="utf-8") as handle:
        for task in task_rows:
            response = {
                "context": "street",
                "difficulty": "medium",
                "label_preferences_by_kind": {"sfx": ["car_horn"]},
                "relations": ["overlap"],
                "rationale": "A horn can plausibly occur while someone speaks on a street.",
            }
            handle.write(json.dumps({"task_id": task["task_id"], "response": response}) + "\n")
    output = tmp_path / "llm.jsonl"
    recipes = compile_llm_responses(
        tasks, responses, output_path=output, model_name="fixture"
    )

    assert [recipe.seed for recipe in recipes] == [7, 38]
    assert all(recipe.template == "speech_with_sfx" for recipe in recipes)
    assert all(recipe.proposal_source == "llm:fixture" for recipe in recipes)
    assert read_recipes(output) == recipes

    bad_rows = [
        json.loads(line) for line in responses.read_text(encoding="utf-8").splitlines()
    ]
    bad_rows[0]["response"]["label_preferences_by_kind"] = {
        "sfx": ["invented_laser"]
    }
    responses.write_text(
        "\n".join(json.dumps(row) for row in bad_rows) + "\n", encoding="utf-8"
    )
    try:
        compile_llm_responses(tasks, responses, output_path=output, model_name="fixture")
    except ValueError as exc:
        assert "invented" in str(exc)
    else:
        raise AssertionError("hallucinated LLM label was accepted")


def test_render_plan_binds_recipe_and_inventory_hashes(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    write_inventory(inventory_path, inventory)
    recipes = generate_rule_recipes(
        inventory,
        count=1,
        seed=12,
        template_weights={"speech_with_sfx": 1},
    )
    # Synthetic sources do not carry catalog labels, so use a template without
    # label constraints to exercise artifact binding independently.
    recipe = recipes[0].model_copy(
        update={
            "template": "overlapping_speakers",
            "label_preferences_by_kind": {},
        }
    )
    recipe_path = tmp_path / "recipes.jsonl"
    write_recipes(recipe_path, [recipe])
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pool": {"kind": "synthetic"},
                "sampler": {"p_rir": 0.0, "p_echo": 0.0},
                "render": {
                    "sample_count": 1,
                    "scene_id_prefix": "recipe",
                    "recipe_plan_path": str(recipe_path),
                    "recipe_inventory_path": str(inventory_path),
                },
            }
        ),
        encoding="utf-8",
    )

    scenes = sample_scene_plan(str(config))

    assert len(scenes) == 1
    assert scenes[0].recipe_metadata["recipe_id"] == recipe.recipe_id
    assert len(str(scenes[0].recipe_metadata["recipe_plan_sha256"])) == 64


def test_sampler_selects_the_recipe_primary_label() -> None:
    class Pool:
        def candidates(self, kind, rng, max_duration=None):
            del rng, max_duration
            if kind == "speech":
                return ["speech"]
            return ["dog", "horn"]

        def metadata(self, key):
            labels = {"dog": ["dog_bark"], "horn": ["car_horn"], "speech": []}
            return {
                "source_group": f"group:{key}",
                "source_labels": labels[key],
                "source_duration_sec": 1.0,
                "text": key,
            }

        def pick(self, kind, rng):
            return rng.choice(self.candidates(kind, rng))

    sampler = SceneGraphSampler(
        Pool(), SceneSamplerConfig(duration_range=(4.0, 4.0), p_rir=0.0, p_echo=0.0)
    )
    scene = sampler.sample(
        "recipe_label",
        123,
        "speech_with_sfx",
        label_preferences_by_kind={"sfx": ["car_horn"]},
    )

    assert next(source.path for source in scene.sources if source.kind == "sfx") == "horn"


def test_recipe_comparison_requires_matched_seed_and_template_schedule(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    left = generate_rule_recipes(
        inventory,
        count=10,
        seed=99,
        template_weights={"speech_with_sfx": 1, "speech_ambience_sfx": 1},
    )
    right = [
        recipe.model_copy(update={"proposal_source": "llm:fixture"}) for recipe in left
    ]
    assert compare_recipe_sets(left, right)["pass"] is True
    right[0] = right[0].model_copy(update={"seed": 1000})
    assert compare_recipe_sets(left, right)["pass"] is False
