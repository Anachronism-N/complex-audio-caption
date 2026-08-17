from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from sceneledger.cli.render import sample_scene_plan
from sceneledger.cli.scene_recipes import main as scene_recipes_main
from sceneledger.data.scene_graph_sampler import SceneGraphSampler, SceneSamplerConfig
from sceneledger.data.scene_recipes import (
    build_label_inventory,
    compare_recipe_sets,
    compile_llm_responses,
    compile_llm_source_timeline_responses,
    export_llm_source_timeline_tasks,
    export_llm_tasks,
    generate_rule_recipes,
    generate_rule_source_timeline_recipes,
    read_recipes,
    validate_recipe_review,
    validate_recipes,
    write_inventory,
    write_recipe_review,
    write_recipes,
)
from sceneledger.data.source_catalog import (
    SourceRecord,
    file_sha256,
    write_source_catalog,
)


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
        duration_sec=1.0,
        file_sha256=f"fixture:{source_id}",
        split="test",
    )


def _inventory(tmp_path: Path):
    catalog = tmp_path / "catalog.jsonl"
    write_source_catalog(
        catalog,
        [
            _record("speech1", "speech", None),
            _record("speech2", "speech", None),
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


def test_complex_rule_recipe_fills_all_slots_without_repeating_available_labels(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "complex_catalog.jsonl"
    write_source_catalog(
        catalog,
        [
            _record("speech1", "speech", None),
            _record("speech2", "speech", None),
            _record("rain", "ambience", "rain"),
            _record("bird", "sfx", "bird_call"),
            _record("dog", "sfx", "dog_bark"),
            _record("water", "sfx", "water_drop"),
        ],
    )
    inventory = build_label_inventory([catalog])

    recipes = generate_rule_recipes(
        inventory,
        count=4,
        seed=17,
        template_weights={"multi_speaker_ambient_events": 1.0},
    )

    assert all(recipe.context == "nature" for recipe in recipes)
    assert all(
        len(recipe.label_preferences_by_kind["sfx"]) == 3
        for recipe in recipes
    )
    assert all(
        len(set(recipe.label_preferences_by_kind["sfx"])) == 3
        for recipe in recipes
    )
    assert all(
        recipe.label_preferences_by_kind["ambience"] == ["rain"]
        for recipe in recipes
    )


def test_recipe_review_must_be_complete_and_meet_pass_rate(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    recipes = generate_rule_recipes(
        inventory,
        count=4,
        seed=3,
        template_weights={"speech_with_sfx": 1.0},
    )
    review = tmp_path / "review.csv"
    write_recipe_review(review, recipes)

    incomplete = validate_recipe_review(review, recipes, min_pass_rate=0.75)
    assert incomplete["pass"] is False
    assert incomplete["n_completed"] == 0

    with review.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for index, row in enumerate(rows):
        row["plausible_y_n"] = "n" if index == 0 else "y"
        row["label_compatible_y_n"] = "y"
        row["notes"] = "one intentional rejection" if index == 0 else "checked"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accepted = validate_recipe_review(review, recipes, min_pass_rate=0.75)
    rejected = validate_recipe_review(review, recipes, min_pass_rate=0.80)
    assert accepted["pass"] is True
    assert accepted["pass_rate"] == 0.75
    assert rejected["pass"] is False

    recipe_path = tmp_path / "recipes.jsonl"
    report_path = tmp_path / "review_report.json"
    write_recipes(recipe_path, recipes)
    assert (
        scene_recipes_main(
            [
                "validate-review",
                "--recipes",
                str(recipe_path),
                "--review-csv",
                str(review),
                "--min-pass-rate",
                "0.75",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["recipe_plan_sha256"] == file_sha256(recipe_path)
    assert report["review_csv_sha256"] == file_sha256(review)

    rows[0]["labels_json"] = json.dumps({"sfx": ["invented_laser"]})
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tampered = validate_recipe_review(review, recipes, min_pass_rate=0.75)
    assert tampered["pass"] is False
    assert tampered["immutable_mismatches"][0]["field"] == "labels_json"


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


def test_source_timeline_tasks_freeze_exact_candidates_and_build_rule_arm(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    tasks = tmp_path / "source_timeline_tasks.jsonl"
    export_llm_source_timeline_tasks(
        inventory,
        count=2,
        seed=21,
        template_weights={"speech_with_sfx": 1.0},
        output_path=tasks,
        candidates_per_slot=2,
        scene_duration_sec=4.0,
    )
    task_rows = [json.loads(line) for line in tasks.read_text(encoding="utf-8").splitlines()]

    assert len(task_rows) == 2
    assert task_rows[0]["required_slots"] == [
        {"slot_id": "speech_1", "kind": "speech"},
        {"slot_id": "sfx_1", "kind": "sfx"},
    ]
    assert len(task_rows[0]["allowed_sources_by_slot"]["speech_1"]) == 2
    assert len(task_rows[0]["task_sha256"]) == 64

    rule_path = tmp_path / "rule_source_timeline.jsonl"
    recipes = generate_rule_source_timeline_recipes(
        tasks,
        inventory=inventory,
        output_path=rule_path,
        strategy="keyword",
    )

    assert all(recipe.schema_version == "sceneledger.scene_recipe.v2" for recipe in recipes)
    assert all(recipe.scene_duration_sec == 4.0 for recipe in recipes)
    assert all(len(recipe.source_plan) == 2 for recipe in recipes)
    assert all(recipe.proposal_metadata["task_sha256"] for recipe in recipes)
    assert validate_recipes(recipes, inventory)["pass"] is True


def test_llm_source_timeline_compile_accepts_allowlist_and_rejects_invention(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    tasks = tmp_path / "source_timeline_tasks.jsonl"
    export_llm_source_timeline_tasks(
        inventory,
        count=1,
        seed=8,
        template_weights={"speech_with_sfx": 1.0},
        output_path=tasks,
        candidates_per_slot=2,
        scene_duration_sec=4.0,
    )
    task = json.loads(tasks.read_text(encoding="utf-8"))
    speech = task["allowed_sources_by_slot"]["speech_1"][0]
    sfx = task["allowed_sources_by_slot"]["sfx_1"][0]
    response = {
        "context": "street",
        "difficulty": "medium",
        "source_plan": [
            {
                "slot_id": "speech_1",
                "kind": "speech",
                "catalog_source_id": speech["catalog_source_id"],
                "onset_sec": 0.2,
            },
            {
                "slot_id": "sfx_1",
                "kind": "sfx",
                "catalog_source_id": sfx["catalog_source_id"],
                "onset_sec": 1.1,
            },
        ],
        "relations": ["overlap"],
        "rationale": "The selected speech and foreground effect form a plausible timed scene.",
    }
    responses = tmp_path / "source_timeline_responses.jsonl"
    responses.write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "response": response,
                "request_metadata": {
                    "model": "fixture",
                    "temperature": 0.2,
                    "json_mode": True,
                    "task_sha256": task["task_sha256"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "llm_source_timeline.jsonl"

    recipes = compile_llm_source_timeline_responses(
        tasks,
        responses,
        inventory=inventory,
        output_path=output,
        model_name="fixture",
    )

    assert recipes[0].source_plan[0].catalog_source_id == speech["catalog_source_id"]
    assert recipes[0].source_plan[1].onset_sec == 1.1
    assert recipes[0].proposal_metadata["request"]["temperature"] == 0.2

    response["source_plan"][1]["catalog_source_id"] = "invented_audio"
    responses.write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "response": response,
                "request_metadata": {
                    "model": "fixture",
                    "temperature": 0.2,
                    "json_mode": True,
                    "task_sha256": task["task_sha256"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        compile_llm_source_timeline_responses(
            tasks,
            responses,
            inventory=inventory,
            output_path=output,
            model_name="fixture",
        )
    except ValueError as exc:
        assert "outside the frozen slate" in str(exc)
    else:
        raise AssertionError("invented catalog source was accepted")

    response["source_plan"][1]["catalog_source_id"] = sfx["catalog_source_id"]
    response["source_plan"][1]["onset_sec"] = round(
        float(sfx["max_onset_sec"]) + 0.1, 1
    )
    responses.write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "response": response,
                "request_metadata": {
                    "model": "fixture",
                    "temperature": 0.2,
                    "json_mode": True,
                    "task_sha256": task["task_sha256"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        compile_llm_source_timeline_responses(
            tasks,
            responses,
            inventory=inventory,
            output_path=output,
            model_name="fixture",
        )
    except ValueError as exc:
        assert "onset exceeds candidate limit" in str(exc)
    else:
        raise AssertionError("out-of-range onset was accepted")

    response["source_plan"][1]["onset_sec"] = 1.1
    responses.write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "response": response,
                "request_metadata": {
                    "model": "fixture",
                    "temperature": 0.2,
                    "json_mode": True,
                    "task_sha256": "0" * 64,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        compile_llm_source_timeline_responses(
            tasks,
            responses,
            inventory=inventory,
            output_path=output,
            model_name="fixture",
        )
    except ValueError as exc:
        assert "response task hash mismatch" in str(exc)
    else:
        raise AssertionError("response bound to another task was accepted")


def test_sampler_obeys_exact_source_and_timeline_after_template_scheduling() -> None:
    class Pool:
        def candidates(self, kind, rng, max_duration=None):
            del rng, max_duration
            return {
                "speech": ["speech_a", "speech_b"],
                "sfx": ["dog", "horn"],
            }[kind]

        def metadata(self, key):
            return {
                "source_kind": "speech" if key.startswith("speech") else "sfx",
                "source_group": f"group:{key}",
                "source_labels": {"dog": ["dog_bark"], "horn": ["car_horn"]}.get(
                    key, []
                ),
                "source_duration_sec": 1.0,
                "text": key,
            }

        def pick(self, kind, rng):
            return rng.choice(self.candidates(kind, rng))

    sampler = SceneGraphSampler(
        Pool(), SceneSamplerConfig(duration_range=(4.0, 4.0), p_rir=0.0, p_echo=0.0)
    )
    scene = sampler.sample(
        "planned",
        4,
        "speech_with_sfx",
        duration_sec=4.0,
        label_preferences_by_kind={"sfx": ["car_horn"]},
        source_plan=[
            {
                "slot_id": "speech_1",
                "kind": "speech",
                "catalog_source_id": "speech_b",
                "onset_sec": 2.0,
            },
            {
                "slot_id": "sfx_1",
                "kind": "sfx",
                "catalog_source_id": "horn",
                "onset_sec": 0.3,
            },
        ],
    )

    by_kind = {source.kind: source for source in scene.sources}
    assert by_kind["speech"].path == "speech_b"
    assert by_kind["speech"].onset == 2.0
    assert by_kind["sfx"].path == "horn"
    assert by_kind["sfx"].onset == 0.3
