"""Auditable rule/LLM scene recipes constrained by source-catalog evidence.

An LLM may propose *which verified source classes plausibly co-occur*.  It may
not invent source IDs, captions, lyrics, or acoustic truth.  The deterministic
sampler remains responsible for selecting audited waveforms and timestamps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sceneledger.data.source_catalog import CatalogKind, file_sha256, read_source_catalog

RECIPE_SCHEMA_VERSION = "sceneledger.scene_recipe.v1"
INVENTORY_SCHEMA_VERSION = "sceneledger.label_inventory.v1"
LLM_TASK_SCHEMA_VERSION = "sceneledger.recipe_llm_task.v1"

TEMPLATE_KIND_COUNTS: dict[str, Counter[str]] = {
    "speech_with_sfx": Counter({"speech": 1, "sfx": 1}),
    "speech_ambience_sfx": Counter({"speech": 1, "ambience": 1, "sfx": 1}),
    "speech_over_music": Counter({"speech": 1, "music": 1}),
    "music_with_sfx": Counter({"music": 1, "sfx": 1}),
    "speech_music_sfx": Counter({"speech": 1, "music": 1, "sfx": 1}),
    "ambient_with_intermittent_sfx": Counter({"ambience": 1, "sfx": 1}),
    "overlapping_speakers": Counter({"speech": 2}),
    "lyrics_over_music": Counter({"music": 1, "vocal": 1}),
    "multi_speaker_ambient_events": Counter(
        {"speech": 2, "ambience": 1, "sfx": 3}
    ),
}
ALLOWED_CONTEXTS = {
    "home",
    "workplace",
    "street",
    "transport",
    "nature",
    "public_indoor",
    "performance",
    "generic",
}
ALLOWED_RELATIONS = {
    "overlap",
    "foreground_over_background",
    "sequential",
    "intermittent",
    "competing_sources",
}

CONTEXT_KEYWORDS = {
    "home": (
        "door", "knock", "dish", "vacuum", "washing", "clock", "water",
        "inside", "room", "kitchen", "household", "sink",
    ),
    "workplace": (
        "keyboard", "printer", "tool", "drill", "machine", "phone",
        "office", "inside", "room",
    ),
    "street": (
        "car", "horn", "siren", "traffic", "roadway", "urban", "engine",
        "vehicle", "jackhammer",
    ),
    "transport": ("train", "bus", "aircraft", "vehicle", "engine", "subway"),
    "nature": (
        "bird", "cricket", "rain", "wind", "water", "thunder", "animal",
        "dog", "rural", "natural", "ocean", "wave",
    ),
    "public_indoor": (
        "crowd", "laugh", "applause", "clap", "cough", "speech", "inside",
        "room",
    ),
    "performance": ("music", "applause", "clap", "instrument", "sing"),
}


class CatalogArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    records: int = Field(..., gt=0)


class LabelInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[INVENTORY_SCHEMA_VERSION] = INVENTORY_SCHEMA_VERSION
    catalogs: list[CatalogArtifact]
    labels_by_kind: dict[CatalogKind, dict[str, int]]
    datasets_by_kind: dict[CatalogKind, dict[str, int]]


class SceneRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[RECIPE_SCHEMA_VERSION] = RECIPE_SCHEMA_VERSION
    recipe_id: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    seed: int = Field(..., ge=0)
    template: str
    context: str
    difficulty: Literal["easy", "medium", "hard"]
    label_preferences_by_kind: dict[CatalogKind, list[str]] = Field(default_factory=dict)
    relations: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=8, max_length=400)
    proposal_source: str = Field(..., min_length=1, max_length=120)

    @model_validator(mode="after")
    def _validate_contract(self) -> SceneRecipe:
        counts = TEMPLATE_KIND_COUNTS.get(self.template)
        if counts is None:
            raise ValueError(f"unsupported recipe template: {self.template}")
        if self.context not in ALLOWED_CONTEXTS:
            raise ValueError(f"unsupported scene context: {self.context}")
        invalid_relations = set(self.relations) - ALLOWED_RELATIONS
        if invalid_relations:
            raise ValueError(f"unsupported relations: {sorted(invalid_relations)}")
        for kind, labels in self.label_preferences_by_kind.items():
            if kind not in counts:
                raise ValueError(f"template {self.template} has no {kind} source")
            if kind in {"speech", "vocal"}:
                raise ValueError(
                    f"recipe cannot select {kind} text/identity; only audited source sampling may"
                )
            if len(labels) > counts[kind]:
                raise ValueError(
                    f"template {self.template} has {counts[kind]} {kind} slot(s), "
                    f"but recipe requested {len(labels)} labels"
                )
            if any(not label.strip() for label in labels):
                raise ValueError("recipe labels must be non-empty")
        for kind, count in counts.items():
            if kind in {"sfx", "ambience"} and len(
                self.label_preferences_by_kind.get(kind, [])
            ) != count:
                raise ValueError(
                    f"recipe must select exactly {count} audited {kind} label(s) "
                    f"for template {self.template}"
                )
        return self


def _jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_no}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def write_recipes(path: str | Path, recipes: list[SceneRecipe]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for recipe in recipes:
            handle.write(json.dumps(recipe.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")


def read_recipes(path: str | Path) -> list[SceneRecipe]:
    recipes: list[SceneRecipe] = []
    for line_no, payload in enumerate(_jsonl(path), 1):
        try:
            recipes.append(SceneRecipe.model_validate(payload))
        except Exception as exc:
            raise ValueError(f"invalid recipe {path}:{line_no}: {exc}") from exc
    ids = [recipe.recipe_id for recipe in recipes]
    if len(ids) != len(set(ids)):
        raise ValueError("recipe IDs must be unique")
    return recipes


def build_label_inventory(catalog_paths: list[str | Path]) -> LabelInventory:
    if not catalog_paths:
        raise ValueError("at least one prepared catalog is required")
    labels: dict[str, Counter[str]] = {}
    datasets: dict[str, Counter[str]] = {}
    artifacts: list[CatalogArtifact] = []
    for value in catalog_paths:
        path = Path(value).expanduser().resolve()
        records = read_source_catalog(path)
        artifacts.append(
            CatalogArtifact(path=str(path), sha256=file_sha256(path), records=len(records))
        )
        for record in records:
            labels.setdefault(record.kind, Counter())
            datasets.setdefault(record.kind, Counter())
            if record.labels:
                labels[record.kind][record.labels[0]] += 1
            datasets[record.kind][record.dataset] += 1
    return LabelInventory(
        catalogs=artifacts,
        labels_by_kind={
            kind: dict(sorted(counter.items())) for kind, counter in labels.items()
        },
        datasets_by_kind={
            kind: dict(sorted(counter.items())) for kind, counter in datasets.items()
        },
    )


def write_inventory(path: str | Path, inventory: LabelInventory) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(inventory.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_inventory(path: str | Path) -> LabelInventory:
    return LabelInventory.model_validate_json(Path(path).read_text(encoding="utf-8"))


def inventory_sha256(inventory: LabelInventory) -> str:
    canonical = json.dumps(
        inventory.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _weighted_templates(
    weights: dict[str, float], count: int, rng: random.Random
) -> list[str]:
    if count <= 0 or not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("recipe count and template weights must be positive")
    unknown = set(weights) - set(TEMPLATE_KIND_COUNTS)
    if unknown:
        raise ValueError(f"unsupported recipe templates: {sorted(unknown)}")
    items = sorted(weights.items())
    total = sum(weight for _, weight in items)
    return [rng.choices([item[0] for item in items], [item[1] / total for item in items])[0] for _ in range(count)]


def _labels_for_context(
    inventory: LabelInventory, kind: str, context: str
) -> list[str]:
    available = list(inventory.labels_by_kind.get(kind, {}))
    keywords = CONTEXT_KEYWORDS.get(context, ())
    matched = [
        label
        for label in available
        if any(keyword in label.casefold() for keyword in keywords)
    ]
    return matched


def generate_rule_recipes(
    inventory: LabelInventory,
    *,
    count: int,
    seed: int,
    template_weights: dict[str, float],
    recipe_prefix: str = "rule",
    strategy: Literal["keyword", "uniform"] = "keyword",
) -> list[SceneRecipe]:
    rng = random.Random(seed)
    templates = _weighted_templates(template_weights, count, rng)
    contexts = sorted(CONTEXT_KEYWORDS)
    recipes: list[SceneRecipe] = []
    for index, template in enumerate(templates):
        counts = TEMPLATE_KIND_COUNTS[template]
        viable = []
        if strategy == "keyword":
            for context in contexts:
                if all(
                    kind in {"speech", "vocal", "music"}
                    or len(_labels_for_context(inventory, kind, context)) >= slots
                    for kind, slots in counts.items()
                ):
                    viable.append(context)
        context = rng.choice(viable) if viable else "generic"
        preferences: dict[str, list[str]] = {}
        for kind, slots in counts.items():
            if kind in {"speech", "vocal", "music"}:
                continue
            contextual = (
                _labels_for_context(inventory, kind, context)
                if strategy == "keyword"
                else []
            )
            all_candidates = list(inventory.labels_by_kind.get(kind, {}))
            candidates = contextual or all_candidates
            if candidates:
                chosen: list[str] = []
                for _ in range(slots):
                    unused = [label for label in candidates if label not in chosen]
                    if not unused:
                        unused = [
                            label for label in all_candidates if label not in chosen
                        ]
                    # Reusing a class is preferable to inventing a label when a
                    # tiny fixture/catalog genuinely has fewer classes than
                    # source slots.  Real complex profiles separately require
                    # broad label coverage.
                    chosen.append(rng.choice(unused or candidates))
                preferences[kind] = chosen
        difficulty = "hard" if sum(counts.values()) >= 3 else "medium"
        recipes.append(
            SceneRecipe(
                recipe_id=f"{recipe_prefix}_{index + 1:06d}",
                seed=seed + index * 31,
                template=template,
                context=context,
                difficulty=difficulty,
                label_preferences_by_kind=preferences,  # type: ignore[arg-type]
                relations=["overlap", "foreground_over_background"],
                rationale=(
                    "Deterministic keyword-compatible rule proposal from audited labels."
                    if strategy == "keyword"
                    else "Deterministic uniform-label control independent of scene context."
                ),
                proposal_source=f"rules:{strategy}_v1",
            )
        )
    validate_recipes(recipes, inventory)
    return recipes


def export_llm_tasks(
    inventory: LabelInventory,
    *,
    count: int,
    seed: int,
    template_weights: dict[str, float],
    output_path: str | Path,
    max_labels_per_kind: int = 120,
) -> None:
    rng = random.Random(seed)
    templates = _weighted_templates(template_weights, count, rng)
    digest = inventory_sha256(inventory)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for index, template in enumerate(templates):
            counts = TEMPLATE_KIND_COUNTS[template]
            allowed = {
                kind: [
                    label
                    for label, _count in sorted(
                        inventory.labels_by_kind.get(kind, {}).items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:max_labels_per_kind]
                ]
                for kind in counts
                if kind not in {"speech", "vocal"}
            }
            task = {
                "schema_version": LLM_TASK_SCHEMA_VERSION,
                "task_id": f"llm_task_{index + 1:06d}",
                "recipe_id": f"llm_{index + 1:06d}",
                "seed": seed + index * 31,
                "required_template": template,
                "required_kind_counts": dict(counts),
                "allowed_contexts": sorted(ALLOWED_CONTEXTS),
                "allowed_relations": sorted(ALLOWED_RELATIONS),
                "allowed_labels_by_kind": allowed,
                "inventory_sha256": digest,
                "system_prompt": (
                    "You design plausible complex-audio mixture recipes. Return one JSON object only. "
                    "Use exact allowed labels. Never invent source IDs, captions, speech text, lyrics, "
                    "timestamps, licenses, people, brands, or facts about an audio waveform."
                ),
                "user_prompt": (
                    f"Required template: {template}. Required source counts: {dict(counts)}. "
                    f"Allowed contexts: {sorted(ALLOWED_CONTEXTS)}. "
                    f"Allowed labels by kind: {allowed}. Return keys context, difficulty "
                    "(easy|medium|hard), label_preferences_by_kind, relations, rationale. "
                    "Use at most one exact label per required non-speech slot."
                ),
            }
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")


def compile_llm_responses(
    tasks_path: str | Path,
    responses_path: str | Path,
    *,
    output_path: str | Path,
    model_name: str,
) -> list[SceneRecipe]:
    tasks = {str(row["task_id"]): row for row in _jsonl(tasks_path)}
    responses = {str(row["task_id"]): row for row in _jsonl(responses_path)}
    missing = sorted(set(tasks) - set(responses))
    extra = sorted(set(responses) - set(tasks))
    if missing or extra:
        raise ValueError(f"LLM task/response mismatch: missing={missing[:10]} extra={extra[:10]}")
    recipes: list[SceneRecipe] = []
    for task_id, task in tasks.items():
        raw = responses[task_id].get("response")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM response is not JSON for {task_id}: {exc}") from exc
        if isinstance(raw, dict) and isinstance(raw.get("recipe"), dict):
            raw = raw["recipe"]
        if not isinstance(raw, dict):
            raise ValueError(f"LLM response must be an object for {task_id}")
        allowed = task.get("allowed_labels_by_kind", {})
        preferences = raw.get("label_preferences_by_kind", {})
        if not isinstance(preferences, dict):
            raise ValueError(f"label_preferences_by_kind must be an object for {task_id}")
        for kind, labels in preferences.items():
            if not isinstance(labels, list) or not set(map(str, labels)) <= set(
                map(str, dict(allowed).get(kind, []))
            ):
                raise ValueError(f"LLM invented a disallowed {kind} label for {task_id}: {labels}")
        payload = {
            **raw,
            "schema_version": RECIPE_SCHEMA_VERSION,
            "recipe_id": task["recipe_id"],
            "seed": task["seed"],
            "template": task["required_template"],
            "proposal_source": f"llm:{model_name}",
        }
        recipes.append(SceneRecipe.model_validate(payload))
    write_recipes(output_path, recipes)
    return recipes


def validate_recipes(
    recipes: list[SceneRecipe], inventory: LabelInventory
) -> dict[str, object]:
    known = {
        kind: set(labels) for kind, labels in inventory.labels_by_kind.items()
    }
    errors: list[dict[str, object]] = []
    for recipe in recipes:
        for kind, labels in recipe.label_preferences_by_kind.items():
            unknown = sorted(set(labels) - known.get(kind, set()))
            if unknown:
                errors.append(
                    {"recipe_id": recipe.recipe_id, "kind": kind, "unknown_labels": unknown}
                )
    ids = [recipe.recipe_id for recipe in recipes]
    if len(ids) != len(set(ids)):
        errors.append({"duplicate_recipe_ids": True})
    report = recipe_summary(recipes)
    report.update(
        {
            "schema_version": "sceneledger.scene_recipe_validation.v1",
            "pass": not errors,
            "inventory_sha256": inventory_sha256(inventory),
            "errors": errors,
        }
    )
    if errors:
        raise ValueError(f"recipe validation failed: {errors[:10]}")
    return report


def recipe_summary(recipes: list[SceneRecipe]) -> dict[str, object]:
    if not recipes:
        raise ValueError("recipe set is empty")
    templates = Counter(recipe.template for recipe in recipes)
    contexts = Counter(recipe.context for recipe in recipes)
    labels = Counter(
        (kind, label)
        for recipe in recipes
        for kind, values in recipe.label_preferences_by_kind.items()
        for label in values
    )
    combinations = {
        (
            recipe.template,
            recipe.context,
            tuple(
                (kind, tuple(values))
                for kind, values in sorted(recipe.label_preferences_by_kind.items())
            ),
        )
        for recipe in recipes
    }
    probabilities = [count / len(recipes) for count in templates.values()]
    entropy = -sum(value * math.log2(value) for value in probabilities)
    return {
        "n_recipes": len(recipes),
        "template_counts": dict(sorted(templates.items())),
        "context_counts": dict(sorted(contexts.items())),
        "unique_primary_labels": len(labels),
        "unique_recipe_combinations": len(combinations),
        "template_entropy_bits": round(entropy, 6),
    }


def compare_recipe_sets(
    left: list[SceneRecipe], right: list[SceneRecipe]
) -> dict[str, object]:
    """Verify a matched design before comparing rule and LLM arms."""
    matched_count = len(left) == len(right)
    matched_seeds = [recipe.seed for recipe in left] == [recipe.seed for recipe in right]
    matched_templates = [recipe.template for recipe in left] == [
        recipe.template for recipe in right
    ]
    checks = [
        {"name": "matched_count", "pass": matched_count, "detail": [len(left), len(right)]},
        {"name": "matched_seeds", "pass": matched_seeds, "detail": None},
        {"name": "matched_templates", "pass": matched_templates, "detail": None},
    ]
    return {
        "schema_version": "sceneledger.scene_recipe_comparison.v1",
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
        "left": recipe_summary(left),
        "right": recipe_summary(right),
    }


def write_recipe_review(path: str | Path, recipes: list[SceneRecipe]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "recipe_id",
                "proposal_source",
                "template",
                "context",
                "difficulty",
                "labels_json",
                "relations_json",
                "rationale",
                "plausible_y_n",
                "label_compatible_y_n",
                "notes",
            ]
        )
        for recipe in recipes:
            writer.writerow(
                [
                    recipe.recipe_id,
                    recipe.proposal_source,
                    recipe.template,
                    recipe.context,
                    recipe.difficulty,
                    json.dumps(recipe.label_preferences_by_kind, ensure_ascii=False),
                    json.dumps(recipe.relations, ensure_ascii=False),
                    recipe.rationale,
                    "",
                    "",
                    "",
                ]
            )


def validate_recipe_review(
    path: str | Path,
    recipes: list[SceneRecipe],
    *,
    min_pass_rate: float = 0.90,
) -> dict[str, object]:
    """Require an exact, completed human plausibility review for a recipe set."""
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be in [0, 1]")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    expected_ids = [recipe.recipe_id for recipe in recipes]
    observed_ids = [str(row.get("recipe_id") or "") for row in rows]
    duplicate_ids = sorted(
        recipe_id
        for recipe_id, count in Counter(observed_ids).items()
        if recipe_id and count > 1
    )
    missing_ids = sorted(set(expected_ids) - set(observed_ids))
    extra_ids = sorted(set(observed_ids) - set(expected_ids) - {""})

    def _answer(value: object) -> bool | None:
        normalized = str(value or "").strip().casefold()
        if normalized in {"y", "yes", "1", "true"}:
            return True
        if normalized in {"n", "no", "0", "false"}:
            return False
        return None

    reviewed: list[dict[str, object]] = []
    invalid_answers: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    immutable_mismatches: list[dict[str, object]] = []
    expected_by_id = {recipe.recipe_id: recipe for recipe in recipes}
    for row in rows:
        recipe_id = str(row.get("recipe_id") or "")
        expected = expected_by_id.get(recipe_id)
        if expected is not None:
            scalar_fields = {
                "proposal_source": expected.proposal_source,
                "template": expected.template,
                "context": expected.context,
                "difficulty": expected.difficulty,
                "rationale": expected.rationale,
            }
            for field, expected_value in scalar_fields.items():
                observed_value = str(row.get(field) or "")
                if observed_value != expected_value:
                    immutable_mismatches.append(
                        {
                            "recipe_id": recipe_id,
                            "field": field,
                            "observed": observed_value,
                            "expected": expected_value,
                        }
                    )
            for field, expected_value in (
                ("labels_json", expected.label_preferences_by_kind),
                ("relations_json", expected.relations),
            ):
                try:
                    observed_json = json.loads(str(row.get(field) or ""))
                except json.JSONDecodeError:
                    observed_json = None
                if observed_json != expected_value:
                    immutable_mismatches.append(
                        {
                            "recipe_id": recipe_id,
                            "field": field,
                            "observed": observed_json,
                            "expected": expected_value,
                        }
                    )
        plausible = _answer(row.get("plausible_y_n"))
        compatible = _answer(row.get("label_compatible_y_n"))
        item = {
            "recipe_id": recipe_id,
            "plausible": plausible,
            "label_compatible": compatible,
            "notes": str(row.get("notes") or ""),
        }
        reviewed.append(item)
        if plausible is None or compatible is None:
            invalid_answers.append(item)
        elif not plausible or not compatible:
            rejected.append(item)
    completed = len(reviewed) - len(invalid_answers)
    passed_rows = completed - len(rejected)
    pass_rate = passed_rows / len(recipes) if recipes else 0.0
    structural_pass = (
        len(rows) == len(recipes)
        and not duplicate_ids
        and not missing_ids
        and not extra_ids
        and not invalid_answers
        and not immutable_mismatches
    )
    recipe_set_sha256 = hashlib.sha256(
        json.dumps(
            [recipe.model_dump() for recipe in recipes],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "sceneledger.recipe_human_review.v1",
        "pass": structural_pass and pass_rate >= min_pass_rate,
        "n_expected": len(recipes),
        "n_rows": len(rows),
        "n_completed": completed,
        "n_passed": passed_rows,
        "pass_rate": round(pass_rate, 6),
        "min_pass_rate": min_pass_rate,
        "recipe_set_sha256": recipe_set_sha256,
        "duplicate_recipe_ids": duplicate_ids,
        "missing_recipe_ids": missing_ids,
        "extra_recipe_ids": extra_ids,
        "invalid_answers": invalid_answers[:50],
        "immutable_mismatches": immutable_mismatches[:50],
        "rejected": rejected[:50],
    }


__all__ = [
    "ALLOWED_CONTEXTS",
    "ALLOWED_RELATIONS",
    "INVENTORY_SCHEMA_VERSION",
    "LLM_TASK_SCHEMA_VERSION",
    "RECIPE_SCHEMA_VERSION",
    "TEMPLATE_KIND_COUNTS",
    "LabelInventory",
    "SceneRecipe",
    "build_label_inventory",
    "compare_recipe_sets",
    "compile_llm_responses",
    "export_llm_tasks",
    "generate_rule_recipes",
    "inventory_sha256",
    "read_inventory",
    "read_recipes",
    "recipe_summary",
    "validate_recipes",
    "validate_recipe_review",
    "write_inventory",
    "write_recipe_review",
    "write_recipes",
]
