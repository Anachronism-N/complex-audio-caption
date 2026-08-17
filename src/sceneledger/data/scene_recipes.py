"""Auditable rule/LLM scene recipes constrained by source-catalog evidence.

The v1 recipe contract lets an LLM propose verified source classes.  The v2
contract additionally lets it select exact allow-listed source IDs and their
0.1-second onset times.  In both cases captions, identities, file hashes and
acoustic truth remain bound to the audited catalog and rendered stems.
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
SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION = "sceneledger.scene_recipe.v2"
INVENTORY_SCHEMA_VERSION = "sceneledger.label_inventory.v1"
LLM_TASK_SCHEMA_VERSION = "sceneledger.recipe_llm_task.v1"
LLM_SOURCE_TIMELINE_TASK_SCHEMA_VERSION = "sceneledger.source_timeline_llm_task.v1"

SOURCE_TIMELINE_TEMPLATES = {
    "speech_with_sfx",
    "speech_ambience_sfx",
    "speech_over_music",
    "music_with_sfx",
    "speech_music_sfx",
    "ambient_with_intermittent_sfx",
    "overlapping_speakers",
}

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
        {"speech": 3, "ambience": 1, "sfx": 3}
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


def _template_slots(template: str) -> list[str]:
    counts = TEMPLATE_KIND_COUNTS.get(template)
    if counts is None:
        raise ValueError(f"unsupported recipe template: {template}")
    return [
        f"{kind}_{index + 1}"
        for kind, count in counts.items()
        for index in range(count)
    ]


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


class SourceTimelineSelection(BaseModel):
    """One exact, allow-listed dry recording and its requested scene onset."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(
        ..., pattern=r"^(speech|vocal|music|sfx|ambience)_[1-9][0-9]*$"
    )
    kind: CatalogKind
    catalog_source_id: str = Field(..., min_length=1, max_length=240)
    onset_sec: float = Field(..., ge=0.0, lt=120.0)

    @model_validator(mode="after")
    def _validate_grid(self) -> SourceTimelineSelection:
        if abs(self.onset_sec * 10.0 - round(self.onset_sec * 10.0)) > 1e-6:
            raise ValueError("source-plan onset_sec must lie on the 0.1-second grid")
        return self


class LLMSourceTimelineProposal(BaseModel):
    """Strict response payload for a bounded source-and-time planning task."""

    model_config = ConfigDict(extra="forbid")

    context: str
    difficulty: Literal["easy", "medium", "hard"]
    source_plan: list[SourceTimelineSelection]
    relations: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=8, max_length=400)

    @model_validator(mode="after")
    def _validate_vocabulary(self) -> LLMSourceTimelineProposal:
        if self.context not in ALLOWED_CONTEXTS:
            raise ValueError(f"unsupported scene context: {self.context}")
        invalid_relations = set(self.relations) - ALLOWED_RELATIONS
        if invalid_relations:
            raise ValueError(f"unsupported relations: {sorted(invalid_relations)}")
        return self


class SceneRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        RECIPE_SCHEMA_VERSION, SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION
    ] = RECIPE_SCHEMA_VERSION
    recipe_id: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    seed: int = Field(..., ge=0)
    template: str
    context: str
    difficulty: Literal["easy", "medium", "hard"]
    label_preferences_by_kind: dict[CatalogKind, list[str]] = Field(default_factory=dict)
    relations: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=8, max_length=400)
    proposal_source: str = Field(..., min_length=1, max_length=120)
    scene_duration_sec: float | None = Field(default=None, ge=2.0, le=120.0)
    source_plan: list[SourceTimelineSelection] = Field(default_factory=list)
    proposal_metadata: dict[str, object] = Field(default_factory=dict)

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
        if self.source_plan:
            if self.schema_version != SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION:
                raise ValueError("source_plan requires the v2 source-timeline recipe schema")
            if self.scene_duration_sec is None:
                raise ValueError("source_plan requires scene_duration_sec")
            expected_slots = _template_slots(self.template)
            observed_slots = [item.slot_id for item in self.source_plan]
            if len(observed_slots) != len(set(observed_slots)):
                raise ValueError("source_plan slot IDs must be unique")
            if set(observed_slots) != set(expected_slots):
                raise ValueError(
                    "source_plan slots do not match template: "
                    f"expected={expected_slots} observed={observed_slots}"
                )
            expected_kind_by_slot = {
                slot_id: slot_id.rsplit("_", 1)[0] for slot_id in expected_slots
            }
            for item in self.source_plan:
                if item.kind != expected_kind_by_slot[item.slot_id]:
                    raise ValueError(
                        f"source_plan kind mismatch for {item.slot_id}: {item.kind}"
                    )
                if item.onset_sec >= self.scene_duration_sec:
                    raise ValueError(
                        f"source_plan onset is outside scene for {item.slot_id}: "
                        f"{item.onset_sec} >= {self.scene_duration_sec}"
                    )
        elif self.schema_version == SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION:
            raise ValueError("v2 source-timeline recipe requires a non-empty source_plan")
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


def _object_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records_from_inventory(inventory: LabelInventory):
    records = []
    for artifact in inventory.catalogs:
        path = Path(artifact.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"inventory catalog is missing: {path}")
        observed_hash = file_sha256(path)
        if observed_hash != artifact.sha256:
            raise ValueError(
                "inventory catalog changed after inventory creation: "
                f"{path} expected={artifact.sha256} observed={observed_hash}"
            )
        catalog_records = read_source_catalog(path)
        if len(catalog_records) != artifact.records:
            raise ValueError(
                f"inventory catalog record count changed: {path} "
                f"expected={artifact.records} observed={len(catalog_records)}"
            )
        records.extend(catalog_records)
    counts = Counter(record.source_id for record in records)
    duplicates = sorted(source_id for source_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "source IDs must be globally unique across inventory catalogs: "
            f"{duplicates[:20]}"
        )
    observed_splits = {record.split for record in records}
    if None in observed_splits or len(observed_splits) != 1:
        raise ValueError(
            "source-timeline inventory must contain exactly one frozen split, "
            f"observed={sorted(str(value) for value in observed_splits)}"
        )
    missing_file_hashes = [record.source_id for record in records if not record.file_sha256]
    if missing_file_hashes:
        raise ValueError(
            "source-timeline inventory contains sources without file SHA-256: "
            f"{missing_file_hashes[:20]}"
        )
    return records


def _candidate_payload(record, scene_duration_sec: float) -> dict[str, object]:
    if record.duration_sec is None:
        raise ValueError(f"source lacks duration_sec: {record.source_id}")
    if record.kind in {"speech", "vocal", "sfx"}:
        raw_max_onset = scene_duration_sec - float(record.duration_sec) - 0.05
    else:
        # Long backgrounds may be cropped; short backgrounds may intentionally
        # enter late. The renderer still truncates them at the scene boundary.
        raw_max_onset = scene_duration_sec - 0.1
    max_onset = max(0.0, math.floor((raw_max_onset + 1e-9) * 10.0) / 10.0)
    return {
        "catalog_source_id": record.source_id,
        "kind": record.kind,
        "primary_label": record.labels[0] if record.labels else None,
        "caption": record.caption,
        "duration_sec": round(float(record.duration_sec), 6),
        "max_onset_sec": round(max_onset, 1),
        "dataset": record.dataset,
        "split": record.split,
        "source_group": record.source_group,
        "leakage_groups": list(record.leakage_groups),
        "identity": record.identity,
        "annotation_origin": record.annotation_origin,
        "file_sha256": record.file_sha256,
    }


def _candidate_slate(
    records,
    *,
    kind: str,
    scene_duration_sec: float,
    rng: random.Random,
    limit: int,
) -> list[dict[str, object]]:
    eligible = [
        record
        for record in records
        if record.kind == kind
        and record.duration_sec is not None
        and (
            kind in {"music", "ambience"}
            or float(record.duration_sec) <= scene_duration_sec - 0.05
        )
        and (kind not in {"sfx", "ambience"} or bool(record.labels))
    ]
    if not eligible:
        raise ValueError(
            f"inventory has no timeline-eligible {kind} source for "
            f"scene_duration_sec={scene_duration_sec}"
        )
    buckets: dict[tuple[str, str], list[object]] = {}
    for record in eligible:
        primary_label = record.labels[0] if record.labels else "<unlabeled>"
        buckets.setdefault((record.dataset, primary_label), []).append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    output = []
    while buckets and len(output) < limit:
        keys = sorted(buckets)
        rng.shuffle(keys)
        for key in keys:
            output.append(_candidate_payload(buckets[key].pop(), scene_duration_sec))
            if not buckets[key]:
                del buckets[key]
            if len(output) >= limit:
                break
    return output


def export_llm_source_timeline_tasks(
    inventory: LabelInventory,
    *,
    count: int,
    seed: int,
    template_weights: dict[str, float],
    output_path: str | Path,
    candidates_per_slot: int = 12,
    scene_duration_sec: float = 12.0,
) -> None:
    """Freeze exact source slates and let an LLM select audio plus onset order.

    The task exposes catalog source IDs, never arbitrary paths.  The compiler
    later checks the selected IDs, kinds, durations and leakage groups against
    both this frozen task and the hashed catalog inventory.
    """

    if candidates_per_slot < 2:
        raise ValueError("candidates_per_slot must be at least 2")
    if scene_duration_sec < 2.0 or scene_duration_sec > 120.0:
        raise ValueError("scene_duration_sec must be in [2, 120]")
    if abs(scene_duration_sec * 10.0 - round(scene_duration_sec * 10.0)) > 1e-6:
        raise ValueError("scene_duration_sec must lie on the 0.1-second grid")
    unsupported = set(template_weights) - SOURCE_TIMELINE_TEMPLATES
    if unsupported:
        raise ValueError(
            "source-timeline v1 does not yet support aligned/repeated/persistent-track "
            f"templates: {sorted(unsupported)}"
        )
    records = _records_from_inventory(inventory)
    rng = random.Random(seed)
    templates = _weighted_templates(template_weights, count, rng)
    digest = inventory_sha256(inventory)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for index, template in enumerate(templates):
            task_seed = seed + index * 31
            task_rng = random.Random(task_seed)
            slots = _template_slots(template)
            slates_by_kind = {
                kind: _candidate_slate(
                    records,
                    kind=kind,
                    scene_duration_sec=scene_duration_sec,
                    rng=task_rng,
                    limit=candidates_per_slot,
                )
                for kind in dict(TEMPLATE_KIND_COUNTS[template])
            }
            allowed = {
                slot_id: slates_by_kind[slot_id.rsplit("_", 1)[0]]
                for slot_id in slots
            }
            for kind, required_count in TEMPLATE_KIND_COUNTS[template].items():
                unique_ids = {
                    str(item["catalog_source_id"])
                    for item in slates_by_kind[kind]
                }
                if len(unique_ids) < required_count:
                    raise ValueError(
                        f"candidate slate cannot fill {template} {kind} slots: "
                        f"required={required_count} available={len(unique_ids)}"
                    )
            # Reject a task before it reaches an LLM when source/leakage-group
            # constraints make the displayed cross-slot slate impossible.
            _rule_source_combination(
                slots,
                allowed,
                context="generic",
                strategy="uniform",
            )
            task: dict[str, object] = {
                "schema_version": LLM_SOURCE_TIMELINE_TASK_SCHEMA_VERSION,
                "task_id": f"llm_source_timeline_task_{index + 1:06d}",
                "recipe_id": f"llmst_{index + 1:06d}",
                "seed": task_seed,
                "required_template": template,
                "required_slots": [
                    {"slot_id": slot_id, "kind": slot_id.rsplit("_", 1)[0]}
                    for slot_id in slots
                ],
                "scene_duration_sec": round(scene_duration_sec, 1),
                "time_resolution_sec": 0.1,
                "allowed_contexts": sorted(ALLOWED_CONTEXTS),
                "allowed_relations": sorted(ALLOWED_RELATIONS),
                "allowed_sources_by_slot": allowed,
                "inventory_sha256": digest,
                "system_prompt": (
                    "You plan plausible complex-audio mixtures. Select one exact allow-listed "
                    "catalog_source_id for every required slot and choose its onset_sec on the "
                    "0.1-second grid. You may decide source combinations, order and overlap. "
                    "Never invent source IDs, captions, transcripts, lyrics, people, paths, "
                    "licenses or acoustic observations. Return one JSON object only."
                ),
            }
            task["user_prompt"] = (
                f"Scene duration: {scene_duration_sec:.1f}s. Required template: {template}. "
                f"Required slots: {task['required_slots']}. Allowed contexts: "
                f"{sorted(ALLOWED_CONTEXTS)}. Allowed relations: {sorted(ALLOWED_RELATIONS)}. "
                "For each slot choose one candidate below and use onset_sec <= its "
                "max_onset_sec. Do not reuse a catalog_source_id or leakage/source group. "
                "Return keys context, difficulty, source_plan, relations, rationale. "
                "Each source_plan item must contain slot_id, kind, catalog_source_id, "
                f"onset_sec. Candidate slates: {json.dumps(allowed, ensure_ascii=False)}"
            )
            task["task_sha256"] = _object_sha256(task)
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_source_timeline_task(
    task: dict[str, object], inventory: LabelInventory
) -> dict[str, list[dict[str, object]]]:
    if task.get("schema_version") != LLM_SOURCE_TIMELINE_TASK_SCHEMA_VERSION:
        raise ValueError(f"unsupported source-timeline task schema: {task.get('schema_version')}")
    expected_task_hash = str(task.get("task_sha256") or "")
    unhashed = {key: value for key, value in task.items() if key != "task_sha256"}
    observed_task_hash = _object_sha256(unhashed)
    if not expected_task_hash or expected_task_hash != observed_task_hash:
        raise ValueError(
            f"source-timeline task hash mismatch for {task.get('task_id')}: "
            f"expected={expected_task_hash!r} observed={observed_task_hash}"
        )
    if task.get("inventory_sha256") != inventory_sha256(inventory):
        raise ValueError(f"inventory hash mismatch for task {task.get('task_id')}")
    template = str(task.get("required_template") or "")
    if template not in SOURCE_TIMELINE_TEMPLATES:
        raise ValueError(f"unsupported source-timeline template in task: {template!r}")
    scene_duration = float(task.get("scene_duration_sec") or 0.0)
    if not 2.0 <= scene_duration <= 120.0 or abs(
        scene_duration * 10.0 - round(scene_duration * 10.0)
    ) > 1e-6:
        raise ValueError(f"invalid source-timeline scene duration: {scene_duration}")
    if float(task.get("time_resolution_sec") or 0.0) != 0.1:
        raise ValueError("source-timeline task resolution must be exactly 0.1 seconds")
    allowed_raw = task.get("allowed_sources_by_slot")
    if not isinstance(allowed_raw, dict):
        raise ValueError(f"allowed_sources_by_slot is invalid for {task.get('task_id')}")
    allowed: dict[str, list[dict[str, object]]] = {}
    for slot_id, values in allowed_raw.items():
        if not isinstance(values, list) or not values or any(
            not isinstance(item, dict) for item in values
        ):
            raise ValueError(f"invalid candidate slate for {task.get('task_id')}:{slot_id}")
        allowed[str(slot_id)] = [dict(item) for item in values]
    expected_slot_list = [
        str(item["slot_id"])
        for item in task.get("required_slots", [])
        if isinstance(item, dict) and item.get("slot_id")
    ]
    canonical_slots = _template_slots(template)
    if expected_slot_list != canonical_slots or set(allowed) != set(canonical_slots):
        raise ValueError(
            f"candidate slots do not match required slots for {task.get('task_id')}"
        )
    for slot_id, candidates in allowed.items():
        expected_kind = slot_id.rsplit("_", 1)[0]
        source_ids = [str(candidate.get("catalog_source_id") or "") for candidate in candidates]
        if any(not source_id for source_id in source_ids) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ValueError(f"candidate slate has empty or duplicate IDs for {slot_id}")
        if any(str(candidate.get("kind") or "") != expected_kind for candidate in candidates):
            raise ValueError(f"candidate slate kind mismatch for {slot_id}")
    return allowed


def _compile_source_timeline_recipe(
    *,
    task: dict[str, object],
    proposal: LLMSourceTimelineProposal,
    inventory: LabelInventory,
    records_by_id: dict[str, object],
    proposal_source: str,
    proposal_metadata: dict[str, object],
) -> SceneRecipe:
    allowed = _validate_source_timeline_task(task, inventory)
    required_slots = [str(item["slot_id"]) for item in task["required_slots"]]
    selected_by_slot = {item.slot_id: item for item in proposal.source_plan}
    if len(selected_by_slot) != len(proposal.source_plan) or set(selected_by_slot) != set(
        required_slots
    ):
        raise ValueError(
            f"source_plan must fill every slot exactly once for {task.get('task_id')}"
        )
    selected_records = []
    ordered_plan = []
    label_preferences: dict[str, list[str]] = {}
    selected_ids: set[str] = set()
    selected_group_tokens: set[str] = set()
    for slot_id in required_slots:
        selection = selected_by_slot[slot_id]
        candidates = {
            str(item["catalog_source_id"]): item for item in allowed[slot_id]
        }
        candidate = candidates.get(selection.catalog_source_id)
        if candidate is None:
            raise ValueError(
                f"LLM selected a source outside the frozen slate for "
                f"{task.get('task_id')}:{slot_id}: {selection.catalog_source_id}"
            )
        expected_kind = slot_id.rsplit("_", 1)[0]
        if selection.kind != expected_kind or candidate.get("kind") != expected_kind:
            raise ValueError(f"source kind mismatch for {task.get('task_id')}:{slot_id}")
        if selection.onset_sec > float(candidate["max_onset_sec"]) + 1e-6:
            raise ValueError(
                f"onset exceeds candidate limit for {task.get('task_id')}:{slot_id}: "
                f"{selection.onset_sec} > {candidate['max_onset_sec']}"
            )
        if selection.catalog_source_id in selected_ids:
            raise ValueError(
                f"source reused inside scene {task.get('task_id')}: "
                f"{selection.catalog_source_id}"
            )
        selected_ids.add(selection.catalog_source_id)
        group_tokens = {
            str(candidate.get("source_group") or ""),
            *(str(item) for item in candidate.get("leakage_groups", [])),
        } - {""}
        overlap = group_tokens & selected_group_tokens
        if overlap:
            raise ValueError(
                f"source/leakage group reused inside scene {task.get('task_id')}: "
                f"{sorted(overlap)}"
            )
        selected_group_tokens.update(group_tokens)
        record = records_by_id.get(selection.catalog_source_id)
        if record is None or record.kind != expected_kind:
            raise ValueError(
                f"selected source is absent or changed kind in frozen inventory: "
                f"{selection.catalog_source_id}"
            )
        expected_candidate = _candidate_payload(
            record, float(task["scene_duration_sec"])
        )
        evidence_fields = (
            "kind",
            "primary_label",
            "caption",
            "duration_sec",
            "max_onset_sec",
            "dataset",
            "split",
            "source_group",
            "leakage_groups",
            "identity",
            "annotation_origin",
            "file_sha256",
        )
        changed_fields = [
            field
            for field in evidence_fields
            if candidate.get(field) != expected_candidate.get(field)
        ]
        if changed_fields:
            raise ValueError(
                f"selected source evidence changed for {selection.catalog_source_id}: "
                f"{changed_fields}"
            )
        if expected_kind in {"sfx", "ambience"}:
            if not record.labels:
                raise ValueError(
                    f"selected {expected_kind} source lacks an audited primary label"
                )
            label_preferences.setdefault(expected_kind, []).append(record.labels[0])
        selected_records.append(record)
        ordered_plan.append(selection)
    if task["required_template"] == "overlapping_speakers":
        speech = [
            (item, record)
            for item, record in zip(ordered_plan, selected_records, strict=True)
            if item.kind == "speech"
        ]
        if len(speech) != 2:
            raise ValueError("overlapping_speakers requires exactly two planned speech sources")
        left_end = speech[0][0].onset_sec + float(speech[0][1].duration_sec)
        right_end = speech[1][0].onset_sec + float(speech[1][1].duration_sec)
        overlap_sec = min(left_end, right_end) - max(
            speech[0][0].onset_sec, speech[1][0].onset_sec
        )
        if overlap_sec < 0.1 - 1e-6:
            raise ValueError("overlapping_speakers plan has less than 0.1s overlap")
    return SceneRecipe(
        schema_version=SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION,
        recipe_id=str(task["recipe_id"]),
        seed=int(task["seed"]),
        template=str(task["required_template"]),
        context=proposal.context,
        difficulty=proposal.difficulty,
        label_preferences_by_kind=label_preferences,  # type: ignore[arg-type]
        relations=proposal.relations,
        rationale=proposal.rationale,
        proposal_source=proposal_source,
        scene_duration_sec=float(task["scene_duration_sec"]),
        source_plan=ordered_plan,
        proposal_metadata=proposal_metadata,
    )


def compile_llm_source_timeline_responses(
    tasks_path: str | Path,
    responses_path: str | Path,
    *,
    inventory: LabelInventory,
    output_path: str | Path,
    model_name: str,
) -> list[SceneRecipe]:
    task_rows = _jsonl(tasks_path)
    response_rows = _jsonl(responses_path)
    for artifact, rows in (("tasks", task_rows), ("responses", response_rows)):
        ids = [str(row.get("task_id") or "") for row in rows]
        duplicates = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
        if not all(ids) or duplicates:
            raise ValueError(
                f"LLM source-timeline {artifact} has empty or duplicate task IDs: "
                f"{duplicates[:10]}"
            )
    tasks = {str(row["task_id"]): row for row in task_rows}
    responses = {str(row["task_id"]): row for row in response_rows}
    missing = sorted(set(tasks) - set(responses))
    extra = sorted(set(responses) - set(tasks))
    if missing or extra:
        raise ValueError(
            f"LLM source-timeline task/response mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    records_by_id = {
        record.source_id: record for record in _records_from_inventory(inventory)
    }
    recipes = []
    for task_id, task in tasks.items():
        response_row = responses[task_id]
        raw = response_row.get("response")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM source-timeline response is not JSON for {task_id}: {exc}"
                ) from exc
        if isinstance(raw, dict) and isinstance(raw.get("recipe"), dict):
            raw = raw["recipe"]
        try:
            proposal = LLMSourceTimelineProposal.model_validate(raw)
        except Exception as exc:
            raise ValueError(
                f"invalid LLM source-timeline response for {task_id}: {exc}"
            ) from exc
        response_digest = _object_sha256(raw)
        request_metadata = response_row.get("request_metadata")
        if not isinstance(request_metadata, dict):
            raise ValueError(f"response request_metadata is required for {task_id}")
        response_task_hash = request_metadata.get("task_sha256")
        if response_task_hash != task["task_sha256"]:
            raise ValueError(f"response task hash mismatch for {task_id}")
        response_model = request_metadata.get("model")
        if response_model != model_name:
            raise ValueError(
                f"response model mismatch for {task_id}: "
                f"{response_model!r} != {model_name!r}"
            )
        safe_request_metadata = {
            key: value
            for key, value in request_metadata.items()
            if key in {"model", "temperature", "json_mode", "task_sha256"}
        }
        recipes.append(
            _compile_source_timeline_recipe(
                task=task,
                proposal=proposal,
                inventory=inventory,
                records_by_id=records_by_id,
                proposal_source=f"llm-source-timeline:{model_name}",
                proposal_metadata={
                    "task_id": task_id,
                    "task_sha256": task["task_sha256"],
                    "inventory_sha256": task["inventory_sha256"],
                    "response_sha256": response_digest,
                    "request": safe_request_metadata,
                },
            )
        )
    write_recipes(output_path, recipes)
    validate_recipes(recipes, inventory)
    return recipes


def _rule_context_for_slates(
    allowed: dict[str, list[dict[str, object]]], strategy: str
) -> str:
    if strategy == "uniform":
        return "generic"
    scores: dict[str, int] = {}
    for context, keywords in CONTEXT_KEYWORDS.items():
        score = 0
        for candidates in allowed.values():
            score += max(
                (
                    sum(
                        keyword in str(candidate.get("primary_label") or "").casefold()
                        for keyword in keywords
                    )
                    for candidate in candidates
                ),
                default=0,
            )
        scores[context] = score
    best = max(scores.values(), default=0)
    return sorted(context for context, score in scores.items() if score == best)[0] if best else "generic"


def _rule_candidate_order(
    candidates: list[dict[str, object]], *, context: str, strategy: str
) -> list[dict[str, object]]:
    if strategy == "uniform" or context == "generic":
        return candidates
    keywords = CONTEXT_KEYWORDS[context]
    return sorted(
        candidates,
        key=lambda candidate: (
            -sum(
                keyword in str(candidate.get("primary_label") or "").casefold()
                for keyword in keywords
            ),
            str(candidate["catalog_source_id"]),
        ),
    )


def _rule_source_combination(
    required_slots: list[str],
    allowed: dict[str, list[dict[str, object]]],
    *,
    context: str,
    strategy: str,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()

    def search(index: int) -> bool:
        if index == len(required_slots):
            return True
        slot_id = required_slots[index]
        for candidate in _rule_candidate_order(
            allowed[slot_id], context=context, strategy=strategy
        ):
            source_id = str(candidate["catalog_source_id"])
            group_tokens = {
                str(candidate.get("source_group") or ""),
                *(str(item) for item in candidate.get("leakage_groups", [])),
            } - {""}
            if source_id in selected_ids or group_tokens & selected_groups:
                continue
            selected.append(candidate)
            selected_ids.add(source_id)
            selected_groups.update(group_tokens)
            if search(index + 1):
                return True
            selected.pop()
            selected_ids.remove(source_id)
            selected_groups.difference_update(group_tokens)
        return False

    if not search(0):
        raise ValueError(
            "frozen source slate has no leakage-safe combination for slots "
            f"{required_slots}"
        )
    return selected


def generate_rule_source_timeline_recipes(
    tasks_path: str | Path,
    *,
    inventory: LabelInventory,
    output_path: str | Path,
    strategy: Literal["uniform", "keyword"] = "keyword",
) -> list[SceneRecipe]:
    """Create the matched non-LLM arm from the exact same frozen task slates."""

    records_by_id = {
        record.source_id: record for record in _records_from_inventory(inventory)
    }
    recipes = []
    for task in _jsonl(tasks_path):
        allowed = _validate_source_timeline_task(task, inventory)
        required_slots = [str(item["slot_id"]) for item in task["required_slots"]]
        context = _rule_context_for_slates(allowed, strategy)
        selected = _rule_source_combination(
            required_slots, allowed, context=context, strategy=strategy
        )
        plan = []
        foreground_index = 0
        for slot_id, candidate in zip(required_slots, selected, strict=True):
            kind = slot_id.rsplit("_", 1)[0]
            max_onset = float(candidate["max_onset_sec"])
            if kind in {"music", "ambience"}:
                onset = 0.0
            else:
                fractions = (0.15, 0.5, 0.8)
                fraction = fractions[min(foreground_index, len(fractions) - 1)]
                onset = math.floor(max_onset * fraction * 10.0 + 1e-9) / 10.0
                foreground_index += 1
            plan.append(
                SourceTimelineSelection(
                    slot_id=slot_id,
                    kind=kind,  # type: ignore[arg-type]
                    catalog_source_id=str(candidate["catalog_source_id"]),
                    onset_sec=round(onset, 1),
                )
            )
        if task["required_template"] == "overlapping_speakers":
            shared_onset = min(item.onset_sec for item in plan)
            plan = [item.model_copy(update={"onset_sec": shared_onset}) for item in plan]
        relations = ["overlap"]
        if any(item.kind in {"music", "ambience"} for item in plan):
            relations.append("foreground_over_background")
        proposal = LLMSourceTimelineProposal(
            context=context,
            difficulty="hard" if len(plan) >= 3 else "medium",
            source_plan=plan,
            relations=relations,
            rationale=(
                "Deterministic exact-source and staggered-timeline control from the "
                f"same frozen candidate slate using the {strategy} strategy."
            ),
        )
        recipes.append(
            _compile_source_timeline_recipe(
                task=task,
                proposal=proposal,
                inventory=inventory,
                records_by_id=records_by_id,
                proposal_source=f"rules:source-timeline-{strategy}_v1",
                proposal_metadata={
                    "task_id": task["task_id"],
                    "task_sha256": task["task_sha256"],
                    "inventory_sha256": task["inventory_sha256"],
                    "strategy": strategy,
                },
            )
        )
    write_recipes(output_path, recipes)
    validate_recipes(recipes, inventory)
    return recipes


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
    records_by_id = (
        {record.source_id: record for record in _records_from_inventory(inventory)}
        if any(recipe.source_plan for recipe in recipes)
        else {}
    )
    for recipe in recipes:
        for kind, labels in recipe.label_preferences_by_kind.items():
            unknown = sorted(set(labels) - known.get(kind, set()))
            if unknown:
                errors.append(
                    {"recipe_id": recipe.recipe_id, "kind": kind, "unknown_labels": unknown}
                )
        if recipe.source_plan:
            selected_ids: set[str] = set()
            selected_groups: set[str] = set()
            planned_labels: dict[str, list[str]] = {}
            for item in recipe.source_plan:
                record = records_by_id.get(item.catalog_source_id)
                if record is None:
                    errors.append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "slot_id": item.slot_id,
                            "unknown_catalog_source_id": item.catalog_source_id,
                        }
                    )
                    continue
                if record.kind != item.kind:
                    errors.append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "slot_id": item.slot_id,
                            "source_kind_mismatch": [record.kind, item.kind],
                        }
                    )
                if item.catalog_source_id in selected_ids:
                    errors.append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "duplicate_catalog_source_id": item.catalog_source_id,
                        }
                    )
                selected_ids.add(item.catalog_source_id)
                group_tokens = {record.source_group, *record.leakage_groups}
                overlap = group_tokens & selected_groups
                if overlap:
                    errors.append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "source_group_reuse": sorted(overlap),
                        }
                    )
                selected_groups.update(group_tokens)
                if (
                    recipe.scene_duration_sec is not None
                    and item.kind in {"speech", "vocal", "sfx"}
                    and record.duration_sec is not None
                    and item.onset_sec + float(record.duration_sec)
                    > recipe.scene_duration_sec + 1e-6
                ):
                    errors.append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "slot_id": item.slot_id,
                            "source_exceeds_scene": {
                                "onset_sec": item.onset_sec,
                                "duration_sec": record.duration_sec,
                                "scene_duration_sec": recipe.scene_duration_sec,
                            },
                        }
                    )
                if item.kind in {"sfx", "ambience"} and record.labels:
                    planned_labels.setdefault(item.kind, []).append(record.labels[0])
            for kind, labels in planned_labels.items():
                if recipe.label_preferences_by_kind.get(kind) != labels:
                    errors.append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "kind": kind,
                            "planned_label_mismatch": {
                                "source_plan": labels,
                                "recipe": recipe.label_preferences_by_kind.get(kind),
                            },
                        }
                    )
            metadata_inventory_hash = recipe.proposal_metadata.get("inventory_sha256")
            if metadata_inventory_hash and metadata_inventory_hash != inventory_sha256(
                inventory
            ):
                errors.append(
                    {
                        "recipe_id": recipe.recipe_id,
                        "proposal_inventory_hash_mismatch": metadata_inventory_hash,
                    }
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
    planned_sources = [
        item.catalog_source_id for recipe in recipes for item in recipe.source_plan
    ]
    onset_patterns = {
        tuple((item.slot_id, item.onset_sec) for item in recipe.source_plan)
        for recipe in recipes
        if recipe.source_plan
    }
    return {
        "n_recipes": len(recipes),
        "template_counts": dict(sorted(templates.items())),
        "context_counts": dict(sorted(contexts.items())),
        "unique_primary_labels": len(labels),
        "unique_recipe_combinations": len(combinations),
        "template_entropy_bits": round(entropy, 6),
        "n_source_timeline_recipes": sum(bool(recipe.source_plan) for recipe in recipes),
        "unique_planned_catalog_sources": len(set(planned_sources)),
        "unique_onset_patterns": len(onset_patterns),
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
    source_timeline_comparison = any(recipe.source_plan for recipe in [*left, *right])
    matched_source_timeline_mode = (
        all(bool(recipe.source_plan) for recipe in [*left, *right])
        if source_timeline_comparison
        else True
    )
    matched_scene_durations = [recipe.scene_duration_sec for recipe in left] == [
        recipe.scene_duration_sec for recipe in right
    ]
    matched_candidate_tasks = [
        recipe.proposal_metadata.get("task_sha256") for recipe in left
    ] == [recipe.proposal_metadata.get("task_sha256") for recipe in right]
    checks = [
        {"name": "matched_count", "pass": matched_count, "detail": [len(left), len(right)]},
        {"name": "matched_seeds", "pass": matched_seeds, "detail": None},
        {"name": "matched_templates", "pass": matched_templates, "detail": None},
        {
            "name": "matched_source_timeline_mode",
            "pass": matched_source_timeline_mode,
            "detail": None,
        },
        {
            "name": "matched_scene_durations",
            "pass": matched_scene_durations,
            "detail": None,
        },
        {
            "name": "matched_candidate_tasks",
            "pass": matched_candidate_tasks,
            "detail": None,
        },
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
                "scene_duration_sec",
                "source_plan_json",
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
                    recipe.scene_duration_sec if recipe.scene_duration_sec is not None else "",
                    json.dumps(
                        [item.model_dump() for item in recipe.source_plan],
                        ensure_ascii=False,
                    ),
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
            if expected.source_plan or "scene_duration_sec" in row:
                observed_duration = str(row.get("scene_duration_sec") or "")
                expected_duration = (
                    str(expected.scene_duration_sec)
                    if expected.scene_duration_sec is not None
                    else ""
                )
                if observed_duration != expected_duration:
                    immutable_mismatches.append(
                        {
                            "recipe_id": recipe_id,
                            "field": "scene_duration_sec",
                            "observed": observed_duration,
                            "expected": expected_duration,
                        }
                    )
            json_fields = [
                ("labels_json", expected.label_preferences_by_kind),
                ("relations_json", expected.relations),
            ]
            if expected.source_plan or "source_plan_json" in row:
                json_fields.insert(
                    0,
                    (
                        "source_plan_json",
                        [item.model_dump() for item in expected.source_plan],
                    ),
                )
            for field, expected_value in json_fields:
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
    "LLM_SOURCE_TIMELINE_TASK_SCHEMA_VERSION",
    "LLM_TASK_SCHEMA_VERSION",
    "RECIPE_SCHEMA_VERSION",
    "SOURCE_TIMELINE_RECIPE_SCHEMA_VERSION",
    "SOURCE_TIMELINE_TEMPLATES",
    "TEMPLATE_KIND_COUNTS",
    "LLMSourceTimelineProposal",
    "LabelInventory",
    "SceneRecipe",
    "SourceTimelineSelection",
    "build_label_inventory",
    "compare_recipe_sets",
    "compile_llm_responses",
    "compile_llm_source_timeline_responses",
    "export_llm_tasks",
    "export_llm_source_timeline_tasks",
    "generate_rule_recipes",
    "generate_rule_source_timeline_recipes",
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
